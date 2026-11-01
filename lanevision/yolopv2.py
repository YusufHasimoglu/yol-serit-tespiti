"""YOLOPv2 ile şerit boyası ve sürülebilir-yol maskesi çıkarımı.

YOLOPv2, BDD100K üzerinde eğitilmiş çok görevli bir yol algılayıcıdır.  Bu
sarmalayıcı yalnızca TorchScript ağırlığını kullanır; proje içine üçüncü taraf
eğitim kodu veya CUDA eklentisi getirmez.  Model maskeleri, mevcut fiziksel
boya/şerit-geometri doğrulamasına kılavuzluk eder.  Böylece model tek başına
görselde olmayan bir çizgiyi şerit diye gösteremez.
"""
from __future__ import annotations

import threading
import warnings
from pathlib import Path

import cv2
import numpy as np

from .continuity import detect_continuous
from .core import Detection, LaneSettings


APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YOLOPV2_WEIGHTS = APP_ROOT / "modeller" / "yolopv2_bdd100k.pt"
# This compact four-class correction model is trained only from the user's
# confirmed drawings.  The large YOLOPv2 backbone remains the official,
# immutable TorchScript model; the feedback layer adapts its rendering to
# local yellow paint, worn markings and dashed/solid styles.
DEFAULT_YOLOPV2_FEEDBACK_WEIGHTS = APP_ROOT / "modeller" / "yolopv2_feedback_unet.pt"
_INPUT_WIDTH = 640
_INPUT_HEIGHT = 384
_VERTICAL_PAD = 12
_MODEL_CACHE: dict[Path, object] = {}
_MODEL_LOCK = threading.RLock()


def yolopv2_available(weights: str | Path = DEFAULT_YOLOPV2_WEIGHTS) -> bool:
    """YOLOPv2 TorchScript ağırlığının bu bilgisayarda hazır olup olmadığını döndür."""
    return Path(weights).is_file()


def yolopv2_feedback_available(weights: str | Path = DEFAULT_YOLOPV2_FEEDBACK_WEIGHTS) -> bool:
    """Kullanıcı çizimleriyle eğitilmiş yerel düzeltici ağırlığı var mı?"""
    return Path(weights).is_file()


def prepare_yolopv2_input(frame: np.ndarray) -> np.ndarray:
    """Resmî YOLOPv2 ön işleme boyutunda RGB NCHW tensör verisi hazırla."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 100:
        raise ValueError("BGR formatında, en az 100x100 piksellik görüntü gereklidir.")
    # Official BDD100K inference resizes to 1280x720, then letterboxes to
    # 640x384.  Keeping exactly this geometry avoids a train/inference aspect
    # mismatch on portrait input images.
    base = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
    resized = cv2.resize(base, (_INPUT_WIDTH, 360), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, _VERTICAL_PAD, _VERTICAL_PAD, 0, 0,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    return rgb.transpose(2, 0, 1)[None]


def _mask_from_outputs(output: object, original_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """TorchScript YOLOPv2 çıktısını kaynak boyutlu yol ve şerit maskesine çevir."""
    if not isinstance(output, (tuple, list)) or len(output) < 3:
        raise RuntimeError("YOLOPv2 çıktısı beklenen üç görev başlığını içermiyor.")
    try:
        road_tensor, lane_tensor = output[1], output[2]
        road = road_tensor.detach().float().cpu().numpy()
        lane = lane_tensor.detach().float().cpu().numpy()
    except (AttributeError, TypeError) as exc:
        raise RuntimeError("YOLOPv2 maske çıktıları tensör biçiminde değil.") from exc
    if road.ndim != 4 or lane.ndim != 4 or road.shape[0] < 1 or lane.shape[0] < 1:
        raise RuntimeError("YOLOPv2 maske çıktılarının boyutu desteklenmiyor.")
    # The model emits 384x640 masks.  Remove the 12-pixel letterbox bars
    # before scaling back to its native 1280x720 coordinate system.
    road_crop = road[0, :, _VERTICAL_PAD:-_VERTICAL_PAD, :]
    lane_crop = lane[0, :, _VERTICAL_PAD:-_VERTICAL_PAD, :]
    if road_crop.shape[0] < 2:
        raise RuntimeError("YOLOPv2 sürülebilir-yol maskesi iki sınıf içermiyor.")
    road_mask = np.argmax(road_crop, axis=0).astype(np.uint8) * 255
    if lane_crop.shape[0] == 1:
        lane_mask = (lane_crop[0] >= .5).astype(np.uint8) * 255
    else:
        lane_mask = np.argmax(lane_crop, axis=0).astype(np.uint8) * 255
    road_mask = cv2.resize(road_mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
    lane_mask = cv2.resize(lane_mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
    height, width = original_shape
    return (cv2.resize(lane_mask, (width, height), interpolation=cv2.INTER_NEAREST),
            cv2.resize(road_mask, (width, height), interpolation=cv2.INTER_NEAREST))


def _mask_visual(frame: np.ndarray, lane_mask: np.ndarray, road_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arayüzde incelenebilen ayrı şerit ve sürülebilir-alan görsellerini üret."""
    lane_view = np.zeros_like(frame)
    lane_view[lane_mask > 0] = (0, 225, 255)
    road_view = frame.copy()
    green = np.zeros_like(frame)
    green[road_mask > 0] = (64, 176, 96)
    road_view = cv2.addWeighted(road_view, 1., green, .42, 0.)
    return lane_view, road_view


class YOLOPv2Detector:
    """CPU'da TorchScript olarak çalışan YOLOPv2 şerit/yol sarmalayıcısı."""

    def __init__(self, weights: str | Path = DEFAULT_YOLOPV2_WEIGHTS,
                 feedback_weights: str | Path = DEFAULT_YOLOPV2_FEEDBACK_WEIGHTS):
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"YOLOPv2 modeli bulunamadı: {self.weights}")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("YOLOPv2 için PyTorch kurulu değil.") from exc
        self.torch = torch
        with _MODEL_LOCK:
            model = _MODEL_CACHE.get(self.weights)
            if model is None:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="`torch.jit.load` is deprecated")
                        model = torch.jit.load(str(self.weights), map_location="cpu").eval()
                except (OSError, RuntimeError, ValueError) as exc:
                    raise RuntimeError(f"YOLOPv2 modeli açılamadı: {exc}") from exc
                _MODEL_CACHE[self.weights] = model
        self.model = model
        self.feedback = None
        self.feedback_warning = None
        feedback_path = Path(feedback_weights)
        if feedback_path.is_file():
            try:
                from .ai_model import AISegmenter
                feedback = AISegmenter(feedback_path, threshold=.42)
                if feedback.output_channels != 4:
                    raise ValueError("geri bildirim modeli dört düz/kesik kanalı içermiyor")
                self.feedback = feedback
            except (ImportError, FileNotFoundError, RuntimeError, ValueError) as exc:
                # The primary YOLOPv2 model remains usable if a locally trained
                # experimental checkpoint is incomplete or incompatible.
                self.feedback_warning = str(exc)

    def masks(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prepared = prepare_yolopv2_input(frame)
        tensor = self.torch.from_numpy(prepared)
        try:
            with _MODEL_LOCK, self.torch.inference_mode():
                output = self.model(tensor)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f"YOLOPv2 çıkarımı başarısız oldu: {exc}") from exc
        return _mask_from_outputs(output, frame.shape[:2])

    def detect(self, frame: np.ndarray, settings: LaneSettings | None = None) -> Detection:
        settings = settings or LaneSettings()
        settings.validate()
        lane_mask, road_mask = self.masks(frame)
        feedback_probabilities = None
        if self.feedback is not None:
            feedback_probabilities = self.feedback.probabilities(frame)
        result = detect_continuous(
            frame, settings, lane_evidence=lane_mask, road_evidence=road_mask,
            probabilities=feedback_probabilities,
            channel_layout=(self.feedback.channel_layout if self.feedback is not None else None),
        )
        lane_view, road_view = _mask_visual(frame, lane_mask, road_mask)
        result.stages.update({
            "09 YOLOPv2 şerit maskesi": lane_view,
            "10 YOLOPv2 sürülebilir alan": road_view,
        })
        if feedback_probabilities is not None:
            feedback_view = np.zeros_like(frame)
            solid = np.maximum(feedback_probabilities[:, :, 0], feedback_probabilities[:, :, 2])
            dashed = np.maximum(feedback_probabilities[:, :, 1], feedback_probabilities[:, :, 3])
            feedback_view[:, :, 0] = (solid * 255).astype(np.uint8)
            feedback_view[:, :, 1] = np.maximum((solid * 255).astype(np.uint8),
                                                 (dashed * 127).astype(np.uint8))
            feedback_view[:, :, 2] = (dashed * 255).astype(np.uint8)
            result.stages["11 Kullanıcı eğitimi düzeltmesi"] = feedback_view
        suffix = "maske ile doğrulandı" if getattr(result, "ai_evidence_used", False) else "fiziksel boya yedeği"
        feedback_note = " + kullanıcı eğitimi" if self.feedback is not None else ""
        result.method = f"Yapay zekâ (YOLOPv2 şerit + alan, {suffix}{feedback_note})"
        result.yolopv2_lane_mask = lane_mask
        result.yolopv2_road_mask = road_mask
        return result
