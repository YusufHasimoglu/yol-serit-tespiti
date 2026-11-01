"""UFLD V2 ONNX ile derin şerit algılama.

Bu modül, UFLD V2'nin satır-çapa (row-anchor) çıktısını uygulamanın
``Detection`` biçimine dönüştürür.  Model, şerit pikseli segmentasyonu yerine
bir dizi anahtar nokta ürettiğinden kesikli boyadaki boşluklarda da eğriyi
kararlı biçimde sürdürebilir.  Çizginin düz/kesik türü modelin çıktısında
bulunmadığı için burada bilinçli olarak ``belirsiz`` bırakılır; kullanıcı
etiketi ve U-Net bu bilgiyi sağlamaya devam eder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .core import Detection, LaneSettings


APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UFLD_WEIGHTS = APP_ROOT / "modeller" / "ufldv2_lane_detection.onnx"
_INPUT_WIDTH = 1600
_INPUT_HEIGHT = 320
_CROP_RATIO = 0.60
_RGB_MEAN = np.asarray((0.485, 0.456, 0.406), np.float32)
_RGB_STD = np.asarray((0.229, 0.224, 0.225), np.float32)


def ufld_available(weights: str | Path = DEFAULT_UFLD_WEIGHTS) -> bool:
    """Yerel ONNX ağırlığının kullanıcı için gerçekten hazır olup olmadığını döndür."""
    return Path(weights).is_file()


def _soft_argmax(values: np.ndarray, radius: int = 1) -> float:
    """En güçlü ızgara hücresi çevresinde kararlı alt-piksel x hesaplar."""
    peak = int(np.argmax(values))
    lo, hi = max(0, peak - radius), min(len(values), peak + radius + 1)
    indexes = np.arange(lo, hi, dtype=np.float32)
    selected = values[lo:hi].astype(np.float32)
    selected -= float(np.max(selected))
    weights = np.exp(selected)
    return float(np.sum(indexes * weights) / max(float(np.sum(weights)), 1e-8) + .5)


def _row_candidates(loc_row: np.ndarray, exist_row: np.ndarray, width: int, height: int) -> list[dict]:
    """UFLD'nin satır çapalarını, her şerit için kaynak-boyutlu noktalara çevir.

    ``loc_row`` biçimi ``[izgara, satır, şerit]``, ``exist_row`` ise
    ``[2, satır, şerit]`` olur.  UFLD V2 dışa aktarımlarında satır sayısı
    değişebildiğinden çapalar sabit bir sayı yerine çıktının uzunluğundan
    üretilir.
    """
    if loc_row.ndim != 3 or exist_row.ndim != 3 or loc_row.shape[1:] != exist_row.shape[1:]:
        raise ValueError("UFLD satır çıktısının boyutu beklenen biçimde değil.")
    grids, rows, lanes = loc_row.shape
    # UFLD CULane ön işlemi görüntünün alt %60'ını kullandığı için satır
    # çapaları kaynağın 0.42..1.00 bölgesine yayılır.
    anchors = np.linspace(.42, 1., rows, dtype=np.float32)
    active = np.argmax(exist_row, axis=0).astype(bool)
    candidates: list[dict] = []
    for lane in range(lanes):
        indices = np.flatnonzero(active[:, lane])
        # Tek/iki noktadan üretilen eğri görüntüde gerçek çizgi gibi
        # gösterilemeyeceği için en az dört gözlem istenir.
        if len(indices) < max(4, int(rows * .16)):
            continue
        points = []
        for row in indices:
            grid_position = _soft_argmax(loc_row[:, row, lane])
            x = grid_position / max(1, grids - 1) * width
            y = float(anchors[row] * height)
            if -width * .05 <= x <= width * 1.05:
                points.append((x, y))
        if len(points) < 4:
            continue
        points_array = np.asarray(points, np.float32)
        # Uzak tekil noktaları iki geçişte ayıkla; bazı ONNX dışa
        # aktarımlarında TTA dalı zayıf uç nokta tahminleri verebiliyor.
        degree = 2 if len(points_array) >= 8 else 1
        try:
            coefficients = np.polyfit(points_array[:, 1], points_array[:, 0], degree)
            for _ in range(2):
                residual = np.abs(points_array[:, 0] - np.polyval(coefficients, points_array[:, 1]))
                keep = residual <= max(18., float(np.median(residual)) * 2.7 + 3.)
                if int(np.count_nonzero(keep)) < max(4, degree + 2):
                    break
                points_array = points_array[keep]
                coefficients = np.polyfit(points_array[:, 1], points_array[:, 0], degree)
        except (ValueError, np.linalg.LinAlgError):
            continue
        low, high = float(points_array[:, 1].min()), float(points_array[:, 1].max())
        if high - low < height * .12:
            continue
        ys = np.linspace(low, high, max(16, int((high - low) / 4)))
        xs = np.polyval(coefficients, ys)
        valid = (xs >= -width * .08) & (xs <= width * 1.08)
        curve = np.column_stack([xs[valid], ys[valid]]).round().astype(np.int32)
        if len(curve) < 8:
            continue
        candidates.append({
            "curve": curve,
            "coeffs": coefficients,
            "support": int(len(points_array)),
            "source_lane": lane,
        })
    return candidates


def _pair_score(left: dict, right: dict, width: int, height: int) -> float | None:
    """Kameranın baktığı ego şeridini saran fiziksel çifti seç."""
    low = max(float(left["curve"][0, 1]), float(right["curve"][0, 1]))
    high = min(float(left["curve"][-1, 1]), float(right["curve"][-1, 1]))
    if high - low < height * .16:
        return None
    ys = np.linspace(low, high, 15)
    left_x = np.polyval(left["coeffs"], ys)
    right_x = np.polyval(right["coeffs"], ys)
    lane_width = right_x - left_x
    if np.count_nonzero(lane_width > width * .035) < len(lane_width) * .90:
        return None
    bottom_width = float(lane_width[-1]) / width
    middle_width = float(np.median(lane_width)) / width
    if not .08 <= bottom_width <= 1.03 or not .06 <= middle_width <= .92:
        return None
    center = float((left_x[-1] + right_x[-1]) * .5) / width
    # Kamera merkezini saran, destek sayısı çok ve makul genişlikte olan
    # çifti seç.  Böylece dört şeritli yollarda yol dışı bariyer yerine
    # kullanıcının bulunduğu şeride en yakın iki sınır tercih edilir.
    center_score = max(0., 1. - abs(center - .5) / .62)
    width_score = max(0., 1. - abs(bottom_width - .44) / .78)
    support_score = min(1., (left["support"] + right["support"]) / 58.)
    return support_score * .55 + center_score * .30 + width_score * .15


def choose_ego_pair(candidates: Iterable[dict], width: int, height: int) -> tuple[dict, dict, float] | None:
    """Aday eğriler arasından sol/sağ ego-şerit sınırlarını döndür."""
    ordered = sorted(candidates, key=lambda item: float(np.polyval(item["coeffs"], height * .94)))
    best: tuple[dict, dict, float] | None = None
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1:]:
            score = _pair_score(left, right, width, height)
            if score is not None and (best is None or score > best[2]):
                best = left, right, score
    return best


def _draw_result(frame: np.ndarray, left: dict | None, right: dict | None, settings: LaneSettings,
                 confidence: float, points: list[np.ndarray]) -> Detection:
    """Seçilen eğrileri uygulamanın standart tarama ve durum görünümüne çiz."""
    h, w = frame.shape[:2]
    output = frame.copy()
    overlay = np.zeros_like(frame)
    left_curve = left["curve"] if left else None
    right_curve = right["curve"] if right else None
    offset = None
    if left_curve is not None and right_curve is not None:
        low = max(int(left_curve[0, 1]), int(right_curve[0, 1]))
        high = min(int(left_curve[-1, 1]), int(right_curve[-1, 1]))
        if high > low + 12:
            ys = np.arange(low, high + 1, 3, dtype=np.float32)
            left_x = np.polyval(left["coeffs"], ys)
            right_x = np.polyval(right["coeffs"], ys)
            valid = (left_x < right_x) & (left_x >= 0) & (right_x < w)
            if int(np.count_nonzero(valid)) >= 5:
                polygon = np.vstack([
                    np.column_stack([left_x[valid], ys[valid]]),
                    np.column_stack([right_x[valid], ys[valid]])[::-1],
                ]).round().astype(np.int32)
                cv2.fillPoly(overlay, [polygon], (68, 182, 97))
                offset = w / 2 - float((left_x[valid][-1] + right_x[valid][-1]) * .5)
    output = cv2.addWeighted(output, 1., overlay, settings.overlay_alpha, 0.)
    for curve in (left_curve, right_curve):
        if curve is not None:
            cv2.polylines(output, [curve], False, (0, 225, 255), settings.line_thickness, cv2.LINE_AA)
    label_left, label_right = ("VAR" if left_curve is not None else "YOK"), ("VAR" if right_curve is not None else "YOK")
    cv2.rectangle(output, (8, 7), (min(w - 5, 520), 47), (18, 28, 44), -1)
    cv2.putText(output, f"SOL: {label_left}  SAG: {label_right} | UFLD V2", (17, 34),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (242, 249, 255), 2, cv2.LINE_AA)
    keypoints = frame.copy()
    for curve in points:
        for x, y in curve:
            cv2.circle(keypoints, (int(x), int(y)), 3, (0, 225, 255), -1, cv2.LINE_AA)
    def ends(curve: np.ndarray | None):
        return (tuple(map(int, curve[-1])), tuple(map(int, curve[0]))) if curve is not None else None
    det = Detection(output, {"01 Orijinal": frame.copy(), "08 Sonuc": output,
                             "11 UFLD V2 anahtar noktalari": keypoints},
                    ends(left_curve), ends(right_curve), offset, confidence,
                    sum(len(curve) for curve in points))
    det.method = "Yapay zekâ (UFLD V2)"
    det.lane_polynomials = ((left_curve, left["coeffs"]) if left else None,
                            (right_curve, right["coeffs"]) if right else None)
    det.marking_types = ("belirsiz" if left else None, "belirsiz" if right else None)
    return det


class UFLDDetector:
    """OpenCV DNN üstünde çalışan, bağımsız UFLD V2 ONNX sarmalayıcısı."""

    def __init__(self, weights: str | Path = DEFAULT_UFLD_WEIGHTS):
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"UFLD V2 ONNX modeli bulunamadı: {self.weights}")
        try:
            self.net = cv2.dnn.readNetFromONNX(str(self.weights))
            self.output_names = tuple(self.net.getUnconnectedOutLayersNames())
        except cv2.error as exc:
            raise RuntimeError(f"UFLD V2 ONNX modeli açılamadı: {exc}") from exc
        expected = {"loc_row", "exist_row"}
        if not expected.issubset(self.output_names):
            raise RuntimeError("UFLD V2 ONNX çıktıları eksik; loc_row ve exist_row bekleniyordu.")

    @staticmethod
    def _blob(frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 100:
            raise ValueError("BGR formatında, en az 100x100 piksellik görüntü gereklidir.")
        resized_height = int(round(_INPUT_HEIGHT / _CROP_RATIO))
        resized = cv2.resize(frame, (_INPUT_WIDTH, resized_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized[-_INPUT_HEIGHT:], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        normalized = (rgb - _RGB_MEAN) / _RGB_STD
        return normalized.transpose(2, 0, 1)[None].astype(np.float32)

    def detect(self, frame: np.ndarray, settings: LaneSettings | None = None) -> Detection:
        settings = settings or LaneSettings()
        settings.validate()
        blob = self._blob(frame)
        try:
            self.net.setInput(blob)
            output = dict(zip(self.output_names, self.net.forward(list(self.output_names))))
        except cv2.error as exc:
            raise RuntimeError(f"UFLD V2 çıkarımı başarısız oldu: {exc}") from exc
        h, w = frame.shape[:2]
        loc_row, exist_row = output["loc_row"], output["exist_row"]
        if loc_row.ndim != 4 or exist_row.ndim != 4 or loc_row.shape[0] != exist_row.shape[0]:
            raise RuntimeError("UFLD V2 ONNX çıktılarının biçimi desteklenmiyor.")
        # Bu ONNX dışa aktarımı giriş başına iki TTA dalı döndürür. Her dalı
        # ayrı değerlendirip fiziksel ego-şerit geometrisi en güçlü olanı al.
        best: tuple[dict, dict, float, list[dict]] | None = None
        all_points: list[np.ndarray] = []
        for branch in range(loc_row.shape[0]):
            candidates = _row_candidates(loc_row[branch], exist_row[branch], w, h)
            pair = choose_ego_pair(candidates, w, h)
            if pair is not None:
                left, right, score = pair
                if best is None or score > best[2]:
                    best = left, right, score, candidates
        if best is None:
            return _draw_result(frame, None, None, settings, 0., all_points)
        left, right, score, candidates = best
        all_points = [item["curve"] for item in candidates]
        return _draw_result(frame, left, right, settings, score, all_points)
