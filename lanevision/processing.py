"""Giriş/çıkış, video kodlama, istatistik kaydı ve Unicode dosya yolu desteği."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .core import Detection, LaneSettings, settings_dict
from .engine import InferenceEngine

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")
    # Türkçe karakter içeren Windows yollarını cv2.imread'den daha iyi destekler.
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        raise ValueError("Görüntü çıktı uzantısı .png veya .jpg olmalıdır.")
    ok, buf = cv2.imencode(".jpg" if ext == ".jpeg" else ext, image)
    if not ok:
        raise OSError(f"Görüntü kodlanamadı: {path}")
    buf.tofile(str(path))
    return path


def save_stages(stages: dict[str, np.ndarray], folder: str | Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, image in stages.items():
        write_image(folder / f"{name.replace(' ', '_')}.png", image)
    return folder


def process_image(source: str | Path, output: str | Path, settings: LaneSettings | None = None,
                  stages_dir: str | Path | None = None, mode: str = "auto") -> Detection:
    if Path(source).resolve() == Path(output).resolve():
        raise ValueError("Giriş ve çıktı dosyası farklı olmalıdır.")
    detection = InferenceEngine(settings,mode=mode).detect(read_image(source))
    write_image(output, detection.frame)
    if stages_dir is not None:
        save_stages(detection.stages, stages_dir)
    return detection


@dataclass
class VideoSummary:
    output_path: str
    frames: int
    detected_both: int
    detected_left: int
    detected_right: int
    fps: float
    cancelled: bool

    def to_dict(self) -> dict:
        return {
            "cikti": self.output_path,
            "islenen_kare": self.frames,
            "iki_serit_tespit_kare": self.detected_both,
            "sol_serit_tespit_kare": self.detected_left,
            "sag_serit_tespit_kare": self.detected_right,
            "iki_serit_tespit_orani_yuzde": round(100 * self.detected_both / self.frames, 2) if self.frames else 0,
            "fps_kaynak": round(self.fps, 2),
            "iptal_edildi": self.cancelled,
        }


def process_video(source: str | Path, output: str | Path, settings: LaneSettings | None = None,
                  progress: Callable[[int, int, Detection], None] | None = None,
                  cancelled: Callable[[], bool] | None = None,
                  report: bool = True, mode: str = "auto") -> VideoSummary:
    """Uzun videoları kare kare işle; tüm videoyu belleğe alma.

    progress callback her 5 karede çağrılır. Her çağrının görüntüsü geçicidir.
    Sonuç MP4 (mp4v) kodlanır; ses aktarılmaz.
    """
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.suffix.lower() != ".mp4":
        raise ValueError("Video çıktı dosyası .mp4 olmalıdır.")
    if source.resolve() == output.resolve():
        raise ValueError("Girdi ve çıktı dosyası farklı olmalıdır.")
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Video açılamadı: {source}")
    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        first_ok, first = cap.read()
        if not first_ok or first is None:
            raise ValueError("Video boş veya ilk kare okunamadı.")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise OSError("MP4 video yazıcısı başlatılamadı. Codec desteklerini kontrol edin.")
        detector = InferenceEngine(settings,mode=mode)
        rows = []
        frames, n_both, n_left, n_right = 0, 0, 0, 0
        is_cancelled = False
        current = first
        while True:
            if cancelled and cancelled():
                is_cancelled = True
                break
            found = detector.detect(current, video=True)
            writer.write(found.frame)
            frames += 1
            n_both += int(found.both_found)
            n_left += int(found.left is not None)
            n_right += int(found.right is not None)
            rows.append({"kare": frames, "saniye": round((frames - 1) / fps, 3), **found.stats()})
            if progress and (frames == 1 or frames % 5 == 0 or (total and frames == total)):
                progress(frames, total, found)
            ok, current = cap.read()
            if not ok:
                break
        summary = VideoSummary(str(output), frames, n_both, n_left, n_right, fps, is_cancelled)
        if report:
            csv_path = output.with_name(output.stem + "_istatistik.csv")
            json_path = output.with_name(output.stem + "_ozet.json")
            with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
                csv_writer = csv.DictWriter(stream, fieldnames=rows[0].keys() if rows else ["kare"])
                csv_writer.writeheader()
                csv_writer.writerows(rows)
            with json_path.open("w", encoding="utf-8") as stream:
                json.dump({"video": summary.to_dict(), "parametreler": settings_dict(detector.settings), "yontem": mode},
                          stream, ensure_ascii=False, indent=2)
        return summary
    finally:
        cap.release()
        if writer is not None:
            writer.release()