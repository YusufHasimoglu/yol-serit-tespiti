"""Açıklanabilir, Canny + Hough tabanlı şerit çizgisi tespiti.

Algoritma örnekleri eğitim amaçlıdır; araç kontrolünde kullanılamaz.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class LaneSettings:
    canny_low: int = 55
    canny_high: int = 160
    blur_kernel: int = 5
    roi_top: float = 0.59
    roi_left: float = 0.06
    roi_right: float = 0.94
    roi_top_left: float = 0.42
    roi_top_right: float = 0.58
    hough_threshold: int = 28
    min_line_length: int = 24
    max_line_gap: int = 36
    min_abs_slope: float = 0.30
    max_abs_slope: float = 8.0
    overlay_alpha: float = 0.34
    line_thickness: int = 8
    smoothing: float = 0.60  # Videoda önceki karenin ağırlığı

    def validate(self) -> None:
        if not 0 <= self.canny_low < self.canny_high <= 255:
            raise ValueError("Canny alt eşik, üst eşikten küçük olmalıdır (0..255).")
        if self.blur_kernel < 1 or self.blur_kernel % 2 != 1:
            raise ValueError("Bulanıklaştırma çekirdeği pozitif tek sayı olmalıdır.")
        if not 0.30 <= self.roi_top < 0.85:
            raise ValueError("ROI yüksekliği 0.30 ile 0.85 arasında olmalıdır.")
        if not (0 <= self.roi_left < self.roi_top_left < self.roi_top_right < self.roi_right <= 1):
            raise ValueError("ROI yatay sınırları artan sırada olmalıdır.")
        if self.hough_threshold < 1 or self.min_line_length < 1 or self.max_line_gap < 0:
            raise ValueError("Hough parametreleri geçersiz.")
        if self.min_abs_slope <= 0 or self.max_abs_slope <= self.min_abs_slope:
            raise ValueError("Eğim aralığı geçersiz.")
        if not 0 <= self.overlay_alpha <= 1 or not 0 <= self.smoothing < 1:
            raise ValueError("Görünürlük / yumuşatma ayarı geçersiz.")
        if self.line_thickness < 1:
            raise ValueError("Çizgi kalınlığı en az 1 olmalıdır.")


@dataclass
class Detection:
    frame: np.ndarray
    stages: dict[str, np.ndarray]
    left: Optional[tuple[tuple[int, int], tuple[int, int]]]
    right: Optional[tuple[tuple[int, int], tuple[int, int]]]
    offset_px: Optional[float]
    confidence: float
    segment_count: int
    method: str = "Klasik Hough"

    @property
    def both_found(self) -> bool:
        return self.left is not None and self.right is not None

    def stats(self) -> dict:
        return {
            "sol_serit": self.left is not None,
            "sag_serit": self.right is not None,
            "iki_serit": self.both_found,
            "merkez_sapmasi_piksel": round(self.offset_px, 2) if self.offset_px is not None else None,
            "gorsel_destek_skoru": round(self.confidence, 3),
            "hough_segment_sayisi": self.segment_count,
            "yontem": self.method,
        }


def _fit(points: list[tuple[int, int, float]], y_bottom: int, y_top: int, width: int):
    if len(points) < 4:
        return None
    xx = np.asarray([p[0] for p in points], dtype=np.float64)
    yy = np.asarray([p[1] for p in points], dtype=np.float64)
    if np.ptp(yy) < 14:
        return None
    weights = np.sqrt(np.asarray([p[2] for p in points], dtype=np.float64))
    try:
        slope, intercept = np.polyfit(yy, xx, 1, w=weights)
    except (ValueError, np.linalg.LinAlgError):
        return None
    xb, xt = slope * y_bottom + intercept, slope * y_top + intercept
    if not (-width * 0.25 <= xb <= width * 1.25 and -width * 0.25 <= xt <= width * 1.25):
        return None
    return ((int(round(xb)), y_bottom), (int(round(xt)), y_top))


def _smooth_line(current, previous, factor: float):
    if current is None:
        return None
    if previous is None or factor <= 0:
        return current
    return tuple(tuple(int(round((1 - factor) * a + factor * b)) for a, b in zip(p, q))
                 for p, q in zip(current, previous))


class LaneDetector:
    def __init__(self, settings: LaneSettings | None = None):
        self.settings = settings or LaneSettings()
        self.settings.validate()
        self.previous_left = None
        self.previous_right = None

    def reset(self):
        self.previous_left = None
        self.previous_right = None

    def detect(self, frame: np.ndarray, video: bool = False) -> Detection:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 100:
            raise ValueError("BGR formatında, en az 100x100 piksellik görüntü gereklidir.")
        s = self.settings
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (s.blur_kernel, s.blur_kernel), 0)
        edges = cv2.Canny(blurred, s.canny_low, s.canny_high)
        mask = np.zeros_like(edges)
        y_top, y_bottom = int(h * s.roi_top), h - 1
        polygon = np.array([[ (int(w * s.roi_left), y_bottom),
                              (int(w * s.roi_top_left), y_top),
                              (int(w * s.roi_top_right), y_top),
                              (int(w * s.roi_right), y_bottom) ]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        roi = cv2.bitwise_and(edges, mask)
        segments = cv2.HoughLinesP(roi, 1, np.pi / 180, s.hough_threshold,
                                   minLineLength=s.min_line_length,
                                   maxLineGap=s.max_line_gap)
        picks: dict[str, list[tuple[int, int, float]]] = {"left": [], "right": []}
        segment_debug = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        n_segments = 0
        if segments is not None:
            for seg in segments[:, 0]:
                x1, y1, x2, y2 = map(int, seg)
                dx, dy = x2 - x1, y2 - y1
                if abs(dx) < 2:
                    continue
                slope = dy / dx
                if not s.min_abs_slope <= abs(slope) <= s.max_abs_slope:
                    continue
                center_x = (x1 + x2) / 2
                side = "left" if slope < 0 and center_x < w / 2 else (
                    "right" if slope > 0 and center_x > w / 2 else None)
                if side is None:
                    continue
                length = float(np.hypot(dx, dy))
                picks[side].extend([(x1, y1, length), (x2, y2, length)])
                cv2.line(segment_debug, (x1, y1), (x2, y2),
                         (255, 190, 0) if side == "left" else (0, 145, 255), 2)
                n_segments += 1
        left = _fit(picks["left"], y_bottom, y_top, w)
        right = _fit(picks["right"], y_bottom, y_top, w)
        if video:
            left = _smooth_line(left, self.previous_left, s.smoothing)
            right = _smooth_line(right, self.previous_right, s.smoothing)
            self.previous_left = left
            self.previous_right = right
        else:
            self.reset()
        # Geçersiz/kesişen çizgilerde alan çizilmesin.
        if left is not None and right is not None:
            if left[0][0] >= right[0][0] or left[1][0] >= right[1][0]:
                # Sahte pozitif durumda hangisinin yanlış olduğunu bilmediğimizden
                # güvenli olarak ikisini de iptal ediyoruz.
                left, right = None, None
        overlay = np.zeros_like(frame)
        offset_px = None
        if left is not None and right is not None:
            quad = np.array([left[0], left[1], right[1], right[0]], dtype=np.int32)
            cv2.fillPoly(overlay, [quad], (68, 182, 97))
            road_center = (left[0][0] + right[0][0]) / 2
            offset_px = w / 2 - road_center
        output = cv2.addWeighted(frame, 1.0, overlay, s.overlay_alpha, 0)
        for line, color in ((left, (0, 225, 255)), (right, (0, 225, 255))):
            if line is not None:
                cv2.line(output, line[0], line[1], color, s.line_thickness, cv2.LINE_AA)
        # Destek skoru, algılanan çizgi varlığına göre basit bir görsel göstergedir;
        # doğruluk olasılığı veya sınıflandırma güveni değildir.
        confidence = 0.5 * float(left is not None) + 0.5 * float(right is not None)
        label = f"SOL: {'VAR' if left else 'YOK'}   SAG: {'VAR' if right else 'YOK'}"
        cv2.rectangle(output, (8, 7), (min(w - 5, 455), 47), (18, 28, 44), -1)
        cv2.putText(output, label, (17, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, (242, 249, 255), 2, cv2.LINE_AA)
        stages = {
            "01 Orijinal": frame.copy(),
            "02 Gri ton": gray,
            "03 Bulaniklastirma": blurred,
            "04 Canny kenarlar": edges,
            "05 ROI maskesi": mask,
            "06 ROI kenarlari": roi,
            "07 Hough dogrulari": segment_debug,
            "08 Sonuc": output,
        }
        return Detection(output, stages, left, right, offset_px, confidence, n_segments)


def settings_dict(settings: LaneSettings) -> dict:
    return asdict(settings)