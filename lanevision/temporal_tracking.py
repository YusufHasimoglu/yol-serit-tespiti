"""Video kareleri arasında şerit geometrisini kararlı tutan güvenlik katmanı.

Bir algılayıcı her karede yeniden ölçüm üretir; ışık, gölge veya bariyer gibi
etkenler tek karelik, büyük bir konum değişimine yol açabilir. Bu modül yeni
ölçümü önceki güvenilir şerit çiftiyle karşılaştırır. Uyumlu ölçümler nazikçe
birleştirilir, ani çizgi ve alan sıçramaları kısa süreliğine reddedilir.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .core import Detection, LaneSettings


def _line_array(line) -> np.ndarray | None:
    """Bir çizgiyi altta başlayan iki noktalı float dizisine dönüştür."""
    if line is None:
        return None
    points = np.asarray(line, dtype=np.float32)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        return None
    if points[0, 1] < points[1, 1]:
        points = points[::-1]
    return points.copy()


def _as_line(points: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    clipped = np.rint(points).astype(np.int32)
    return tuple((int(point[0]), int(point[1])) for point in clipped)


@dataclass
class _LanePair:
    left: np.ndarray
    right: np.ndarray
    marking_types: tuple[object, object]
    confidence: float
    velocity_left: np.ndarray
    velocity_right: np.ndarray


class VideoLaneStabilizer:
    """Tek karelik hataları tutup geometrik olarak tutarlı bir video izi üret.

    Bu bir tahmin üreticisi değildir: güvenilir bir çift yoksa kısa süre önceki
    geometriyi en fazla ``max_hold_frames`` kare gösterir. Böylece başka şerit,
    kaldırım veya yansıma bir anda taralı alanın yerine geçemez.
    """

    def __init__(self, settings: LaneSettings | None = None, max_hold_frames: int = 8,
                 challenger_frames: int = 3):
        self.settings = settings or LaneSettings()
        self.settings.validate()
        self.max_hold_frames = max(1, int(max_hold_frames))
        self.challenger_frames = max(2, int(challenger_frames))
        self._pair: _LanePair | None = None
        self._hold_frames = 0
        self._challenger: _LanePair | None = None
        self._challenger_count = 0
        self.last_status = "başlatılıyor"

    def reset(self) -> None:
        self._pair = None
        self._hold_frames = 0
        self._challenger = None
        self._challenger_count = 0
        self.last_status = "başlatılıyor"

    @property
    def active(self) -> bool:
        """Kararlı bir çift geometri bellekteyse ``True`` döndür."""
        return self._pair is not None

    def seed(self, detection: Detection, frame: np.ndarray | None = None) -> None:
        """Kullanıcı tarafından doğrulanmış bir kareyi tereddütsüz başlangıç kabul et."""
        shape = frame.shape[:2] if frame is not None else detection.frame.shape[:2]
        pair = self._pair_from_detection(detection, shape)
        if pair is None:
            return
        self._pair = pair
        self._hold_frames = 0
        self._challenger = None
        self._challenger_count = 0
        self.last_status = "doğrulanmış başlangıç"

    @staticmethod
    def _marking_types(detection: Detection) -> tuple[object, object]:
        types = getattr(detection, "marking_types", (None, None))
        if not isinstance(types, tuple) or len(types) != 2:
            return (None, None)
        return types

    def _pair_from_detection(self, detection: Detection, shape: tuple[int, int]) -> _LanePair | None:
        left, right = _line_array(detection.left), _line_array(detection.right)
        if left is None or right is None:
            return None
        height, width = shape
        if height < 1 or width < 1:
            return None
        if min(abs(float(left[0, 1] - left[1, 1])), abs(float(right[0, 1] - right[1, 1]))) < height * .10:
            return None
        xs = np.concatenate((left[:, 0], right[:, 0]))
        if float(xs.min()) < -width * .04 or float(xs.max()) > width * 1.04:
            return None
        widths = right[:, 0] - left[:, 0]
        if np.any(widths <= max(5.0, width * .012)):
            return None
        # Uzak noktada şeritler doğal olarak birbirine çok yaklaşır. Otoyol
        # perspektifinde alt/üst genişlik oranı 5'i geçebildiği için burada
        # cömert bir perspektif sınırı kullanılır; asıl sıçrama denetimi
        # ardışık karelerdeki konum ve alan değişiminde yapılır.
        if np.any(widths > width * .96) or float(np.max(widths) / max(np.min(widths), 1.0)) > 8.0:
            return None
        return _LanePair(left, right, self._marking_types(detection),
                         float(np.clip(detection.confidence, 0.0, 1.0)),
                         np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32))

    @staticmethod
    def _widths(pair: _LanePair) -> np.ndarray:
        return pair.right[:, 0] - pair.left[:, 0]

    @staticmethod
    def _distance(first: _LanePair, second: _LanePair) -> float:
        offsets = np.concatenate((first.left[:, 0] - second.left[:, 0],
                                  first.right[:, 0] - second.right[:, 0]))
        return float(np.median(np.abs(offsets)))

    def _position_limit(self, pair: _LanePair, width: int) -> float:
        lane_width = float(np.median(self._widths(pair)))
        motion = float(np.median(np.abs(np.concatenate((pair.velocity_left[:, 0], pair.velocity_right[:, 0])))))
        return max(12.0, width * .026, lane_width * .075) + motion * 1.8

    @staticmethod
    def _width_is_compatible(candidate: _LanePair, reference: _LanePair) -> bool:
        reference_width = np.maximum(VideoLaneStabilizer._widths(reference), 1.0)
        ratio = VideoLaneStabilizer._widths(candidate) / reference_width
        return bool(np.all((ratio >= .72) & (ratio <= 1.36)))

    def _compatible(self, candidate: _LanePair, reference: _LanePair, width: int) -> bool:
        predicted = _LanePair(reference.left + reference.velocity_left,
                              reference.right + reference.velocity_right,
                              reference.marking_types, reference.confidence,
                              reference.velocity_left, reference.velocity_right)
        return (self._width_is_compatible(candidate, reference) and
                self._distance(candidate, predicted) <= self._position_limit(reference, width))

    def _blend(self, candidate: _LanePair, previous: _LanePair, previous_weight: float | None = None) -> _LanePair:
        weight = self.settings.smoothing if previous_weight is None else previous_weight
        weight = float(np.clip(weight, .20, .82))
        left = previous.left * weight + candidate.left * (1.0 - weight)
        right = previous.right * weight + candidate.right * (1.0 - weight)
        left_velocity = previous.velocity_left * .55 + (left - previous.left) * .45
        right_velocity = previous.velocity_right * .55 + (right - previous.right) * .45
        kinds = candidate.marking_types
        if kinds == (None, None):
            kinds = previous.marking_types
        return _LanePair(left, right, kinds, candidate.confidence,
                         left_velocity.astype(np.float32), right_velocity.astype(np.float32))

    def _advance_challenger(self, candidate: _LanePair, width: int) -> bool:
        if self._challenger is not None and self._compatible(candidate, self._challenger, width):
            self._challenger_count += 1
        else:
            self._challenger = candidate
            self._challenger_count = 1
        return self._challenger_count >= self.challenger_frames

    @staticmethod
    def _draw_marking(image: np.ndarray, line: np.ndarray, kind: object, thickness: int) -> None:
        color = (0, 220, 255) if kind != "kesik" else (0, 170, 255)
        if kind != "kesik":
            cv2.line(image, tuple(np.rint(line[0]).astype(int)), tuple(np.rint(line[1]).astype(int)),
                     color, thickness, cv2.LINE_AA)
            return
        # Kesik çizgiler için birbirinden ayrı dört görünür boya parçası çiz.
        for start, end in ((.00, .17), (.29, .46), (.58, .75), (.87, 1.00)):
            first = line[0] * (1.0 - start) + line[1] * start
            second = line[0] * (1.0 - end) + line[1] * end
            cv2.line(image, tuple(np.rint(first).astype(int)), tuple(np.rint(second).astype(int)),
                     color, thickness, cv2.LINE_AA)

    def _render(self, frame: np.ndarray, source: Detection, pair: _LanePair,
                status: str, held: bool = False) -> Detection:
        height, width = frame.shape[:2]
        overlay = np.zeros_like(frame)
        area = np.rint(np.asarray((pair.left[0], pair.left[1], pair.right[1], pair.right[0]))).astype(np.int32)
        cv2.fillPoly(overlay, [area], (68, 174, 100))
        output = cv2.addWeighted(frame, 1.0, overlay, self.settings.overlay_alpha, 0)
        self._draw_marking(output, pair.left, pair.marking_types[0], self.settings.line_thickness)
        self._draw_marking(output, pair.right, pair.marking_types[1], self.settings.line_thickness)
        cv2.rectangle(output, (8, 7), (min(width - 5, 650), 47), (18, 28, 44), -1)
        label = "KARARLI VIDEO TAKIBI" if not held else "KARARLI VIDEO TAKIBI | ONCEKI GEOMETRI"
        cv2.putText(output, label, (17, 34), cv2.FONT_HERSHEY_SIMPLEX, .58,
                    (242, 249, 255), 2, cv2.LINE_AA)
        stages = dict(source.stages)
        stages["08 Sonuc"] = output
        confidence = pair.confidence * (.88 if held else 1.0)
        result = Detection(output, stages, _as_line(pair.left), _as_line(pair.right),
                           width / 2 - float((pair.left[0, 0] + pair.right[0, 0]) * .5),
                           float(np.clip(confidence, 0.0, 1.0)), source.segment_count)
        result.marking_types = pair.marking_types
        result.scene_profile = source.scene_profile
        result.temporal_status = status
        result.method = source.method + " · video takibi: " + status
        return result

    def update(self, frame: np.ndarray, detection: Detection) -> Detection:
        """Yeni ölçümü kabul et, yumuşat veya güvenilir son geometriyi koru."""
        candidate = self._pair_from_detection(detection, frame.shape[:2])
        if self._pair is None:
            if candidate is not None:
                self._pair = candidate
                self.last_status = "başlangıç ölçümü"
            return detection

        if candidate is not None and self._compatible(candidate, self._pair, frame.shape[1]):
            self._pair = self._blend(candidate, self._pair)
            self._hold_frames = 0
            self._challenger = None
            self._challenger_count = 0
            self.last_status = "ölçüm yumuşatıldı"
            return self._render(frame, detection, self._pair, self.last_status)

        if candidate is not None and self._advance_challenger(candidate, frame.shape[1]):
            # Üç ardışık, kendi içinde tutarlı ölçüm gerçek bir yol değişimi
            # olabilir. Birden sıçramak yerine kontrollü biçimde bağlan.
            self._pair = self._blend(candidate, self._pair, previous_weight=.38)
            self._hold_frames = 0
            self._challenger = None
            self._challenger_count = 0
            self.last_status = "yeni geometri onaylandı"
            return self._render(frame, detection, self._pair, self.last_status)

        if candidate is None:
            # Kesinti sonrasında eski, yarım kalmış adayın yeni bir yol
            # geometrisi olarak onaylanmasına izin vermeyiz.
            self._challenger = None
            self._challenger_count = 0
        self._hold_frames += 1
        if self._hold_frames <= self.max_hold_frames:
            self.last_status = "ani sıçrama engellendi" if candidate is not None else "ölçüm bekleniyor"
            return self._render(frame, detection, self._pair, self.last_status, held=True)

        # Uzun süre kanıt yoksa eski alanı sonsuza kadar göstermeyiz. Sonraki
        # güvenilir çift yeni bir başlangıç olarak kabul edilir.
        self.reset()
        return detection
