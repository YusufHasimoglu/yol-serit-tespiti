"""Kullanıcının düzelttiği bir video karesinden süreklilik tabanlı şerit takibi."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .core import Detection, LaneSettings


_SIDE_CODES = {"left": (1, 2), "right": (3, 4)}


def _marking_type(mask: np.ndarray, side: str) -> str:
    solid, dashed = _SIDE_CODES[side]
    if np.any(mask == solid):
        return "duz"
    if np.any(mask == dashed):
        return "kesik"
    raise ValueError(f"{side} şerit çizgisi kullanıcı düzeltmesinde yok.")


def _sample_centres(mask: np.ndarray, codes: tuple[int, int], maximum: int = 80) -> np.ndarray:
    """Sınıf maskesinden LK'nin izleyebileceği yayılmış merkez noktaları çıkar."""
    selected = np.isin(mask, codes)
    rows = np.flatnonzero(selected.any(axis=1))
    if len(rows) < 8:
        raise ValueError("Kullanıcı çizgisi takip için yeterli nokta içermiyor.")
    groups = np.split(rows, np.where(np.diff(rows) > 3)[0] + 1)
    points = []
    for group in groups:
        # Çizgi kalınlığının aynı y değerinde ürettiği tekrarları tek merkeze
        # indir; kesik parçalar arası boşluklar bilinçli olarak korunur.
        step = max(1, len(group) // max(2, maximum // max(1, len(groups))))
        for y in group[::step]:
            xs = np.flatnonzero(selected[y])
            if len(xs):
                points.append((float(np.median(xs)), float(y)))
    if len(points) < 8:
        raise ValueError("Kullanıcı çizgisinden yeterli takip noktası çıkarılamadı.")
    points = np.asarray(points, np.float32)
    if len(points) > maximum:
        points = points[np.linspace(0, len(points) - 1, maximum).round().astype(int)]
    return points.reshape(-1, 1, 2)


def _dash_spans(mask: np.ndarray, codes: tuple[int, int]) -> tuple[float, float, tuple[tuple[float, float], ...]]:
    """Kesik parçaları y koordinatına göre, ölçeklenebilir aralıklarda sakla."""
    selected = np.isin(mask, codes)
    rows = np.flatnonzero(selected.any(axis=1))
    if not len(rows):
        raise ValueError("Şerit için gözlenen piksel yok.")
    low, high = float(rows.min()), float(rows.max())
    denom = max(1.0, high - low)
    groups = np.split(rows, np.where(np.diff(rows) > 3)[0] + 1)
    spans = tuple(((float(group[0]) - low) / denom, (float(group[-1]) - low) / denom)
                  for group in groups if len(group))
    return low, high, spans


def _robust_poly(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Optik akıştaki tekil atlamaları ikinci derece eğri uydurmadan çıkar."""
    if len(points) < 6:
        return None
    ys, xs = points[:, 1], points[:, 0]
    degree = 2 if len(points) >= 9 else 1
    poly = np.polyfit(ys, xs, degree)
    residual = np.abs(xs - np.polyval(poly, ys))
    median = float(np.median(residual))
    keep = residual <= max(5.0, median * 2.8 + 2.0)
    if int(keep.sum()) < 6:
        return None
    kept = points[keep]
    poly = np.polyfit(kept[:, 1], kept[:, 0], 2 if len(kept) >= 9 else 1)
    return poly, kept


def _curve(poly: np.ndarray, low: float, high: float, count: int = 80) -> np.ndarray:
    ys = np.linspace(low, high, max(8, count))
    return np.column_stack([np.polyval(poly, ys), ys]).round().astype(np.int32)


@dataclass
class _SideState:
    kind: str
    points: np.ndarray
    spans: tuple[tuple[float, float], ...]
    low: float
    high: float


class ManualLaneAnchor:
    """İlk doğru çizimden sonra iki şeridi kareden kareye optik akışla taşır."""

    def __init__(self, frame: np.ndarray, line_mask: np.ndarray, settings: LaneSettings | None = None,
                 deep_flow=None):
        if frame is None or frame.ndim != 3 or line_mask is None or line_mask.shape != frame.shape[:2]:
            raise ValueError("Aynı boyutta özgün video karesi ve dört sınıflı çizgi maskesi gerekli.")
        self.settings = settings or LaneSettings()
        self.settings.validate()
        self.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.previous_frame = frame.copy()
        self.deep_flow = deep_flow
        self._used_deep_flow = False
        self.sides: dict[str, _SideState] = {}
        for side, codes in _SIDE_CODES.items():
            low, high, spans = _dash_spans(line_mask, codes)
            self.sides[side] = _SideState(_marking_type(line_mask, side), _sample_centres(line_mask, codes), spans, low, high)
        self.last_polynomials = {
            side: _robust_poly(state.points.reshape(-1, 2))[0]
            for side, state in self.sides.items()
        }
        self.prediction_frames = 0

    def current_detection(self, frame: np.ndarray, method: str = "Kaydedilmiş kullanıcı düzeltmesi") -> Detection | None:
        """Kaydedilmiş düzeltmenin ait olduğu karede bire bir görünümünü üret."""
        if frame is None or frame.ndim != 3 or frame.shape[:2] != self.previous_gray.shape:
            return None
        return self._render(frame, self.last_polynomials, self.sides, 1.0, 1.0, method)

    @staticmethod
    def _ordered_endpoints(line: tuple[tuple[int, int], tuple[int, int]] | None):
        """Bir tespit doğrusunun alt ve üst uçlarını, verilen sıradan bağımsız al."""
        if line is None or len(line) != 2:
            return None
        first, second = line
        if len(first) != 2 or len(second) != 2:
            return None
        bottom, top = sorted((first, second), key=lambda point: point[1], reverse=True)
        return np.asarray(bottom, np.float32), np.asarray(top, np.float32)

    def _candidate_is_compatible(self, candidate: Detection, frame: np.ndarray) -> bool:
        """Yeni ölçümün kullanıcı yolunu mantıksız biçimde değiştirmediğini denetle.

        Genel algılayıcı yalnızca takip koptuktan sonra ölçüm adayıdır. Adayın
        kullanıcının son güvenilir eğrisiyle hem alt/üst konumu hem de şerit
        genişliği bakımından tutarlı olması gerekir. Böylece kaldırım, bariyer
        veya karşı yöndeki yol yeni şerit gibi kabul edilmez.
        """
        if candidate is None or candidate.left is None or candidate.right is None:
            return False
        left = self._ordered_endpoints(candidate.left)
        right = self._ordered_endpoints(candidate.right)
        if left is None or right is None:
            return False
        (left_bottom, left_top), (right_bottom, right_top) = left, right
        h, w = frame.shape[:2]
        # Aday iki sınır birbiriyle kesişmemeli ve yeterli ortak yüksekliği
        # paylaşmalıdır. Bu, tek başına çok sayıda yanlış Hough eşleşmesini eler.
        common_top = max(float(left_top[1]), float(right_top[1]))
        common_bottom = min(float(left_bottom[1]), float(right_bottom[1]))
        if common_bottom - common_top < h * .12:
            return False
        ys = np.linspace(common_top, common_bottom, 16)
        candidate_left = np.interp(ys, [left_top[1], left_bottom[1]], [left_top[0], left_bottom[0]])
        candidate_right = np.interp(ys, [right_top[1], right_bottom[1]], [right_top[0], right_bottom[0]])
        if np.any(candidate_left >= candidate_right):
            return False

        expected_left = np.polyval(self.last_polynomials["left"], ys)
        expected_right = np.polyval(self.last_polynomials["right"], ys)
        # Genişliğe ölçeklenen tolerans, hafif virajda yeniden bağlanmaya izin
        # verir; ama tüm görüntüde başka bir şeride sıçramayı engeller.
        position_limit = max(32.0, w * .105) + self.prediction_frames * 2.0
        width_limit = max(52.0, w * .16)
        candidate_width = candidate_right - candidate_left
        expected_width = expected_right - expected_left
        return (float(np.median(np.abs(candidate_left - expected_left))) <= position_limit and
                float(np.median(np.abs(candidate_right - expected_right))) <= position_limit and
                float(np.median(np.abs(candidate_width - expected_width))) <= width_limit)

    def reanchor_if_compatible(self, frame: np.ndarray, candidate: Detection) -> Detection | None:
        """Uyumlu bir ölçümü kullanıcı çizim türlerini koruyarak yeni ankraj yap.

        Dönen görünümde yeni algılayıcının çizgi türü kullanılmaz: kullanıcı
        tarafından çizilmiş düz/kesik bilgisi korunur. Uyum yoksa ``None``
        döner; çağıran taraf alakasız aday çizgiyi göstermemelidir.
        """
        if frame is None or frame.ndim != 3 or frame.shape[:2] != self.previous_gray.shape:
            return None
        if not self._candidate_is_compatible(candidate, frame):
            return None
        new_states: dict[str, _SideState] = {}
        new_polynomials: dict[str, np.ndarray] = {}
        for side, line in (("left", candidate.left), ("right", candidate.right)):
            endpoints = self._ordered_endpoints(line)
            if endpoints is None:
                return None
            bottom, top = endpoints
            ys = np.linspace(float(top[1]), float(bottom[1]), 36)
            xs = np.linspace(float(top[0]), float(bottom[0]), 36)
            points = np.column_stack([xs, ys]).astype(np.float32)
            fitted = _robust_poly(points)
            if fitted is None:
                return None
            poly, kept = fitted
            previous = self.sides[side]
            new_states[side] = _SideState(previous.kind, kept.reshape(-1, 1, 2).astype(np.float32),
                                          previous.spans, float(kept[:, 1].min()), float(kept[:, 1].max()))
            new_polynomials[side] = poly
        self.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.previous_frame = frame.copy()
        self.sides = new_states
        self.last_polynomials = new_polynomials
        self.prediction_frames = 0
        return self._render(frame, new_polynomials, new_states, .68, .68,
                            "Kullanıcı ankrajı: uyumlu ölçümle yeniden bağlandı")

    def warning_detection(self, frame: np.ndarray) -> Detection:
        """Ankraj güveni bitince yanlış çizgi çizmek yerine açık uyarı üret."""
        output = frame.copy()
        h, w = output.shape[:2]
        cv2.rectangle(output, (8, 7), (min(w - 5, 760), 48), (48, 31, 18), -1)
        cv2.putText(output, "KULLANICI ANKRAJI KOPTU | DUZELTME BEKLENIYOR", (17, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, .54, (242, 244, 248), 2, cv2.LINE_AA)
        det = Detection(output, {"01 Orijinal": frame.copy(), "08 Sonuc": output,
                                 "11 Kullanıcı ankrajı": output},
                        None, None, None, .0, 0)
        det.method = "Kullanıcı ankrajı güveni düşük — düzeltme bekliyor"
        det.marking_types = (self.sides["left"].kind, self.sides["right"].kind)
        det.lane_polynomials = (None, None)
        return det

    def _track_side(self, state: _SideState, current_gray: np.ndarray,
                    current_frame: np.ndarray) -> tuple[np.ndarray, _SideState, float, bool] | None:
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, current_gray, state.points, None,
            winSize=(25, 25), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, .01),
        )
        candidate = None
        if next_points is not None and status is not None:
            valid = status.reshape(-1).astype(bool)
            if errors is not None:
                valid &= errors.reshape(-1) < 38
            candidate = next_points.reshape(-1, 2)[valid]
        used_deep = False
        if candidate is not None:
            h, w = current_gray.shape[:2]
            inside = ((candidate[:, 0] >= 1) & (candidate[:, 0] < w - 1) &
                      (candidate[:, 1] >= 1) & (candidate[:, 1] < h - 1))
            candidate = candidate[inside]
        fitted = _robust_poly(candidate) if candidate is not None else None
        coverage = (len(fitted[1]) / max(1, len(state.points))) if fitted is not None else 0.0
        # Lucas-Kanade küçük ve dokulu hareketlerde hızlıdır. Kayıp, büyük
        # hareket ya da düşük doku yüzünden güveni azalırsa RAFT'in öğrenilmiş
        # yoğun hareket alanı aynı noktalar için ikinci ölçüm olur.
        if (fitted is None or coverage < .28) and self.deep_flow is not None:
            deep_points = self.deep_flow.track_points(self.previous_frame, current_frame, state.points)
            if deep_points is not None:
                candidate = deep_points.reshape(-1, 2)
                used_deep = True
                fitted = _robust_poly(candidate)
                coverage = (len(fitted[1]) / max(1, len(state.points))) if fitted is not None else 0.0
        if fitted is None:
            return None
        poly, kept = fitted
        h, w = current_gray.shape[:2]
        inside = ((kept[:, 0] >= 1) & (kept[:, 0] < w - 1) &
                  (kept[:, 1] >= 1) & (kept[:, 1] < h - 1))
        kept = kept[inside]
        if len(kept) < 6:
            return None
        refined = _robust_poly(kept)
        if refined is None:
            return None
        poly, kept = refined
        coverage = len(kept) / max(1, len(state.points))
        # Takip noktaları aşırı seyrekleştiğinde eski eğriyi ileri taşımak,
        # bir sonraki karede yol dışına sıçramaya yol açar. Güvenle yedeğe dön.
        if coverage < .28:
            return None
        low, high = float(kept[:, 1].min()), float(kept[:, 1].max())
        new_state = _SideState(state.kind, kept.reshape(-1, 1, 2).astype(np.float32), state.spans, low, high)
        return poly, new_state, coverage, used_deep

    def _recover_side(self, side: str, state: _SideState, current_gray: np.ndarray) -> tuple[np.ndarray, _SideState, float] | None:
        """Öngörülen eğrinin dar çevresinde gerçek çizgi kenarı ara.

        Optik akış gölge, parlama veya kısa örtülme nedeniyle koptuğunda tüm
        görüntüde yeni bir şerit aramak yanlış bariyer seçimine yol açabilir.
        Bu arama yalnızca kullanıcının çizdiği yolun beklenen devamında yapılır.
        """
        prediction = self.last_polynomials[side]
        h, w = current_gray.shape[:2]
        edges = cv2.Canny(cv2.GaussianBlur(current_gray, (5, 5), 0), 55, 155)
        corridor = min(max(20, int(w * .055)), 34 + self.prediction_frames * 3)
        spans = ((0.0, 1.0),) if state.kind == "duz" else state.spans
        candidates = []
        for start, end in spans:
            y0 = state.low + (state.high - state.low) * start
            y1 = state.low + (state.high - state.low) * end
            for y in np.linspace(y0, y1, max(6, int((y1 - y0) / 12))):
                yi = int(round(y))
                if not 2 <= yi < h - 2:
                    continue
                expected = float(np.polyval(prediction, y))
                lo, hi = max(2, int(expected - corridor)), min(w - 2, int(expected + corridor + 1))
                if hi - lo < 3:
                    continue
                band = edges[yi - 1:yi + 2, lo:hi].max(axis=0)
                if not np.any(band):
                    continue
                # Yakındaki güçlü kenarı seç; uzak, alakasız yol çizgisi veya
                # kaldırım kenarı tahmin eğrisinin yerine geçemez.
                offsets = np.arange(lo, hi) - expected
                scores = band.astype(np.float32) - np.abs(offsets) * 6.0
                x = float(np.arange(lo, hi)[int(np.argmax(scores))])
                candidates.append((x, float(y)))
        fitted = _robust_poly(np.asarray(candidates, np.float32)) if len(candidates) >= 6 else None
        if fitted is None:
            return None
        poly, kept = fitted
        coverage = min(.72, len(kept) / max(1, len(state.points)))
        if coverage < .22:
            return None
        recovered = _SideState(state.kind, kept.reshape(-1, 1, 2).astype(np.float32), state.spans,
                               float(kept[:, 1].min()), float(kept[:, 1].max()))
        return poly, recovered, coverage

    @staticmethod
    def _draw_side(output: np.ndarray, poly: np.ndarray, state: _SideState, thickness: int) -> np.ndarray:
        full = _curve(poly, state.low, state.high)
        spans = ((0.0, 1.0),) if state.kind == "duz" else state.spans
        color = (255, 223, 30) if state.kind == "duz" else (0, 176, 255)
        for start, end in spans:
            y0 = state.low + (state.high - state.low) * start
            y1 = state.low + (state.high - state.low) * end
            part = _curve(poly, y0, y1, max(4, int(len(full) * max(.04, end - start))))
            if len(part) > 1:
                cv2.polylines(output, [part], False, color, thickness, cv2.LINE_AA)
        return full

    def _render(self, frame: np.ndarray, polynomials: dict[str, np.ndarray],
                states: dict[str, _SideState], left_coverage: float, right_coverage: float,
                method: str) -> Detection | None:
        """Kullanıcı çizim türleri ve kapalı alanla ankraj sonucunu oluştur."""
        left_poly, right_poly = polynomials["left"], polynomials["right"]
        left_state, right_state = states["left"], states["right"]
        left_curve = _curve(left_poly, left_state.low, left_state.high)
        right_curve = _curve(right_poly, right_state.low, right_state.high)
        low, high = max(left_state.low, right_state.low), min(left_state.high, right_state.high)
        if high - low < frame.shape[0] * .12:
            return None
        ys = np.linspace(low, high, 60)
        left_x, right_x = np.polyval(left_poly, ys), np.polyval(right_poly, ys)
        h, w = frame.shape[:2]
        ordered = (left_x < right_x) & (left_x >= 0) & (right_x < w)
        if int(ordered.sum()) < 20:
            return None
        ys = ys[ordered]
        left_x, right_x = left_x[ordered], right_x[ordered]
        overlay = np.zeros_like(frame)
        lane_area = np.vstack([np.column_stack([left_x, ys]), np.column_stack([right_x[::-1], ys[::-1]])]).round().astype(np.int32)
        cv2.fillPoly(overlay, [lane_area], (70, 177, 105))
        output = cv2.addWeighted(frame, 1.0, overlay, self.settings.overlay_alpha, 0)
        self._draw_side(output, left_poly, left_state, self.settings.line_thickness)
        self._draw_side(output, right_poly, right_state, self.settings.line_thickness)
        cv2.rectangle(output, (8, 7), (min(w - 5, 650), 48), (17, 29, 42), -1)
        cv2.putText(output, "KULLANICI ANKRAJI | SERIT TAKIBI", (17, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, .62, (241, 249, 255), 2, cv2.LINE_AA)
        bottom_y = float(ys.max())
        left_bottom, right_bottom = float(np.polyval(left_poly, bottom_y)), float(np.polyval(right_poly, bottom_y))
        left_end = (tuple(map(int, left_curve[-1])), tuple(map(int, left_curve[0])))
        right_end = (tuple(map(int, right_curve[-1])), tuple(map(int, right_curve[0])))
        confidence = min(1.0, (left_coverage + right_coverage) * .5)
        det = Detection(output, {"01 Orijinal": frame.copy(), "08 Sonuc": output,
                                 "11 Kullanıcı ankrajı": output},
                        left_end, right_end, w / 2 - (left_bottom + right_bottom) / 2,
                        confidence, len(left_state.points) + len(right_state.points))
        det.method = method
        det.lane_polynomials = ((left_curve, left_poly), (right_curve, right_poly))
        det.marking_types = (left_state.kind, right_state.kind)
        return det

    def track(self, frame: np.ndarray) -> Detection | None:
        """İlk çizimden türeyen sınırları yeni karede çiz; güven kaybında ``None`` döndür."""
        if frame is None or frame.ndim != 3 or frame.shape[:2] != self.previous_gray.shape:
            return None
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._used_deep_flow = False
        tracked = {side: self._track_side(state, current_gray, frame) for side, state in self.sides.items()}
        predicted = set()
        for side, state in self.sides.items():
            if tracked[side] is not None:
                continue
            tracked[side] = self._recover_side(side, state, current_gray)
            if tracked[side] is not None:
                tracked[side] = (*tracked[side], False)
                continue
            # Kısa bir örtülme veya kesik çizginin boşluğu sırasında önceki
            # yol geometrisi güvenli bir köprü sağlar. Bu köprü sınırlıdır;
            # kalıcı kanıt yoksa aşağıda normal algılamaya geri dönülür.
            if self.prediction_frames >= 10:
                self.previous_gray = current_gray
                self.previous_frame = frame.copy()
                return None
            predicted.add(side)
            tracked[side] = (self.last_polynomials[side], state, .30, False)
        left_poly, left_state, left_coverage, left_deep = tracked["left"]
        right_poly, right_state, right_coverage, right_deep = tracked["right"]
        self.previous_gray = current_gray
        self.previous_frame = frame.copy()
        self.sides = {"left": left_state, "right": right_state}
        self.last_polynomials = {"left": left_poly, "right": right_poly}
        self.prediction_frames = self.prediction_frames + 1 if predicted else 0
        self._used_deep_flow = left_deep or right_deep
        return self._render(frame, self.last_polynomials, self.sides, left_coverage, right_coverage,
                            "Kullanıcı ankrajlı öngörülü şerit takibi" if predicted
                            else ("Kullanıcı ankrajlı derin optik şerit takibi" if self._used_deep_flow
                                  else "Kullanıcı ankrajlı optik şerit takibi"))

