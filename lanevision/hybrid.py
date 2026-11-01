"""Beyaz/sarı işaret segmentasyonu ve eğrisel şerit takibi (eğitimsiz yedek).

Klasik Hough'un neredeyse dik çizgileri elemesi ve düz çizgi varsayımı
nedeniyle kaçırdığı örneklerde aday şeritleri satır satır izler.
"""
from __future__ import annotations
import cv2
import numpy as np
from .core import LaneSettings, Detection


def color_candidates(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Gün ışığı beyaz ve sarı yol işaretleri; parlaklık eşiği lokal karşıtlığı destekler.
    white = ((v >= 154) & (s <= 100)).astype(np.uint8) * 255
    yellow = (((h >= 13) & (h <= 38) & (s >= 68) & (v >= 110))).astype(np.uint8) * 255
    result = cv2.bitwise_or(white, yellow)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return result


def _collect_row_peaks(mask, ymin, ymax, min_x, max_x):
    strip = mask[max(0, ymin):min(mask.shape[0], ymax), max(0, min_x):min(mask.shape[1], max_x)]
    if strip.size == 0:
        return []
    hist = np.count_nonzero(strip, axis=0).astype(np.float32)
    if hist.size < 5:
        return []
    hist = np.convolve(hist, np.ones(9)/9, mode='same')
    peaks = []
    for pos in np.argsort(hist)[::-1]:
        if hist[pos] < 2.5:
            break
        x = int(pos + min_x)
        if all(abs(x - q[0]) > max(10, mask.shape[1] * .035) for q in peaks):
            peaks.append((x, float(hist[pos])))
        if len(peaks) >= 8:
            break
    return peaks


def _trace(mask, x_seed, y_start, y_end, side):
    h, w = mask.shape
    points = []
    x_prev = float(x_seed)
    velocity = 0.55 if side == "left" else -1.8  # dx/dy, yukari giderken sag cizgi sola ilerler
    last_y = None
    last_x = None
    miss = 0
    step = max(9, h // 47)
    for y in range(y_start, y_end, -step):
        y0, y1 = max(0, y - step//2), min(h, y + step//2)
        predicted = x_prev + velocity * step
        radius = int(max(w*.058, 32) + min(miss, 5)*w*.018)
        x0, x1 = max(0, round(predicted-radius)), min(w, round(predicted+radius))
        strip = mask[y0:y1, x0:x1]
        if strip.size == 0:
            break
        hist = np.count_nonzero(strip, axis=0).astype(np.float32)
        if len(hist) < 4:
            break
        hist = np.convolve(hist, np.ones(7)/7, mode='same')
        ids = np.flatnonzero(hist >= max(1.5, hist.max()*0.48))
        if ids.size > 0 and hist.max() >= 1.6:
            # önceki tahmine en yakın kuvvetli beyaz/sarı şerit noktası
            weights = hist[ids] * np.exp(-((ids + x0 - predicted)/(radius*.65))**2)
            best = ids[np.argmax(weights)]
            pos = float(best + x0)
            if points and abs(pos - predicted) > radius * .85:
                miss += 1
                continue
            points.append((pos, float(y), float(hist[best])))
            if last_x is not None and last_y is not None and last_y != y:
                estimated = (pos-last_x) / (last_y-y)
                if -6 < estimated < 6:
                    velocity = .65*velocity + .35*estimated
            last_x, last_y = pos, y
            x_prev = pos
            miss = 0
        else:
            miss += 1
            if miss > 6 and len(points) >= 5:
                break
    if len(points) < 6 or (max(p[1] for p in points) - min(p[1] for p in points)) < h * .12:
        return None
    coords = np.array(points, dtype=np.float64)
    yy, xx = coords[:, 1], coords[:, 0]
    # Ağır uç değerler için iki adımlı ağırlıklı ikinci derece uyum.
    deg = 2 if np.ptp(yy) > h*.27 and len(points) >= 10 else 1
    weight = np.sqrt(coords[:, 2] + .5)
    try:
        coefficients = np.polyfit(yy, xx, deg, w=weight)
        for _ in range(2):
            residual = np.abs(xx - np.polyval(coefficients, yy))
            keep = residual < max(16, np.median(residual)*2.8)
            if np.count_nonzero(keep) < max(5, deg+2):
                break
            coefficients = np.polyfit(yy[keep], xx[keep], deg, w=weight[keep])
    except (ValueError, np.linalg.LinAlgError):
        return None
    curve_bottom = int(max(yy))
    curve_top = int(max(y_end, min(yy)))
    # Yalnızca gözlemlendiği yere çiz; ekstrapolasyon kesikli çizgilerde sorun yaratır.
    y_draw = np.arange(curve_top, curve_bottom + 1, 3)
    x_draw = np.polyval(coefficients, y_draw)
    valid = (x_draw >= 0) & (x_draw < w)
    curve = np.stack([x_draw[valid], y_draw[valid]], 1).round().astype(np.int32)
    if len(curve) < 2:
        return None
    return {'curve': curve, 'support': len(points), 'coeffs': coefficients,
            'span': float(np.ptp(yy)/h), 'seed': x_seed}


def detect_hybrid(frame: np.ndarray, settings: LaneSettings | None = None, mask_override=None) -> Detection:
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 100:
        raise ValueError('BGR formatında, en az 100x100 piksellik görüntü gereklidir.')
    s = settings or LaneSettings()
    s.validate()
    h, w = frame.shape[:2]
    candidate = color_candidates(frame) if mask_override is None else mask_override.copy()
    roi = np.zeros((h, w), np.uint8)
    ytop = int(h * min(.60, max(.40, s.roi_top - .08)))
    # Hough ROI'ye kıyasla sağ omuz çizgisi için genişletilmiş bölge.
    polygon = np.array([[(0, h-1), (int(w*.33), ytop),
                         (int(w*.75), ytop), (w-1, h-1)]], dtype=np.int32)
    cv2.fillPoly(roi, polygon, 255)
    candidate = cv2.bitwise_and(candidate, roi)
    # En alt 10% üzerine kaymış olabilen şeritleri de bulabilmek için çoklu başlangıç.
    seeds = {'left': [], 'right': []}
    for ya, yb in ((.64, .91), (.58, .80), (.73, .98)):
        for side, bounds in (('left', (int(w*.12), int(w*.56))),
                             ('right', (int(w*.53), w))):
            for x, strength in _collect_row_peaks(candidate, int(h*ya), int(h*yb), *bounds):
                if not any(abs(x - p[0]) < w*.065 for p in seeds[side]):
                    seeds[side].append((x, strength))
    curves = {}
    for side in ('left', 'right'):
        best, best_score = None, -1e6
        for x, strength in seeds[side][:9]:
            result = _trace(candidate, x, int(h*.965), ytop, side)
            if result is None:
                continue
            # uç pozisyonu ve destek uzunluğu, sahte parmaklıklara karşı prior.
            center_hint = .40*w if side == 'left' else .85*w
            score = result['support'] * 1.8 + result['span']*55 + min(strength, 30)\
                    - abs(x-center_hint)/w*19
            if score > best_score:
                best, best_score = result, score
        curves[side] = best
    left, right = curves['left'], curves['right']
    # Aynı çizginin iki tarafa atanmasını önle.
    if left is not None and right is not None:
        if abs(left['seed'] - right['seed']) < w*.11:
            if left['support'] >= right['support']:
                right = None
            else:
                left = None
        elif left['seed'] >= right['seed']:
            right = None
    output = frame.copy()
    overlay = np.zeros_like(frame)
    offset = None
    if left is not None and right is not None:
        # Yalnızca her iki eğri için gözlemlenen ortak y aralığını doldur.
        ya = max(int(left['curve'][0, 1]), int(right['curve'][0, 1]))
        yb = min(int(left['curve'][-1, 1]), int(right['curve'][-1, 1]))
        if yb > ya + 20:
            ys = np.arange(ya, yb+1, 4)
            lx = np.polyval(left['coeffs'], ys)
            rx = np.polyval(right['coeffs'], ys)
            valid = (lx < rx) & (lx >= 0) & (rx < w)
            if valid.sum() >= 5:
                lp = np.stack([lx[valid], ys[valid]], 1).round().astype(np.int32)
                rp = np.stack([rx[valid], ys[valid]], 1).round().astype(np.int32)
                cv2.fillPoly(overlay, [np.vstack([lp, rp[::-1]])], (68, 182, 97))
                offset = w / 2 - float((lp[-1,0] + rp[-1,0])/2)
    output = cv2.addWeighted(output, 1., overlay, s.overlay_alpha, 0)
    for obj, color in ((left, (0, 225, 255)), (right, (0, 225, 255))):
        if obj is not None:
            cv2.polylines(output, [obj['curve']], False, color, max(3, s.line_thickness), cv2.LINE_AA)
    def endpoints(obj):
        if obj is None:
            return None
        return (tuple(map(int, obj['curve'][-1])), tuple(map(int, obj['curve'][0])))
    l, r = endpoints(left), endpoints(right)
    cv2.rectangle(output, (8, 7), (min(w-5, 520), 47), (18, 28, 44), -1)
    tag = f"SOL: {'VAR' if l else 'YOK'}   SAG: {'VAR' if r else 'YOK'}  |  EGIRISEL"
    cv2.putText(output, tag, (17, 34), cv2.FONT_HERSHEY_SIMPLEX, .65, (242,249,255), 2, cv2.LINE_AA)
    stages = {'01 Orijinal': frame.copy(), '02 Gri ton': cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
              '03 Bulaniklastirma': cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0),
              '04 Canny kenarlar': cv2.Canny(frame, s.canny_low, s.canny_high),
              '05 ROI maskesi': roi, '06 ROI kenarlari': candidate,
              '07 Hough dogrulari': cv2.cvtColor(candidate, cv2.COLOR_GRAY2BGR),
              '08 Sonuc': output}
    return Detection(output, stages, l, r, offset, float(l is not None)*.5+float(r is not None)*.5, 0)