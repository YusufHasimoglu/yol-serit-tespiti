"""Eğitim amaçlı, şerit maskesi de üreten senaryo tabanlı sentetik yol örnekleri.

Sentetik testler gerçek yol performansı değildir. Aynı tohumla aynı örnek üretilir.
"""
from __future__ import annotations
import cv2
import numpy as np
from .ai_model import INPUT_WIDTH, INPUT_HEIGHT

SCENARIOS = ('duz_gunduz', 'viraj_gunduz', 'duz_gece', 'viraj_gece', 'yagmurlu_gece')


def generate_scenario(seed: int, scenario: str):
    if scenario not in SCENARIOS:
        raise ValueError(f'Bilinmeyen senaryo: {scenario}')
    rng = np.random.default_rng(seed)
    w, h = INPUT_WIDTH, INPUT_HEIGHT
    night = 'gece' in scenario
    curved = 'viraj' in scenario
    wet = 'yagmurlu' in scenario
    horizon = int(rng.integers(45, 80))
    van_x = float(rng.uniform(.37, .63) * w)
    ygrid, xgrid = np.mgrid[:h, :w]
    t_all = np.clip((ygrid - horizon) / max(1, h - horizon), 0, 1)

    # Arka plan: ufuk gradyanı, orman ve gürültü. Sol/sağ etiketler tüm eğriliklerde tutarlı.
    base = np.zeros((h, w, 3), np.float32)
    if night:
        top = np.array([23, 17, 10], np.float32)
        bottom = np.array([45, 38, 32], np.float32)
    else:
        top = np.array([194, 142, 82], np.float32)
        bottom = np.array([213, 191, 150], np.float32)
    sky_t = (ygrid / max(1, horizon)).clip(0, 1)[..., None]
    base[:] = top * (1 - sky_t) + bottom * sky_t
    base[horizon:] = np.array([24, 45, 28] if night else [58, 102, 51], np.float32)
    img = np.clip(base, 0, 255).astype(np.uint8)
    for _ in range(75):
        x = int(rng.integers(0, w)); y = int(rng.integers(horizon - 12, h))
        rad = int(rng.integers(2, 13))
        val = int(rng.integers(9, 42 if night else 103))
        cv2.circle(img, (x, y), rad, (val//2, val, val//2), -1, cv2.LINE_AA)

    yy = np.arange(horizon, h, dtype=np.float32)
    t = (yy - horizon) / max(1, h - horizon)
    depth = .09 + .91 * (t ** 1.15)
    curve = float(rng.choice([-1, 1]) * rng.uniform(30, 75)) if curved else float(rng.uniform(-8, 8))
    curve_shape = t ** 1.6 * (1.05 - .20*t)
    center = van_x + curve * curve_shape
    lane_span = float(rng.uniform(70, 120)) * depth
    road_span = lane_span * float(rng.uniform(2.2, 2.8))
    lx = center - lane_span*.5
    rx = center + lane_span*.5
    road_l = center - road_span*.5
    road_r = center + road_span*.5

    path = np.vstack((np.stack([road_l, yy], -1), np.stack([road_r[::-1], yy[::-1]], -1))).round().astype(np.int32)
    roadmask = np.zeros((h,w),np.uint8)
    cv2.fillPoly(roadmask, [path], 255)
    asphalt = int(rng.integers(24, 56)) if night else int(rng.integers(62, 131))
    color_road = np.stack((np.full((h,w),asphalt+3),np.full((h,w),asphalt),np.full((h,w),asphalt-3)),axis=-1).astype(np.float32)
    grain = rng.normal(0, 7 if night else 11, (h,w,1))
    color_road = np.clip(color_road+grain,0,255).astype(np.uint8)
    img[roadmask > 0] = color_road[roadmask > 0]

    # Yolun dış kenarları, bariyer ve yansıma; hedef maskeye dahil değil.
    for xx in (road_l, road_r):
        pts = np.stack([xx, yy],-1).round().astype(np.int32)
        cv2.polylines(img,[pts],False,(75,79,79) if night else (148,154,155),1,cv2.LINE_AA)

    labels_list = [np.zeros((h,w),np.uint8), np.zeros((h,w),np.uint8)]
    for side,xcurve in enumerate((lx,rx)):
        points = np.stack((xcurve,yy),-1).round().astype(np.int32)
        # Çizgi görünürlüğünün gecede ve kesikli çizgide değişmesi beklenir.
        dashed = bool(rng.random()<(.53 if side==0 else .38))
        paint = (168,183,193) if night else (235,238,244)
        if rng.random()<.28:
            paint=(42,181,207) if night else (38,208,245)
        thick = 2 if night else int(rng.integers(2,4))
        if dashed:
            segment = int(rng.integers(10, 19)); period = segment+int(rng.integers(9,18))
            segments = [(k,min(len(points),k+segment)) for k in range(int(rng.integers(0,period)),len(points),period)]
        else:
            segments=[(0,len(points))]
        for a,b in segments:
            if b-a<2:continue
            cv2.polylines(img,[points[a:b]],False,paint,thick,cv2.LINE_AA)
            cv2.polylines(labels_list[side],[points[a:b]],False,255,3,cv2.LINE_8)

    if night:
        # Kamera farlarının aydınlattığı bölge: kısa mesafede parlak, uzakta karanlık.
        beam=np.exp(-((xgrid-(van_x+curve*.73))/(w*.47))**2/2) * (.08+.94*t_all**1.4)
        illuminated=img.astype(np.float32)
        illuminated *= (.34+1.36*beam[...,None])
        illuminated += np.maximum(0,beam[...,None]-.10)*np.array([16,20,29],np.float32)
        img=np.clip(illuminated,0,255).astype(np.uint8)
        # Far/sokak lambası parlama simülasyonu (hedef maske bozulmaz).
        for _ in range(int(rng.integers(0,3))):
            x=int(rng.integers(0,w));y=int(rng.integers(max(0,horizon-10),min(h,horizon+38)))
            amp=float(rng.uniform(18,64));sigma=float(rng.uniform(5,16))
            glow=amp*np.exp(-((xgrid-x)**2+(ygrid-y)**2)/(2*sigma**2))
            img=np.clip(img.astype(np.float32)+glow[...,None],0,255).astype(np.uint8)
    if wet:
        # Yağmurun yansıması ve görüşü zorlaştıran yağmur çizgileri.
        sheen=16+24*np.exp(-((xgrid-(van_x+curve*.7))/(w*.13))**2)
        img[roadmask>0]=np.clip(img[roadmask>0].astype(np.float32)+sheen[roadmask>0,None],0,255).astype(np.uint8)
        for _ in range(85):
            x=int(rng.integers(0,w));y=int(rng.integers(0,h));length=int(rng.integers(3,11))
            cv2.line(img,(x,y),(min(w-1,x-2+length//3),min(h-1,y+length)),(85,93,98),1,cv2.LINE_AA)
    if rng.random()<.6:
        img=cv2.GaussianBlur(img,(3,3),float(rng.uniform(.3,1.0)))
    img=np.clip(img.astype(np.float32)+rng.normal(0,2.2 if night else 1.4,(h,w,1)),0,255).astype(np.uint8)
    return img,np.stack(labels_list,axis=-1)