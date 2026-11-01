"""Lisans gerektirmeyen sentetik test görüntüsü ve videosu üretimi.

Bunlar algoritmayı doğrulamak için üretilmiş örneklerdir; gerçek yol verisi değildir.
"""
from __future__ import annotations
from pathlib import Path
import math
import cv2
import numpy as np


def draw_scene(width=1280, height=720, shift=0, dark=False):
    rng = np.random.default_rng(123)
    scene = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = int(height * .57)
    if dark:
        scene[:] = (32, 19, 12)
        scene[:horizon] = (65, 42, 22)
    else:
        for y in range(horizon):
            v = y / max(horizon, 1)
            scene[y, :] = (int(224-35*v), int(187-38*v), int(138-44*v))
        scene[horizon:] = (55, 92, 73)
    # Uzak tepeler / ufuk, test için görsel çeşitlilik
    hills = np.array([(0, horizon + 18), (140, horizon - 16), (310, horizon + 5),
                      (480, horizon - 13), (700, horizon + 7),
                      (980, horizon - 21), (1280, horizon + 12), (1280, horizon + 65),
                      (0, horizon + 65)], dtype=np.int32)
    cv2.fillPoly(scene, [hills], (81, 116, 104) if not dark else (40, 50, 60))
    # Yoldaki asfalt yüzey
    road_poly = np.array([(width//2 - 115 + shift//3, horizon),
                          (width//2 + 115 + shift//3, horizon),
                          (width + 80, height), (-80, height)], dtype=np.int32)
    cv2.fillPoly(scene, [road_poly], (81, 84, 86) if not dark else (45, 47, 51))
    noise = rng.normal(0, 4 if not dark else 2, size=(height-horizon, width, 1))
    scene[horizon:] = np.clip(scene[horizon:].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Şerit çizgileri testte açıkça seçilebilsin diye sabit kontrastta.
    top_y, bottom_y = horizon + 14, height - 1
    vanishing_x = width // 2 + shift // 3
    left_top = (vanishing_x - 74, top_y)
    right_top = (vanishing_x + 74, top_y)
    left_bottom = (width // 2 - 405 + shift, bottom_y)
    right_bottom = (width // 2 + 405 + shift, bottom_y)
    outline = (25, 28, 30) if not dark else (12, 15, 16)
    paint = (249, 249, 249) if not dark else (206, 206, 206)
    cv2.line(scene, left_top, left_bottom, outline, 18, cv2.LINE_AA)
    cv2.line(scene, right_top, right_bottom, outline, 18, cv2.LINE_AA)
    cv2.line(scene, left_top, left_bottom, paint, 10, cv2.LINE_AA)
    cv2.line(scene, right_top, right_bottom, paint, 10, cv2.LINE_AA)
    # Kesikli orta işaret (merkez hizasında; iki ana şerit değil)
    for yf, yf2 in [(0.67, 0.69), (0.73, 0.77), (0.83, 0.89), (0.96, 0.99)]:
        x = vanishing_x + int((shift * 2 / 3) * ((yf - .57) / .43))
        cv2.line(scene, (x, int(height*yf)), (x, int(height*yf2)),
                 (220, 220, 220), int(4 + 8 * (yf-.57)/.43), cv2.LINE_AA)
    if dark:
        scene = cv2.GaussianBlur(scene, (3, 3), 0)
    return scene


def create_samples(output_dir: str | Path):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path / 'ornek_yol_duz.png'), draw_scene())
    cv2.imwrite(str(path / 'ornek_yol_dusuk_isik.png'), draw_scene(dark=True))
    video = path / 'ornek_yol_video.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 12.0,
                             (1280, 720))
    if not writer.isOpened():
        raise OSError('Sentetik örnek video oluşturulamadı')
    try:
        for n in range(36):
            shift = round(32 * math.sin(n / 8))
            writer.write(draw_scene(shift=shift))
    finally:
        writer.release()
    return path


if __name__ == '__main__':
    create_samples(Path(__file__).resolve().parent.parent / 'ornekler')
