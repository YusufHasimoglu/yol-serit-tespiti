"""Grafik arayüzü olmayan ortamlarda komut satırı aracı."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from lanevision.core import LaneSettings
from lanevision.processing import VIDEO_EXTENSIONS, process_image, process_video


def main():
    parser = argparse.ArgumentParser(description="Yol şerit çizgisi tespiti - OpenCV")
    parser.add_argument("--input", required=True, type=Path, help="Görüntü/video yolu")
    parser.add_argument("--output", type=Path, help="Çıktı yolu")
    parser.add_argument("--steps", action="store_true", help="Görselde ara adımları da kaydet")
    parser.add_argument("--canny-low", type=int, default=55)
    parser.add_argument("--canny-high", type=int, default=160)
    parser.add_argument("--roi-top", type=float, default=0.59)
    parser.add_argument("--hough-threshold", type=int, default=28)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Girdi bulunamadı: {args.input}")
    settings = LaneSettings(canny_low=args.canny_low, canny_high=args.canny_high,
                            roi_top=args.roi_top, hough_threshold=args.hough_threshold)
    try:
        settings.validate()
        video = args.input.suffix.lower() in VIDEO_EXTENSIONS
        output = args.output or Path("sonuc.mp4" if video else "sonuc.png")
        if video:
            data = process_video(args.input, output, settings)
            print(json.dumps(data.to_dict(), ensure_ascii=False, indent=2))
        else:
            stages = output.with_name(output.stem + "_adimlar") if args.steps else None
            data = process_image(args.input, output, settings, stages)
            print("Görüntü kaydedildi:", output)
            print(json.dumps(data.stats(), ensure_ascii=False, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Hata: {exc}\n")


if __name__ == "__main__":
    main()
