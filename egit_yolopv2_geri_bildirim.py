"""YOLOPv2 için kullanıcının düzeltmelerinden dört sınıflı yardımcı model eğit.

YOLOPv2'nin resmî indirilen ağırlığı yalnızca çıkarım için TorchScript olarak
sunulur. Bu betik onu değiştirmez; kullanıcının doğruladığı sol/sağ ve
düz/kesik etiketlerinden ayrı bir düzeltici U-Net üretir. Uygulama, ağırlık
başarıyla oluştuğunda YOLOPv2 sonucuna otomatik olarak ekler.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from lanevision.training_data import load_user_samples
from lanevision.yolopv2 import DEFAULT_YOLOPV2_FEEDBACK_WEIGHTS


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOPv2 kullanıcı geri bildirim düzelticisi eğit")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--synthetic", type=int, default=160)
    parser.add_argument("--temporal-augment", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_YOLOPV2_FEEDBACK_WEIGHTS)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.synthetic < 0 or args.temporal_augment < 0:
        parser.error("Eğitim sayıları geçersiz.")
    user_samples = list(load_user_samples(ROOT / "data" / "gercek", style_aware=True))
    if len(user_samples) < 24:
        parser.error(f"En az 24 tam etiketli kullanıcı örneği gerekir; bulundu: {len(user_samples)}.")
    report = args.out.parent / "yolopv2_feedback_egitim_raporu.json"
    command = [
        sys.executable, str(ROOT / "egit.py"), "--style-aware",
        "--synthetic", str(args.synthetic), "--epochs", str(args.epochs),
        "--batch", str(args.batch), "--temporal-augment", str(args.temporal_augment),
        "--real-folder", str(ROOT / "data" / "gercek"), "--out", str(args.out),
        "--report", str(report),
    ]
    print(f"YOLOPv2 geri bildirim eğitimi: {len(user_samples)} kullanıcı etiketi")
    return subprocess.run(command, cwd=str(ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
