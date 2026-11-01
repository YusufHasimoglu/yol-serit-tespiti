"""Resmî YOLOPv2 TorchScript ağırlığını güvenli olarak yerelde indir.

Model GitHub deposuna eklenmez. Çalıştırma: ``python indir_yolopv2.py``.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


URL = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
TARGET = Path(__file__).resolve().parent / "modeller" / "yolopv2_bdd100k.pt"
MINIMUM_BYTES = 140 * 1024 * 1024


def main() -> int:
    if TARGET.is_file() and TARGET.stat().st_size >= MINIMUM_BYTES:
        print(f"YOLOPv2 modeli zaten hazır: {TARGET}")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = TARGET.with_suffix(".pt.part")
    try:
        print("YOLOPv2 modeli indiriliyor (yaklaşık 150 MB)...")
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as stream:
            total = int(response.headers.get("Content-Length") or 0)
            copied = 0
            while True:
                part = response.read(1024 * 1024)
                if not part:
                    break
                stream.write(part)
                copied += len(part)
                if total:
                    print(f"\rYOLOPv2 indiriliyor: %{min(100, int(copied * 100 / total))}", end="", flush=True)
        print()
        if not temporary.is_file() or temporary.stat().st_size < MINIMUM_BYTES:
            raise RuntimeError("İndirme tamamlanmadı veya model dosyası beklenenden küçük.")
        temporary.replace(TARGET)
        print(f"YOLOPv2 modeli hazır: {TARGET}")
        return 0
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        print(f"YOLOPv2 indirilemedi: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
