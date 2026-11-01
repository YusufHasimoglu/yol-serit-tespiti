"""UFLD V2 ONNX ağırlığını güvenli olarak yerel modele indir.

Bu büyük dosya kaynak depoya konulmaz.  Betik, resmî model dağıtım
adresinden indirir, geçici bir dosya kullanır ve tamamlanmadan model adını
almaz.  Çalıştırma: ``python indir_ufld_v2.py``.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


URL = ("https://www.modelscope.cn/models/cix/ai_model_hub_25_Q3/resolve/master/"
       "models/ComputeVision/Lane_Detection/onnx_Ultra_Fast_Lane_Detection_v2/"
       "model/Ultra_Fast_Lane_Detection_v2.onnx")
TARGET = Path(__file__).resolve().parent / "modeller" / "ufldv2_lane_detection.onnx"
MINIMUM_BYTES = 300 * 1024 * 1024


def main() -> int:
    if TARGET.is_file() and TARGET.stat().st_size >= MINIMUM_BYTES:
        print(f"UFLD V2 modeli zaten hazır: {TARGET}")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = TARGET.with_suffix(".onnx.part")
    try:
        print("UFLD V2 modeli indiriliyor (yaklaşık 368 MB)...")
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
                    print(f"\rUFLD V2 indiriliyor: %{min(100, int(copied * 100 / total))}", end="", flush=True)
        print()
        if not temporary.is_file() or temporary.stat().st_size < MINIMUM_BYTES:
            raise RuntimeError("İndirme tamamlanmadı veya model dosyası beklenenden küçük.")
        temporary.replace(TARGET)
        print(f"UFLD V2 modeli hazır: {TARGET}")
        return 0
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        print(f"UFLD V2 indirilemedi: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
