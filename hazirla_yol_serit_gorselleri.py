"""Yerel BDD100K şerit örneklerinden ayrı bir inceleme klasörü hazırla.

Bu araç yalnızca zaten kullanıcının bilgisayarında bulunan resmî BDD100K
görüntülerinin on kopyasını ``ornek_medya/gorseller`` altında toplar. Ham veri
ve örnek kopyaları GitHub'a eklenmez.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from lanevision.training_data import _bdd_image_path


ROOT = Path(__file__).resolve().parent
DEFAULT_BDD_ROOT = ROOT / "data" / "external" / "bdd100k"
DEFAULT_MANIFEST = DEFAULT_BDD_ROOT / "bdd100_manifest.json"
DEFAULT_OUTPUT = ROOT / "ornek_medya" / "gorseller"


def main() -> int:
    parser = argparse.ArgumentParser(description="10 BDD100K yol şeridi görselini ayrı klasöre hazırla")
    parser.add_argument("--root", type=Path, default=DEFAULT_BDD_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.count < 1 or args.count > 100:
        parser.error("--count 1..100 arasında olmalıdır.")
    entries = json.loads(args.manifest.read_text(encoding="utf-8")).get("entries", [])
    if len(entries) < args.count:
        raise RuntimeError("Manifest istenen sayıda şeritli görsel içermiyor.")
    args.out.mkdir(parents=True, exist_ok=True)
    selected = []
    # Manifest farklı video kaynaklarından geldiği için eşit aralıklı seçim,
    # gündüz/gece ve yol türlerinde tekrarı azaltır.
    indexes = [round(i * (len(entries) - 1) / max(1, args.count - 1)) for i in range(args.count)]
    for number, index in enumerate(indexes, 1):
        entry = entries[index]
        name = str(entry["name"] if isinstance(entry, dict) else entry)
        source = _bdd_image_path(args.root, name)
        if source is None:
            raise FileNotFoundError(f"BDD100K görüntüsü bulunamadı: {name}")
        target = args.out / f"{number:02d}_{source.name}"
        shutil.copy2(source, target)
        selected.append({"file": target.name, "bdd100k_name": name, "source": "BDD100K"})
        print(f"[{number}/{args.count}] hazır: {target.name}")
    (args.out / "KAYNAKLAR.json").write_text(json.dumps({
        "dataset": "BDD100K", "license_notice": "BDD100K kullanım koşulları veri setinin resmî lisansına tabidir.",
        "items": selected,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
