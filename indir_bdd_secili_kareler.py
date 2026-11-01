"""BDD100K görüntü arşivinden manifestteki 100 etiketli ana kareyi al.

5 GB arşivin tamamını indirmek yerine ZIP merkez dizini ile yalnızca seçilen
dosyaların byte aralıkları indirilir. Ham görüntüler ``data/external`` altında
kalmalı ve Git'e eklenmemelidir.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from indir_bdd_video_ornekleri import HTTPRangeReader


DEFAULT_URL='http://128.32.162.150/bdd100k/bdd100k_images_100k.zip'


def _wanted_names(manifest: Path, count: int) -> list[str]:
    data=json.loads(manifest.read_text(encoding='utf-8'))
    names=[]
    for entry in data.get('entries',data):
        name=entry.get('name') if isinstance(entry,dict) else str(entry)
        name=Path(name).name
        if not Path(name).suffix:
            name+='.jpg'
        if name not in names:
            names.append(name)
        if len(names)>=count:
            break
    if len(names)<count:
        raise ValueError(f'Manifestte yalnızca {len(names)} farklı kare var.')
    return names


def main() -> None:
    parser=argparse.ArgumentParser(description='BDD100K seçili ana kare indiricisi')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--count',type=int,default=100)
    parser.add_argument('--url',default=DEFAULT_URL)
    args=parser.parse_args()
    if args.count<1 or args.count>100:
        parser.error('--count 1..100 arasında olmalıdır.')
    names=_wanted_names(args.manifest,args.count)
    args.out.mkdir(parents=True,exist_ok=True)
    print('BDD100K görüntü merkez dizini okunuyor...',flush=True)
    reader=HTTPRangeReader(args.url)
    with zipfile.ZipFile(reader) as archive:
        indexed={Path(info.filename).name:info for info in archive.infolist()
                 if Path(info.filename).suffix.lower() in {'.jpg','.jpeg','.png'}}
        missing=[name for name in names if name not in indexed]
        if missing:
            raise RuntimeError(f'Arşivde {len(missing)} seçili ana kare bulunamadı; ilk örnek: {missing[0]}')
        for index,name in enumerate(names,1):
            target=args.out/name
            info=indexed[name]
            if target.is_file() and target.stat().st_size==info.file_size:
                print(f'[{index}/{len(names)}] var: {name}',flush=True)
                continue
            with archive.open(info) as source, target.open('wb') as destination:
                shutil.copyfileobj(source,destination,1024*1024)
            print(f'[{index}/{len(names)}] indirildi: {name}',flush=True)


if __name__=='__main__':
    main()
