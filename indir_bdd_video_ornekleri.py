"""Resmî BDD100K arşivinden manifestteki sınırlı sayıdaki videoyu al.

Büyük (TB ölçeğindeki) arşivin tamamı indirilmez. HTTP byte-range desteğiyle
merkez dizin okunur, sonra sadece istenen video üyeleri çekilir. Zipfile, her
çıktının ZIP CRC bilgisini okuma sırasında denetler. Ham videoların hedefi
``data/external`` olmalıdır ve bu klasör Git tarafından yok sayılır.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL='http://128.32.162.150/bdd100k/bdd100k_videos.zip'


class HTTPRangeReader(io.RawIOBase):
    """A small seekable reader backed by authenticated-size HTTP ranges."""
    def __init__(self,url: str, timeout: int=90):
        self.url=url;self.timeout=timeout;self.position=0
        request=urllib.request.Request(url,method='HEAD')
        with urllib.request.urlopen(request,timeout=timeout) as response:
            self.size=int(response.headers['Content-Length'])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self,offset,whence=io.SEEK_SET):
        if whence==io.SEEK_CUR: offset=self.position+offset
        elif whence==io.SEEK_END: offset=self.size+offset
        if offset<0: raise ValueError('Aralık okuyucusunda negatif konum yok.')
        self.position=min(offset,self.size)
        return self.position
    def read(self,size=-1):
        if size is None or size<0: size=self.size-self.position
        if size==0 or self.position>=self.size: return b''
        end=min(self.size-1,self.position+size-1)
        request=urllib.request.Request(self.url,headers={'Range':f'bytes={self.position}-{end}'})
        with urllib.request.urlopen(request,timeout=self.timeout) as response:
            data=response.read()
        if len(data)!=end-self.position+1:
            raise OSError('Sunucu beklenen byte aralığının tamamını döndürmedi.')
        self.position=end+1
        return data


def _wanted_stems(manifest: Path, limit: int) -> list[str]:
    data=json.loads(manifest.read_text(encoding='utf-8'))
    entries=data.get('entries',data)
    stems=[]
    for entry in entries:
        name=entry.get('source_video') if isinstance(entry,dict) else entry
        stem=Path(str(name)).stem
        if stem and stem not in stems:
            stems.append(stem)
        if len(stems)>=limit:
            break
    if len(stems)<limit:
        raise ValueError(f'Manifestte yalnızca {len(stems)} farklı video anahtarı var.')
    return stems


def main() -> None:
    parser=argparse.ArgumentParser(description='BDD100K manifest video indiricisi')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--count',type=int,default=100)
    parser.add_argument('--url',default=DEFAULT_URL)
    args=parser.parse_args()
    if args.count<1 or args.count>100:
        parser.error('--count 1..100 arasında olmalıdır.')
    stems=_wanted_stems(args.manifest,args.count)
    args.out.mkdir(parents=True,exist_ok=True)
    reader=HTTPRangeReader(args.url)
    with zipfile.ZipFile(reader) as archive:
        by_stem={Path(info.filename).stem:info for info in archive.infolist()
                 if Path(info.filename).suffix.lower() in {'.mov','.mp4'}}
        missing=[stem for stem in stems if stem not in by_stem]
        if missing:
            raise RuntimeError(f'Arşivde {len(missing)} manifest videosu bulunamadı; ilk örnek: {missing[0]}')
        for index,stem in enumerate(stems,1):
            target=args.out/f'{stem}{Path(by_stem[stem].filename).suffix.lower()}'
            if target.is_file() and target.stat().st_size==by_stem[stem].file_size:
                print(f'[{index}/{len(stems)}] var: {target.name}',flush=True)
                continue
            with archive.open(by_stem[stem]) as source, target.open('wb') as destination:
                shutil.copyfileobj(source,destination,1024*1024)
            print(f'[{index}/{len(stems)}] indirildi: {target.name}',flush=True)

    metadata = {
        'dataset': 'BDD100K',
        'source': 'https://github.com/bdd100k/bdd100k',
        'download_documentation': 'https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst',
        'archive': args.url,
        'count': len(stems),
        'videos': [
            f'{stem}{Path(by_stem[stem].filename).suffix.lower()}' for stem in stems
        ],
        'notice': (
            'Bu ornekler egitim ve arastirma amaciyla hazirlanmistir. '
            'Kullanim BDD100K veri setinin lisans ve kullanim kosullarina tabidir.'
        ),
    }
    (args.out / 'KAYNAKLAR.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8'
    )


if __name__=='__main__':
    main()
