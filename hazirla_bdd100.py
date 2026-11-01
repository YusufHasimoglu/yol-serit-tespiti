"""BDD100K'den düz/kesik şerit içeren 100 farklı video ana karesi seç.

Bu araç yalnızca yerel veri alanında bir seçim listesi yazar. Görüntü, video
ve BDD100K etiketlerini depoya kopyalamaz veya Git'e eklemez.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from lanevision.training_data import _bdd_lane_entries


def _styles(record: dict) -> set[str]:
    styles=set()
    for label in record.get('labels') or []:
        category=str(label.get('category','')).lower()
        if category!='lane' and not category.startswith('lane/'):
            continue
        attrs=label.get('attributes') or {}
        direction=str(attrs.get('laneDirection',attrs.get('direction','parallel'))).lower()
        style=str(attrs.get('laneStyle',attrs.get('style',''))).lower()
        if direction=='parallel' and style in {'solid','dashed'}:
            styles.add(style)
    return styles


def _iter_records(source: Path):
    """Read legacy per-video labels lazily; 100 choices need not load 30K files."""
    source=Path(source)
    if source.is_dir():
        for path in source.rglob('*.json'):
            video=json.loads(path.read_text(encoding='utf-8'))
            base=str(video.get('name',''))
            for frame in video.get('frames') or []:
                name=base if Path(base).suffix else f'{base}.jpg'
                yield {'name':name,'labels':frame.get('objects') or []}
        return
    raw=json.loads(source.read_text(encoding='utf-8'))
    for record in (raw if isinstance(raw,list) else [raw]):
        if 'labels' in record:
            yield record
        else:
            base=str(record.get('name',''))
            for frame in record.get('frames') or []:
                name=base if Path(base).suffix else f'{base}.jpg'
                yield {'name':name,'labels':frame.get('objects') or []}


def main() -> None:
    parser=argparse.ArgumentParser(description='BDD100K lane-style ana kare seçici')
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--count',type=int,default=100)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.count<10:
        parser.error('--count en az 10 olmalıdır.')
    both=[];solid=[];dashed=[]
    seen=set()
    for record in _iter_records(args.labels):
        name=str(record.get('name',''))
        if not name or name in seen:
            continue
        seen.add(name)
        # Only keep samples that can provide both ego-lane sides.  A frame may
        # contain solid and dashed markings elsewhere on the road while all
        # usable near-camera traces fall on one side; that is not a truthful
        # left/right training target for this application.
        selected=_bdd_lane_entries(record,1280,720)
        if len(selected)<2:
            continue
        styles={style for _,style,_,_ in selected}
        entry={'name':name,'styles':sorted(styles),'source_video':Path(name).stem}
        if styles=={'solid','dashed'}:
            both.append(entry)
            if len(both)>=args.count:
                break
        elif 'solid' in styles:
            solid.append(entry)
        elif 'dashed' in styles:
            dashed.append(entry)
    # Prefer frames that contain both target styles.  If those alone cannot
    # fill the requested count, keep the remaining styles balanced.
    chosen=both[:args.count]
    pools=(solid,dashed)
    turn=0
    while len(chosen)<args.count and any(pools):
        pool=pools[turn%2]
        if pool:
            chosen.append(pool.pop(0))
        turn+=1
        if turn>args.count*4:
            break
    if len(chosen)<args.count:
        raise SystemExit(f'Yalnızca {len(chosen)} uygun, farklı video ana karesi bulundu.')
    styles={'solid':sum('solid' in entry['styles'] for entry in chosen),
            'dashed':sum('dashed' in entry['styles'] for entry in chosen)}
    payload={'dataset':'BDD100K','purpose':'düz/kesik şerit eğitim seçimi',
             'count':len(chosen),'unique_source_videos':len({e['source_video'] for e in chosen}),
             'style_coverage':styles,'entries':chosen}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:payload[k] for k in ('count','unique_source_videos','style_coverage')},ensure_ascii=False))


if __name__=='__main__':
    main()
