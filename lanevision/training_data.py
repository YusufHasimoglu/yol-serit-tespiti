"""Gerçek etiket (TuSimple JSON Lines) ve tekrar üretilebilir sentetik eğitim örnekleri."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from .ai_model import INPUT_WIDTH,INPUT_HEIGHT


def synthetic_sample(seed:int, style_aware:bool=False):
    """Basitleştirilmiş sentetik yol örneği.

    ``style_aware`` olduğunda kanallar sırasıyla sol-düz, sol-kesik,
    sağ-düz ve sağ-kesiktir. Varsayılan iki kanallı biçim eski eğitim
    kontrolleriyle uyumluluğu korur.
    """
    rng=np.random.default_rng(seed)
    w,h=INPUT_WIDTH,INPUT_HEIGHT
    sky=int(rng.integers(52,91))
    img=np.zeros((h,w,3),np.uint8)
    color=rng.integers(130,218,size=3,dtype=np.uint8)
    img[:sky]=color
    green=(int(rng.integers(34,100)),int(rng.integers(55,140)),int(rng.integers(20,105)))
    img[sky:]=green
    # ağaç ve çalıların yol çizgileriyle karıştırılmaması için çeşitli dokular
    for _ in range(32):
        x=int(rng.integers(-5,w+5));y=int(rng.integers(max(0,sky-12),h));
        radius=int(rng.integers(3,25));cv2.circle(img,(x,y),radius,tuple(int(np.clip(c+rng.normal(0,20),0,255)) for c in green),-1)
    vanishing=float(rng.uniform(.35,.65)*w)
    center_shift=float(rng.uniform(-8,8)); curve=float(rng.uniform(-40,40))
    road_bottom_width=float(rng.uniform(.9,1.55)*w)
    lane_width=float(rng.uniform(.26,.46)*w)
    ys=np.arange(sky-3,h)
    t=np.clip((ys-sky)/(h-sky),0,1)
    center=vanishing+center_shift*t+curve*t*t
    left=center-(lane_width*.22+lane_width*t*.82)
    right=center+(lane_width*.22+lane_width*t*.82)
    road_l=center-road_bottom_width*t*.5-17
    road_r=center+road_bottom_width*t*.5+17
    poly=np.concatenate([np.stack([road_l,ys],1), np.stack([road_r[::-1],ys[::-1]],1)]).round().astype(np.int32)
    asphalt=int(rng.integers(42,140))
    cv2.fillPoly(img,[poly],(asphalt+int(rng.integers(-8,9)),asphalt,asphalt+int(rng.integers(-5,6))))
    img=np.clip(img.astype(np.float32)+rng.normal(0,rng.uniform(2,9),(h,w,1)),0,255).astype(np.uint8)
    masks=np.zeros((h,w,4 if style_aware else 2),np.uint8)
    thickness=int(rng.integers(2,5))
    for i,xx in enumerate((left,right)):
        pts=np.stack([xx,ys],1).round().astype(np.int32)
        # kesikli veya sürekli; ilk işaret bazen kesikli, diğeri düz
        dashed=bool(rng.integers(0,2)) if i==0 else bool(rng.random()<.2)
        mask_channel=np.zeros((h,w),np.uint8)
        if dashed:
            gap=int(rng.integers(14,27)); length=int(rng.integers(9,15))
            for j in range(0,len(pts)-length,gap):
                cv2.polylines(img,[pts[j:j+length]],False,(245,245,245),thickness,cv2.LINE_AA)
                cv2.polylines(mask_channel,[pts[j:j+length]],False,255,thickness+2)
        else:
            color=(235,238,239) if rng.random()<.75 else (40,210,242)
            cv2.polylines(img,[pts],False,color,thickness,cv2.LINE_AA)
            cv2.polylines(mask_channel,[pts],False,255,thickness+2)
        if style_aware:
            # Sol: düz=0/kesik=1, sağ: düz=2/kesik=3.
            masks[:,:,({0: (1 if dashed else 0), 1: (3 if dashed else 2)}[i])]=mask_channel
        else:
            masks[:,:,i]=mask_channel
    # Gölge / ışık değişimleri.
    if rng.random()<.6:
        x=int(rng.integers(0,w)); y=int(rng.integers(sky,h));
        xx=np.arange(w)[None];yy=np.arange(h)[:,None]
        shade=1-.32*np.exp(-((xx-x)*.013+(yy-y)*.024)**2/2)
        img=np.clip(img.astype(np.float32)*shade[:,:,None],0,255).astype(np.uint8)
    if rng.random()<.3:
        img=cv2.GaussianBlur(img,(3,3),0)
    return img,masks


def tusimple_samples(root:Path, jsonl:Path, limit:int|None=None):
    """Resim ve maskeleri yükle; -2 etiketsiz nokta, eğitim maskesine işlenmez."""
    count=0
    with Path(jsonl).open('r',encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item=json.loads(line)
            p=(root/item['raw_file']).resolve()
            if not p.is_file():continue
            src=cv2.imread(str(p));
            if src is None:continue
            sh,sw=src.shape[:2]
            masks=np.zeros((INPUT_HEIGHT,INPUT_WIDTH,2),np.uint8)
            # Referans merkezine en yakın iki çizgiyi tespit et (soldaki/sağdaki)
            lanes=[]
            for line_points in item['lanes']:
                pts=[(x,y) for x,y in zip(line_points,item['h_samples']) if x>=0 and y>=0]
                if len(pts)<3:continue
                lower=max(pts,key=lambda pt:pt[1]);lanes.append((lower[0],pts))
            lefts=sorted((k,p) for k,p in lanes if k<sw/2)
            rights=sorted((k,p) for k,p in lanes if k>=sw/2)
            for i,item_lane in enumerate((lefts[-1] if lefts else None,rights[0] if rights else None)):
                if item_lane is None: continue
                pts=np.array([[int(x*INPUT_WIDTH/sw),int(y*INPUT_HEIGHT/sh)] for x,y in item_lane[1]],np.int32)
                canvas=np.zeros((INPUT_HEIGHT,INPUT_WIDTH),np.uint8)
                cv2.polylines(canvas,[pts],False,255,3,cv2.LINE_AA)
                masks[:,:,i]=canvas
            im=cv2.resize(src,(INPUT_WIDTH,INPUT_HEIGHT))
            yield im,masks
            count+=1
            if limit and count>=limit:break


def _bdd_image_path(root: Path, name: str) -> Path | None:
    """Find a BDD100K image without assuming one extraction layout."""
    clean=Path(name.replace('\\\\','/'))
    candidates=(root/clean, root/'images'/'100k'/'train'/clean.name,
                root/'images'/'100k'/'val'/clean.name,
                root/'bdd100k'/'images'/'100k'/'train'/clean.name,
                root/'bdd100k'/'images'/'100k'/'val'/clean.name,
                root/'train'/clean.name, root/'val'/clean.name)
    return next((p for p in candidates if p.is_file()),None)


def _bdd_records(labels_source: Path):
    """Normalize current and legacy BDD100K label layouts to frame records."""
    labels_source=Path(labels_source)
    if labels_source.is_dir():
        raw_records=[]
        for path in sorted(labels_source.rglob('*.json')):
            raw_records.append(json.loads(path.read_text(encoding='utf-8')))
    else:
        raw=json.loads(labels_source.read_text(encoding='utf-8'))
        raw_records=raw if isinstance(raw,list) else [raw]
    records=[]
    for record in raw_records:
        # New BDD layout: one record already has `name` and `labels`.
        if 'labels' in record:
            records.append(record)
            continue
        # Legacy BDD layout: a video-level record with its labelled keyframe
        # inside `frames`, and lane paths inside `objects`.
        base=str(record.get('name',''))
        for frame in record.get('frames') or []:
            name=base if Path(base).suffix else f'{base}.jpg'
            records.append({'name':name,'labels':frame.get('objects') or []})
    return records


def _bdd_lane_entries(record: dict, width: int, height: int):
    """Return closest parallel solid/dashed lane paths from one BDD record."""
    lanes=[]
    for label in record.get('labels') or []:
        category=str(label.get('category','')).lower()
        if category!='lane' and not category.startswith('lane/'):
            continue
        attributes=label.get('attributes') or {}
        direction=str(attributes.get('laneDirection',attributes.get('direction','parallel'))).lower()
        style=str(attributes.get('laneStyle',attributes.get('style',''))).lower()
        if direction!='parallel' or style not in {'solid','dashed'}:
            continue
        paths=label.get('poly2d') or label.get('seg2d') or []
        # Legacy labels keep one polyline as [[x,y,"L|C"], ...], while the
        # newer layout stores a list of {vertices: ...} dictionaries.
        if paths and isinstance(paths[0],(list,tuple)):
            paths=[{'vertices':[point[:2] for point in paths]}]
        for path in paths:
            vertices=path.get('vertices',[]) if isinstance(path,dict) else []
            points=np.asarray(vertices,dtype=np.float32)
            if points.ndim!=2 or points.shape[0]<3 or points.shape[1]<2:
                continue
            # BDD stores Bezier vertices/control points in order.  Joining the
            # given path is a deliberate lightweight rasterisation; this model
            # is not claimed to reconstruct an exact Bezier curve.
            points=points[:,:2]
            lowest=points[np.argmax(points[:,1])]
            side='left' if lowest[0]<width*.5 else 'right'
            lanes.append((side,style,float(abs(lowest[0]-width*.5)),points))
    chosen=[]
    for side in ('left','right'):
        side_lanes=sorted((lane for lane in lanes if lane[0]==side),key=lambda lane:lane[2])
        if side_lanes:
            chosen.append(side_lanes[0])
    return chosen


def bdd_style_samples(root: Path, labels_json: Path, limit: int|None=None,
                      manifest: Path|None=None):
    """Load BDD100K frames with official solid/dashed lane attributes.

    The output mask channels are ``left_solid``, ``left_dashed``,
    ``right_solid``, ``right_dashed``.  A manifest can restrict the loader to
    a reproducibly selected set of distinct source video keyframes.
    """
    root=Path(root)
    labels_json=Path(labels_json)
    records=[]
    wanted=None
    if manifest:
        data=json.loads(Path(manifest).read_text(encoding='utf-8'))
        wanted=[entry['name'] if isinstance(entry,dict) else str(entry)
                for entry in data.get('entries',data)]
        # Legacy BDD labels are individual files named after the video ID.
        # Reading only the manifest's hundred files avoids loading tens of
        # thousands of unrelated JSON records before a small fine-tune.
        if labels_json.is_dir():
            selected=[]
            for name in wanted:
                label_file=labels_json/f'{Path(name).stem}.json'
                if label_file.is_file():
                    selected.extend(_bdd_records(label_file))
            records=selected
        else:
            records=_bdd_records(labels_json)
            by_name={str(item.get('name')):item for item in records}
            records=[by_name[name] for name in wanted if name in by_name]
    else:
        records=_bdd_records(labels_json)
    count=0
    for record in records:
        name=str(record.get('name',''))
        path=_bdd_image_path(root,name)
        if path is None:
            continue
        source=cv2.imread(str(path))
        if source is None:
            continue
        sh,sw=source.shape[:2]
        selected=_bdd_lane_entries(record,sw,sh)
        if len(selected)<2:
            continue
        masks=np.zeros((INPUT_HEIGHT,INPUT_WIDTH,4),np.uint8)
        for side,style,_,points in selected:
            points=points.copy()
            points[:,0]*=INPUT_WIDTH/sw;points[:,1]*=INPUT_HEIGHT/sh
            index={('left','solid'):0,('left','dashed'):1,
                   ('right','solid'):2,('right','dashed'):3}[(side,style)]
            # NumPy'nin ``masks[:, :, index]`` görünümü OpenCV için bitişik
            # bellek değildir; bağımsız kanalda çizip geri yerleştiriyoruz.
            channel=masks[:,:,index].copy()
            cv2.polylines(channel,[np.rint(points).astype(np.int32)],False,255,4,cv2.LINE_AA)
            masks[:,:,index]=channel
        if not np.any(masks):
            continue
        yield cv2.resize(source,(INPUT_WIDTH,INPUT_HEIGHT)),masks,Path(name).stem
        count+=1
        if limit and count>=limit:
            break


def load_user_samples(folder:Path):
    """data/gercek/images/*.png ve data/gercek/masks/*_mask.png.

    Maske kırmızı=sol, yeşil=sağ. Poligon etiketleri eğitim maskesine çevrilir.
    """
    folder=Path(folder)
    imgroot=folder/'images';maskroot=folder/'masks'
    if not imgroot.is_dir():return
    for p in sorted(imgroot.iterdir()):
        if p.suffix.lower() not in {'.png','.jpg','.jpeg','.webp'}:continue
        label=maskroot/(p.stem+'_mask.png')
        if not label.is_file():continue
        image=cv2.imread(str(p));mask=cv2.imread(str(label))
        if image is None or mask is None:continue
        image=cv2.resize(image,(INPUT_WIDTH,INPUT_HEIGHT));mask=cv2.resize(mask,(INPUT_WIDTH,INPUT_HEIGHT),interpolation=cv2.INTER_NEAREST)
        # OpenCV BGR: kırmızı kanal 2 sol, yeşil kanal 1 sağ
        labels=np.stack([(mask[:,:,2]>150).astype(np.uint8)*255,(mask[:,:,1]>150).astype(np.uint8)*255],-1)
        yield image,labels
