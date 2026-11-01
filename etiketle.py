"""Gerçek yol fotoğrafını eğitim için etiketle.

Klasik mod: 1=sol, 2=sağ.
--style-aware: 1=sol düz, 2=sol kesik, 3=sağ düz, 4=sağ kesik.
5=iki sınır arasındaki alan. Sol tık nokta ekler; U geri alır, S kaydeder.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from lanevision.processing import read_image, write_image
from lanevision.feedback import area_mask_from_boundaries, record_review


CLASSIC_CLASSES=(("SOL",(0,0,255),1),("SAG",(0,255,0),2))
STYLE_CLASSES=(("SOL DUZ",(255,223,30),1),("SOL KESIK",(0,176,255),2),
               ("SAG DUZ",(255,80,190),3),("SAG KESIK",(0,235,120),4))


def _style_mask(shape,strokes,classes,thickness):
    """Kayıpsız sınıf-ID maskesi; anti-aliasing sınıf değerini bozmaz."""
    mask=np.zeros(shape,np.uint8)
    for index,(_,_,value) in enumerate(classes):
        for stroke in strokes[index]:
            if len(stroke)>1:
                cv2.polylines(mask,[np.asarray(stroke,np.int32)],False,value,thickness,cv2.LINE_8)
    return mask


def annotate(image_path:Path,name:str,root:Path,style_aware:bool=False):
    original=read_image(image_path)
    h,w=original.shape[:2]
    width=min(1100,w)
    scale=width/w
    height=int(h*scale)
    classes=STYLE_CLASSES if style_aware else CLASSIC_CLASSES
    # Her türde birden fazla parça tutulur. Kesik çizgide N ile yeni parça
    # başlatıldığında aradaki asfalt boşluğu eğitim maskesinde korunur.
    strokes={index:[[]] for index in range(len(classes))}
    area_points=[]
    area_index=len(classes)
    current=[0]

    def boundary(indexes):
        for index in indexes:
            parts=[np.asarray(stroke,np.int32) for stroke in strokes[index] if len(stroke)>1]
            if parts:
                return np.vstack(parts)
        return None

    def click(event,x,y,flags,param):
        if event!=cv2.EVENT_LBUTTONDOWN:return
        point=(round(x/scale),round(y/scale))
        if current[0]==area_index:area_points.append(point)
        else:strokes[current[0]][-1].append(point)
    shortcuts=('1 SOL-DUZ / 2 SOL-KESIK / 3 SAG-DUZ / 4 SAG-KESIK / 5 ALAN'
               if style_aware else '1 SOL / 2 SAG / 5 ALAN')
    win='Serit etiketleme - U GERI / S KAYDET / ESC CIK'
    cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win,click)
    while True:
        out=original.copy()
        left_indexes=(0,1) if style_aware else (0,)
        right_indexes=(2,3) if style_aware else (1,)
        manual=np.asarray(area_points,np.int32) if len(area_points)>=3 else None
        area=area_mask_from_boundaries((h,w),boundary(left_indexes),boundary(right_indexes),manual)
        if np.any(area):
            tint=out.copy();tint[area>0]=(70,177,105)
            out=cv2.addWeighted(out,.70,tint,.30,0)
        if len(area_points)>=2:
            cv2.polylines(out,[np.asarray(area_points,np.int32)],len(area_points)>=3,(75,235,145),3,cv2.LINE_AA)
        for point in area_points:cv2.circle(out,point,5,(75,235,145),-1,cv2.LINE_AA)
        for k,(_,color,_) in enumerate(classes):
            for stroke in strokes[k]:
                if len(stroke)>1:cv2.polylines(out,[np.array(stroke,np.int32)],False,color,8,cv2.LINE_AA)
                for p in stroke:cv2.circle(out,p,5,color,-1,cv2.LINE_AA)
        show=cv2.resize(out,(width,height))
        cv2.rectangle(show,(0,0),(width,58),(12,24,38),-1)
        selected='ARADAKI ALAN' if current[0]==area_index else classes[current[0]][0]
        cv2.putText(show,f"SECILI: {selected} | {shortcuts}",(10,23),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(show,'5: alan | N: yeni kesik parca | U: geri al | S: kaydet',(10,46),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
        cv2.imshow(win,show)
        key=cv2.waitKey(35)&255
        if key in range(ord('1'),ord(str(len(classes)))+1):current[0]=key-ord('1')
        elif key==ord('5'):current[0]=area_index
        elif key in (ord('u'),ord('U')):
            if current[0]==area_index and area_points:area_points.pop()
            elif current[0]!=area_index and strokes[current[0]][-1]:strokes[current[0]][-1].pop()
        elif key in (ord('n'),ord('N')) and current[0]!=area_index and strokes[current[0]][-1]:
            strokes[current[0]].append([])
        elif key in (27,ord('q')):break
        elif key in (ord('s'),ord('S')):
            has_line=lambda index:any(len(stroke)>1 for stroke in strokes[index])
            left_count=sum(has_line(index) for index in left_indexes)
            right_count=sum(has_line(index) for index in right_indexes)
            if left_count!=1 or right_count!=1:
                print('Uyari: Her taraf icin yalnizca bir, en az iki noktalı cizgi etiketleyin.');continue
            if not np.any(area):
                print('Uyari: Alan olusturulamadi. Sinirlari kontrol edin veya 5 ile alan cokgeni cizin.');continue
            if style_aware:
                mask=_style_mask((h,w),strokes,classes,max(5,round(w/160)))
            else:
                mask=np.zeros_like(original)
                for k,(_,color,_) in enumerate(classes):
                    for stroke in strokes[k]:
                        if len(stroke)>1:cv2.polylines(mask,[np.array(stroke,np.int32)],False,color,max(5,round(w/160)),cv2.LINE_8)
            (root/'images').mkdir(parents=True,exist_ok=True)
            (root/'masks').mkdir(parents=True,exist_ok=True)
            write_image(root/'images'/f'{name}.png',original)
            write_image(root/'masks'/f'{name}_mask.png',mask)
            write_image(root/'areas'/f'{name}_area.png',area)
            record_review(root,original,name,image_path.name,'manual_correction')
            print('Kaydedildi:',root/'images'/f'{name}.png');break
    cv2.destroyAllWindows()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--image',required=True,type=Path)
    p.add_argument('--name',default=None)
    p.add_argument('--outdir',type=Path,default=Path('data/gercek'))
    p.add_argument('--style-aware',action='store_true',help='Duz/kesik ile sol/sag siniflarini ayri etiketle.')
    args=p.parse_args()
    annotate(args.image,args.name or args.image.stem,args.outdir,args.style_aware)
