"""Gerçek yol fotoğrafını eğitim için etiketle.

Klasik mod: 1=sol, 2=sağ.
--style-aware: 1=sol düz, 2=sol kesik, 3=sağ düz, 4=sağ kesik.
Sol tık nokta ekler; U geri alır, S kaydeder, ESC çıkar.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from lanevision.processing import read_image, write_image


CLASSIC_CLASSES=(("SOL",(0,0,255),1),("SAG",(0,255,0),2))
STYLE_CLASSES=(("SOL DUZ",(255,223,30),1),("SOL KESIK",(0,176,255),2),
               ("SAG DUZ",(255,80,190),3),("SAG KESIK",(0,235,120),4))


def _style_mask(shape,points,classes,thickness):
    """Kayıpsız sınıf-ID maskesi; anti-aliasing sınıf değerini bozmaz."""
    mask=np.zeros(shape,np.uint8)
    for index,(_,_,value) in enumerate(classes):
        if len(points[index])>1:
            cv2.polylines(mask,[np.asarray(points[index],np.int32)],False,value,thickness,cv2.LINE_8)
    return mask


def annotate(image_path:Path,name:str,root:Path,style_aware:bool=False):
    original=read_image(image_path)
    h,w=original.shape[:2]
    width=min(1100,w)
    scale=width/w
    height=int(h*scale)
    classes=STYLE_CLASSES if style_aware else CLASSIC_CLASSES
    points={index:[] for index in range(len(classes))}
    current=[0]
    def click(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:points[current[0]].append((round(x/scale),round(y/scale)))
    shortcuts='1 SOL-DUZ / 2 SOL-KESIK / 3 SAG-DUZ / 4 SAG-KESIK' if style_aware else '1 SOL / 2 SAG'
    win='Serit etiketleme - U GERI / S KAYDET / ESC CIK'
    cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win,click)
    while True:
        out=original.copy()
        for k,(_,color,_) in enumerate(classes):
            if len(points[k])>1:cv2.polylines(out,[np.array(points[k],np.int32)],False,color,8,cv2.LINE_AA)
            for p in points[k]:cv2.circle(out,p,5,color,-1,cv2.LINE_AA)
        show=cv2.resize(out,(width,height))
        cv2.rectangle(show,(0,0),(width,58),(12,24,38),-1)
        cv2.putText(show,f"SECILI: {classes[current[0]][0]} | {shortcuts}",(10,23),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(show,'Sol tik: nokta ekle | U: geri al | S: kaydet',(10,46),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
        cv2.imshow(win,show)
        key=cv2.waitKey(35)&255
        if key in range(ord('1'),ord(str(len(classes)))+1):current[0]=key-ord('1')
        elif key in (ord('u'),ord('U')) and points[current[0]]:points[current[0]].pop()
        elif key in (27,ord('q')):break
        elif key in (ord('s'),ord('S')):
            left_indexes=(0,1) if style_aware else (0,)
            right_indexes=(2,3) if style_aware else (1,)
            left_count=sum(len(points[index])>1 for index in left_indexes)
            right_count=sum(len(points[index])>1 for index in right_indexes)
            if left_count!=1 or right_count!=1:
                print('Uyari: Her taraf icin yalnizca bir, en az iki noktalı cizgi etiketleyin.');continue
            if style_aware:
                mask=_style_mask((h,w),points,classes,max(5,round(w/160)))
            else:
                mask=np.zeros_like(original)
                for k,(_,color,_) in enumerate(classes):
                    if len(points[k])>1:cv2.polylines(mask,[np.array(points[k],np.int32)],False,color,max(5,round(w/160)),cv2.LINE_8)
            (root/'images').mkdir(parents=True,exist_ok=True)
            (root/'masks').mkdir(parents=True,exist_ok=True)
            write_image(root/'images'/f'{name}.png',original)
            write_image(root/'masks'/f'{name}_mask.png',mask)
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
