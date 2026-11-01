"""Gerçek yol görüntüsünde elle sol/sağ şerit çizme ve eğitim maskesi kaydetme.

Kullanım: python etiketle.py --image yol.png --name yol_01
1=sol (kırmızı), 2=sağ (yeşil), sol tık=kontrol noktası,
U=geri al, S=kaydet, ESC=çık. Yalnızca eğitim için.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from lanevision.processing import read_image, write_image


def annotate(image_path:Path,name:str,root:Path):
    original=read_image(image_path)
    h,w=original.shape[:2]
    width=min(1100,w)
    scale=width/w
    height=int(h*scale)
    display=cv2.resize(original,(width,height))
    points={0:[],1:[]}
    current=[0]
    def click(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:points[current[0]].append((round(x/scale),round(y/scale)))
    win='Serit etiketleme - 1 SOL / 2 SAG / U GERI / S KAYDET / ESC CIK'
    cv2.namedWindow(win,cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win,click)
    while True:
        out=original.copy()
        for k,color in ((0,(0,0,255)),(1,(0,255,0))):
            if len(points[k])>1:cv2.polylines(out,[np.array(points[k],np.int32)],False,color,8,cv2.LINE_AA)
            for p in points[k]:cv2.circle(out,p,5,color,-1)
        show=cv2.resize(out,(width,height))
        cv2.rectangle(show,(0,0),(width,36),(12,24,38),-1)
        cv2.putText(show,f"SECILI: {'SOL (KIRMIZI)' if current[0]==0 else 'SAG (YESIL)'} | U: GERI | S: KAYDET",(10,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
        cv2.imshow(win,show)
        key=cv2.waitKey(35)&255
        if key==ord('1'):current[0]=0
        elif key==ord('2'):current[0]=1
        elif key in (ord('u'),ord('U')) and points[current[0]]:points[current[0]].pop()
        elif key in (27,ord('q')):break
        elif key in (ord('s'),ord('S')):
            mask=np.zeros_like(original)
            for k,color in ((0,(0,0,255)),(1,(0,255,0))):
                if len(points[k])>1:cv2.polylines(mask,[np.array(points[k],np.int32)],False,color,max(5,round(w/160)),cv2.LINE_AA)
            if min(len(points[0]),len(points[1]))<2:
                print('Uyarı: En az iki noktadan oluşan iki şeridi etiketleyin.');continue
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
    args=p.parse_args()
    annotate(args.image,args.name or args.image.stem,args.outdir)