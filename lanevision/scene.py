"""Hafif, açıklanabilir görüntü profili analizi.

Bu sınıflandırma öğrenilmiş bir hava/viraj sınıflandırıcısı DEĞİLDİR. Kameradaki
aydınlatma ve görüntüdeki olası çizgi geometrisinin tahminini verir; belirsizliği
raporlar. Sahnede tespit edilmiş gerçek şerit olmadan viraj kesin söylenemez.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import cv2
import numpy as np


@dataclass(frozen=True)
class SceneProfile:
    lighting: Literal['gece', 'gunduz', 'alacakaranlik']
    geometry: Literal['viraj', 'duz', 'belirsiz']
    brightness: float
    curve_score: float
    geometry_evidence: int

    @property
    def label(self):
        light={'gece':'GECE','gunduz':'GÜNDÜZ','alacakaranlik':'ALACAKARANLIK'}[self.lighting]
        shape={'viraj':'VİRAJLI','duz':'DÜZ','belirsiz':'YOL ŞEKLİ BELİRSİZ'}[self.geometry]
        return f'{light}  ·  {shape}'

    @property
    def short_lighting(self):
        return {'gece':'GECE','gunduz':'GÜNDÜZ','alacakaranlik':'ALACAKARANLIK'}[self.lighting]

    @property
    def short_geometry(self):
        return {'viraj':'VİRAJ','duz':'DÜZ','belirsiz':'BELİRSİZ'}[self.geometry]


def analyze_scene(frame: np.ndarray) -> SceneProfile:
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 64:
        raise ValueError('Sahne analizi için en az 64x64 BGR görüntü gereklidir.')
    h,w=frame.shape[:2]
    small=cv2.resize(frame,(320,180),interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    v=hsv[:,:,2]
    # Parlak gökyüzünü tek başına gündüz sayma: alt ve orta bölgeleri de dikkate al.
    upper=float(np.median(v[:82,:]))
    road=float(np.median(v[100:,:]))
    brightness=round(.37*upper+.63*road,1)
    if brightness < 69:
        lighting='gece'
    elif brightness > 110:
        lighting='gunduz'
    else:
        lighting='alacakaranlik'

    # Şerit adayı piksellerin kolon merkezlerini yatay dilimlerde izle.
    # Bu yaklaşım zor vakalarda belirsiz üretir; tek başına şerit tespiti DEĞİLDİR.
    hh,ww=180,320
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(6,6)).apply(gray)
    enhanced=cv2.GaussianBlur(clahe,(3,3),0)
    white=(enhanced>np.maximum(165,cv2.GaussianBlur(enhanced,(0,0),7)+37))
    hsvs=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    yellow=(hsvs[:,:,0]>10)&(hsvs[:,:,0]<38)&(hsvs[:,:,1]>75)&(hsvs[:,:,2]>95)
    cand=((white|yellow).astype(np.uint8)*255)
    # Gökyüzü, ağaç ve korkulukların yol dışı piksellerini azaltan trapez ROI.
    roi=np.zeros_like(cand)
    cv2.fillPoly(roi,[np.array([[5,179],[115,79],[205,79],[315,179]],np.int32)],255)
    cand=cv2.bitwise_and(cand,roi)
    pts={'left':[],'right':[]}
    for y in range(95,175,8):
        ya,yb=y-4,y+5
        for side,(xmin,xmax) in (('left',(35,163)),('right',(163,294))):
            col=np.count_nonzero(cand[ya:yb,xmin:xmax],axis=0).astype(np.float32)
            if col.size<4:continue
            smooth=np.convolve(col,np.ones(5)/5,mode='same')
            idx=int(np.argmax(smooth))
            if smooth[idx]>=3.0:
                pts[side].append((y,float(xmin+idx)))
    residuals=[]
    evidence=0
    for side in ('left','right'):
        track=np.array(pts[side],dtype=np.float64)
        if len(track)<6:continue
        ys,xs=track[:,0],track[:,1]
        try:
            linear=np.polyfit(ys,xs,1)
            curve=np.polyfit(ys,xs,2)
            rms_linear=np.sqrt(np.mean((xs-np.polyval(linear,ys))**2))
            rms_curve=np.sqrt(np.mean((xs-np.polyval(curve,ys))**2))
            # Geometrik kanıt, polinomun hata düşürmesine ve sapma büyüklüğüne bağlı.
            gain=max(0.0,rms_linear-rms_curve)
            deviation=abs(curve[0])*80**2 / 4
            if rms_curve<18 and gain>1.2:
                residuals.append(deviation)
                evidence+=len(track)
            elif rms_linear<10:
                residuals.append(0.)
                evidence+=len(track)
        except (np.linalg.LinAlgError,ValueError):
            pass
    score=round(float(max(residuals,default=0.)),1)
    geometry='belirsiz'
    if evidence>=7:
        geometry='viraj' if score>=11 else 'duz'
    return SceneProfile(lighting,geometry,brightness,score,evidence)