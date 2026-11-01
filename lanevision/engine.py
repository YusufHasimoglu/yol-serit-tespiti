"""Dört yöntem: klasik, eğrisel renk, yapay zekâ ve otomatik hibrit."""
from __future__ import annotations
import cv2
import numpy as np
from .core import LaneDetector,LaneSettings,Detection
from .hybrid import detect_hybrid

MODES={'auto':'Otomatik (AI + eğrisel yedek)', 'ai':'Yapay zekâ (U-Net)',
       'hybrid':'Gelişmiş eğrisel takip', 'classic':'Klasik Canny + Hough'}


def _fit_mask(mask:np.ndarray,side:str):
    h,w=mask.shape
    # Yanlış pozitifleri azalt: küçücük bileşenleri kaldır ve çizgiyi sınıfa göre ara.
    labels_count,labels,stats,centers=cv2.connectedComponentsWithStats((mask>0).astype('uint8'),8)
    selected=np.zeros_like(mask)
    # Büyük ana çizgi bileşenini seç; soldaki kesik çizgiler için güçlü 2 parça koru.
    valid_parts=[]
    for i in range(1,labels_count):
        x,y,bw,bh,area=stats[i]
        if area<max(22,h*w*.00007):continue
        cx=centers[i,0]
        if side=='left' and cx>w*.68:continue
        if side=='right' and cx<w*.39:continue
        valid_parts.append((int(area),i))
    valid_parts.sort(reverse=True)
    if valid_parts:
        threshold=max(32,int(valid_parts[0][0]*.09))
        primary_x=centers[valid_parts[0][1]][0]
        kept=0
        for area,i in valid_parts:
            if area>=threshold and (side=='right' and kept==0 or side=='left' and abs(centers[i][0]-primary_x)<w*.105):
                selected[labels==i]=255
                kept+=1
                if kept >= (3 if side=='left' else 1):break
    yy,xx=np.nonzero(selected)
    if len(xx)<max(85,int(h*w*.00015)) or np.ptp(yy)<h*.10:return None
    # Kalın maskelerde merkez çizgiyi çıkar; sıra sıra ağırlıklı orta nokta.
    y_steps=np.arange(max(int(h*.38),int(np.min(yy))),min(h-1,int(np.max(yy)))+1,4)
    points=[]
    for y in y_steps:
        xvals=np.flatnonzero(np.any(selected[max(0,y-2):min(h,y+3)]>0,axis=0))
        if len(xvals)>=2:
            # Başka çizgi küçük kümelerinin etkisini en aza indirir.
            points.append((float(np.median(xvals)),float(y)))
    if len(points)<6:return None
    xy=np.array(points);xs,ys=xy[:,0],xy[:,1]
    deg=2 if np.ptp(ys)>h*.26 and len(points)>=13 else 1
    try:
        coeffs=np.polyfit(ys,xs,deg)
        for _ in range(2):
            err=np.abs(xs-np.polyval(coeffs,ys))
            keep=err<max(13,np.median(err)*2.7)
            if keep.sum()>=max(6,deg+3):coeffs=np.polyfit(ys[keep],xs[keep],deg)
    except (ValueError,np.linalg.LinAlgError):return None
    start,end=int(np.min(ys)),int(np.max(ys))
    ydraw=np.arange(start,end+1,3)
    xdraw=np.polyval(coeffs,ydraw)
    valid=(xdraw>=0)&(xdraw<w)
    pts=np.stack([xdraw[valid],ydraw[valid]],1).round().astype(np.int32)
    if len(pts)<5:return None
    return pts,coeffs


def detect_ai(frame,settings,segmenter):
    masks,probs=segmenter.masks(frame)
    h,w=frame.shape[:2]
    detected=[_fit_mask(masks[0],'left'),_fit_mask(masks[1],'right')]
    # Modelin belirsiz olduğu yerleri gerçek çizgi diye göstermeyelim.
    output=frame.copy();overlay=np.zeros_like(frame)
    left,right=detected
    offset=None
    if left is not None and right is not None:
        ya=max(left[0][0,1],right[0][0,1]);yb=min(left[0][-1,1],right[0][-1,1])
        if yb>ya+20:
            ys=np.arange(ya,yb+1,4);lx=np.polyval(left[1],ys);rx=np.polyval(right[1],ys)
            valid=(lx<rx)&(lx>=0)&(rx<w)
            if valid.sum()>5:
                lp=np.stack([lx[valid],ys[valid]],1).round().astype(np.int32)
                rp=np.stack([rx[valid],ys[valid]],1).round().astype(np.int32)
                cv2.fillPoly(overlay,[np.vstack([lp,rp[::-1]])],(68,182,97))
                offset=w/2-float((lp[-1,0]+rp[-1,0])/2)
    output=cv2.addWeighted(output,1.,overlay,settings.overlay_alpha,0)
    for item in (left,right):
        if item is not None:cv2.polylines(output,[item[0]],False,(0,225,255),settings.line_thickness,cv2.LINE_AA)
    def ends(obj):
        return (tuple(map(int,obj[0][-1])),tuple(map(int,obj[0][0]))) if obj is not None else None
    le,re=ends(left),ends(right)
    cv2.rectangle(output,(8,7),(min(w-5,520),47),(18,28,44),-1)
    cv2.putText(output,f"SOL: {'VAR' if le else 'YOK'}  SAG: {'VAR' if re else 'YOK'} | U-NET",
                (17,34),cv2.FONT_HERSHEY_SIMPLEX,.65,(242,249,255),2,cv2.LINE_AA)
    mask_viz=np.zeros_like(frame);mask_viz[:,:,2]=masks[0];mask_viz[:,:,1]=masks[1]
    prob_viz=np.zeros_like(frame);prob_viz[:,:,2]=(probs[:,:,0]*255).astype('uint8');prob_viz[:,:,1]=(probs[:,:,1]*255).astype('uint8')
    stages={'01 Orijinal':frame.copy(),'08 Sonuc':output,'09 AI olasilik haritasi':prob_viz,'10 AI sinif maskeleri':mask_viz}
    det=Detection(output,stages,le,re,offset,float(le is not None)*.5+float(re is not None)*.5,0)
    det.method='Yapay zekâ (U-Net)'
    return det


class InferenceEngine:
    def __init__(self,settings:LaneSettings|None=None,mode='auto',weights=None):
        if mode not in MODES:raise ValueError(f'Yöntem desteklenmiyor: {mode}')
        self.settings=settings or LaneSettings()
        self.settings.validate()
        self.mode=mode
        self.classic=LaneDetector(self.settings)
        self.segmenter=None
        self.model_warning=None
        if mode in {'auto','ai'}:
            try:
                from .ai_model import AISegmenter,DEFAULT_WEIGHTS
                self.segmenter=AISegmenter(weights=weights or DEFAULT_WEIGHTS,threshold=.70)
            except (ImportError,FileNotFoundError,RuntimeError,ValueError) as exc:
                self.model_warning=str(exc)
                if mode=='ai':raise RuntimeError('Yapay zekâ modeli yüklenemedi: '+str(exc)) from exc

    def detect(self,frame,video=False):
        if self.mode=='classic':
            result=self.classic.detect(frame,video)
            result.method='Klasik Hough'
            return result
        if self.mode=='hybrid':
            result=detect_hybrid(frame,self.settings)
            result.method='Eğrisel renk takibi'
            return result
        ai=None
        if self.segmenter is not None:
            ai=detect_ai(frame,self.settings,self.segmenter)
            if self.mode=='ai':return ai
        # Otomatik: model ikisini de bulursa onunla göster; değilse eğrisel renk yaklaşımı.
        if ai is not None and ai.both_found:
            # Kısa bir mesafede aşırı yatay kayma, bariyer/şerit karışmasına işaret edebilir.
            def plausible(line):
                (xb,yb),(xt,yt)=line
                return abs(xb-xt) / max(1,abs(yb-yt)) < 3.4
            if plausible(ai.left) and plausible(ai.right):
                return ai
        color=detect_hybrid(frame,self.settings)
        color.method='Eğrisel renk takibi (AI yetersiz/yok)'
        if color.both_found or (ai is None and (color.left is not None or color.right is not None)):
            if ai is not None:
                color.stages.update({k:v for k,v in ai.stages.items() if k.startswith(('09','10'))})
            return color
        classic=self.classic.detect(frame,video)
        classic.method='Klasik Hough (yedek)'
        if classic.confidence>=color.confidence:
            if ai is not None:classic.stages.update({k:v for k,v in ai.stages.items() if k.startswith(('09','10'))})
            return classic
        return color