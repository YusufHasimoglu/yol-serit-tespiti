"""Dört yöntem: klasik, eğrisel renk, yapay zekâ ve otomatik hibrit."""
from __future__ import annotations
from dataclasses import replace
import cv2
import numpy as np
from .core import LaneDetector,LaneSettings,Detection
from .hybrid import detect_hybrid
from .scene import analyze_scene
from .continuity import detect_continuous
from .ufld import UFLDDetector
from .yolopv2 import YOLOPv2Detector

MODES={'auto':'Akıllı otomatik (gece / viraj / AI)', 'ai':'Yapay zekâ (U-Net)',
       'ufld':'Yapay zekâ (UFLD V2 derin şerit)',
       'yolopv2':'Gelişmiş AI (YOLOPv2 şerit + yol maskesi)',
       'hybrid':'Gelişmiş eğrisel takip', 'continuity':'Kesikli ve sürekli şerit takibi', 'classic':'Klasik Canny + Hough'}


def _curves_keep_order(result,width,height):
    """Reject a fitted pair that crosses inside its observed image range."""
    pair=getattr(result,'lane_polynomials',None)
    if not pair or len(pair)!=2 or pair[0] is None or pair[1] is None:
        return True

    def model(item):
        if isinstance(item,dict):
            # Continuity candidates are already validated in their resized
            # coordinate system.  Hybrid candidates retain a source-sized
            # curve and coefficients and can be checked here.
            if 'curve' not in item or 'coeffs' not in item:return None
            return item['curve'],item['coeffs']
        if isinstance(item,tuple) and len(item)==2:return item
        return None

    left,right=model(pair[0]),model(pair[1])
    if left is None or right is None:return True
    left_points,left_poly=left;right_points,right_poly=right
    if len(left_points)<3 or len(right_points)<3:return False
    low=max(float(np.min(left_points[:,1])),float(np.min(right_points[:,1])))
    high=min(float(np.max(left_points[:,1])),float(np.max(right_points[:,1])))
    if high-low<height*.10:return False
    ys=np.linspace(low,high,17)
    widths=np.polyval(right_poly,ys)-np.polyval(left_poly,ys)
    # A vanishing point may make the far end narrow, but neither curves may
    # cross or leap into the neighbouring lane in the usable road area.
    if np.count_nonzero(widths>max(5.,width*.018))<len(widths)*.82:return False
    if not width*.07<=float(np.median(widths))<=width*.78:return False
    if float(widths[-1])<width*.11:return False
    return True


def _unknown_detection(frame):
    """Prefer an explicit uncertain result over a visually unrelated overlay."""
    output=frame.copy()
    h,w=frame.shape[:2]
    cv2.rectangle(output,(8,7),(min(w-5,620),47),(18,28,44),-1)
    cv2.putText(output,'GUVENILIR SERIT CIFTI BULUNAMADI',(17,34),cv2.FONT_HERSHEY_SIMPLEX,
                .62,(242,249,255),2,cv2.LINE_AA)
    result=Detection(output,{'01 Orijinal':frame.copy(),'08 Sonuc':output},None,None,None,0.,0)
    result.method='Belirsiz - güvenilir şerit çifti yok'
    return result


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
    style_aware=(getattr(segmenter,'output_channels',2)==4)
    if style_aware:
        # Four channels: sol-düz, sol-kesik, sağ-düz, sağ-kesik.
        left_mask=cv2.bitwise_or(masks[0],masks[1])
        right_mask=cv2.bitwise_or(masks[2],masks[3])
    else:
        left_mask,right_mask=masks[0],masks[1]
    detected=[_fit_mask(left_mask,'left'),_fit_mask(right_mask,'right')]
    # Modelin belirsiz olduğu yerleri gerçek çizgi diye göstermeyelim.
    output=frame.copy();overlay=np.zeros_like(frame)
    left,right=detected
    offset=None
    def marking_type(item,side):
        if item is None:
            return None
        if not style_aware:
            return 'belirsiz'
        solid_idx,dashed_idx=(0,1) if side=='left' else (2,3)
        points=item[0]
        yy=np.clip(points[:,1],0,h-1);xx=np.clip(points[:,0],0,w-1)
        solid=float(np.mean(probs[yy,xx,solid_idx]));dashed=float(np.mean(probs[yy,xx,dashed_idx]))
        # The visual model contributes only when it has actual evidence.  A
        # close score is safer as "belirsiz" than a fabricated line type.
        if max(solid,dashed)<.18 or abs(solid-dashed)<.055:
            return 'belirsiz'
        return 'kesik' if dashed>solid else 'duz'

    left_type,right_type=marking_type(left,'left'),marking_type(right,'right')
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
    for item,kind in ((left,left_type),(right,right_type)):
        if item is not None:
            color=(0,176,255) if kind=='kesik' else ((255,223,30) if kind=='duz' else (0,225,255))
            cv2.polylines(output,[item[0]],False,color,settings.line_thickness,cv2.LINE_AA)
    def ends(obj):
        return (tuple(map(int,obj[0][-1])),tuple(map(int,obj[0][0]))) if obj is not None else None
    le,re=ends(left),ends(right)
    cv2.rectangle(output,(8,7),(min(w-5,520),47),(18,28,44),-1)
    def label(kind,end):
        return (kind.upper() if kind and kind!='belirsiz' else ('VAR' if end else 'YOK'))
    cv2.putText(output,f"SOL: {label(left_type,le)}  SAG: {label(right_type,re)} | U-NET",
                (17,34),cv2.FONT_HERSHEY_SIMPLEX,.65,(242,249,255),2,cv2.LINE_AA)
    mask_viz=np.zeros_like(frame);prob_viz=np.zeros_like(frame)
    if style_aware:
        # Turkuaz=düz, turuncu=kesik; sol/sağ bilgisi modelin kanal
        # düzeninde korunur, görsel ise türleri anlaşılır kılmaya odaklanır.
        mask_viz[:,:,1]=np.maximum(masks[0],masks[2]);mask_viz[:,:,0]=np.maximum(masks[0],masks[2])
        mask_viz[:,:,2]=np.maximum(masks[1],masks[3]);mask_viz[:,:,1]=np.maximum(mask_viz[:,:,1],np.maximum(masks[1],masks[3])//2)
        solid=np.maximum(probs[:,:,0],probs[:,:,2]);dashed=np.maximum(probs[:,:,1],probs[:,:,3])
        prob_viz[:,:,1]=(solid*255).astype('uint8');prob_viz[:,:,0]=(solid*255).astype('uint8')
        prob_viz[:,:,2]=(dashed*255).astype('uint8');prob_viz[:,:,1]=np.maximum(prob_viz[:,:,1],(dashed*127).astype('uint8'))
    else:
        mask_viz[:,:,2]=masks[0];mask_viz[:,:,1]=masks[1]
        prob_viz[:,:,2]=(probs[:,:,0]*255).astype('uint8');prob_viz[:,:,1]=(probs[:,:,1]*255).astype('uint8')
    stages={'01 Orijinal':frame.copy(),'08 Sonuc':output,'09 AI olasilik haritasi':prob_viz,'10 AI sinif maskeleri':mask_viz}
    det=Detection(output,stages,le,re,offset,float(le is not None)*.5+float(re is not None)*.5,0)
    det.method='Yapay zekâ (U-Net)'
    det.lane_polynomials=(left,right)
    det.marking_types=(left_type,right_type)
    det.ai_probabilities=probs
    det.ai_channel_layout=getattr(segmenter,'channel_layout',('left','right'))
    return det


class InferenceEngine:
    def __init__(self,settings:LaneSettings|None=None,mode='auto',weights=None):
        if mode not in MODES:raise ValueError(f'Yöntem desteklenmiyor: {mode}')
        self.settings=settings or LaneSettings()
        self.settings.validate()
        self.mode=mode
        self.classic=LaneDetector(self.settings)
        self.segmenter=None
        self.ufld=None
        self.yolopv2=None
        self.model_warning=None
        if mode=='ufld':
            try:
                self.ufld=UFLDDetector()
            except (FileNotFoundError,RuntimeError) as exc:
                self.model_warning=str(exc)
                raise RuntimeError('UFLD V2 şerit modeli yüklenemedi: '+str(exc)) from exc
        if mode=='yolopv2':
            try:
                self.yolopv2=YOLOPv2Detector()
            except (FileNotFoundError,RuntimeError) as exc:
                self.model_warning=str(exc)
                raise RuntimeError('YOLOPv2 şerit modeli yüklenemedi: '+str(exc)) from exc
        if mode in {'auto','ai'}:
            try:
                from .ai_model import AISegmenter,DEFAULT_WEIGHTS
                self.segmenter=AISegmenter(weights=weights or DEFAULT_WEIGHTS,threshold=.70)
            except (ImportError,FileNotFoundError,RuntimeError,ValueError) as exc:
                self.model_warning=str(exc)
                if mode=='ai':raise RuntimeError('Yapay zekâ modeli yüklenemedi: '+str(exc)) from exc

    def detect(self,frame,video=False):
        scene=analyze_scene(frame)

        def attach(result):
            result.scene_profile=scene
            result.method+=f" · {scene.short_lighting.lower()} / {scene.short_geometry.lower()}"
            return result

        if self.mode=='continuity':
            return attach(detect_continuous(frame,self.settings))
        if self.mode=='classic':
            return attach(self.classic.detect(frame,video))
        if self.mode=='hybrid':
            result=detect_hybrid(frame,self.settings)
            result.method='Eğrisel renk takibi'
            return attach(result)
        if self.mode=='ufld':
            return attach(self.ufld.detect(frame,self.settings))
        if self.mode=='yolopv2':
            return attach(self.yolopv2.detect(frame,self.settings))

        ai=None
        if self.segmenter is not None:
            # Koşula göre eşik uyarlaması: gece daha az güçlü çizgi piksellerini
            # korur. Daha düşük eşik yanlış pozitif yaratabileceği için çift
            # şerit geometrisi ayrıca kontrol edilir.
            self.segmenter.threshold={'gece':.60,'alacakaranlik':.65,'gunduz':.70}[scene.lighting]
            ai=detect_ai(frame,self.settings,self.segmenter)
            if self.mode=='ai':
                return attach(ai)

        def plausible(obj):
            if obj is None:return False
            if obj.left is None or obj.right is None:return False
            (lbottom,_),(ltop,_) = obj.left
            (rbottom,_),(rtop,_) = obj.right
            w,h=frame.shape[1],frame.shape[0]
            # Yanlış şerit geçişi, olası bariyer ve aşırı geniş şeritleri ele.
            if lbottom>=rbottom:return False
            marking_types=getattr(obj,'marking_types',None)
            # Continuity tracer already checks the complete sampled pair.
            # Its small drawing extension can cross at the far endpoint on a
            # bend; rejecting it again here would discard a valid curve.
            pair=getattr(obj,'lane_polynomials',None)
            is_validated_continuity=(isinstance(pair,tuple) and len(pair)==2
                                     and all(isinstance(item,dict) for item in pair if item is not None))
            # Other methods only expose endpoints, so retain a small
            # vanishing-point tolerance while rejecting a real reversal.
            if not is_validated_continuity and ltop>rtop+w*.06:return False
            width=(rbottom-lbottom)/w
            if not .08 <= width <= .92:return False
            for line in (obj.left,obj.right):
                (xb,yb),(xt,yt)=line
                if abs(xb-xt)>3.4*max(1,abs(yb-yt)):return False
            if not _curves_keep_order(obj,w,h):return False
            return True

        # Current strategy: prioritize the ego lane's center dividing paint + road edge.
        # The trace avoids silently substituting a left-side guardrail as a lane.
        continuous=detect_continuous(frame,self.settings,
                                     probabilities=(getattr(ai,'ai_probabilities',None) if ai is not None else None),
                                     channel_layout=(getattr(ai,'ai_channel_layout',None) if ai is not None else None))
        if plausible(continuous):
            continuous.method='Ego-şerit devamlılık analizi'
            if ai is not None:
                continuous.stages.update({k:v for k,v in ai.stages.items() if k.startswith(('09','10'))})
            return attach(continuous)

        if ai is not None and plausible(ai):
            # Gerçek modelin bulduğu iki eğriden, salt renk tahmininden daha
            # anlamlı bir ikinci-geometri kontrolü elde edilir.
            left,right=ai.lane_polynomials
            if left is not None and right is not None:
                low=max(int(left[0][0,1]),int(right[0][0,1]))
                high=min(int(left[0][-1,1]),int(right[0][-1,1]))
                if high-low > frame.shape[0]*.18:
                    ys=np.linspace(low,high,18)
                    centers=(np.polyval(left[1],ys)+np.polyval(right[1],ys))*.5
                    chord=np.linspace(centers[0],centers[-1],len(ys))
                    deviation=float(np.max(np.abs(centers-chord)))
                    # Çok küçük sapmaları viraj olarak aşırı yorumlama.
                    geometry=('viraj' if deviation>frame.shape[1]*.012 else ('belirsiz' if deviation>frame.shape[1]*.006 else 'duz'))
                    scene=replace(scene,geometry=geometry,
                                  curve_score=round(deviation,1),geometry_evidence=18)
            ai.method='Sahneye uyarlanmış U-Net'
            return attach(ai)


        color=detect_hybrid(frame,self.settings)
        color.method='Eğrisel izleme (yedek)'
        if plausible(color):
            if ai is not None:
                color.stages.update({k:v for k,v in ai.stages.items() if k.startswith(('09','10'))})
            return attach(color)

        # Gece ve viraj adaylarında Hough'un düz çizgilerle yol dışına
        # sapma riski yüksek. Modelin kısmi tespitini, uygun olduğu zaman
        # tek çizgili klasik Hough'a tercih et.
        classic=self.classic.detect(frame,video)
        classic.method='Klasik Hough (son yedek)'
        candidates=[c for c in (ai,continuous,color,classic) if c is not None and plausible(c)]
        if not candidates:
            # Tek bir fiziksel yol işareti, iki sınırdan oluşan taralı alan
            # üretmeye yetmez; fakat onu tamamen "bulunamadı" diye gizlemek
            # de kullanıcı açısından yanıltıcıdır.  Sağlam tek izi çizip
            # alanın taranmadığını açıkça bildir.
            def plausible_single(obj):
                if obj is None or (obj.left is None and obj.right is None):
                    return False
                line=obj.left or obj.right
                if abs(line[0][1]-line[1][1])<frame.shape[0]*.12:
                    return False
                return obj.confidence>=.16

            partials=[c for c in (continuous,color,ai,classic) if plausible_single(c)]
            if partials:
                # Continuity is the only partial path that also preserves
                # physical dashed/solid evidence; prefer it over a generic
                # colour curve when both follow the same single marking.
                partial=max(partials,key=lambda c:c.confidence+(.24 if c is continuous else 0.))
                partial.method='Tek şerit işareti (alan taranmadı)'
                if ai is not None and partial is not ai:
                    partial.stages.update({k:v for k,v in ai.stages.items() if k.startswith(('09','10'))})
                return attach(partial)
            unknown=_unknown_detection(frame)
            if ai is not None:
                unknown.stages.update({k:v for k,v in ai.stages.items() if k.startswith(('09','10'))})
            return attach(unknown)
        if scene.geometry=='viraj' or scene.lighting=='gece':
            weight={id(ai):.2} if ai is not None else {}
            weight[id(color)]=.06
            weight[id(continuous)]=.15
            weight[id(classic)]=-.16
        else:
            weight={id(color):.05,id(classic):-.07,id(continuous):.14}
            if ai is not None:weight[id(ai)]=.12
        final=max(candidates,key=lambda c:c.confidence+weight.get(id(c),0.))
        if ai is not None and final is not ai:
            final.stages.update({k:v for k,v in ai.stages.items() if k.startswith(('09','10'))})
        return attach(final)
