"""Geometrik şerit devamlılığı: çizgi parçalarını sahneye göre grupla.

Ego şeridinin sol sınırı çoğu iki yönlü yolda orta kesikli çizgidir;
sağ sınır ise yol kenarındaki sürekli çizgi olabilir. Bu modül görüntünün
mutlak solundaki bariyeri 'sol şerit' kabul etmemeye çalışır.

Öngörülen/boyasız kısımlar ayrı bir görsel stilde gösterilir. Hiçbir algoritma
olmayan çizgiyi kesin olarak bildiğini iddia etmez.
"""
from __future__ import annotations
import cv2,numpy as np
from .core import LaneSettings,Detection
from .hybrid import color_candidates


def enhanced_paint_candidates(im):
    """Yol boyası HSV + yerel parlaklık: düşük ışıkta soluk turuncuyu korur."""
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    v=v.astype(np.float32)
    local=cv2.GaussianBlur(v,(0,0),5.5)
    # White / gray-painted lane marks, asphalt is rejected by local contrast.
    white=((v>95)&(s<120)&((v-local)>max(9,float(np.percentile(v,75))*.055)))
    # Night/dusk: colors desaturate, painted lane edges can be low brightness.
    orange=((h>=4)&(h<=38)&(s>49)&(v>78)&((v-local)>5))
    base=np.uint8((white|orange)*255)
    return cv2.morphologyEx(base,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))


def _peaks(mask,y,step=7):
    h,w=mask.shape;y0=max(0,y-step//2);y1=min(h,y+step//2+1)
    values=np.count_nonzero(mask[y0:y1],axis=0).astype(np.float32)
    values=cv2.GaussianBlur(values[None],(13,1),2.5)[0]
    maximum=float(values.max())
    idx=np.where((values>max(1.6,maximum*.23)) & (values>=np.roll(values,1)) & (values>=np.roll(values,-1)))[0]
    idx=sorted(idx,key=lambda k:-values[k]);found=[]
    for k in idx:
        if all(abs(k-x)>max(11,w*.023) for x,_ in found):
            found.append((int(k),float(values[k])))
        if len(found)==16:break
    return found


def _fit_track(rows,seed_x,seed_y,side,w,h):
    """Dynamic local color peaks with unobserved bands bridged, not painted."""
    # first find path upward (gaps allowed), then downward
    def walk(order):
        xx=float(seed_x);v=0.0;last_y=seed_y;miss=0;track=[]
        for y in order:
            delta=y-last_y
            guess=xx+v*delta
            radius=w*(.060+min(miss,6)*.020)
            options=[(x,weight) for x,weight in rows[y] if abs(x-guess)<radius]
            if options:
                def scorer(item):
                    x,a=item
                    deviation=abs(x-guess)/radius
                    return a/(1+1.5*deviation**2)
                x,a=max(options,key=scorer)
                if a<1.7: miss+=1;continue
                if track and abs(delta)>1:
                    v=np.clip(.78*v+.22*(x-xx)/delta,-3,3)
                track.append((x,y,a))
                xx=x;last_y=y;miss=0
            else:
                miss+=1
                if miss>22:break
        return track
    up=walk(sorted((k for k in rows if k<=seed_y),reverse=True))
    down=walk(sorted((k for k in rows if k>seed_y)))
    pts=up+down
    if len(pts)<7:return None
    arr=np.array(pts,dtype=np.float64)
    ys,xs=arr[:,1],arr[:,0]
    if np.ptp(ys)<h*.11:return None
    weight=np.sqrt(np.maximum(arr[:,2],1))
    try:
        deg=2 if np.ptp(ys)>h*.23 else 1
        poly=np.polyfit(ys,xs,deg,w=weight)
        for _ in range(3):
            residual=np.abs(xs-np.polyval(poly,ys))
            keep=residual<max(w*.022,1.8*np.median(residual)+w*.005)
            if keep.sum() < max(6,deg+2):break
            poly=np.polyfit(ys[keep],xs[keep],deg,w=weight[keep]);arr=arr[keep];xs,ys=xs[keep],ys[keep];weight=weight[keep]
    except (np.linalg.LinAlgError,ValueError):return None
    ylow,yhigh=int(np.min(ys)),int(np.max(ys))
    if yhigh-ylow<h*.10:return None
    # İncelenen dilimlerde renk maskesiyle destek oranı; kaybolan parçalar tahmin.
    xs_pred=np.polyval(poly,ys)
    rms=float(np.sqrt(np.mean((xs-xs_pred)**2)))
    if rms>w*.035:return None
    samples=int(len(np.unique(ys)))
    extent=(yhigh-ylow)/h
    # Prior evaluating at reference row: left should near center dividing line,
    # right near road shoulder. Allows wide curves via low weight.
    reference_y=float(h*.76)
    ref_x=float(np.polyval(poly,reference_y))/w
    expected=.47 if side=='left' else .83
    deviation=abs(ref_x-expected)
    # Horizon-side anchor: no bridge towards a barrier or forest.
    xh=float(np.polyval(poly,h*.55))/w
    if side=='left' and not (.24<=xh<=.70):return None
    if side=='right' and not (.45<=xh<=.83):return None
    if side=='left' and not (.12<=ref_x<=.73):return None
    if side=='right' and not (.52<=ref_x<=1.28):return None
    score= samples*1.55+extent*35-min(50,deviation*95)-rms/w*95
    return {'poly':poly,'points':arr,'score':float(score),'span':(ylow,yhigh),'ref_x':ref_x,'samples':samples,'rms':rms,'side':side}


def _candidates(mask,side):
    h,w=mask.shape
    yt=int(h*.40);ys=list(range(int(h*.97),yt-1,-7))
    rows={y:_peaks(mask,y) for y in ys}
    seeds=[]
    for y in ys:
        if not .53<=y/h<=.94:continue
        for x,a in rows[y]:
            xf=x/w
            if side=='left' and .16<=xf<=.65:
                seeds.append((x,y,a - 13*abs(xf-.47)))
            if side=='right' and .55<=xf<=.995:
                seeds.append((x,y,a - 9*abs(xf-.85)))
    seeds.sort(key=lambda p:-p[2])
    # Avoid many starting pixels from the same continuous line.
    selected=[]
    for x,y,a in seeds:
        if all(abs(x-t[0])>w*.035 or abs(y-t[1])>h*.09 for t in selected):selected.append((x,y,a))
        if len(selected)>=60:break
    tracks=[]
    for x,y,_ in selected:
        item=_fit_track(rows,x,y,side,w,h)
        if item is None:continue
        if any(abs(item['ref_x']-v['ref_x'])<.035 and abs(item['score']-v['score'])<20 for v in tracks):continue
        tracks.append(item)
    tracks.sort(key=lambda q:-q['score'])
    return tracks[:10]


def _pair(lefts,rights,h,w):
    picks=[]
    for left in lefts:
        for right in rights:
            ylow=max(left['span'][0],right['span'][0]);yhigh=min(left['span'][1],right['span'][1])
            if yhigh-ylow<.12*h:continue
            yy=np.linspace(ylow,yhigh,11)
            lx=np.polyval(left['poly'],yy);rx=np.polyval(right['poly'],yy)
            widths=(rx-lx)/w
            if np.any(widths<.09) or np.any(widths>1.):continue
            # Road should narrow toward vanishing point (upward), except severe curve.
            narrow=widths[0];wide=widths[-1]
            if wide < narrow*.68:continue
            score=left['score']+right['score']+24*(yhigh-ylow)/h - 15*abs(wide-.43)
            picks.append((score,left,right))
    if picks:
        _,left,right=max(picks,key=lambda x:x[0]);return left,right
    return (lefts[0] if lefts and lefts[0]['score']>12 else None,
            rights[0] if rights and rights[0]['score']>12 else None)


def detect_continuous(frame,settings:LaneSettings|None=None,probabilities=None):
    if frame is None or frame.ndim!=3 or min(frame.shape[:2])<100:raise ValueError('BGR resim (en az 100x100) gerekli')
    s=settings or LaneSettings();s.validate()
    h,w=frame.shape[:2]
    # Work at a fixed size for predictable step widths and speed; preserve aspect.
    tw=640;th=max(160,int(h*tw/w));im=cv2.resize(frame,(tw,th))
    road_value=float(np.median(cv2.cvtColor(im,cv2.COLOR_BGR2HSV)[int(th*.75):,:,2]))
    raw=color_candidates(im) if road_value>107 else enhanced_paint_candidates(im)
    # Clear sky/forest and far-off guardrails outside the road-focused region.
    road=np.zeros((th,tw),np.uint8)
    cv2.fillPoly(road,[np.array([[0,th-1],[int(tw*.23),int(th*.45)],[int(tw*.91),int(th*.45)],[tw-1,th-1]],np.int32)],255)
    raw=cv2.bitwise_and(raw,road)
    if probabilities is not None:
        prob=cv2.resize(probabilities,(tw,th))
        for ch in range(min(2,prob.shape[-1])):
            # AI only helps locate paint-colored candidate pixels; cannot override
            # zero physical lane evidence in this experimental tracing mode.
            confidence=((prob[:,:,ch]>.55)&(raw>0)).astype(np.uint8)*255
            raw=cv2.bitwise_or(raw,confidence)
    lefts=_candidates(raw,'left');rights=_candidates(raw,'right')
    left,right=_pair(lefts,rights,th,tw)
    output=frame.copy();overlay=np.zeros_like(frame)
    sx=w/tw;sy=h/th
    def bounds(item):
        if item is None:return None
        lo,hi=item['span'];
        # Only small extension past measured evidence; don't invent road beyond image.
        lo=max(int(th*.40),lo-7);hi=min(th-1,hi+7)
        ys=np.arange(lo,hi+1,2)
        xx=np.polyval(item['poly'],ys)
        ok=(xx>=0)&(xx<tw)
        xx=xx[ok];ys=ys[ok]
        if len(xx)<5:return None
        return np.column_stack([np.round(xx*sx),np.round(ys*sy)]).astype(np.int32)
    lp,rp=bounds(left),bounds(right)
    offset=None
    if lp is not None and rp is not None:
        y0=max(lp[0,1],rp[0,1]);y1=min(lp[-1,1],rp[-1,1]);ys=np.arange(y0,y1+1,4)
        if len(ys)>8:
            lx=np.polyval(left['poly'],ys/sy)*sx;rx=np.polyval(right['poly'],ys/sy)*sx
            good=(lx<rx)&(lx>=0)&(rx<w)
            if good.sum()>7:
                poly=np.vstack([np.column_stack([lx[good],ys[good]]),np.column_stack([rx[good][::-1],ys[good][::-1]])]).round().astype(np.int32)
                cv2.fillPoly(overlay,[poly],(70,177,105))
                offset=(w/2-(lx[good][-1]+rx[good][-1])/2)
    output=cv2.addWeighted(output,1,overlay,s.overlay_alpha,0)
    for item,pts in ((left,lp),(right,rp)):
        if item is None or pts is None:continue
        # Model estimate is shown as orange dashed centerline, observed pixels as solid cyan.
        # We draw estimated geometry in 12px segments, so a gap is not confused
        # with actually detected road paint.
        for j in range(0,len(pts)-3,7):
            cv2.polylines(output,[pts[j:j+4]],False,(32,176,255),max(2,s.line_thickness-2),cv2.LINE_AA)
        local=cv2.resize(raw,(w,h),interpolation=cv2.INTER_NEAREST)
        yy=pts[:,1];xx=pts[:,0]
        evidence=(local[yy,xx]>0)
        for j in range(len(pts)-1):
            if evidence[j] and evidence[j+1]:
                cv2.line(output,tuple(pts[j]),tuple(pts[j+1]),(255,223,30),max(2,s.line_thickness-2),cv2.LINE_AA)
    def endpoints(pts):
        return (tuple(map(int,pts[-1])),tuple(map(int,pts[0]))) if pts is not None else None
    l,r=endpoints(lp),endpoints(rp)
    cv2.rectangle(output,(8,7),(min(w-5,560),48),(17,29,42),-1)
    cv2.putText(output,f"SOL: {'VAR' if l else 'YOK'}  SAG: {'VAR' if r else 'YOK'} | DEVAMLILIK",(17,34),cv2.FONT_HERSHEY_SIMPLEX,.62,(241,249,255),2,cv2.LINE_AA)
    stages={'01 Orijinal':frame.copy(),'06 ROI kenarlari':cv2.resize(raw,(w,h)),'08 Sonuc':output}
    count=(left['samples'] if left else 0)+(right['samples'] if right else 0)
    det=Detection(output,stages,l,r,offset,float(bool(l))*min(1,count/50)*.5+float(bool(r))*min(1,count/50)*.5,count)
    det.method='Parca-eslemeli egrisel takip (tahmini bosluklar kesikli)'
    det.lane_polynomials=(left,right)
    return det