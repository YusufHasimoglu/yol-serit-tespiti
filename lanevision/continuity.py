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


def _marking_metrics(mask,item):
    """Measure paint/gap runs along a fitted curve.

    Track points are sampled every seven pixels and may jump over a dash gap.
    They are suitable for curve fitting but not for deciding whether paint is
    dashed.  Here the *raw paint mask* is sampled row-by-row along the fitted
    curve, preserving the physical gaps that distinguish a dashed divider
    from a continuous edge line.
    """
    h,w=mask.shape
    low,high=item['span']
    yy=np.arange(max(0,low),min(h-1,high)+1)
    xx=np.rint(np.polyval(item['poly'],yy)).astype(np.int32)
    valid=(xx>=0)&(xx<w)
    yy,xx=yy[valid],xx[valid]
    if len(yy)<8:
        return {'marking':'belirsiz','dashed_score':0.,'solid_score':0.,'coverage':0.}
    # Horizontal tolerance covers curve-fit rounding without vertically
    # bridging natural gaps between dash segments.
    nearby=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,3)))
    observed=nearby[yy,xx]>0
    coverage=float(observed.mean())
    transitions=np.flatnonzero(np.diff(observed.astype(np.int8))!=0)+1
    runs=np.split(np.arange(len(observed)),transitions)
    gap_min=max(6,int(h*.018))
    long_gaps=[len(run) for run in runs if not observed[run[0]] and len(run)>=gap_min]
    gap_count=len(long_gaps)
    gap_fraction=float(sum(long_gaps)/max(1,len(observed)))
    dashed_score=float(np.clip(.18*(1-coverage)+.64*min(1.,gap_count/2.)
                                      +.55*min(1.,gap_fraction/.18),0.,1.))
    solid_score=float(np.clip(coverage*(1-.75*min(1.,gap_count/2.))+.05,0.,1.))
    marking='kesik' if dashed_score>=.45 and dashed_score>.75*solid_score else 'duz'
    return {'marking':marking,'dashed_score':dashed_score,'solid_score':solid_score,
            'coverage':coverage,'dash_gap_count':gap_count}


def _apply_ai_style(item, probabilities, channel_layout, side):
    """Fuse a 4-channel learned style estimate into a physical paint track.

    The network is allowed to distinguish solid/dashed only along a candidate
    that has already been supported by real paint pixels.  This avoids turning
    a bright guardrail or a hallucinated segmentation blob into a lane merely
    because a neural score is high.
    """
    if item is None or probabilities is None or probabilities.ndim!=3:
        return
    layout=tuple(channel_layout or ())
    wanted=(f'{side}_solid',f'{side}_dashed')
    if not all(name in layout for name in wanted):
        return
    solid_ch,dashed_ch=(layout.index(name) for name in wanted)
    h,w=probabilities.shape[:2]
    lo,hi=item['span'];yy=np.arange(max(0,lo),min(h-1,hi)+1)
    xx=np.rint(np.polyval(item['poly'],yy)).astype(np.int32)
    valid=(xx>=0)&(xx<w)
    if valid.sum()<8:
        return
    yy,xx=yy[valid],xx[valid]
    # A small cross-track maximum tolerates sub-pixel curve fitting error.
    solid=cv2.dilate(probabilities[:,:,solid_ch],np.ones((3,5),np.uint8))[yy,xx]
    dashed=cv2.dilate(probabilities[:,:,dashed_ch],np.ones((3,5),np.uint8))[yy,xx]
    solid_score=float(np.mean(solid));dashed_score=float(np.mean(dashed))
    item['ai_solid_score']=solid_score;item['ai_dashed_score']=dashed_score
    if max(solid_score,dashed_score)<.20 or abs(solid_score-dashed_score)<.055:
        return
    # The learned type can correct a raw-gap misread, but cannot erase strong
    # visual evidence without a clear margin.  Scores are kept for diagnostic
    # use and pair ranking below.
    native_solid=float(item.get('solid_score',0.));native_dashed=float(item.get('dashed_score',0.))
    fused_solid=.46*native_solid+.54*solid_score
    fused_dashed=.46*native_dashed+.54*dashed_score
    item['solid_score']=fused_solid;item['dashed_score']=fused_dashed
    item['marking']='kesik' if fused_dashed>fused_solid else 'duz'


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
    # Horizon-side anchor: keep the search on the road, while allowing a
    # painted shoulder line to converge late on a tight bend.  The pair
    # geometry below remains the stronger guard against guardrails.
    xh=float(np.polyval(poly,h*.55))/w
    if side=='left' and not (.17<=xh<=.74):return None
    if side=='right' and not (.40<=xh<=.96):return None
    if side=='left' and not (.08<=ref_x<=.76):return None
    if side=='right' and not (.45<=ref_x<=1.32):return None
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
        item.update(_marking_metrics(mask,item))
        if any(abs(item['ref_x']-v['ref_x'])<.035 and abs(item['score']-v['score'])<20 for v in tracks):continue
        tracks.append(item)
    tracks.sort(key=lambda q:-q['score'])
    return tracks[:10]


def _pair(lefts,rights,h,w):
    """Select the lane around the camera, not merely the brightest two traces.

    Road furniture (especially a sunlit guardrail) often produces a much
    stronger colour response than worn lane paint.  A valid ego-lane pair must
    therefore have a plausible width at the near and far ends *and* keep its
    centre close enough to the camera projection.  The previous score mostly
    rewarded paint support, which made a centre marking + guardrail pair win
    on real multi-lane roads.
    """
    picks=[]
    has_dashed_left=any(item.get('dashed_score',0.)>=.45 for item in lefts)
    has_solid_right=any(item.get('solid_score',0.)>=.45 for item in rights)
    for left in lefts:
        for right in rights:
            ylow=max(left['span'][0],right['span'][0]);yhigh=min(left['span'][1],right['span'][1])
            if yhigh-ylow<.12*h:continue
            yy=np.linspace(ylow,yhigh,15)
            lx=np.polyval(left['poly'],yy);rx=np.polyval(right['poly'],yy)
            widths=(rx-lx)/w
            # A centre dash and road edge naturally meet near the vanishing
            # point.  Tiny negative values there come from independent curve
            # fits, not a crossing at the near (drivable) part of the road.
            # Independent curves can meet a few pixels before the vanishing
            # point on a bend.  A crossing in the near, drivable region is
            # still invalid, but a tiny far-end overlap is not.
            near_widths=widths[len(widths)//3:]
            if np.any(widths<-.060) or np.any(near_widths<.018) or np.any(widths>1.):continue
            positive_widths=widths[widths>.02]
            if len(positive_widths)<len(widths)*.75:continue
            # Perspective: the same lane becomes narrower toward the horizon.
            far_width=float(max(.02,np.percentile(positive_widths,20)))
            near_width=float(widths[-1])
            near_center=float((lx[-1]+rx[-1])/(2*w))
            if not (.12<=near_width<=.76):continue
            if not (.020<=far_width<=min(.55,near_width*.98)):continue
            if near_width < far_width*1.05:continue
            # The selected pair is the ego lane: it should contain the camera
            # projection near the bottom.  This rejects an adjacent lane made
            # of a centre dash and a bright shoulder/guardrail.
            # A dashcam can be mounted noticeably off-centre, so this remains
            # deliberately broad.  Pair ranking below, rather than a hard
            # centre cut-off, decides between a real edge and a bright rail.
            if not (.20<=near_center<=.80):continue
            if lx[-1] > w*.68 or rx[-1] < w*.32:continue
            # A lane width must change smoothly.  Sudden widening is typical
            # of a track jumping from paint to a guardrail.
            if float(np.max(np.abs(np.diff(widths))))>.12:continue
            extent=(yhigh-ylow)/h
            # A camera may sit off-centre, while an implausibly broad road
            # pair is a much stronger signal of a curb/guardrail selection.
            # Prefer the common dashed-divider + solid-edge lane width.
            shape_error=(.9*abs(near_center-.50)+3.0*abs(near_width-.34)
                         +.7*abs(far_width/max(near_width,.01)-.42))
            # Paint strength alone must not make a guardrail win; geometric
            # consistency is intentionally the dominant tie-breaker.
            # A dashed centre divider plus a solid shoulder is common, but
            # line type is only a soft preference: double-yellow, dashed
            # shoulder and all-solid roads are valid lane boundaries too.
            style_match=left.get('dashed_score',0.)*right.get('solid_score',0.)
            reversed_style=left.get('solid_score',0.)*right.get('dashed_score',0.)
            both_dashed=left.get('dashed_score',0.)*right.get('dashed_score',0.)
            if has_dashed_left and has_solid_right:
                style_bonus=38*style_match-10*reversed_style-6*both_dashed
            else:
                style_bonus=16*style_match-5*reversed_style-3*both_dashed
            score=left['score']+right['score']+24*extent-190*shape_error+style_bonus
            picks.append((score,left,right))
    if picks:
        _,left,right=max(picks,key=lambda x:x[0]);return left,right
    # Never combine the independent best candidates when no geometrically
    # valid pair exists: that exact fallback was how a centre dash and a
    # guardrail could be reported as two confident lane boundaries.
    best_left=lefts[0] if lefts and lefts[0]['score']>12 else None
    best_right=rights[0] if rights and rights[0]['score']>12 else None
    if best_left is not None and (best_right is None or best_left['score']>=best_right['score']):
        return best_left,None
    return None,best_right


def detect_continuous(frame,settings:LaneSettings|None=None,probabilities=None,channel_layout=None,
                      lane_evidence=None,road_evidence=None):
    if frame is None or frame.ndim!=3 or min(frame.shape[:2])<100:raise ValueError('BGR resim (en az 100x100) gerekli')
    s=settings or LaneSettings();s.validate()
    h,w=frame.shape[:2]
    # Work at a fixed size for predictable step widths and speed; preserve aspect.
    tw=640;th=max(160,int(h*tw/w));im=cv2.resize(frame,(tw,th))
    road_value=float(np.median(cv2.cvtColor(im,cv2.COLOR_BGR2HSV)[int(th*.75):,:,2]))
    # Parlak sahnede renk maskesi temiz sarı/beyaz boya izleri verir.  Yol
    # yüzeyi koyulaştığında yerel karşıtlık maskesine geçerek gölgede kalan
    # soluk çizgileri korur.  İkisini körlemesine birleştirmek, parlak
    # görsellerde asfalt parlamasını kesik çizgi boşluğu gibi yorumlayabilir.
    raw=color_candidates(im) if road_value>107 else enhanced_paint_candidates(im)
    # Clear sky/forest and far-off guardrails outside the road-focused region.
    road=np.zeros((th,tw),np.uint8)
    cv2.fillPoly(road,[np.array([[0,th-1],[int(tw*.17),int(th*.45)],[int(tw*.95),int(th*.45)],[tw-1,th-1]],np.int32)],255)
    raw=cv2.bitwise_and(raw,road)
    # Keep the physical colour mask as the drawing evidence.  A learned lane
    # model may guide *which* painted trace is selected, but it must never
    # fabricate line segments where the source image contains no paint.
    observed_raw=raw.copy()
    if probabilities is not None:
        prob=cv2.resize(probabilities,(tw,th))
        # AI only helps locate paint-colored candidate pixels; it cannot
        # override zero physical lane evidence in this experimental tracer.
        confidence=((np.max(prob,axis=-1)>.55)&(raw>0)).astype(np.uint8)*255
        raw=cv2.bitwise_or(raw,confidence)
    evidence_used=False
    # YOLOPv2 supplies two masks: likely lane markings and drivable road.  Use
    # them only as a spatial gate for the already observed paint pixels.  If
    # their restricted candidate set cannot form a valid ego-lane pair, fall
    # back to the original physical-paint candidates instead of returning an
    # empty or unrelated result.
    guided=raw
    if road_evidence is not None:
        road_mask=cv2.resize((np.asarray(road_evidence)>0).astype(np.uint8)*255,(tw,th),
                             interpolation=cv2.INTER_NEAREST)
        road_support=cv2.dilate(road_mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,13)))
        road_guided=cv2.bitwise_and(guided,road_support)
        if np.count_nonzero(road_guided)>=max(65,int(np.count_nonzero(guided)*.012)):
            guided=road_guided
    if lane_evidence is not None:
        lane_mask=cv2.resize((np.asarray(lane_evidence)>0).astype(np.uint8)*255,(tw,th),
                             interpolation=cv2.INTER_NEAREST)
        # This width tolerates the small difference between the segmentation
        # contour and the centre of a painted marking after resizing.
        lane_support=cv2.dilate(lane_mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(29,11)))
        lane_guided=cv2.bitwise_and(guided,lane_support)
        if np.count_nonzero(lane_guided)>=max(55,int(np.count_nonzero(guided)*.009)):
            guided=lane_guided
        guided_lefts=_candidates(guided,'left');guided_rights=_candidates(guided,'right')
        guided_left,guided_right=_pair(guided_lefts,guided_rights,th,tw)
        if guided_left is not None and guided_right is not None:
            raw=guided
            lefts,rights=guided_lefts,guided_rights
            left,right=guided_left,guided_right
            evidence_used=True
        else:
            raw=observed_raw
            lefts=_candidates(raw,'left');rights=_candidates(raw,'right')
            left,right=_pair(lefts,rights,th,tw)
    else:
        raw=observed_raw
        lefts=_candidates(raw,'left');rights=_candidates(raw,'right')
        left,right=_pair(lefts,rights,th,tw)
    if probabilities is not None:
        for item in lefts:
            _apply_ai_style(item,prob,channel_layout,'left')
        for item in rights:
            _apply_ai_style(item,prob,channel_layout,'right')
    if evidence_used:
        # Dashed/solid reporting must be measured from the original paint
        # pixels, not from the model gate (which can omit a dash entirely).
        if left is not None:left.update(_marking_metrics(observed_raw,left))
        if right is not None:right.update(_marking_metrics(observed_raw,right))
    # Geometri doğrulanmışsa düz/düz, düz/kesik ve kesik/kesik çiftleri de
    # geçerlidir.  Tür bilgisi, çizgiyi elemek için değil arayüzde belirtmek
    # için kullanılır.
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
    observed_mask=cv2.resize(observed_raw,(w,h),interpolation=cv2.INTER_NEAREST)
    # A small dilation absorbs resize/fit rounding but does not turn a long
    # unobserved curve into measured road paint.
    observed_mask=cv2.dilate(observed_mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

    def observed_points(pts):
        if pts is None:return None
        yy=np.clip(pts[:,1],0,h-1);xx=np.clip(pts[:,0],0,w-1)
        return observed_mask[yy,xx]>0

    left_observed,right_observed=observed_points(lp),observed_points(rp)
    offset=None
    # Do not present a filled lane when one of its boundaries is almost wholly
    # extrapolated.  It is safer to show an incomplete result than a confident
    # but shifted green polygon.
    def evidence_threshold(item):
        # A dashed boundary is deliberately absent between paint segments;
        # requiring the same dense support as a continuous edge prevents a
        # perfectly valid dashed/solid lane from being shaded.
        return .11 if item is not None and item.get('marking')=='kesik' else .18

    enough_evidence=(left_observed is not None and right_observed is not None
                     and left_observed.mean()>=evidence_threshold(left)
                     and right_observed.mean()>=evidence_threshold(right))
    if lp is not None and rp is not None and enough_evidence:
        y0=max(lp[0,1],rp[0,1]);y1=min(lp[-1,1],rp[-1,1]);ys=np.arange(y0,y1+1,4)
        if len(ys)>8:
            lx=np.polyval(left['poly'],ys/sy)*sx;rx=np.polyval(right['poly'],ys/sy)*sx
            good=(lx<rx)&(lx>=0)&(rx<w)
            if good.sum()>7:
                poly=np.vstack([np.column_stack([lx[good],ys[good]]),np.column_stack([rx[good][::-1],ys[good][::-1]])]).round().astype(np.int32)
                cv2.fillPoly(overlay,[poly],(70,177,105))
                offset=(w/2-(lx[good][-1]+rx[good][-1])/2)
    output=cv2.addWeighted(output,1,overlay,s.overlay_alpha,0)
    for item,pts,evidence in ((left,lp,left_observed),(right,rp,right_observed)):
        if item is None or pts is None or evidence is None:continue
        # Kesik çizgi turuncu, düz çizgi turkuaz gösterilir. Böylece ekrandaki
        # iki sınırın türü, taranan alanla birlikte doğrudan anlaşılır.
        is_dashed=item.get('marking')=='kesik'
        measured_color=(0,176,255) if is_dashed else (255,223,30)
        bridge_color=(0,132,210) if is_dashed else (220,180,20)
        measured=np.flatnonzero(evidence)
        if not len(measured):continue
        groups=np.split(measured,np.where(np.diff(measured)>1)[0]+1)
        for group in groups:
            if len(group)>=2:
                cv2.polylines(output,[pts[group]],False,measured_color,max(2,s.line_thickness-2),cv2.LINE_AA)
        # Natural dashed gaps are intentionally left open.  A continuous
        # marking may receive a short bridge only where an observed highlight
        # or shadow interrupted otherwise adjacent evidence.
        if is_dashed:continue
        max_bridge=max(4,int(h*.045/2))
        for first,second in zip(groups,groups[1:]):
            start,end=int(first[-1]),int(second[0])
            if end-start>max_bridge:continue
            for j in range(start,end-1,3):
                cv2.line(output,tuple(pts[j]),tuple(pts[min(j+1,end)]),
                         bridge_color,max(2,s.line_thickness-2),cv2.LINE_AA)
    def endpoints(pts):
        return (tuple(map(int,pts[-1])),tuple(map(int,pts[0]))) if pts is not None else None
    l,r=endpoints(lp),endpoints(rp)
    left_type=left.get('marking','belirsiz').upper() if left else 'YOK'
    right_type=right.get('marking','belirsiz').upper() if right else 'YOK'
    area='TARALI' if offset is not None else 'TARANMADI'
    cv2.rectangle(output,(8,7),(min(w-5,650),48),(17,29,42),-1)
    cv2.putText(output,f"SOL: {left_type}  SAG: {right_type} | {area}",(17,34),cv2.FONT_HERSHEY_SIMPLEX,.62,(241,249,255),2,cv2.LINE_AA)
    stages={'01 Orijinal':frame.copy(),'06 ROI kenarlari':cv2.resize(raw,(w,h)),'08 Sonuc':output}
    count=(left['samples'] if left else 0)+(right['samples'] if right else 0)
    det=Detection(output,stages,l,r,offset,float(bool(l))*min(1,count/50)*.5+float(bool(r))*min(1,count/50)*.5,count)
    det.method='Kesik-düz eşlemeli şerit takibi'
    det.lane_polynomials=(left,right)
    det.marking_types=(left.get('marking') if left else None,right.get('marking') if right else None)
    det.ai_evidence_used=evidence_used
    return det
