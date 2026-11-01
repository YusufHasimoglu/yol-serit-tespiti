"""Manually labelled real images + synthetic scenes. Val split NEVER enters training.

--promote only when held-out average improves AND per-scene deterioration is small.
"""
from __future__ import annotations
import argparse,json,random,copy
from pathlib import Path
import cv2,numpy as np
from lanevision.ai_model import DEFAULT_WEIGHTS,build_model,INPUT_WIDTH,INPUT_HEIGHT
from lanevision.scenario_data import SCENARIOS,generate_scenario

ROOT=Path(__file__).parent

def load_splits(root=ROOT/'data'/'gercek'):
    records=json.loads((root/'annotations.json').read_text(encoding='utf-8'))
    train=[];hold=[]
    for r in records:
        im=cv2.imread(str(root/'images'/(r['name']+'.png')))
        ma=cv2.imread(str(root/'masks'/(r['name']+'_mask.png')))
        if im is None or ma is None: raise FileNotFoundError(r['name'])
        im=cv2.resize(im,(INPUT_WIDTH,INPUT_HEIGHT))
        ma=cv2.resize(ma,(INPUT_WIDTH,INPUT_HEIGHT),interpolation=cv2.INTER_NEAREST)
        label=np.stack([ma[:,:,2],ma[:,:,1]],axis=-1)
        (hold if r['split']=='holdout' else train).append((r['name'],im,label))
    return train,hold

def augment(im,lab,rng):
    im=im.copy();lab=lab.copy()
    if rng.random()<.5:
        im=cv2.flip(im,1);lab=cv2.flip(lab,1)[:,:,[1,0]]
    # Photometric transforms, model should not learn sky brightness as lane existence.
    gain=rng.uniform(.42,1.3);bias=rng.uniform(-30,16)
    im=np.clip(im.astype(np.float32)*gain+bias,0,255).astype(np.uint8)
    if rng.random()<.4:
        im=cv2.GaussianBlur(im,(3,3),rng.uniform(.5,1.5))
    if rng.random()<.35:
        overlay=np.zeros_like(im,dtype=np.float32)
        yy,xx=np.mgrid[:im.shape[0],:im.shape[1]]
        cx=rng.uniform(0,im.shape[1]);cy=rng.uniform(.55*im.shape[0],im.shape[0])
        glow=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*rng.uniform(15,65)**2))
        im=np.clip(im.astype(np.float32)+glow[:,:,None]*rng.uniform(10,60),0,255).astype(np.uint8)
    return im,lab

def train(args):
    import torch
    from torch import nn
    torch.set_num_threads(min(4,torch.get_num_threads()));torch.manual_seed(1207);random.seed(1207)
    rng=np.random.default_rng(1207)
    real,holdout=load_splits()
    print(f'Gerçek eğitim: {len(real)} | Ayrı tutulan gerçek test: {[p[0] for p in holdout]}',flush=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=build_model().to(device)
    ckpt=torch.load(args.weights,weights_only=True,map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    checkpoint={k:val.detach().cpu().clone() for k,val in model.state_dict().items()}
    def eval_hold():
        model.eval();res={}
        with torch.inference_mode():
            for name,im,mask in holdout:
                inp=torch.from_numpy(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1)[None].float().to(device)/255.
                gt=mask>127;pr=(model(inp).sigmoid()[0].permute(1,2,0).cpu().numpy()>.5)
                den=np.logical_or(gt,pr).sum(axis=(0,1));num=np.logical_and(gt,pr).sum(axis=(0,1))
                res[name]=round(float(np.mean(num/np.maximum(den,1))),4)
        return res
    baseline=eval_hold();print('Gerçek ayrılmış örnekler / önce:',baseline,flush=True)
    # Validation uses only real samples not train; synthetic evaluation separately.
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    pos=torch.tensor([9.,9.],device=device).view(1,2,1,1)
    lossfunc=nn.BCEWithLogitsLoss(pos_weight=pos)
    best=baseline;bestscore=np.mean(list(best.values()));bestweights=checkpoint
    history=[]
    for epoch in range(args.epochs):
        # Oversample distinct real images with heavy photometric augmentation.
        pairs=[]
        for name,im,m in real:
            for i in range(args.real_repeats):
                pairs.append(augment(im,m,rng))
        for scenario in SCENARIOS:
            for i in range(args.synthetic_per_scenario):
                pairs.append(generate_scenario(431000 + epoch*999+ i*3 + SCENARIOS.index(scenario)*120,scenario))
        rng.shuffle(pairs);model.train();losses=[]
        for off in range(0,len(pairs),args.batch):
            chunk=pairs[off:off+args.batch]
            xs=np.stack([cv2.cvtColor(i,cv2.COLOR_BGR2RGB).transpose(2,0,1) for i,m in chunk]);ys=np.stack([m.transpose(2,0,1) for i,m in chunk]);
            xs=torch.from_numpy(xs.copy()).float().to(device)/255.;ys=torch.from_numpy(ys.copy()).float().to(device)/255.
            opt.zero_grad(set_to_none=True);logits=model(xs);p=logits.sigmoid()
            dice=1-((2*(p*ys).sum((0,2,3))+1)/((p+ys).sum((0,2,3))+1)).mean()
            loss=lossfunc(logits,ys)+dice
            loss.backward();opt.step();losses.append(float(loss.item()))
        ev=eval_hold();score=np.mean(list(ev.values()))
        history.append({'epoch':epoch+1,'loss':round(float(np.mean(losses)),3),'real_holdout':ev,'mean_iou':round(float(score),4)})
        print('Eğitim',history[-1],flush=True)
        if score>bestscore:
            bestscore=score;best=ev;bestweights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    accepted=(bestscore>=np.mean(list(baseline.values()))+.005 and all(best[k]>=baseline[k]-.045 for k in baseline))
    out=ROOT/'modeller'/'real_candidate.pt'
    torch.save({'model':bestweights,'training':'manually approximated real examples + synthetic','val_iou':best,'eval_note':'two withheld real example images, small exploratory set; not independent general benchmark'},out)
    if args.promote and accepted:
        import shutil
        backup=ROOT/'modeller'/'serit_unet_yedek.pt'
        shutil.copy2(args.weights,backup)
        shutil.copy2(out,args.weights)
    report={'train_names':[x[0] for x in real],'holdout_names':[x[0] for x in holdout],'baseline':baseline,'best':best,'baseline_mean':float(np.mean(list(baseline.values()))),'best_mean':float(bestscore),'accepted':bool(accepted),'promoted':bool(accepted and args.promote),'history':history,'note':'Small hand-annotated dataset, do not interpret as general real-world accuracy. Test image annotations approximate.'}
    (ROOT/'modeller'/'gercek_egitim_raporu.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('SONUÇ',{'accepted':accepted,'promoted':accepted and args.promote,'best':round(float(bestscore),4)},flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path,default=DEFAULT_WEIGHTS)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--real-repeats',type=int,default=14)
    p.add_argument('--synthetic-per-scenario',type=int,default=16)
    p.add_argument('--lr',type=float,default=.00014)
    p.add_argument('--batch',type=int,default=10)
    p.add_argument('--promote',action='store_true')
    train(p.parse_args())