"""Viraj/gece/yağmur senaryolarında deneysel U-Net eğitimi ve korumalı terfi.

Örnek: python egit_senaryo.py --samples-per-scenario 70 --epochs 4 --batch 12
Not: sentetik test IoU gerçek yol başarısını ölçmez.
"""
from __future__ import annotations
import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from lanevision.ai_model import DEFAULT_WEIGHTS, build_model
from lanevision.scenario_data import SCENARIOS, generate_scenario
from lanevision.training_data import load_user_samples


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--samples-per-scenario',type=int,default=70)
    p.add_argument('--validation-per-scenario',type=int,default=16)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--batch',type=int,default=12)
    p.add_argument('--lr',type=float,default=0.00035)
    p.add_argument('--seed',type=int,default=42,help='Her tekrarda değişebilir, doğrulama sabit kalır')
    p.add_argument('--weights',type=Path,default=DEFAULT_WEIGHTS)
    p.add_argument('--real-folder',type=Path,default=Path('data/gercek'))
    p.add_argument('--report',type=Path,default=Path('modeller/senaryo_egitim_raporu.json'))
    p.add_argument('--candidate',type=Path,default=Path('modeller/senaryo_aday.pt'))
    p.add_argument('--min-gain',type=float,default=0.003)
    p.add_argument('--max-scenario-drop',type=float,default=0.05)
    p.add_argument('--promote',action='store_true',help='Yalnızca kalite kapısından geçen adayı esas ağırlık dosyasına kopyala')
    args=p.parse_args()
    if min(args.samples_per_scenario,args.validation_per_scenario,args.epochs,args.batch)<1:
        p.error('Eğitim, doğrulama, epoch ve batch sayıları pozitif olmalıdır')

    import torch
    from torch.utils.data import Dataset, DataLoader
    torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    # Sabit doğrulama tohumları eğitim tohumlarından ayrıdır, senaryo başına raporlanır.
    validation={s:[generate_scenario(900000+k,s) for k in range(args.validation_per_scenario)] for s in SCENARIOS}
    training=[]
    for j,s in enumerate(SCENARIOS):
        training.extend(generate_scenario(args.seed*10000+j*1000+i,s) for i in range(args.samples_per_scenario))
    real=list(load_user_samples(args.real_folder) or [])
    training.extend(real)

    class DS(Dataset):
        def __init__(self,data,augment):self.data=data;self.augment=augment
        def __len__(self):return len(self.data)
        def __getitem__(self,i):
            im,mask=self.data[i];im=im.copy();mask=mask.copy()
            if self.augment:
                if random.random()<.5:
                    im=cv2.flip(im,1); mask=cv2.flip(mask,1)[:,:,[1,0]]
                if random.random()<.58:
                    alpha=random.uniform(.72,1.25);beta=random.uniform(-11,13)
                    im=np.clip(im.astype(np.float32)*alpha+beta,0,255).astype(np.uint8)
            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            return torch.from_numpy(im.copy()).permute(2,0,1).float().div(255),torch.from_numpy(mask.copy()).permute(2,0,1).float().div(255)

    device='cuda' if torch.cuda.is_available() else 'cpu'
    train_loader=DataLoader(DS(training,True),batch_size=args.batch,shuffle=True,num_workers=0)
    val_loaders={s:DataLoader(DS(data,False),batch_size=args.batch,num_workers=0) for s,data in validation.items()}

    def evaluate(model):
        model.eval();metrics={}
        with torch.inference_mode():
            for scenario,loader in val_loaders.items():
                intersection=np.zeros(2,np.float64);union=np.zeros(2,np.float64)
                for xs,ys in loader:
                    p=(model(xs.to(device)).sigmoid()>.5).cpu().numpy()
                    y=(ys>.5).numpy()
                    intersection+=np.logical_and(p,y).sum(axis=(0,2,3))
                    union+=np.logical_or(p,y).sum(axis=(0,2,3))
                metrics[scenario]=round(float(np.mean(intersection/np.maximum(union,1))),4)
        return metrics

    model=build_model().to(device)
    has_baseline=args.weights.is_file()
    if has_baseline:
        ckpt=torch.load(args.weights,map_location='cpu',weights_only=True)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    baseline=evaluate(model) if has_baseline else None
    if baseline is not None: print('Başlangıç IoU:',json.dumps(baseline,ensure_ascii=False),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.0001)
    bce=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.,10.],device=device).view(1,2,1,1))
    history=[];best_metrics=None;best_macro=-1;best_state=None
    for epoch in range(1,args.epochs+1):
        model.train();loss_sum=0.
        for xs,ys in train_loader:
            xs=xs.to(device);ys=ys.to(device)
            opt.zero_grad(set_to_none=True)
            logits=model(xs);pr=logits.sigmoid()
            dice=1-((2*(pr*ys).sum(dim=(0,2,3))+1)/((pr+ys).sum(dim=(0,2,3))+1)).mean()
            loss=bce(logits,ys)+dice
            loss.backward();opt.step();loss_sum+=float(loss.item())
        metrics=evaluate(model);macro=round(sum(metrics.values())/len(metrics),4)
        history.append({'epoch':epoch,'loss':round(loss_sum/len(train_loader),4),'macro_iou':macro,'by_scenario':metrics})
        print('Tur:',json.dumps(history[-1],ensure_ascii=False),flush=True)
        if macro>best_macro:
            best_macro=macro;best_metrics=metrics
            best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}

    base_macro=round(sum(baseline.values())/len(baseline),4) if baseline else None
    gain=round(best_macro-base_macro,4) if base_macro is not None else None
    no_regression=all(best_metrics[s]+args.max_scenario_drop>=baseline[s] for s in SCENARIOS) if baseline else True
    accepted=(base_macro is None or (gain>=args.min_gain and no_regression))
    args.candidate.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':best_state,'scenario_val_macro_iou':best_macro,'scenario_val_iou':best_metrics,
                'epoch':args.epochs,'synthetic_train_images':args.samples_per_scenario*len(SCENARIOS),
                'real_train_images':len(real),'note':'Sentetik senaryo doğrulaması; gerçek yol doğruluğu değildir.'},args.candidate)
    if args.promote and accepted:
        args.weights.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(args.candidate,args.weights)
    report={'timestamp':datetime.now(timezone.utc).isoformat(),'kind':'sentetik-senaryo-deneyi',
            'training_scenarios':{s:args.samples_per_scenario for s in SCENARIOS},'real_train_images':len(real),
            'val_scenarios':{s:args.validation_per_scenario for s in SCENARIOS},
            'baseline':baseline,'baseline_macro_iou':base_macro,'best':best_metrics,'best_macro_iou':best_macro,
            'gain':gain,'no_scenario_large_regression':no_regression,'accepted':accepted,'promoted':bool(accepted and args.promote),
            'candidate_path':str(args.candidate),'weights_path':str(args.weights),'history':history,
            'disclaimer':'IoU yalnızca sabit sentetik senaryo doğrulaması için geçerlidir; gerçek yol testi değildir.'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Karar:', 'KABUL' if accepted else 'RED','| iyileşme:',gain,'| model güncellendi:',bool(accepted and args.promote),flush=True)

if __name__=='__main__':main()