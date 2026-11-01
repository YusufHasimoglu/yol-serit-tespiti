"""Gerçek veya sentetik etiketlerle küçük U-Net eğitimi.

Örnek: python egit.py --synthetic 400 --epochs 8
Gerçek : python egit.py --tusimple-root C:/TuSimple --tusimple-json C:/TuSimple/label_data_0313.json --synthetic 0 --epochs 20
"""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import cv2
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--synthetic',type=int,default=300)
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--batch',type=int,default=12)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--tusimple-root',type=Path)
    p.add_argument('--tusimple-json',type=Path)
    p.add_argument('--real-folder',type=Path,default=Path('data/gercek'))
    p.add_argument('--bdd-root',type=Path,
                   help='BDD100K görüntü kökü (etiketli düz/kesik eğitim için)')
    p.add_argument('--bdd-json',type=Path,
                   help='BDD100K lane-style etiket JSON dosyası')
    p.add_argument('--bdd-manifest',type=Path,
                   help='Yerelde seçilmiş, farklı video anahtar karelerinin JSON listesi')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--out',type=Path,default=Path('modeller/serit_unet_demo.pt'))
    p.add_argument('--resume',action='store_true',help='Son en iyi ağırlıklarla ek eğitim turları')
    p.add_argument('--style-aware',action='store_true',
                   help='Kullanıcı etiketlerindeki sol/sağ düz-kesik dört sınıfını eğit')
    p.add_argument('--temporal-augment',type=int,default=0,
                   help='Her kullanıcı düzeltmesinden üretilecek video-kare benzeri çeşit sayısı')
    args=p.parse_args()
    if args.synthetic<0 or args.epochs<1 or args.batch<1 or args.temporal_augment<0: p.error('Eğitim sayıları geçersiz')
    import torch
    from torch.utils.data import DataLoader,Dataset
    from lanevision.training_data import (synthetic_sample,load_user_samples,
                                          tusimple_samples,bdd_style_samples,temporal_frame_variants)
    from lanevision.ai_model import build_model
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.set_num_threads(min(5,torch.get_num_threads()))
    if bool(args.bdd_root) != bool(args.bdd_json):
        p.error('BDD100K için --bdd-root ve --bdd-json birlikte verilmelidir.')
    style_aware=bool((args.bdd_root and args.bdd_json) or args.style_aware)
    output_channels=4 if style_aware else 2
    channel_layout=(['left_solid','left_dashed','right_solid','right_dashed']
                    if style_aware else ['left','right'])
    examples=[];origins=[];groups=[]
    for i in range(args.synthetic):
        examples.append(synthetic_sample(args.seed+i,style_aware=style_aware));origins.append('sentetik');groups.append(f'sentetik-{i}')
    if style_aware and args.tusimple_json:
        # Two-channel masks cannot truthfully provide solid/dashed labels.
        # Do not silently turn unknown styles into fabricated targets.
        if args.tusimple_json:
            p.error('Düz/kesik BDD eğitimi ile etiketsiz TuSimple maskeleri aynı modelde karıştırılamaz.')
    for user_index,e in enumerate(load_user_samples(args.real_folder,style_aware=style_aware) or []):
        # Türetilen komşu kareler aynı kaynak grubunda kalır; böylece aynı
        # görüntünün versiyonları eğitim ve doğrulama arasında sızmaz.
        group=f'kullanici-{user_index}'
        examples.append(e);origins.append('etiketli-gercek');groups.append(group)
        for variant in temporal_frame_variants(e[0],e[1],args.temporal_augment,args.seed+user_index):
            examples.append(variant);origins.append('etiketli-video-cesitlendirme');groups.append(group)
    if args.tusimple_json and args.tusimple_root:
        for e in tusimple_samples(args.tusimple_root,args.tusimple_json,args.limit or None):
            examples.append(e);origins.append('TuSimple');groups.append(f'tusimple-{len(groups)}')
    if args.bdd_root and args.bdd_json:
        for image,mask,group in bdd_style_samples(args.bdd_root,args.bdd_json,args.limit or None,args.bdd_manifest):
            examples.append((image,mask));origins.append('BDD100K-duz-kesik');groups.append(group)
        if args.bdd_manifest:
            manifest=json.loads(args.bdd_manifest.read_text(encoding='utf-8'))
            expected=len(manifest.get('entries',manifest))
            if args.limit:
                expected=min(expected,args.limit)
            actual=origins.count('BDD100K-duz-kesik')
            if actual!=expected:
                p.error(f'BDD100K manifestindeki {expected} örneğin yalnızca {actual} tanesi görüntü+iki-taraflı etiket olarak okunabildi. Eksik görüntüleri indirin veya manifest seçimini doğrulayın.')
    if len(examples)<4: p.error('En az 4 etiketli görüntü gerekli. --synthetic 100 veya gerçek etiketler ekleyin.')
    # Bir video kaynağının komşu karelerini eğitim ve doğrulama arasında
    # ayırmamak gerekir. BDD manifestinde her kaynak bir ana kare olduğundan
    # sabit grup bölmesi tekrar üretilebilirdir.
    if style_aware:
        valids=[j for j,group in enumerate(groups) if sum(map(ord,group))%5==0]
    else:
        valids=[j for j,v in enumerate(origins) if v=='sentetik' and j%7==0]
    if len(valids)<2:valids=list(range(0,len(examples),max(1,len(examples)//6)))
    train=[j for j in range(len(examples)) if j not in valids]
    class DS(Dataset):
        def __init__(self,ids,augment):self.ids=ids;self.augment=augment
        def __len__(self):return len(self.ids)
        def __getitem__(self,i):
            im,mask=examples[self.ids[i]]
            im=im.copy();mask=mask.copy()
            if self.augment:
                if random.random()<.5:
                    im=cv2.flip(im,1);mask=cv2.flip(mask,1)
                    # Yatay çevirme, sol/sağ sınırları değiştirir ama
                    # düz/kesik türünü değiştirmez.
                    mask=mask[:,:,[2,3,0,1]] if mask.shape[-1]==4 else mask[:,:,[1,0]]
                if random.random()<.8:
                    im=np.clip(im.astype(np.float32)*random.uniform(.65,1.3)+random.uniform(-16,16),0,255).astype(np.uint8)
            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB).copy()
            return torch.from_numpy(im).permute(2,0,1).float()/255,torch.from_numpy(mask.copy()).permute(2,0,1).float()/255
    train_loader=DataLoader(DS(train,True),batch_size=args.batch,shuffle=True,num_workers=0)
    val_loader=DataLoader(DS(valids,False),batch_size=args.batch,num_workers=0)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=build_model(output_channels).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr)
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.full((output_channels,),12.,device=device).view(1,output_channels,1,1))
    best=-1; history=[];start_epoch=0
    args.out.parent.mkdir(exist_ok=True,parents=True)
    if args.resume and args.out.is_file():
        cp=torch.load(args.out,map_location='cpu',weights_only=True)
        saved_channels=int(cp.get('output_channels',2)) if isinstance(cp,dict) else 2
        if saved_channels!=output_channels:
            p.error('Devam edilecek modelin kanal sayısı seçilen eğitim türüyle uyuşmuyor.')
        model.load_state_dict(cp['model'] if 'model' in cp else cp)
        start_epoch=int(cp.get('epoch',0))
        best=float(cp.get('synthetic_val_iou',-1))
    for epoch in range(start_epoch+1,start_epoch+args.epochs+1):
        model.train();total=0
        for x,y in train_loader:
            x=x.to(device);y=y.to(device)
            optim.zero_grad(set_to_none=True)
            logits=model(x);probs=logits.sigmoid()
            dice=1-((2*(probs*y).sum((0,2,3))+1)/((probs+y).sum((0,2,3))+1)).mean()
            loss=loss_fn(logits,y)+dice
            loss.backward();optim.step();total+=loss.item()
        model.eval();inter=union=0
        with torch.inference_mode():
            for x,y in val_loader:
                pred=(model(x.to(device)).sigmoid()>.42).cpu();actual=y>.5
                inter+=(pred&actual).sum().item();union+=(pred|actual).sum().item()
        iou=inter/max(union,1)
        entry={'epoch':epoch,'train_loss':round(total/max(len(train_loader),1),4),'synthetic_val_iou':round(iou,4)}
        history.append(entry);print(json.dumps(entry),flush=True)
        if iou>best:
            best=iou
            torch.save({'model':model.cpu().state_dict(),'epoch':epoch,'synthetic_val_iou':iou,
                        'synthetic_train_images':origins.count('sentetik'),
                        'real_train_images':len(examples)-origins.count('sentetik'),
                        'output_channels':output_channels,'channel_layout':channel_layout,
                        'training_sources':sorted(set(origins)),
                        'note':'Doğrulama görüntüleri eğitimden ayrıdır; bu ölçüm gerçek sürüş güvenliği garantisi değildir.'},args.out)
            model.to(device)
    report={'train_images':len(train),'validation_images':len(valids),'source_counts':{k:origins.count(k) for k in sorted(set(origins))},
            'temporal_augment_per_user_frame':args.temporal_augment,
            'best_validation_iou':best,'output_channels':output_channels,'channel_layout':channel_layout,
            'device':device,'history':history,'weights':str(args.out)}
    (args.out.parent/'egitim_raporu.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Eğitim tamamlandı:',args.out,' en iyi sentetik val IoU:',round(best,3))
if __name__=='__main__':main()
