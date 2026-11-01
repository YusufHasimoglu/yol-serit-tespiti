"""Eğitime katılmamış, etiketli test verisinde AI maske IoU değerlendirmesi."""
from pathlib import Path
import argparse,json
import numpy as np
from lanevision.training_data import load_user_samples
from lanevision.ai_model import AISegmenter

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--folder',type=Path,default=Path('data/test'))
    p.add_argument('--weights',type=Path,default=Path('modeller/serit_unet_demo.pt'))
    p.add_argument('--out',type=Path,default=Path('degerlendirme.json'))
    a=p.parse_args()
    if a.folder.resolve()==Path('data/gercek').resolve():p.error('Eğitim klasörüyle test yapılmaz. Ayrı test görüntüleri kullanın.')
    data=list(load_user_samples(a.folder))
    if not data:p.error('Test klasöründe etiketli görüntü bulunamadı. data/test/images ve masks/ oluşturun.')
    model=AISegmenter(a.weights)
    ious=[]
    for i,(image,label) in enumerate(data):
        pred=model.probabilities(image)>.70
        gt=label>127
        inter=np.logical_and(pred,gt).sum(axis=(0,1))
        union=np.logical_or(pred,gt).sum(axis=(0,1))
        score=[float(a/b) if b else None for a,b in zip(inter,union)]
        ious.append(score)
    scores=np.array([[float('nan') if v is None else v for v in row] for row in ious])
    payload={'test_adedi':len(data),'ortalama_sol_iou':float(np.nanmean(scores[:,0])),
             'ortalama_sag_iou':float(np.nanmean(scores[:,1])),
             'ortalama_iou':float(np.nanmean(scores)), 'goruntu_skorlari':ious}
    a.out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__':main()