"""BDD100K düz/kesik ana karelerinde yeniden üretilebilir eşik değerlendirmesi."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lanevision.ai_model import AISegmenter
from lanevision.training_data import bdd_style_samples


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--weights',type=Path,default=Path('modeller/serit_unet_demo.pt'))
    parser.add_argument('--out',type=Path,default=Path('modeller/bdd100_degerlendirme.json'))
    args=parser.parse_args()
    rows=list(bdd_style_samples(args.root,args.labels,manifest=args.manifest,limit=100))
    valid=[row for row in rows if sum(map(ord,row[2]))%5==0]
    if len(valid)<2:
        raise SystemExit('Bağımsız BDD100K doğrulama karesi bulunamadı.')
    model=AISegmenter(args.weights,threshold=.42)
    if model.output_channels!=4:
        raise SystemExit('Bu değerlendirme dört kanallı düz/kesik modeli gerektirir.')
    probabilities=[]
    labels=[]
    for image,mask,_ in valid:
        probabilities.append(model.probabilities(image))
        labels.append(mask>127)
    probabilities=np.asarray(probabilities);labels=np.asarray(labels)
    results=[]
    for threshold in (.05,.10,.20,.30,.42,.55,.70):
        predicted=probabilities>=threshold
        intersection=np.logical_and(predicted,labels).sum(axis=(0,1,2))
        union=np.logical_or(predicted,labels).sum(axis=(0,1,2))
        per_channel=[float(a/b) if b else 0.0 for a,b in zip(intersection,union)]
        results.append({'threshold':threshold,'mean_iou':round(float(np.mean(per_channel)),4),
                        'per_channel_iou':[round(value,4) for value in per_channel]})
    best=max(results,key=lambda item:item['mean_iou'])
    payload={'validation_images':len(valid),'channel_layout':list(model.channel_layout),
             'threshold_results':results,'best':best}
    args.out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
