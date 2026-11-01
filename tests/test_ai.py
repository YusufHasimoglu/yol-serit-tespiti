"""Şerit tespiti AI eğitim ve demo regresyon testleri."""
from pathlib import Path
import cv2
import numpy as np
from lanevision.hybrid import detect_hybrid
from lanevision.engine import InferenceEngine
from lanevision.training_data import synthetic_sample, tusimple_samples

ROOT=Path(__file__).resolve().parent.parent

def test_hybrid_user_uploaded_curved_example():
    image=cv2.imread(str(ROOT/'ornekler/kullanici_yolu.png'))
    detection=detect_hybrid(image)
    assert detection.both_found
    assert len(detection.stages)>=8


def test_trained_weights_and_detection():
    detector=InferenceEngine(mode='ai')
    image=cv2.imread(str(ROOT/'ornekler/kullanici_yolu.png'))
    found=detector.detect(image)
    assert found.both_found
    assert '10 AI sinif maskeleri' in found.stages
    assert detector.segmenter.meta['real_train_images']>=1


def test_synthetic_sample_annotations():
    image,mask=synthetic_sample(31)
    assert image.shape==(144,256,3)
    assert mask.shape==(144,256,2)
    assert np.count_nonzero(mask[:,:,0])>20
    assert np.count_nonzero(mask[:,:,1])>20


def test_tusimple_loader(tmp_path):
    import json
    (tmp_path/'clips').mkdir()
    cv2.imwrite(str(tmp_path/'clips/example.jpg'),np.zeros((144,256,3),np.uint8))
    record={'raw_file':'clips/example.jpg','h_samples':[70,90,110,130],
            'lanes':[[90,88,80,75],[145,155,168,184],[-2,-2,-2,-2]]}
    labels=tmp_path/'labels.json'
    labels.write_text(json.dumps(record)+'\n',encoding='utf-8')
    out=list(tusimple_samples(tmp_path,labels))
    assert len(out)==1
    assert np.count_nonzero(out[0][1][:,:,0])>0
    assert np.count_nonzero(out[0][1][:,:,1])>0