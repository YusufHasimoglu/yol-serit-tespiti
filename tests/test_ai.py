"""Şerit tespiti AI eğitim ve demo regresyon testleri."""
import json
from pathlib import Path
import cv2
import numpy as np
from lanevision.hybrid import detect_hybrid
from lanevision.core import Detection
from lanevision.engine import InferenceEngine,_curves_keep_order
from lanevision.training_data import synthetic_sample, tusimple_samples, bdd_style_samples, load_user_samples
from lanevision.ai_model import build_model

ROOT=Path(__file__).resolve().parent.parent

def test_hybrid_with_public_synthetic_example():
    image=cv2.imread(str(ROOT/'ornekler/ornek_yol_duz.png'))
    detection=detect_hybrid(image)
    assert detection.both_found
    assert len(detection.stages)>=8


def test_trained_weights_and_detection():
    detector=InferenceEngine(mode='ai')
    image=cv2.imread(str(ROOT/'ornekler/ornek_yol_duz.png'))
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


def test_style_aware_synthetic_sample_and_model_head():
    image,mask=synthetic_sample(44,style_aware=True)
    assert image.shape==(144,256,3)
    assert mask.shape==(144,256,4)
    assert np.count_nonzero(mask)>20
    assert build_model(4).head.out_channels==4


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


def test_bdd_loader_preserves_solid_and_dashed_classes(tmp_path):
    image_dir=tmp_path/'images'/'100k'/'train';image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir/'frame.jpg'),np.full((144,256,3),90,np.uint8))
    record={'name':'frame.jpg','labels':[
        {'category':'lane','attributes':{'laneDirection':'parallel','laneStyle':'dashed'},
         'poly2d':[{'vertices':[[90,80],[82,110],[72,140]]}]},
        {'category':'lane','attributes':{'laneDirection':'parallel','laneStyle':'solid'},
         'poly2d':[{'vertices':[[150,80],[172,110],[198,140]]}]},
    ]}
    labels=tmp_path/'lane.json';labels.write_text(json.dumps([record]),encoding='utf-8')
    out=list(bdd_style_samples(tmp_path,labels))
    assert len(out)==1
    _,mask,group=out[0]
    assert group=='frame'
    assert np.count_nonzero(mask[:,:,1])>0  # sol-kesik
    assert np.count_nonzero(mask[:,:,2])>0  # sağ-düz


def test_bdd_legacy_per_video_labels_are_supported(tmp_path):
    image_dir=tmp_path/'images'/'100k'/'train';image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir/'legacy.jpg'),np.full((144,256,3),90,np.uint8))
    label_dir=tmp_path/'labels'/'train';label_dir.mkdir(parents=True)
    record={'name':'legacy','frames':[{'objects':[
        {'category':'lane/single white','attributes':{'direction':'parallel','style':'dashed'},
         'poly2d':[[90,80,'C'],[82,110,'C'],[72,140,'L']]},
        {'category':'lane/single white','attributes':{'direction':'parallel','style':'solid'},
         'poly2d':[[150,80,'C'],[172,110,'C'],[198,140,'L']]},
    ]}]}
    (label_dir/'legacy.json').write_text(json.dumps(record),encoding='utf-8')
    out=list(bdd_style_samples(tmp_path,label_dir))
    assert len(out)==1
    assert np.count_nonzero(out[0][1][:,:,1])>0
    assert np.count_nonzero(out[0][1][:,:,2])>0


def test_style_aware_user_mask_loader_preserves_all_four_classes(tmp_path):
    (tmp_path/'images').mkdir();(tmp_path/'masks').mkdir()
    cv2.imwrite(str(tmp_path/'images'/'road.png'),np.full((144,256,3),80,np.uint8))
    mask=np.zeros((144,256),np.uint8)
    cv2.line(mask,(70,140),(100,72),1,4)
    cv2.line(mask,(92,140),(116,72),2,4)
    cv2.line(mask,(170,140),(145,72),3,4)
    cv2.line(mask,(205,140),(162,72),4,4)
    cv2.imwrite(str(tmp_path/'masks'/'road_mask.png'),mask)
    samples=list(load_user_samples(tmp_path,style_aware=True))
    assert len(samples)==1
    _,labels=samples[0]
    assert labels.shape==(144,256,4)
    assert all(np.count_nonzero(labels[:,:,index])>0 for index in range(4))


def test_crossing_fallback_curves_are_not_a_lane_pair():
    """Regression for the unrelated yellow curves shown in the first report."""
    frame=np.zeros((300,500,3),np.uint8)
    left_curve=np.array([[130,100],[210,180],[310,270]],np.int32)
    right_curve=np.array([[350,100],[260,180],[180,270]],np.int32)
    result=Detection(frame,{},((310,270),(130,100)),((180,270),(350,100)),None,1.,0)
    result.lane_polynomials=({'curve':left_curve,'coeffs':np.polyfit(left_curve[:,1],left_curve[:,0],2)},
                             {'curve':right_curve,'coeffs':np.polyfit(right_curve[:,1],right_curve[:,0],2)})
    assert not _curves_keep_order(result,500,300)
