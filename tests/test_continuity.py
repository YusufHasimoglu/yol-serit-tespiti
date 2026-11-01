"""Ego-lane continuity tests: detection, gaps and safe empty scenes."""
import cv2
import numpy as np
from pathlib import Path
from lanevision.continuity import _pair,_peaks,detect_continuous
from lanevision.engine import InferenceEngine,MODES

def test_peaks_find_two_separated_paint_stripes():
    mask=np.zeros((200,400),np.uint8)
    cv2.rectangle(mask,(184,135),(196,152),255,-1)
    cv2.rectangle(mask,(320,135),(331,152),255,-1)
    xs=[p[0] for p in _peaks(mask,143)]
    assert any(abs(x-190)<10 for x in xs)
    assert any(abs(x-325)<10 for x in xs)

def test_black_image_cannot_hallucinate_two_lanes():
    im=np.full((240,480,3),30,np.uint8)
    out=detect_continuous(im)
    assert not out.both_found
    assert out.left is None and out.right is None

def test_reconstructed_dashed_center_line_and_road_edge():
    h,w=400,640
    im=np.full((h,w,3),72,np.uint8)
    # left dashed center line (some gaps), right solid road edge
    for y in range(200,395,28):
        x=int(305+(.72*(350-y)))
        cv2.line(im,(x,y),(x,y+14),(245,246,247),8)
    cv2.line(im,(480,200),(635,399),(240,240,240),8)
    out=detect_continuous(im)
    assert out.left is not None
    assert out.right is not None
    assert out.both_found
    assert '08 Sonuc' in out.stages
    assert out.marking_types == ('kesik','duz')
    assert out.offset_px is not None

def test_pair_prefers_dashed_divider_and_solid_road_edge():
    """The desired pair is the same dashed/solid structure as example four."""
    h,w=400,640

    def candidate(top_x,bottom_x,score,side,dashed,solid):
        ys=np.array([180.,390.])
        return {'poly':np.polyfit(ys,[top_x,bottom_x],1), 'span':(180,390),
                'score':score, 'side':side, 'dashed_score':dashed,
                'solid_score':solid}

    bright_outer_edge=candidate(240,100,100,'left',.02,.96)
    dashed_divider=candidate(290,250,45,'left',.98,.10)
    solid_road_edge=candidate(350,480,70,'right',.03,.98)
    picked_left,picked_right=_pair([bright_outer_edge,dashed_divider],
                                   [solid_road_edge],h,w)

    assert picked_left is dashed_divider
    assert picked_right is solid_road_edge

def test_project_example_prefers_dashed_divider_and_solid_edge():
    root=Path(__file__).resolve().parent.parent
    image=cv2.imread(str(root/'ornekler'/'ornek_yol_duz.png'))
    out=detect_continuous(image)
    assert out.both_found
    assert out.marking_types == ('kesik','duz')
    assert out.offset_px is not None

def test_auto_mode_keeps_valid_dashed_solid_pair():
    root=Path(__file__).resolve().parent.parent
    image=cv2.imread(str(root/'ornekler'/'ornek_yol_duz.png'))
    out=InferenceEngine(mode='auto').detect(image)
    assert out.marking_types == ('kesik','duz')
    assert out.offset_px is not None

def test_pair_rejects_bright_guardrail_outside_ego_lane():
    """A stronger shoulder trace must not replace the camera's lane edge."""
    h,w=400,640

    def candidate(top_x,bottom_x,score,side):
        ys=np.array([180.,390.])
        return {'poly':np.polyfit(ys,[top_x,bottom_x],1), 'span':(180,390),
                'score':score, 'side':side}

    left=candidate(290,250,64,'left')
    real_right=candidate(350,480,46,'right')
    # The guardrail has more bright pixels but sits outside the ego lane.
    guardrail=candidate(415,630,130,'right')
    picked_left,picked_right=_pair([left],[guardrail,real_right],h,w)

    assert picked_left is left
    assert picked_right is real_right

def test_pair_does_not_join_unrelated_traces_when_no_lane_geometry_exists():
    h,w=400,640
    def candidate(top_x,bottom_x,score,side):
        ys=np.array([180.,390.])
        return {'poly':np.polyfit(ys,[top_x,bottom_x],1), 'span':(180,390),
                'score':score, 'side':side}

    left=candidate(355,510,80,'left')
    right=candidate(460,635,90,'right')
    picked_left,picked_right=_pair([left],[right],h,w)

    assert (picked_left is None) != (picked_right is None)

def test_explicit_mode_exists():
    assert 'continuity' in MODES
    engine=InferenceEngine(mode='continuity')
    assert engine.mode=='continuity'
