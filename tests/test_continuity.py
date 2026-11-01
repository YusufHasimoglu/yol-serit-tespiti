"""Ego-lane continuity tests: detection, gaps and safe empty scenes."""
import cv2
import numpy as np
from lanevision.continuity import _peaks,detect_continuous
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

def test_explicit_mode_exists():
    assert 'continuity' in MODES
    engine=InferenceEngine(mode='continuity')
    assert engine.mode=='continuity'