import cv2
import numpy as np
import pytest

from lanevision.core import Detection
from lanevision.feedback import area_mask_from_boundaries, confirmed_mask, save_confirmed_detection, style_sample_count


def _reviewed_detection(frame):
    detection=Detection(frame,{},((65,115),(92,48)),((155,115),(128,48)),0.,1.,0)
    detection.marking_types=("DÜZ","KESİK")
    return detection


def test_confirmed_detection_becomes_four_class_mask(tmp_path):
    frame=np.full((128,220,3),80,np.uint8)
    detection=_reviewed_detection(frame)
    mask=confirmed_mask(frame,detection)
    assert np.any(mask==1)
    assert np.any(mask==4)
    record=save_confirmed_detection(frame,detection,tmp_path,"source image.jpg")
    assert record["image"].is_file()
    assert record["mask"].is_file()
    assert record["area"].is_file()
    assert style_sample_count(tmp_path)==1


def test_ambiguous_confirmed_detection_requires_manual_correction():
    frame=np.full((128,220,3),80,np.uint8)
    detection=_reviewed_detection(frame)
    detection.marking_types=("belirsiz","kesik")
    with pytest.raises(ValueError,match="belirsiz"):
        confirmed_mask(frame,detection)


def test_area_mask_can_be_derived_or_drawn_manually():
    left=np.array([[70,115],[92,48]],np.int32)
    right=np.array([[155,115],[128,48]],np.int32)
    derived=area_mask_from_boundaries((128,220),left,right)
    manual=area_mask_from_boundaries((128,220),manual_points=np.array([[70,115],[92,48],[128,48],[155,115]],np.int32))
    assert np.count_nonzero(derived)>100
    assert np.count_nonzero(manual)>100
