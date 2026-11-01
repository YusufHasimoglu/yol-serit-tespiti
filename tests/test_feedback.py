import cv2
import numpy as np
import pytest

from lanevision.core import Detection
from lanevision.feedback import (area_mask_from_boundaries, confirmed_mask,
                                 find_saved_review, render_reviewed_annotation, save_confirmed_detection,
                                 style_sample_count)
from lanevision.processing import write_image


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
    recalled=find_saved_review(frame,tmp_path)
    assert recalled is not None
    assert np.any(recalled["line_mask"]==1)


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


def test_review_preview_preserves_separate_dashed_strokes():
    image=np.full((120,180,3),80,np.uint8)
    mask=np.zeros((120,180),np.uint8)
    cv2.line(mask,(70,104),(76,82),2,4)
    cv2.line(mask,(80,65),(86,43),2,4)
    preview=render_reviewed_annotation(image,mask)
    assert preview.shape==image.shape
    # The wide asphalt gap remains unpainted rather than being joined.
    assert not np.any(np.all(preview[70,74:84] == (0,176,255),axis=1))


def test_review_lookup_migrates_previously_saved_manual_annotation(tmp_path):
    frame=np.full((96,160,3),90,np.uint8)
    mask=np.zeros((96,160),np.uint8)
    cv2.line(mask,(48,90),(68,40),1,4)
    cv2.line(mask,(112,90),(92,40),4,4)
    write_image(tmp_path/'images'/'old_label.png',frame)
    write_image(tmp_path/'masks'/'old_label_mask.png',mask)
    recalled=find_saved_review(frame,tmp_path)
    assert recalled is not None
    assert recalled["review"]=="saved_manual_review"
