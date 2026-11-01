import numpy as np
import pytest
from lanevision.core import LaneDetector, LaneSettings
from lanevision.sample_data import draw_scene


def test_straight_road_detects_both():
    result = LaneDetector().detect(draw_scene())
    assert result.both_found, result.stats()
    assert result.offset_px is not None
    assert abs(result.offset_px) < 45
    assert result.segment_count > 0
    assert len(result.stages) == 8
    assert result.frame.shape == (720, 1280, 3)


def test_blank_image_does_not_create_lanes():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = LaneDetector().detect(frame)
    assert not result.both_found
    assert result.offset_px is None
    assert result.segment_count == 0


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        LaneSettings(canny_low=200, canny_high=100).validate()
    with pytest.raises(ValueError):
        LaneSettings(blur_kernel=4).validate()


def test_invalid_frame_rejected():
    with pytest.raises(ValueError):
        LaneDetector().detect(np.zeros((80, 80, 3), dtype=np.uint8))


def test_video_mode_stable():
    detector = LaneDetector()
    for i in range(5):
        outcome = detector.detect(draw_scene(shift=i * 2), video=True)
        assert outcome.both_found
