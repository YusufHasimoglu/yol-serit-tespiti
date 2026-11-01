import numpy as np

from lanevision.core import Detection, LaneSettings
from lanevision.temporal_tracking import VideoLaneStabilizer


def _frame():
    return np.full((360, 640, 3), 58, np.uint8)


def _detection(frame, shift=0, both=True):
    left = ((125 + shift, 350), (275 + shift, 120))
    right = ((515 + shift, 350), (365 + shift, 120)) if both else None
    output = frame.copy()
    result = Detection(output, {"01 Orijinal": frame.copy(), "08 Sonuc": output},
                       left, right, 0.0 if both else None, 1.0 if both else .5, 18)
    result.marking_types = ("kesik", "duz")
    result.method = "test ölçümü"
    return result


def test_temporal_tracker_blocks_single_frame_lane_and_area_jump():
    frame = _frame()
    tracker = VideoLaneStabilizer(LaneSettings())
    tracker.update(frame, _detection(frame))

    # Aynı genişlikte ama başka şeride sıçrayan tek karelik ölçüm kabul edilmez.
    held = tracker.update(frame, _detection(frame, shift=100))

    assert held.both_found
    assert held.left[0][0] == 125
    assert held.right[0][0] == 515
    assert getattr(held, "temporal_status") == "ani sıçrama engellendi"
    assert held.frame.shape == frame.shape


def test_temporal_tracker_accepts_consistent_new_geometry_only_after_confirmation():
    frame = _frame()
    tracker = VideoLaneStabilizer(LaneSettings(), challenger_frames=3)
    tracker.update(frame, _detection(frame))

    first = tracker.update(frame, _detection(frame, shift=100))
    second = tracker.update(frame, _detection(frame, shift=100))
    accepted = tracker.update(frame, _detection(frame, shift=100))

    assert first.left[0][0] == 125
    assert second.left[0][0] == 125
    assert accepted.left[0][0] > 160
    assert getattr(accepted, "temporal_status") == "yeni geometri onaylandı"


def test_temporal_tracker_keeps_pair_during_short_missing_measurement():
    frame = _frame()
    tracker = VideoLaneStabilizer(LaneSettings())
    tracker.update(frame, _detection(frame))

    held = tracker.update(frame, _detection(frame, both=False))

    assert held.both_found
    assert held.marking_types == ("kesik", "duz")
    assert getattr(held, "temporal_status") == "ölçüm bekleniyor"
