import cv2
import numpy as np

import lanevision.anchored_tracking as anchored_tracking
from lanevision.anchored_tracking import ManualLaneAnchor
from lanevision.core import Detection, LaneSettings
from lanevision.live_video import LiveVideoSession
from lanevision.sample_data import draw_scene
from lanevision.video_feedback import load_video_anchors, save_video_anchor


def test_live_session_yields_each_original_frame_with_detection(tmp_path):
    source = tmp_path / "canli.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (640, 360))
    assert writer.isOpened()
    try:
        for shift in (0, 7, -5):
            writer.write(cv2.resize(draw_scene(shift=shift), (640, 360)))
    finally:
        writer.release()

    session = LiveVideoSession(source, LaneSettings(), mode="classic")
    packets = []
    try:
        while (packet := session.read_next()) is not None:
            packets.append(packet)
    finally:
        session.close()

    assert [packet.index for packet in packets] == [1, 2, 3]
    assert all(packet.original.shape == (360, 640, 3) for packet in packets)
    assert all(packet.detection.frame.shape == packet.original.shape for packet in packets)
    assert session.closed
    assert session.read_next() is None

    # Kare seçicinin verdiği sıfır tabanlı konumdan canlı oynatma devam eder.
    selected = LiveVideoSession(source, LaneSettings(), mode="classic", start_frame=1)
    try:
        first = selected.read_next()
    finally:
        selected.close()
    assert first is not None
    assert first.index == 2


def test_manual_anchor_keeps_user_lane_types_and_moves_with_frame(monkeypatch):
    height, width = 360, 640
    frame = np.full((height, width, 3), 55, np.uint8)
    # Doku, optik akışın kullanıcı çizimini güvenle takip etmesini sağlar.
    for x in range(0, width, 32):
        cv2.line(frame, (x, 120), (x - 55, height - 1), (70, 70, 70), 1)
    mask = np.zeros((height, width), np.uint8)
    left_top, left_bottom = (282, 122), (112, 350)
    right_top, right_bottom = (358, 122), (534, 350)
    # Soldaki kesik parçalar özellikle ayrı sınıf olarak korunur.
    for a, b in ((.0, .22), (.35, .58), (.72, 1.0)):
        p0 = tuple(np.rint(np.array(left_top) * (1 - a) + np.array(left_bottom) * a).astype(int))
        p1 = tuple(np.rint(np.array(left_top) * (1 - b) + np.array(left_bottom) * b).astype(int))
        cv2.line(frame, p0, p1, (245, 245, 245), 9, cv2.LINE_AA)
        cv2.line(mask, p0, p1, 2, 8, cv2.LINE_8)
    cv2.line(frame, right_top, right_bottom, (245, 245, 245), 9, cv2.LINE_AA)
    cv2.line(mask, right_top, right_bottom, 3, 8, cv2.LINE_8)
    shifted = cv2.warpAffine(frame, np.float32([[1, 0, 5], [0, 1, 0]]), (width, height))

    tracker = ManualLaneAnchor(frame, mask, LaneSettings())
    detection = tracker.track(shifted)

    assert detection is not None
    assert detection.marking_types == ("kesik", "duz")
    assert detection.method == "Kullanıcı ankrajlı optik şerit takibi"
    assert detection.left[0][0] > left_bottom[0]
    assert detection.right[0][0] > right_bottom[0]
    assert detection.frame.shape == shifted.shape

    # Çizgi anlık olarak örtülürse önceki yol eğrisi en fazla on kare boyunca
    # kullanılır; böylece model başka bir şeride sıçramaz, ama eski çizgiyi
    # video boyunca sonsuza dek de uydurmaz.
    fallback = ManualLaneAnchor(frame, mask, LaneSettings())
    monkeypatch.setattr(anchored_tracking.cv2, "calcOpticalFlowPyrLK", lambda *args, **kwargs: (None, None, None))
    blank = np.zeros_like(frame)
    predicted = [fallback.track(blank) for _ in range(10)]
    assert all(item is not None for item in predicted)
    assert all("öngörülü" in item.method for item in predicted)
    assert fallback.track(blank) is None


def test_manual_anchor_accepts_only_geometrically_compatible_reconnection():
    height, width = 360, 640
    frame = np.full((height, width, 3), 60, np.uint8)
    mask = np.zeros((height, width), np.uint8)
    cv2.line(mask, (118, 350), (282, 122), 2, 8)
    cv2.line(mask, (532, 350), (358, 122), 3, 8)
    anchor = ManualLaneAnchor(frame, mask, LaneSettings())
    near = Detection(frame.copy(), {}, ((122, 350), (286, 122)), ((536, 350), (362, 122)), 0., 1., 2)
    reconnected = anchor.reanchor_if_compatible(frame, near)
    assert reconnected is not None
    assert reconnected.marking_types == ("kesik", "duz")
    assert "yeniden bağlandı" in reconnected.method
    far = Detection(frame.copy(), {}, ((20, 350), (50, 122)), ((240, 350), (300, 122)), 0., 1., 2)
    assert anchor.reanchor_if_compatible(frame, far) is None


def test_live_video_does_not_display_unverified_detector_after_anchor_loss(tmp_path):
    source = tmp_path / "tek-kare.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (320, 180))
    assert writer.isOpened()
    writer.write(np.full((180, 320, 3), 55, np.uint8));writer.release()

    class LostAnchor:
        def track(self, frame): return None
        def reanchor_if_compatible(self, frame, candidate): return None
        def warning_detection(self, frame):
            det=Detection(frame.copy(),{},None,None,None,0.,0)
            det.method="Kullanıcı ankrajı güveni düşük — düzeltme bekliyor"
            return det

    session = LiveVideoSession(source, LaneSettings(), mode="classic")
    session.anchor = LostAnchor()
    try:
        packet=session.read_next()
    finally:
        session.close()
    assert packet is not None
    assert packet.anchored
    assert "düzeltme bekliyor" in packet.detection.method
    assert isinstance(session.anchor, LostAnchor)


def test_saved_video_anchor_is_reapplied_on_the_same_frame(tmp_path):
    source = tmp_path / "duzeltmeli.avi"
    frame = np.full((180, 320, 3), 58, np.uint8)
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (320, 180))
    assert writer.isOpened()
    writer.write(frame);writer.write(frame);writer.release()
    mask = np.zeros((180, 320), np.uint8)
    cv2.line(mask, (65, 175), (138, 76), 2, 5)
    cv2.line(mask, (255, 175), (182, 76), 3, 5)
    stored = save_video_anchor(tmp_path / "data", source, 1, frame, mask)
    assert stored.is_file()
    anchors = load_video_anchors(tmp_path / "data", source)
    assert list(anchors) == [1]
    session = LiveVideoSession(source, LaneSettings(), mode="classic", saved_anchors=anchors)
    try:
        packet = session.read_next()
    finally:
        session.close()
    assert packet is not None
    assert packet.anchored
    assert packet.detection.method == "Kaydedilmiş kullanıcı düzeltmesi"
    assert packet.detection.marking_types == ("kesik", "duz")


def test_legacy_video_feedback_is_loaded_as_an_anchor(tmp_path):
    source = tmp_path / "eski.avi"
    source.write_bytes(b"placeholder")
    image_root = tmp_path / "data" / "images";mask_root = tmp_path / "data" / "masks"
    image_root.mkdir(parents=True);mask_root.mkdir()
    frame = np.full((120, 200, 3), 60, np.uint8)
    mask = np.zeros((120, 200), np.uint8)
    cv2.line(mask, (40, 114), (82, 52), 2, 4);cv2.line(mask, (160, 114), (118, 52), 3, 4)
    stem = "eski_kare_000017_20261101"
    cv2.imwrite(str(image_root / f"{stem}.png"), frame)
    cv2.imwrite(str(mask_root / f"{stem}_mask.png"), mask)
    anchors = load_video_anchors(tmp_path / "data", source)
    assert list(anchors) == [17]


def test_manual_anchor_uses_deep_flow_when_lk_loses_points(monkeypatch):
    height, width = 360, 640
    frame = np.full((height, width, 3), 55, np.uint8)
    mask = np.zeros((height, width), np.uint8)
    cv2.line(mask, (116, 350), (282, 122), 2, 8)
    cv2.line(mask, (534, 350), (358, 122), 3, 8)

    class FakeDeepFlow:
        def track_points(self, previous, current, points):
            moved = points.copy()
            moved[:, :, 0] += 6
            return moved

    monkeypatch.setattr(anchored_tracking.cv2, "calcOpticalFlowPyrLK", lambda *args, **kwargs: (None, None, None))
    tracker = ManualLaneAnchor(frame, mask, LaneSettings(), deep_flow=FakeDeepFlow())
    detection = tracker.track(frame.copy())
    assert detection is not None
    assert detection.method == "Kullanıcı ankrajlı derin optik şerit takibi"

