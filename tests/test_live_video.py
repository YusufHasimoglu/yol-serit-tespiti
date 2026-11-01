import cv2
import numpy as np

from lanevision.anchored_tracking import ManualLaneAnchor
from lanevision.core import LaneSettings
from lanevision.live_video import LiveVideoSession
from lanevision.sample_data import draw_scene


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


def test_manual_anchor_keeps_user_lane_types_and_moves_with_frame():
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

