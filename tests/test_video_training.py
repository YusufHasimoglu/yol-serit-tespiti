import cv2
import numpy as np

from lanevision.video_feedback import save_video_anchor
from lanevision.video_training import (load_video_anchor_samples, video_anchor_groups,
                                       video_training_stats)


def test_verified_video_anchor_is_counted_and_grouped_for_training(tmp_path):
    source = tmp_path / "egitim_videosu.avi"
    source.write_bytes(b"video-identity")
    frame = np.full((180, 320, 3), 64, np.uint8)
    mask = np.zeros((180, 320), np.uint8)
    cv2.line(mask, (62, 176), (140, 78), 2, 5)
    cv2.line(mask, (258, 176), (181, 78), 3, 5)
    save_video_anchor(tmp_path / "data", source, 4, frame, mask, training_name="video_kare_004")

    stats = video_training_stats(tmp_path / "data")
    assert stats["verified_video_frames"] == 1
    assert stats["source_videos"] == 1
    assert not stats["ready"]
    assert video_anchor_groups(tmp_path / "data")['video_kare_004'].startswith('video-')

    image, labels, group, name = next(load_video_anchor_samples(tmp_path / "data"))
    assert image.shape[:2] == labels.shape[:2]
    assert labels.shape[2] == 4
    assert np.any(labels[:, :, 1]) and np.any(labels[:, :, 2])
    assert group.startswith('video-')
    assert name == 'video_kare_004'
