"""YOLOPv2 entegrasyonunun model dosyası olmadan çalışabilen birim testleri."""
import cv2
import numpy as np

from lanevision.continuity import detect_continuous
from lanevision.engine import MODES
from lanevision.gui import USER_VISIBLE_MODES
from lanevision.yolopv2 import _mask_from_outputs, prepare_yolopv2_input


class _Tensor:
    """Torch yüklemeden çıktı dönüştürücüsünü test eden küçük tensör taklidi."""
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def test_yolopv2_uses_official_letterbox_input_shape():
    frame = np.full((240, 480, 3), 64, np.uint8)
    prepared = prepare_yolopv2_input(frame)
    assert prepared.shape == (1, 3, 384, 640)
    assert prepared.dtype == np.float32


def test_yolopv2_masks_are_cropped_then_returned_to_source_shape():
    road = np.zeros((1, 2, 384, 640), np.float32)
    road[:, 0] = .1
    road[:, 1, 12:372, 200:440] = .9
    lane = np.zeros((1, 1, 384, 640), np.float32)
    lane[:, :, 12:372, 300:310] = .8
    lane_mask, road_mask = _mask_from_outputs((None, _Tensor(road), _Tensor(lane)), (240, 480))
    assert lane_mask.shape == (240, 480)
    assert road_mask.shape == (240, 480)
    assert np.count_nonzero(lane_mask) > 0
    assert np.count_nonzero(road_mask) > 0


def test_yolopv2_guidance_keeps_a_valid_dashed_solid_pair():
    height, width = 400, 640
    image = np.full((height, width, 3), 72, np.uint8)
    for y in range(200, 395, 28):
        cv2.line(image, (int(305 + .72 * (350 - y)), y),
                 (int(305 + .72 * (350 - y)), y + 14), (245, 246, 247), 8)
    cv2.line(image, (480, 200), (635, 399), (240, 240, 240), 8)
    lane = np.zeros((height, width), np.uint8)
    cv2.line(lane, (413, 200), (270, 399), 255, 32)
    cv2.line(lane, (480, 200), (635, 399), 255, 32)
    road = np.zeros((height, width), np.uint8)
    cv2.fillPoly(road, [np.array([[150, 190], [530, 190], [639, 399], [120, 399]], np.int32)], 255)
    result = detect_continuous(image, lane_evidence=lane, road_evidence=road)
    assert result.both_found
    assert result.marking_types == ("kesik", "duz")
    assert result.ai_evidence_used


def test_yolopv2_is_an_explicit_user_selectable_mode():
    assert "yolopv2" in MODES
    assert "YOLOPv2" in MODES["yolopv2"]


def test_desktop_menu_exposes_only_clear_user_choices():
    assert tuple(USER_VISIBLE_MODES) == ("auto", "yolopv2", "classic")
    assert "görsele göre" in USER_VISIBLE_MODES["auto"]
