"""Kullanıcı onaylı şerit tespitlerini yerel eğitim örneklerine dönüştür."""
from __future__ import annotations

import json
import unicodedata
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .processing import write_image


STYLE_CODE = {
    ("left", "duz"): 1,
    ("left", "kesik"): 2,
    ("right", "duz"): 3,
    ("right", "kesik"): 4,
}


def _normal_style(value) -> str | None:
    value = unicodedata.normalize("NFKD", str(value or "").lower()).encode("ascii", "ignore").decode("ascii")
    return value if value in {"duz", "kesik"} else None


def _points_from_item(item, shape: tuple[int, int]) -> np.ndarray | None:
    """Recover a source-image curve from every supported detector format."""
    h, w = shape
    if isinstance(item, dict):
        if "poly" in item and "span" in item:
            # Continuity tracks are fitted in a 640px-wide working image.
            tw = 640
            th = max(160, int(h * tw / w))
            low, high = item["span"]
            yy = np.arange(max(0, int(low)), min(th - 1, int(high)) + 1, 3)
            xx = np.polyval(item["poly"], yy)
            points = np.column_stack([xx * w / tw, yy * h / th])
        elif "curve" in item:
            points = np.asarray(item["curve"], dtype=np.float32)
        else:
            return None
    elif isinstance(item, tuple) and len(item) == 2:
        points = np.asarray(item[0], dtype=np.float32)
    else:
        return None
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return None
    valid = ((points[:, 0] >= 0) & (points[:, 0] < w) &
             (points[:, 1] >= 0) & (points[:, 1] < h))
    points = points[valid, :2]
    return np.rint(points).astype(np.int32) if len(points) >= 2 else None


def _fallback_points(line) -> np.ndarray | None:
    if line is None:
        return None
    points = np.asarray(line, dtype=np.int32)
    return points if points.shape == (2, 2) else None


def confirmed_mask(image: np.ndarray, detection) -> np.ndarray:
    """Create a four-class label only from a user-confirmed complete result.

    Ambiguous type labels are intentionally rejected.  A wrong target harms
    subsequent training more than a skipped sample helps it.
    """
    if image is None or image.ndim != 3:
        raise ValueError("Özgün BGR görüntü gerekli.")
    types = getattr(detection, "marking_types", (None, None))
    if detection.left is None or detection.right is None or len(types) != 2:
        raise ValueError("Eğitime eklemek için iki şerit sınırı gerekli.")
    left_type, right_type = (_normal_style(types[0]), _normal_style(types[1]))
    if left_type is None or right_type is None:
        raise ValueError("Çizgi türü belirsiz. 'Hatalı — doğrusunu çiz' ile etiketleyin.")
    pair = getattr(detection, "lane_polynomials", (None, None))
    left_item, right_item = pair if isinstance(pair, tuple) and len(pair) == 2 else (None, None)
    left = _points_from_item(left_item, image.shape[:2])
    right = _points_from_item(right_item, image.shape[:2])
    if left is None:
        left = _fallback_points(detection.left)
    if right is None:
        right = _fallback_points(detection.right)
    if left is None or right is None:
        raise ValueError("Tespit eğrileri eğitim maskesine dönüştürülemedi.")
    mask = np.zeros(image.shape[:2], np.uint8)
    thickness = max(5, round(image.shape[1] / 160))
    cv2.polylines(mask, [left], False, STYLE_CODE[("left", left_type)], thickness, cv2.LINE_8)
    cv2.polylines(mask, [right], False, STYLE_CODE[("right", right_type)], thickness, cv2.LINE_8)
    return mask


def _unique_name(source_name: str) -> str:
    stem = Path(source_name).stem or "serit"
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem)
    return f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def save_confirmed_detection(image: np.ndarray, detection, root: Path, source_name: str) -> dict:
    """Save a reviewed image+mask locally and append non-sensitive audit data."""
    root = Path(root)
    name = _unique_name(source_name)
    mask = confirmed_mask(image, detection)
    image_path = write_image(root / "images" / f"{name}.png", image)
    mask_path = write_image(root / "masks" / f"{name}_mask.png", mask)
    record = {
        "name": name,
        "source_name": Path(source_name).name,
        "review": "user_confirmed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "classes": ["left_solid_or_dashed", "right_solid_or_dashed"],
    }
    with (root / "feedback.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"name": name, "image": image_path, "mask": mask_path}


def style_sample_count(root: Path) -> int:
    """Count complete four-class masks without reading or exporting photos."""
    root = Path(root) / "masks"
    if not root.is_dir():
        return 0
    count = 0
    for path in root.glob("*_mask.png"):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2:
            continue
        has_left = np.any((mask == 1) | (mask == 2))
        has_right = np.any((mask == 3) | (mask == 4))
        if has_left and has_right:
            count += 1
    return count
