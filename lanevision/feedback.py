"""Kullanıcı onaylı şerit tespitlerini yerel eğitim örneklerine dönüştür."""
from __future__ import annotations

import json
import hashlib
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


def area_mask_from_boundaries(shape: tuple[int, int], left: np.ndarray | None = None,
                              right: np.ndarray | None = None,
                              manual_points: np.ndarray | None = None) -> np.ndarray:
    """Return a binary drivable-area mask from a manual polygon or two boundaries."""
    area = np.zeros(shape, np.uint8)
    if manual_points is not None:
        polygon = np.asarray(manual_points, dtype=np.int32)
        if polygon.ndim == 2 and len(polygon) >= 3:
            cv2.fillPoly(area, [polygon], 255, cv2.LINE_8)
        return area
    if left is None or right is None:
        return area
    left = np.asarray(left, dtype=np.int32)
    right = np.asarray(right, dtype=np.int32)
    if left.ndim != 2 or right.ndim != 2 or len(left) < 2 or len(right) < 2:
        return area
    # Curves can originate at either end depending on the detector.  Sorting
    # both upward gives a stable non-crossing polygon for the lane interior.
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]
    if np.median(left[:, 0]) >= np.median(right[:, 0]):
        return area
    cv2.fillPoly(area, [np.vstack([left, right[::-1]])], 255, cv2.LINE_8)
    return area


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


def image_fingerprint(image: np.ndarray) -> str:
    """Stable fingerprint for the decoded pixels of one original photograph."""
    if image is None or image.ndim != 3:
        raise ValueError("Özgün BGR görüntü gerekli.")
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def record_review(root: Path, image: np.ndarray, name: str, source_name: str,
                  review: str) -> dict:
    """Index a local review so the same photo can be recalled without inference."""
    root = Path(root)
    record = {
        "name": name,
        "source_name": Path(source_name).name,
        "review": review,
        "fingerprint": image_fingerprint(image),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "feedback.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _read_mask(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)


def find_saved_review(image: np.ndarray, root: Path) -> dict | None:
    """Find the latest local correction for exactly the same decoded image.

    The fallback scan also recognizes corrections created before the review
    index existed, because their lossless PNG source copy has identical pixels.
    """
    root = Path(root)
    fingerprint = image_fingerprint(image)
    records_path = root / "feedback.jsonl"
    if records_path.is_file():
        try:
            records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            records = []
        for record in reversed(records):
            if record.get("fingerprint") != fingerprint:
                continue
            name = str(record.get("name", ""))
            line_mask = _read_mask(root / "masks" / f"{name}_mask.png")
            if line_mask is not None and line_mask.ndim == 2:
                return {"name": name, "review": record.get("review", "user_review"),
                        "line_mask": line_mask,
                        "area_mask": _read_mask(root / "areas" / f"{name}_area.png")}
    image_root = root / "images"
    if not image_root.is_dir():
        return None
    for candidate in sorted(image_root.glob("*.png"), key=lambda item: item.stat().st_mtime, reverse=True):
        copy = _read_mask(candidate)
        if copy is None or copy.ndim != 3 or image_fingerprint(copy) != fingerprint:
            continue
        name = candidate.stem
        line_mask = _read_mask(root / "masks" / f"{name}_mask.png")
        if line_mask is not None and line_mask.ndim == 2:
            return {"name": name, "review": "saved_manual_review", "line_mask": line_mask,
                    "area_mask": _read_mask(root / "areas" / f"{name}_area.png")}
    return None


def save_confirmed_detection(image: np.ndarray, detection, root: Path, source_name: str) -> dict:
    """Save a reviewed image+mask locally and append non-sensitive audit data."""
    root = Path(root)
    name = _unique_name(source_name)
    mask = confirmed_mask(image, detection)
    pair = getattr(detection, "lane_polynomials", (None, None))
    left_item, right_item = pair if isinstance(pair, tuple) and len(pair) == 2 else (None, None)
    left = _points_from_item(left_item, image.shape[:2])
    right = _points_from_item(right_item, image.shape[:2])
    if left is None:
        left = _fallback_points(detection.left)
    if right is None:
        right = _fallback_points(detection.right)
    area = area_mask_from_boundaries(image.shape[:2], left, right)
    image_path = write_image(root / "images" / f"{name}.png", image)
    mask_path = write_image(root / "masks" / f"{name}_mask.png", mask)
    area_path = write_image(root / "areas" / f"{name}_area.png", area)
    record_review(root, image, name, source_name, "user_confirmed")
    return {"name": name, "image": image_path, "mask": mask_path, "area": area_path}


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


def render_reviewed_annotation(image: np.ndarray, line_mask: np.ndarray,
                               area_mask: np.ndarray | None = None) -> np.ndarray:
    """Render a user-drawn correction without connecting dashed components."""
    if image is None or image.ndim != 3:
        raise ValueError("Özgün BGR görüntü gerekli.")
    if line_mask is None or line_mask.ndim != 2:
        raise ValueError("Dört sınıflı çizgi maskesi gerekli.")
    output = image.copy()
    if area_mask is not None and area_mask.shape == line_mask.shape and np.any(area_mask):
        tint = output.copy()
        tint[area_mask > 0] = (70, 177, 105)
        output = cv2.addWeighted(output, .70, tint, .30, 0)
    colors = {1: (255, 223, 30), 2: (0, 176, 255),
              3: (255, 80, 190), 4: (0, 235, 120)}
    for code, color in colors.items():
        pixels = (line_mask == code).astype(np.uint8) * 255
        # Only dilate sideways: separate dashed strokes remain visibly apart.
        pixels = cv2.dilate(pixels, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3)))
        output[pixels > 0] = color
    h, w = output.shape[:2]
    cv2.rectangle(output, (8, 7), (min(w - 5, 620), 47), (17, 29, 42), -1)
    cv2.putText(output, "KULLANICI ONAYLI | EGITIME EKLENDI", (17, 34),
                cv2.FONT_HERSHEY_SIMPLEX, .62, (241, 249, 255), 2, cv2.LINE_AA)
    return output
