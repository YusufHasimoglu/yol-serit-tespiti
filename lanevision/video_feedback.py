"""Kullanıcının video karelerinde yaptığı düzeltmeleri aynı videoya bağla."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2


def _video_id(source: str | Path) -> str:
    path = Path(source).resolve()
    try:
        stat = path.stat()
        identity = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        identity = str(path)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _paths(data_root: str | Path) -> tuple[Path, Path]:
    folder = Path(data_root) / "video_ankrajlari"
    return folder, folder / "index.json"


def save_video_anchor(data_root: str | Path, source: str | Path, frame_index: int,
                      frame: np.ndarray, line_mask: np.ndarray) -> Path:
    """Bir düzeltmeyi, yalnızca ait olduğu video ve kare numarasıyla sakla."""
    if frame_index < 1:
        raise ValueError("Video kare numarası 1 veya daha büyük olmalıdır.")
    if frame is None or frame.ndim != 3 or line_mask is None or line_mask.shape != frame.shape[:2]:
        raise ValueError("Video karesi ve çizgi maskesi aynı boyutta olmalıdır.")
    folder, index_path = _paths(data_root)
    folder.mkdir(parents=True, exist_ok=True)
    video_id = _video_id(source)
    artifact = f"{video_id}_kare_{frame_index:06d}.npz"
    artifact_path = folder / artifact
    np.savez_compressed(artifact_path, frame=frame, line_mask=line_mask)
    try:
        records = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else []
    except (OSError, json.JSONDecodeError):
        records = []
    records = [record for record in records
               if not (record.get("video_id") == video_id and int(record.get("frame_index", -1)) == frame_index)]
    records.append({"video_id": video_id, "frame_index": frame_index, "artifact": artifact,
                    "updated_at": datetime.now(timezone.utc).isoformat()})
    records.sort(key=lambda record: (record["video_id"], int(record["frame_index"])))
    index_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact_path


def load_video_anchors(data_root: str | Path, source: str | Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Bir video için kaydedilmiş kullanıcı ankrajlarını kare numarasıyla getir."""
    folder, index_path = _paths(data_root)
    try:
        records = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else []
    except (OSError, json.JSONDecodeError):
        records = []
    video_id = _video_id(source)
    anchors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for record in records:
        if record.get("video_id") != video_id:
            continue
        try:
            frame_index = int(record["frame_index"])
            with np.load(folder / record["artifact"], allow_pickle=False) as saved:
                frame, line_mask = saved["frame"], saved["line_mask"]
            if frame.ndim != 3 or line_mask.ndim != 2 or frame.shape[:2] != line_mask.shape:
                continue
            anchors[frame_index] = (frame.copy(), line_mask.copy())
        except (KeyError, OSError, ValueError):
            continue
    # Kalıcı kayıt sistemi eklenmeden önce, video etiketleyicisi görüntü ve
    # maskeyi ``video_adi_kare_000123_tarih`` adıyla saklıyordu. Bu eski
    # düzeltmeleri de görünür ankraja dönüştürerek kullanıcı emeğini koru.
    root = Path(data_root)
    image_root, mask_root = root / "images", root / "masks"
    prefix = f"{Path(source).stem}_kare_"
    if image_root.is_dir() and mask_root.is_dir():
        for image_path in image_root.iterdir():
            if not image_path.is_file() or not image_path.stem.startswith(prefix):
                continue
            match = re.match(rf"^{re.escape(prefix)}(\d+)_", image_path.stem)
            if match is None:
                continue
            frame_index = int(match.group(1))
            if frame_index in anchors:
                continue
            mask_path = mask_root / f"{image_path.stem}_mask.png"
            if not mask_path.is_file():
                continue
            frame = cv2.imread(str(image_path))
            line_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if frame is not None and line_mask is not None and line_mask.ndim == 2 and frame.shape[:2] == line_mask.shape:
                anchors[frame_index] = (frame, line_mask)
    return anchors
