"""Doğrulanmış video karelerini dört sınıflı şerit eğitimine hazırla.

Ham video, denetimli eğitim etiketi değildir. Bu modül yalnızca kullanıcının
etiketleyicide kaydettiği video ankrajlarını kabul eder; aynı videodan gelen
kareler eğitim/doğrulama bölmesinde ortak bir grupta tutulur.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .ai_model import INPUT_HEIGHT, INPUT_WIDTH


def _index_path(root: Path) -> Path:
    return root / "video_ankrajlari" / "index.json"


def _records(root: Path) -> list[dict]:
    path = _index_path(root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    except (OSError, json.JSONDecodeError):
        raw = []
    return [item for item in raw if isinstance(item, dict)]


def _labels(mask: np.ndarray) -> np.ndarray | None:
    """1..4 değerli maskeyi U-Net'in dört ikili kanalına dönüştür."""
    if mask.ndim != 2:
        return None
    labels = np.stack([(mask == value).astype(np.uint8) * 255 for value in range(1, 5)], axis=-1)
    if not np.any(labels[:, :, :2]) or not np.any(labels[:, :, 2:]):
        return None
    return labels


def _valid_records(root: Path):
    folder = root / "video_ankrajlari"
    for record in _records(root):
        artifact = folder / str(record.get("artifact", ""))
        try:
            with np.load(artifact, allow_pickle=False) as saved:
                frame = saved["frame"]
                mask = saved["line_mask"]
        except (KeyError, OSError, ValueError):
            continue
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[:2] != mask.shape:
            continue
        labels = _labels(mask)
        if labels is None:
            continue
        yield record, frame.copy(), labels


def video_training_stats(data_root: str | Path) -> dict:
    """Kullanıcıya yalnızca gerçek, doğrulanmış video etiketi sayısını ver."""
    root = Path(data_root)
    valid = list(_valid_records(root))
    video_ids = {str(record.get("video_id", "")) for record, _, _ in valid if record.get("video_id")}
    return {
        "verified_video_frames": len(valid),
        "source_videos": len(video_ids),
        "recommended_frames": 10,
        "recommended_videos": 3,
        "ready": len(valid) >= 10 and len(video_ids) >= 3,
    }


def video_anchor_groups(data_root: str | Path) -> dict[str, str]:
    """Kaydedilmiş eğitim görüntüsü adını, video-kaynak grubuna eşleştir."""
    groups: dict[str, str] = {}
    for record, _, _ in _valid_records(Path(data_root)):
        name = str(record.get("training_name", "")).strip()
        video_id = str(record.get("video_id", "")).strip()
        if name and video_id:
            groups[name] = f"video-{video_id}"
    return groups


def load_video_anchor_samples(data_root: str | Path):
    """Gerçek onaylı video ankrajlarını, video grubu ile birlikte üret."""
    for record, frame, labels in _valid_records(Path(data_root)):
        image = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        target = cv2.resize(labels, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
        video_id = str(record.get("video_id", "bilinmeyen"))
        name = str(record.get("training_name") or record.get("artifact") or "video_ankraji")
        yield image, target, f"video-{video_id}", name
