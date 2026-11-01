"""Canlı video karelerini, zaman yumuşatmalı şerit tespitiyle sırayla üretir."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .core import Detection, LaneSettings
from .anchored_tracking import ManualLaneAnchor
from .engine import InferenceEngine


@dataclass(frozen=True)
class LiveFrame:
    """Bir özgün video karesi ve ona ait anlık tespit."""

    index: int
    total: int
    fps: float
    original: np.ndarray
    detection: Detection
    anchored: bool = False


class LiveVideoSession:
    """Videoyu belleğe almadan canlı görüntülemede birer kare işler.

    Her oturum kendi ``InferenceEngine`` örneğini tutar. Böylece videodaki
    zaman yumuşatma, tek fotoğraf analizinden bağımsız ve kararlı kalır.
    """

    def __init__(self, source: str | Path, settings: LaneSettings | None = None,
                 mode: str = "auto", start_frame: int = 0):
        self.source = Path(source)
        if not self.source.is_file():
            raise FileNotFoundError(self.source)
        self.capture = cv2.VideoCapture(str(self.source))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError(f"Video açılamadı: {self.source}")
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.fps = fps if np.isfinite(fps) and fps > 0 else 25.0
        self.total = max(0, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        start_frame = int(start_frame)
        if start_frame < 0 or (self.total and start_frame >= self.total):
            self.capture.release()
            raise ValueError(f"Başlangıç karesi 0 ile {max(0, self.total - 1)} arasında olmalı.")
        if start_frame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.index = start_frame
        self.closed = False
        self.settings = settings or LaneSettings()
        self.settings.validate()
        self.detector = InferenceEngine(self.settings, mode=mode)
        self.anchor: ManualLaneAnchor | None = None

    def set_manual_anchor(self, frame: np.ndarray, line_mask: np.ndarray) -> None:
        """Use a user-corrected frame as the active pair of lane boundaries."""
        self.anchor = ManualLaneAnchor(frame, line_mask, self.settings)

    def read_next(self) -> LiveFrame | None:
        """Bir sonraki kareyi tespitli olarak ver; video bittiyse ``None`` döndür."""
        if self.closed:
            return None
        ok, original = self.capture.read()
        if not ok or original is None:
            self.close()
            return None
        self.index += 1
        anchored = False
        if self.anchor is not None:
            detection = self.anchor.track(original)
            if detection is not None:
                anchored = True
            else:
                # Normal algılayıcı burada yalnızca "ölçüm adayı" üretir.
                # Kullanıcı çiziminden uzak veya ters yöndeki çizgiler asla
                # doğrudan gösterilmez; ankrajın geometrik denetiminden
                # geçerse kullanıcı düz/kesik türleriyle yeniden bağlanır.
                candidate = self.detector.detect(original, video=True)
                detection = self.anchor.reanchor_if_compatible(original, candidate)
                if detection is None:
                    detection = self.anchor.warning_detection(original)
                anchored = True
        if not anchored:
            detection = self.detector.detect(original, video=True)
        return LiveFrame(self.index, self.total, self.fps, original, detection, anchored)

    def close(self) -> None:
        if not self.closed:
            self.capture.release()
            self.closed = True

