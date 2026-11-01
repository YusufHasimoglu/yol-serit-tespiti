"""Torchvision RAFT ile kullanıcı çizim noktaları için derin optik akış."""
from __future__ import annotations

import cv2
import numpy as np


class RaftPointFlow:
    """LK başarısız olduğunda kullanılan, tembel yüklenen derin akış yedeği.

    Ağırlıklar yalnızca gerçekten gerekirse Torchvision tarafından indirilir.
    Yükleme ya da çıkarım başarısız olursa çağıran takipçi klasik yönteme
    güvenli biçimde geri döner; rastgele hareket üretilmez.
    """

    def __init__(self):
        self._model = None
        self._torch = None
        self._error: str | None = None
        self._pair: tuple[int, int] | None = None
        self._flow: np.ndarray | None = None

    @property
    def error(self) -> str | None:
        return self._error

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._error is not None:
            return False
        try:
            import torch
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
            self._torch = torch
            self._model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=True).eval()
            torch.set_num_threads(min(4, torch.get_num_threads()))
            return True
        except Exception as exc:  # ağ / ağırlık / sürüm sorunu canlı videoyu durdurmamalı
            self._error = str(exc)
            return False

    def _dense_flow(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray | None:
        pair = (id(previous), id(current))
        if pair == self._pair and self._flow is not None:
            return self._flow
        if not self._load() or self._torch is None:
            return None
        h, w = previous.shape[:2]
        # RAFT girişleri sekizin katı olmalı. CPU'da canlı kullanım için
        # ölçüyü makul tutuyoruz; akış sonra özgün çözünürlüğe ölçeklenir.
        scale = min(1.0, 384.0 / max(w, 1), 256.0 / max(h, 1))
        rw = max(128, int(round(w * scale / 8)) * 8)
        rh = max(128, int(round(h * scale / 8)) * 8)
        try:
            old = cv2.resize(previous, (rw, rh), interpolation=cv2.INTER_AREA)
            new = cv2.resize(current, (rw, rh), interpolation=cv2.INTER_AREA)
            old_tensor = self._torch.from_numpy(cv2.cvtColor(old, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float()
            new_tensor = self._torch.from_numpy(cv2.cvtColor(new, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float()
            with self._torch.inference_mode():
                flow = self._model(old_tensor, new_tensor, num_flow_updates=6)[-1][0]
            dense = flow.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            dense = cv2.resize(dense, (w, h), interpolation=cv2.INTER_LINEAR)
            dense[:, :, 0] *= w / rw
            dense[:, :, 1] *= h / rh
        except Exception as exc:
            self._error = str(exc)
            return None
        self._pair, self._flow = pair, dense
        return dense

    def track_points(self, previous: np.ndarray, current: np.ndarray, points: np.ndarray) -> np.ndarray | None:
        dense = self._dense_flow(previous, current)
        if dense is None or points is None:
            return None
        h, w = dense.shape[:2]
        flat = points.reshape(-1, 2).astype(np.float32)
        x = flat[:, 0].reshape(-1, 1)
        y = flat[:, 1].reshape(-1, 1)
        valid = (x[:, 0] >= 0) & (x[:, 0] < w - 1) & (y[:, 0] >= 0) & (y[:, 0] < h - 1)
        if int(valid.sum()) < 6:
            return None
        dx = cv2.remap(dense[:, :, 0], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).reshape(-1)
        dy = cv2.remap(dense[:, :, 1], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).reshape(-1)
        moved = flat.copy()
        moved[:, 0] += dx
        moved[:, 1] += dy
        return moved[valid].reshape(-1, 1, 2).astype(np.float32)
