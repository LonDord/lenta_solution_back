"""Простой трек без Kalman: сопоставление bbox между кадрами по IoU."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _iou_matrix(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    nm = np.zeros((len(prev), len(cur)), dtype=np.float64)
    for i in range(len(prev)):
        for j in range(len(cur)):
            ax1, ay1, ax2, ay2 = prev[i]
            bx1, by1, bx2, by2 = cur[j]
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            nm[i, j] = inter / max(ua, 1e-6)
    return nm


class IoUBBoxTracker:
    """Жадное сопоставление с порогом; новые объекту выдаются новые id."""

    def __init__(self, iou_thresh: float = 0.2) -> None:
        self.iou_thresh = float(iou_thresh)
        self._next_id = 1
        self._prev_xyxy = np.zeros((0, 4), dtype=np.float64)
        self._prev_ids: np.ndarray = np.zeros((0,), dtype=np.int32)

    def update(self, detections_xyxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: ids (n,), xyxy того же порядка
        """
        if detections_xyxy.size == 0:
            self._prev_xyxy = np.zeros((0, 4))
            self._prev_ids = np.zeros((0,), dtype=np.int32)
            return np.zeros((0,), dtype=np.int32), detections_xyxy

        cur = detections_xyxy.astype(np.float64)
        ids = np.zeros(len(cur), dtype=np.int32)
        used_prev = np.zeros(len(self._prev_xyxy), dtype=bool)

        if len(self._prev_xyxy) == 0:
            for j in range(len(cur)):
                ids[j] = self._next_id
                self._next_id += 1
            self._prev_xyxy = cur.copy()
            self._prev_ids = ids.copy()
            return ids, cur

        mat = _iou_matrix(self._prev_xyxy, cur)
        pairs: List[Tuple[float, int, int]] = []
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] >= self.iou_thresh:
                    pairs.append((float(mat[i, j]), i, j))
        pairs.sort(reverse=True)

        matched_cur = np.zeros(len(cur), dtype=bool)
        for _, pi, pj in pairs:
            if used_prev[pi] or matched_cur[pj]:
                continue
            ids[pj] = int(self._prev_ids[pi])
            used_prev[pi] = True
            matched_cur[pj] = True

        for j in range(len(cur)):
            if not matched_cur[j]:
                ids[j] = self._next_id
                self._next_id += 1

        self._prev_xyxy = cur.copy()
        self._prev_ids = ids.copy()
        return ids, cur
