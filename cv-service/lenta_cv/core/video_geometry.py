"""Коррекция ориентации и геометрии кадра до детекции (до YOLO)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

_ROT_MAP = {
    "none": None,
    "cw90": cv2.ROTATE_90_CLOCKWISE,
    "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "rot180": cv2.ROTATE_180,
}


class _Undistorter:
    """Ленивая инициализация remap для текущего размера кадра."""

    def __init__(self, k_path: Path, d_path: Path) -> None:
        self.k_path = Path(k_path)
        self.d_path = Path(d_path)
        self._shape: Tuple[int, int] | None = None
        self._map1: Optional[np.ndarray] = None
        self._map2: Optional[np.ndarray] = None

    def _ensure_maps(self, h: int, w: int) -> None:
        if self._shape == (h, w) and self._map1 is not None:
            return
        k = np.load(self.k_path)
        d = np.load(self.d_path).reshape(-1)
        if k.shape != (3, 3):
            raise ValueError("K должен быть 3x3 (camera matrix)")
        new_k, _ = cv2.getOptimalNewCameraMatrix(k, d, (w, h), alpha=1.0, newImgSize=(w, h))
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            k, d, None, new_k, (w, h), cv2.CV_16SC2
        )
        self._shape = (h, w)

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        self._ensure_maps(h, w)
        return cv2.remap(frame_bgr, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)


def apply_frame_geometry(
    frame_bgr: np.ndarray,
    *,
    rotate: str = "none",
    flip: str = "none",
    scale_x: float = 1.0,
    undistort_k_path: str | Path | None = None,
    undistort_d_path: str | Path | None = None,
    undistorter: Any = None,
) -> np.ndarray:
    """
    Порядок: undistort (если задано) → масштаб по X (SAR / «растянутые» пиксели) → поворот → отражение.

    :param rotate: ``none`` | ``cw90`` | ``ccw90`` | ``rot180`` (ландшафт → портрет: ``cw90`` или ``ccw90``).
    :param flip: ``none`` | ``h`` (горизонтально) | ``v`` (вертикально) | ``hv``.
    :param scale_x: умножитель ширины кадра (например 1.09 если картинка сжата по горизонтали).
    """
    out = np.ascontiguousarray(frame_bgr)

    if undistorter is not None:
        out = undistorter.apply(out)
    elif undistort_k_path is not None and undistort_d_path is not None:
        u = _Undistorter(Path(undistort_k_path), Path(undistort_d_path))
        out = u.apply(out)

    if abs(float(scale_x) - 1.0) > 1e-6:
        h, w = out.shape[:2]
        nw = max(1, int(round(w * float(scale_x))))
        out = cv2.resize(out, (nw, h), interpolation=cv2.INTER_LINEAR)

    if rotate not in _ROT_MAP:
        raise ValueError(f"Неизвестный rotate={rotate!r}, допустимо: {list(_ROT_MAP)}")
    rot = _ROT_MAP[rotate]
    if rot is not None:
        out = cv2.rotate(out, rot)

    if flip == "h":
        out = cv2.flip(out, 1)
    elif flip == "v":
        out = cv2.flip(out, 0)
    elif flip == "hv":
        out = cv2.flip(out, -1)
    elif flip != "none":
        raise ValueError("flip должен быть none | h | v | hv")

    return out


def make_undistorter(k_path: str | Path | None, d_path: str | Path | None) -> _Undistorter | None:
    if k_path is None or d_path is None:
        return None
    return _Undistorter(Path(k_path), Path(d_path))


def _scaled_width(orig_w: int, scale_x: float) -> int:
    return max(1, int(round(orig_w * float(scale_x))))


def _bbox_corners_xyxy(xyxy: Sequence[float]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _xyxy_from_corners(corners: Sequence[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _point_flip_inv(x: float, y: float, w: int, h: int, flip: str) -> tuple[float, float]:
    if flip == "h":
        return float(w - 1) - x, y
    if flip == "v":
        return x, float(h - 1) - y
    if flip == "hv":
        return float(w - 1) - x, float(h - 1) - y
    return x, y


def _point_rotate_inv(x: float, y: float, pre_rotate_w: int, pre_rotate_h: int, rotate: str) -> tuple[float, float]:
    if rotate == "ccw90":
        return float(pre_rotate_w - 1) - y, x
    if rotate == "cw90":
        return y, float(pre_rotate_h - 1) - x
    if rotate == "rot180":
        return float(pre_rotate_w - 1) - x, float(pre_rotate_h - 1) - y
    return x, y


def _point_scale_x_inv(x: float, y: float, orig_w: int, scale_x: float) -> tuple[float, float]:
    scaled_w = _scaled_width(orig_w, scale_x)
    if scaled_w == orig_w:
        return x, y
    return x * float(orig_w) / float(scaled_w), y


def bbox_processed_to_original(
    xyxy: Sequence[float],
    orig_wh: Tuple[int, int],
    *,
    rotate: str = "none",
    flip: str = "none",
    scale_x: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Обратное преобразование bbox из координат кадра после ``apply_frame_geometry`` в координаты
    исходного видео (без undistort).

    :param orig_wh: (width, height) исходного кадра до геометрии.
    """
    if rotate not in _ROT_MAP:
        raise ValueError(f"Неизвестный rotate={rotate!r}, допустимо: {list(_ROT_MAP)}")
    if flip not in ("none", "h", "v", "hv"):
        raise ValueError("flip должен быть none | h | v | hv")

    orig_w, orig_h = int(orig_wh[0]), int(orig_wh[1])
    pre_w = _scaled_width(orig_w, scale_x)
    pre_h = orig_h

    if rotate in ("ccw90", "cw90"):
        proc_w, proc_h = pre_h, pre_w
    elif rotate == "rot180":
        proc_w, proc_h = pre_w, pre_h
    else:
        proc_w, proc_h = pre_w, pre_h

    out_corners: list[tuple[float, float]] = []
    for x_p, y_p in _bbox_corners_xyxy(xyxy):
        x, y = float(x_p), float(y_p)
        x, y = _point_flip_inv(x, y, proc_w, proc_h, flip)
        x, y = _point_rotate_inv(x, y, pre_w, pre_h, rotate)
        x, y = _point_scale_x_inv(x, y, orig_w, scale_x)
        out_corners.append((x, y))
    return _xyxy_from_corners(out_corners)


def bbox_original_to_processed(
    xyxy: Sequence[float],
    orig_wh: Tuple[int, int],
    *,
    rotate: str = "none",
    flip: str = "none",
    scale_x: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Прямое преобразование bbox из координат исходного видео в кадр после ``apply_frame_geometry``
    (порядок: масштаб по X → поворот → отражение). Взаимно обратно к ``bbox_processed_to_original``.
    """
    if rotate not in _ROT_MAP:
        raise ValueError(f"Неизвестный rotate={rotate!r}, допустимо: {list(_ROT_MAP)}")
    if flip not in ("none", "h", "v", "hv"):
        raise ValueError("flip должен быть none | h | v | hv")

    orig_w, orig_h = int(orig_wh[0]), int(orig_wh[1])
    pre_w = _scaled_width(orig_w, scale_x)
    pre_h = orig_h

    if rotate in ("ccw90", "cw90"):
        proc_w, proc_h = pre_h, pre_w
    elif rotate == "rot180":
        proc_w, proc_h = pre_w, pre_h
    else:
        proc_w, proc_h = pre_w, pre_h

    out_corners: list[tuple[float, float]] = []
    for x_o, y_o in _bbox_corners_xyxy(xyxy):
        x, y = float(x_o), float(y_o)
        x = x * float(pre_w) / float(orig_w)
        if rotate == "ccw90":
            xp, yp = y, float(pre_w - 1) - x
        elif rotate == "cw90":
            xp, yp = float(pre_h - 1) - y, x
        elif rotate == "rot180":
            xp, yp = float(pre_w - 1) - x, float(pre_h - 1) - y
        else:
            xp, yp = x, y
        xp, yp = _point_flip_inv(xp, yp, proc_w, proc_h, flip)
        out_corners.append((xp, yp))
    return _xyxy_from_corners(out_corners)
