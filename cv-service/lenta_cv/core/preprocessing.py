"""Метрика резкости и лёгкая подготовка изображений для OCR."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from lenta_cv.core.perspective import unwarp_roi_perspective


def laplacian_variance(roi_bgr: np.ndarray) -> float:
    """
    Дисперсия отклика Лапласиана по grayscale ROI.
    Выше значение коррелирует с более резким изображением.
    """
    if roi_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def bbox_clip_xyxy(xyxy: np.ndarray, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    x1 = max(0, min(w - 1, x1))
    x2 = max(1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, 1, 1
    return x1, y1, x2, y2


def write_crop_bmp(path: str | Path, img_bgr: np.ndarray) -> bool:
    """Сохранение кропа в несжатом BMP (BGR, типичное для cv2 без потерь)."""
    return bool(cv2.imwrite(str(path), np.ascontiguousarray(img_bgr)))


def pad_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    shape: Tuple[int, int],
    pad_frac: float = 0.05,
) -> Tuple[int, int, int, int]:
    px = max(2, int((x2 - x1) * pad_frac))
    py = max(2, int((y2 - y1) * pad_frac))
    h, w = shape[:2]
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(w, x2 + px),
        min(h, y2 + py),
    )


def prepare_for_ocr(
    frame_bgr: np.ndarray,
    xyxy: np.ndarray,
    pad_frac: float = 0.05,
    enhance: bool = True,
    longest_side_max: int = 1600,
    dewarp_perspective: bool = False,
    dewarp_epsilon_frac: float = 0.032,
    dewarp_min_quad_area_frac: float = 0.22,
    dewarp_max_infer_side: int = 640,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Вырезка ROI с паддингом, опционально выравнивание перспективы, CLAHE,
    ограничение максимальной стороны перед инференсом OCR.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_clip_xyxy(xyxy, (h, w))
    x1, y1, x2, y2 = pad_bbox(x1, y1, x2, y2, (h, w), pad_frac=pad_frac)
    roi = frame_bgr[y1:y2, x1:x2].copy()

    if dewarp_perspective and roi.size > 0:
        roi = unwarp_roi_perspective(
            roi,
            epsilon_frac=dewarp_epsilon_frac,
            min_area_frac=dewarp_min_quad_area_frac,
            max_infer_side=dewarp_max_infer_side,
        )

    if enhance and roi.size > 0:
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        roi = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    if roi.size > 0 and longest_side_max > 0:
        rh, rw = roi.shape[:2]
        long_side = max(rh, rw)
        if long_side > longest_side_max:
            scale = longest_side_max / float(long_side)
            roi = cv2.resize(
                roi,
                (int(rw * scale), int(rh * scale)),
                interpolation=cv2.INTER_AREA,
            )

    return roi, (x1, y1, x2, y2)
