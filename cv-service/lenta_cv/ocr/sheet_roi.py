"""Крупные ROI «листа/наклейки» по контурам всего кадра — затем OCR внутри кропов."""

from __future__ import annotations

import cv2
import numpy as np

from lenta_cv.core.preprocessing import bbox_clip_xyxy


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ar = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    br = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / (ar + br - inter))


def _collect_boxes_from_binary(
    binary_u8: np.ndarray,
    img_area: float,
    *,
    min_area_frac: float,
    max_area_frac: float,
    epsilon_frac: float,
    min_aspect_span: float,
    max_aspect_span: float,
) -> list[np.ndarray]:
    binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    out: list[np.ndarray] = []
    for cnt in contours[:28]:
        a = cv2.contourArea(cnt)
        if a < min_area_frac * img_area or a > max_area_frac * img_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-6:
            continue
        eps = epsilon_frac * peri

        bbox = cv2.minAreaRect(cnt)
        w_r, h_r = bbox[1]
        w_, h_ = float(w_r), float(h_r)
        if min(w_, h_) < 1e-3:
            continue
        ar_span = max(w_, h_) / max(min(w_, h_), 1e-6)
        if ar_span < min_aspect_span or ar_span > max_aspect_span:
            continue

        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) >= 4:
            pts = approx.reshape(-1, 2).astype(np.float64)
            x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        else:
            x, y, w0, h0 = cv2.boundingRect(cnt)
            x1, y1, x2, y2 = float(x), float(y), float(x + w0), float(y + h0)

        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) * (y2 - y1) < min_area_frac * img_area:
            continue
        out.append(np.array([x1, y1, x2, y2], dtype=np.float64))
    return out


def _nms_xyxy(boxes: list[np.ndarray], iou_thresh: float) -> list[np.ndarray]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept: list[np.ndarray] = []
    for b in boxes:
        if any(_iou_xyxy(b, k) >= iou_thresh for k in kept):
            continue
        kept.append(b)
    return kept


def find_sheet_roi_boxes_xyxy(
    img_bgr: np.ndarray,
    *,
    max_infer_side: int = 960,
    min_area_frac: float = 0.015,
    max_area_frac: float = 0.92,
    max_rois: int = 6,
    roi_pad_frac: float = 0.035,
    epsilon_frac: float = 0.028,
    min_aspect_span: float = 1.02,
    max_aspect_span: float = 8.0,
    nms_iou: float = 0.45,
    min_roi_side_px: int = 64,
) -> np.ndarray:
    """
    Контуры по Canny (+ fallback adaptive) на даунскейле → ось-параллельные bbox крупных прямых регионов.

    Пустой массив означает «нет кандидатов» — вызывающий код обычно тогда OCR по всему кадру.
    """
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    h0, w0 = img_bgr.shape[:2]
    m = max(h0, w0)
    scale = 1.0 if max_infer_side <= 0 or m <= max_infer_side else float(max_infer_side) / float(m)
    if scale >= 1.0:
        small = img_bgr
        sx = sy = 1.0
    else:
        sw, sh = int(round(w0 * scale)), int(round(h0 * scale))
        small = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        sx = w0 / float(sw)
        sy = h0 / float(sh)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    small_area = float(gray.shape[0] * gray.shape[1])

    binaries: list[np.ndarray] = []
    e = cv2.Canny(blurred, 35, 100)
    binaries.append(cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=2))
    if min(gray.shape) >= 48:
        bs = min(int(gray.shape[0]), int(gray.shape[1]), 51)
        if bs % 2 == 0:
            bs -= 1
        bs = max(21, bs)
        th = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=bs,
            C=9,
        )
        binaries.append(th)

    kw = dict(
        min_area_frac=min_area_frac,
        max_area_frac=max_area_frac,
        epsilon_frac=epsilon_frac,
        min_aspect_span=min_aspect_span,
        max_aspect_span=max_aspect_span,
    )
    pooled: list[np.ndarray] = []
    for bn in binaries:
        pooled.extend(_collect_boxes_from_binary(bn, small_area, **kw))

    pooled_full: list[np.ndarray] = []
    for b in pooled:
        x1, y1, x2, y2 = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
        w = x2 - x1
        h_ = y2 - y1
        x1 -= w * roi_pad_frac
        x2 += w * roi_pad_frac
        y1 -= h_ * roi_pad_frac
        y2 += h_ * roi_pad_frac
        cx1, cy1, cx2, cy2 = bbox_clip_xyxy(np.array([x1, y1, x2, y2]), (h0, w0))
        if cx2 - cx1 < min_roi_side_px or cy2 - cy1 < min_roi_side_px:
            continue
        pooled_full.append(np.array([cx1, cy1, cx2, cy2], dtype=np.float64))

    kept = _nms_xyxy(pooled_full, nms_iou)
    if not kept:
        return np.zeros((0, 4), dtype=np.float64)
    kept = sorted(kept, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[: max(1, max_rois)]
    return np.stack(kept, axis=0)
