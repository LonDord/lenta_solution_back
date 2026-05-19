"""Выравнивание перспективы ROI: поиск четырёхугольника по контурам → warpPerspective."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _order_quad_clockwise(pts: np.ndarray) -> np.ndarray:
    """Уголки в порядке: top-left → top-right → bottom-right → bottom-left (совместимо с warp)."""
    pts = pts.reshape(-1, 2).astype(np.float32)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_dimensions(quad: np.ndarray) -> Tuple[int, int]:
    tl, tr, br, bl = quad
    w = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) * 0.5))
    h = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) * 0.5))
    return max(w, 8), max(h, 8)


def _find_quad_gray(gray_u8: np.ndarray, epsilon_frac: float, min_area_frac: float) -> Optional[np.ndarray]:
    h_, w_ = gray_u8.shape[:2]
    img_a = float(h_ * w_)

    def try_contours(binary: np.ndarray) -> Optional[np.ndarray]:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:12]:
            a = cv2.contourArea(cnt)
            if a < min_area_frac * img_a:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri < 1e-6:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon_frac * peri, True)
            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx):
                continue
            return approx.reshape(4, 2).astype(np.float32)
        return None

    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    e1 = cv2.Canny(blurred, 35, 100)
    e1 = cv2.dilate(e1, np.ones((3, 3), np.uint8), iterations=1)
    q = try_contours(e1)
    if q is not None:
        return q

    m = min(h_, w_)
    if m < 35:
        return None
    th = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=9,
    )
    return try_contours(th)


def unwarp_roi_perspective(
    roi_bgr: np.ndarray,
    *,
    epsilon_frac: float = 0.032,
    min_area_frac: float = 0.22,
    max_infer_side: int = 640,
) -> np.ndarray:
    """
    Если в ROI удаётся найти выпуклый четырёхугольник контурами — вытягивает его в прямоугольник.
    Иначе возвращает вход без изменений.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return roi_bgr
    h0, w0 = roi_bgr.shape[:2]
    if min(h0, w0) < 16:
        return roi_bgr

    scale = min(1.0, float(max_infer_side) / max(h0, w0))
    if scale < 1.0:
        small_w, small_h = int(round(w0 * scale)), int(round(h0 * scale))
        gray = cv2.cvtColor(
            cv2.resize(roi_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
    else:
        small_w, small_h = w0, h0
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    quad = _find_quad_gray(gray, epsilon_frac, min_area_frac)
    if quad is None:
        return roi_bgr

    sx = w0 / float(small_w)
    sy = h0 / float(small_h)
    quad_full = quad.copy()
    quad_full[:, 0] *= sx
    quad_full[:, 1] *= sy

    ordered = _order_quad_clockwise(quad_full)
    out_w, out_h = _quad_dimensions(ordered)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(
        roi_bgr,
        m,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
