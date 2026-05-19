"""Детекция ценников на кадре: YOLO + ByteTrack, при пустом кадре — OCR и кластеризация."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from lenta_cv.config.settings import PipelineSettings
from lenta_cv.core.iou_tracker import IoUBBoxTracker
from lenta_cv.core.preprocessing import bbox_clip_xyxy
from lenta_cv.ocr.sheet_roi import find_sheet_roi_boxes_xyxy
from lenta_cv.ocr.text_cluster import cluster_text_boxes
from lenta_cv.ocr.text_det import PROXY_CLASS_ID, get_proxy_word_boxes, get_text_det_engine

FALLBACK_TRACK_ID_OFFSET = 1_000_000


@dataclass
class FrameDetections:
    xyxy: np.ndarray
    track_ids: np.ndarray
    class_ids: np.ndarray
    confidences: np.ndarray
    source: Literal["yolo", "cluster", "yolo_empty", "none"]


def gather_proxy_words_xyxy(
    img_bgr: np.ndarray,
    *,
    text_eng: object,
    text_det_backend_flag: str,
    settings: PipelineSettings,
) -> np.ndarray:
    fh, fw = img_bgr.shape[:2]
    if settings.text_sheet_contour_first:
        rois = find_sheet_roi_boxes_xyxy(
            img_bgr,
            max_infer_side=settings.text_sheet_max_infer_side,
            min_area_frac=settings.text_sheet_min_area_frac,
            max_area_frac=settings.text_sheet_max_area_frac,
            max_rois=settings.text_sheet_max_rois,
            roi_pad_frac=settings.text_sheet_pad_frac,
            epsilon_frac=settings.text_sheet_epsilon_frac,
            min_aspect_span=settings.text_sheet_min_aspect_span,
            max_aspect_span=settings.text_sheet_max_aspect_span,
            nms_iou=settings.text_sheet_nms_iou,
            min_roi_side_px=settings.text_sheet_min_side_px,
        )
        if rois.size > 0:
            parts: list[np.ndarray] = []
            for ri in range(rois.shape[0]):
                bx = rois[ri]
                x1, y1, x2, y2 = bbox_clip_xyxy(bx, (fh, fw))
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue
                crop = img_bgr[y1:y2, x1:x2]
                wb, _ = get_proxy_word_boxes(
                    crop,
                    engine=text_eng,
                    backend=text_det_backend_flag,
                    max_long_side=settings.text_max_long_side,
                    anchor_mode=settings.text_anchor_mode,
                    rec_min_conf=settings.text_ocr_rec_min_conf,
                    fallback_all_if_empty=settings.text_anchor_fallback_all,
                )
                if wb.size == 0:
                    continue
                wc = wb.copy()
                wc[:, [0, 2]] += x1
                wc[:, [1, 3]] += y1
                parts.append(wc)
            if parts:
                return np.vstack(parts)
    words, _ = get_proxy_word_boxes(
        img_bgr,
        engine=text_eng,
        backend=text_det_backend_flag,
        max_long_side=settings.text_max_long_side,
        anchor_mode=settings.text_anchor_mode,
        rec_min_conf=settings.text_ocr_rec_min_conf,
        fallback_all_if_empty=settings.text_anchor_fallback_all,
    )
    return words


def _empty_detections(*, source: Literal["yolo", "cluster", "yolo_empty", "none"] = "none") -> FrameDetections:
    return FrameDetections(
        xyxy=np.zeros((0, 4), dtype=np.float64),
        track_ids=np.zeros(0, dtype=np.int64),
        class_ids=np.zeros(0, dtype=np.int64),
        confidences=np.zeros(0, dtype=np.float64),
        source=source,
    )


def detect_pricetag_boxes_cluster_fallback(
    img_bgr: np.ndarray,
    *,
    text_eng: object,
    text_det_backend_flag: str,
    fallback_tracker: IoUBBoxTracker,
    settings: PipelineSettings,
) -> FrameDetections:
    words = gather_proxy_words_xyxy(
        img_bgr,
        text_eng=text_eng,
        text_det_backend_flag=text_det_backend_flag,
        settings=settings,
    )
    clusters = cluster_text_boxes(
        words,
        gap_mul=settings.text_cluster_gap_mul,
        horizontal_gap_mul=settings.text_cluster_horizontal_gap_mul,
        min_y_overlap_frac=settings.text_cluster_min_y_overlap_frac,
        pad_y_top_frac=settings.text_cluster_pad_y_top_frac,
        pad_y_bottom_frac=settings.text_cluster_pad_y_bottom_frac,
        pad_x_frac=settings.text_cluster_pad_x_frac,
        split_wide_if_width_gt_mul=settings.text_cluster_split_wide_width_mul,
        split_x_gap_mul=settings.text_cluster_split_x_gap_mul,
    )
    if clusters.size == 0:
        return _empty_detections()
    ids_np, boxes_np = fallback_tracker.update(clusters)
    n = len(boxes_np)
    return FrameDetections(
        xyxy=boxes_np,
        track_ids=ids_np.astype(np.int64) + FALLBACK_TRACK_ID_OFFSET,
        class_ids=np.full(n, PROXY_CLASS_ID, dtype=np.int64),
        confidences=np.ones(n, dtype=np.float64),
        source="cluster",
    )


def detect_pricetag_boxes(
    img_bgr: np.ndarray,
    *,
    model: object,
    track_kwargs: dict,
    settings: PipelineSettings,
    text_eng: object | None = None,
    text_det_backend_flag: str | None = None,
    fallback_tracker: IoUBBoxTracker | None = None,
) -> tuple[FrameDetections, list | None]:
    """
    YOLO ``track`` на кадре; если детекций нет — OCR → ``cluster_text_boxes`` → IoU-трекер.

    Возвращает детекции и список результатов Ultralytics (для профайлинга), если был вызов YOLO.
    """
    results = model.track(img_bgr, **track_kwargs)
    if results:
        result = results[0]
        boxes = result.boxes
        if boxes is not None and boxes.id is not None and len(boxes) > 0:
            return (
                FrameDetections(
                    xyxy=boxes.xyxy.cpu().numpy(),
                    track_ids=boxes.id.cpu().numpy().astype(np.int64),
                    class_ids=boxes.cls.cpu().numpy().astype(np.int64),
                    confidences=boxes.conf.cpu().numpy(),
                    source="yolo",
                ),
                results,
            )

    if text_eng is None or text_det_backend_flag is None or fallback_tracker is None:
        return _empty_detections(source="yolo_empty"), results if results else None

    return (
        detect_pricetag_boxes_cluster_fallback(
            img_bgr,
            text_eng=text_eng,
            text_det_backend_flag=text_det_backend_flag,
            fallback_tracker=fallback_tracker,
            settings=settings,
        ),
        results if results else None,
    )


def ensure_text_engine(
    settings: PipelineSettings,
    text_state: tuple[object | None, str | None],
) -> tuple[object, str]:
    eng, flag = text_state
    if eng is not None and flag is not None:
        return eng, flag
    eng, flag = get_text_det_engine(settings.text_det_backend)
    return eng, flag
