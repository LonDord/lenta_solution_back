"""Настройки пайплайна детекции ценников (YOLO + fallback на кластеризацию OCR)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class PipelineSettings:
    """Параметры инференса и постобработки; веса модели задаются отдельно в ``process_video``."""

    conf: float = 0.2
    imgsz: int = 640
    device: str = ""
    tracker: str = "bytetrack.yaml"
    coco_class_ids: int | Sequence[int] | None = None
    use_geom_filter: bool = False
    min_area_frac: float = 0.0008
    max_area_frac: float = 0.15
    min_aspect_span: float = 1.15
    pad_frac: float = 0.06
    ocr_prep: bool = True
    frame_rotate: str = "ccw90"
    frame_flip: str = "none"
    frame_scale_x: float = 1.0
    undistort_k_path: Path | None = None
    undistort_d_path: Path | None = None
    # Обычно False: кроп по bbox YOLO, контраст CLAHE (см. ocr_prep), лучший кадр по лапласиану.
    # True — экспериментальный quad-dewarp по контуру внутри ROI (часто хуже на полке).
    roi_dewarp_perspective: bool = False
    roi_dewarp_epsilon_frac: float = 0.032
    roi_dewarp_min_quad_area_frac: float = 0.22
    roi_dewarp_max_infer_side: int = 640
    text_det_backend: str = "rapid"
    text_cluster_gap_mul: float = 1.25
    text_cluster_horizontal_gap_mul: float | None = 0.55
    text_cluster_min_y_overlap_frac: float = 0.35
    text_cluster_pad_y_top_frac: float = 0.65
    text_cluster_pad_y_bottom_frac: float = 0.25
    text_cluster_pad_x_frac: float = 0.02
    text_cluster_split_wide_width_mul: float = 6.0
    text_cluster_split_x_gap_mul: float = 1.05
    text_max_long_side: int = 1280
    text_anchor_mode: str = "digits"
    text_ocr_rec_min_conf: float = 0.35
    text_anchor_fallback_all: bool = True
    text_sheet_contour_first: bool = False
    text_sheet_max_infer_side: int = 960
    text_sheet_min_area_frac: float = 0.012
    text_sheet_max_area_frac: float = 0.93
    text_sheet_max_rois: int = 6
    text_sheet_pad_frac: float = 0.04
    text_sheet_epsilon_frac: float = 0.028
    text_sheet_min_aspect_span: float = 1.02
    text_sheet_max_aspect_span: float = 8.0
    text_sheet_nms_iou: float = 0.45
    text_sheet_min_side_px: int = 56
    text_track_iou: float = 0.22
    log_yolo_ms: bool = False
    log_yolo_ms_every_n: int = 30
