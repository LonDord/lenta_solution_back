"""
Видео → YOLO (веса) → трекинг → при пустом кадре fallback OCR+кластеризация → лучшие кропы.

Пример: ``python lenta_cv/example.py`` или ``from lenta_cv import process_video``.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import cv2
import numpy as np
from ultralytics import YOLO

from lenta_cv.config.settings import PipelineSettings
from lenta_cv.core.detect import (
    detect_pricetag_boxes,
    detect_pricetag_boxes_cluster_fallback,
    ensure_text_engine,
)
from lenta_cv.core.iou_tracker import IoUBBoxTracker
from lenta_cv.core.preprocessing import bbox_clip_xyxy, laplacian_variance, prepare_for_ocr, write_crop_bmp
from lenta_cv.core.video_geometry import apply_frame_geometry, bbox_processed_to_original, make_undistorter


def _coco_class_filter(coco_class_ids: int | Sequence[int] | None) -> Optional[Set[int]]:
    if coco_class_ids is None:
        return None
    if isinstance(coco_class_ids, int):
        return {coco_class_ids}
    return set(coco_class_ids)


def _yolo_track_timing_accum_init() -> dict[str, float | int]:
    return {"n": 0, "pre": 0.0, "inf": 0.0, "post": 0.0, "other": 0.0, "wall": 0.0}


def _yolo_track_timing_update(
    results: list,
    wall_s: float,
    frame_idx: int,
    *,
    enabled: bool,
    every_n: int,
    accum: dict[str, float | int] | None,
) -> None:
    if not enabled or not results:
        return
    sp = getattr(results[0], "speed", None)
    if not isinstance(sp, dict):
        return
    pre = float(sp.get("preprocess") or 0.0)
    inf = float(sp.get("inference") or 0.0)
    post = float(sp.get("postprocess") or 0.0)
    wall_ms = wall_s * 1000.0
    other = max(0.0, wall_ms - pre - inf - post)
    if accum is not None:
        accum["n"] = int(accum["n"]) + 1
        accum["pre"] = float(accum["pre"]) + pre
        accum["inf"] = float(accum["inf"]) + inf
        accum["post"] = float(accum["post"]) + post
        accum["other"] = float(accum["other"]) + other
        accum["wall"] = float(accum["wall"]) + wall_ms
    if every_n > 0 and frame_idx % every_n == 0:
        print(
            f"[yolo timing] frame={frame_idx} preprocess={pre:.2f}ms inference={inf:.2f}ms "
            f"postprocess={post:.2f}ms track_and_overhead={other:.2f}ms wall={wall_ms:.2f}ms"
        )


def _yolo_track_timing_print_summary(accum: dict[str, float | int] | None, *, label: str = "yolo timing") -> None:
    if accum is None or int(accum["n"]) <= 0:
        return
    n = int(accum["n"])
    print(
        f"[{label}] mean over {n} frames: preprocess={float(accum['pre']) / n:.2f}ms "
        f"inference={float(accum['inf']) / n:.2f}ms postprocess={float(accum['post']) / n:.2f}ms "
        f"track_and_overhead={float(accum['other']) / n:.2f}ms wall={float(accum['wall']) / n:.2f}ms"
    )


def _box_area_frac(xyxy: np.ndarray, frame_h: int, frame_w: int) -> float:
    x1, y1, x2, y2 = bbox_clip_xyxy(xyxy, (frame_h, frame_w))
    return float((y2 - y1) * (x2 - x1)) / float(max(1, frame_h * frame_w))


def _aspect_ratio(xyxy: np.ndarray, frame_h: int, frame_w: int) -> float:
    x1, y1, x2, y2 = bbox_clip_xyxy(xyxy, (frame_h, frame_w))
    w_box = max(1, x2 - x1)
    h_box = max(1, y2 - y1)
    return (w_box / h_box) if w_box >= h_box else (h_box / w_box)


def _passes_heuristic(
    xyxy: np.ndarray,
    frame_h: int,
    frame_w: int,
    min_area_frac: float,
    max_area_frac: float,
    min_aspect_span: float,
) -> bool:
    af = _box_area_frac(xyxy, frame_h, frame_w)
    asp = _aspect_ratio(xyxy, frame_h, frame_w)
    if af < min_area_frac or af > max_area_frac:
        return False
    if asp < min_aspect_span:
        return False
    return True


def _frame_index_to_timestamp_ms(frame_idx: int, fps: float) -> int:
    """Как в GT CSV: ``frame_timestamp`` в миллисекундах от начала ролика."""
    if fps > 1e-6:
        return int(round(float(frame_idx) / fps * 1000.0))
    return int(frame_idx)


def _coord_gt_csv_style(v: float) -> str:
    """Строка координаты как в ``VIDEO/*.csv``: десятичная запятая."""
    s = f"{float(v):.4f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


def ensure_bgr(orig: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(orig)


@dataclass
class TrackObservation:
    frame_idx: int
    bbox_xyxy: np.ndarray
    sharpness: float
    cls_id: int
    cls_conf: float


@dataclass
class TrackState:
    best_sharpness: float = -1.0
    best_observation: Optional[TrackObservation] = None
    best_crop_for_ocr: Optional[np.ndarray] = None

    def consider(self, obs: TrackObservation, crop_bgr: np.ndarray) -> None:
        if crop_bgr.size == 0:
            return
        if obs.sharpness > self.best_sharpness:
            self.best_sharpness = obs.sharpness
            self.best_observation = obs
            self.best_crop_for_ocr = crop_bgr.copy()


def _consume_frame_detections(
    img_bgr: np.ndarray,
    frame_idx: int,
    xyxy: np.ndarray,
    ids_arr: np.ndarray,
    cls_arr: np.ndarray,
    sc_arr: np.ndarray,
    allowed: Optional[Set[int]],
    settings: PipelineSettings,
    tracks: Dict[int, TrackState],
) -> None:
    fh, fw = img_bgr.shape[:2]
    if xyxy.size == 0:
        return
    for bi in range(xyxy.shape[0]):
        tid = int(ids_arr[bi])
        cid = int(cls_arr[bi])
        if allowed is not None and cid >= 0 and cid not in allowed:
            continue
        bbox = xyxy[bi].astype(np.float64)

        geom_ok = (allowed is not None or not settings.use_geom_filter) or _passes_heuristic(
            bbox,
            fh,
            fw,
            settings.min_area_frac,
            settings.max_area_frac,
            settings.min_aspect_span,
        )
        if not geom_ok:
            continue

        bx1, by1, bx2, by2 = bbox_clip_xyxy(bbox, (fh, fw))
        roi_vis = img_bgr[by1:by2, bx1:bx2]
        sharp = laplacian_variance(roi_vis)

        obs = TrackObservation(
            frame_idx=frame_idx,
            bbox_xyxy=bbox,
            sharpness=float(sharp),
            cls_id=cid,
            cls_conf=float(sc_arr[bi]),
        )

        crop_for_ocr, _ = prepare_for_ocr(
            img_bgr,
            bbox,
            pad_frac=settings.pad_frac,
            enhance=settings.ocr_prep,
            dewarp_perspective=settings.roi_dewarp_perspective,
            dewarp_epsilon_frac=settings.roi_dewarp_epsilon_frac,
            dewarp_min_quad_area_frac=settings.roi_dewarp_min_quad_area_frac,
            dewarp_max_infer_side=settings.roi_dewarp_max_infer_side,
        )

        tracks.setdefault(tid, TrackState()).consider(obs, crop_for_ocr)


def process_video(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    weights: str,
    settings: PipelineSettings | None = None,
    conf: float | None = None,
    device: str | None = None,
    frame_rotate: str | None = None,
) -> Path:
    """
    Обрабатывает видео: YOLO+ByteTrack; на кадрах без детекций — OCR и кластеризация.

    :param weights: Путь к ``.pt`` (дообученная модель или базовый чекпойнт Ultralytics).
    :param settings: Полные настройки; если ``None`` — ``PipelineSettings()`` с переопределением
        отдельных полей через ``conf``, ``device``, ``frame_rotate``.
    """
    cfg = settings or PipelineSettings()
    if conf is not None:
        cfg.conf = conf
    if device is not None:
        cfg.device = device
    if frame_rotate is not None:
        cfg.frame_rotate = frame_rotate

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    crops_dir = output_dir / "crops_ocr_ready"
    crops_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "tracks_best_manifest.json"
    csv_path = output_dir / "tracks_best_summary.csv"

    timing_accum = _yolo_track_timing_accum_init() if cfg.log_yolo_ms else None

    model = YOLO(weights)
    allowed = _coco_class_filter(cfg.coco_class_ids)
    undistorter = make_undistorter(cfg.undistort_k_path, cfg.undistort_d_path)
    tracks: Dict[int, TrackState] = {}
    fallback_tracker = IoUBBoxTracker(iou_thresh=cfg.text_track_iou)

    track_kwargs: dict = dict(
        conf=float(cfg.conf),
        imgsz=int(cfg.imgsz),
        verbose=False,
        persist=True,
        tracker=cfg.tracker,
        stream=False,
    )
    if cfg.device:
        track_kwargs["device"] = cfg.device
    predict_classes = list(allowed) if allowed is not None else None
    if predict_classes is not None:
        track_kwargs["classes"] = predict_classes

    text_eng: object | None = None
    text_det_backend_flag: str | None = None
    text_state: tuple[object | None, str | None] = (None, None)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    video_orig_wh = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    frame_idx = 0
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            raw_bgr = ensure_bgr(raw)
            img_bgr = apply_frame_geometry(
                raw_bgr,
                rotate=cfg.frame_rotate,
                flip=cfg.frame_flip,
                scale_x=cfg.frame_scale_x,
                undistorter=undistorter,
            )

            t_yolo = time.perf_counter()
            det, yolo_results = detect_pricetag_boxes(
                img_bgr,
                model=model,
                track_kwargs=track_kwargs,
                settings=cfg,
                text_eng=text_eng,
                text_det_backend_flag=text_det_backend_flag,
                fallback_tracker=fallback_tracker,
            )
            wall_yolo = time.perf_counter() - t_yolo

            if det.source == "yolo_empty":
                text_eng, text_det_backend_flag = ensure_text_engine(cfg, text_state)
                text_state = (text_eng, text_det_backend_flag)
                t_fb = time.perf_counter()
                det = detect_pricetag_boxes_cluster_fallback(
                    img_bgr,
                    text_eng=text_eng,
                    text_det_backend_flag=text_det_backend_flag,
                    fallback_tracker=fallback_tracker,
                    settings=cfg,
                )
                wall_yolo += time.perf_counter() - t_fb

            if cfg.log_yolo_ms and yolo_results:
                _yolo_track_timing_update(
                    yolo_results,
                    wall_yolo,
                    frame_idx,
                    enabled=True,
                    every_n=cfg.log_yolo_ms_every_n,
                    accum=timing_accum,
                )

            if det.xyxy.size == 0:
                frame_idx += 1
                continue

            _consume_frame_detections(
                img_bgr,
                frame_idx,
                det.xyxy,
                det.track_ids,
                det.class_ids,
                det.confidences,
                allowed,
                cfg,
                tracks,
            )
            frame_idx += 1
    finally:
        cap.release()

    manifest: List[dict] = []
    for tid, ts in sorted(tracks.items()):
        best = ts.best_observation
        if best is None or ts.best_crop_for_ocr is None:
            continue
        fp = crops_dir / f"track_{tid:06d}_frame_{best.frame_idx:06d}.bmp"
        write_crop_bmp(fp, ts.best_crop_for_ocr)
        bx_orig = list(
            bbox_processed_to_original(
                best.bbox_xyxy,
                video_orig_wh,
                rotate=cfg.frame_rotate,
                flip=cfg.frame_flip,
                scale_x=cfg.frame_scale_x,
            )
        )
        manifest.append(
            {
                "track_id": int(tid),
                "best_frame": int(best.frame_idx),
                "bbox_xyxy": bx_orig,
                "laplacian_variance": round(best.sharpness, 3),
                "coco_class_id": int(best.cls_id),
                "det_conf": round(best.cls_conf, 4),
                "crop_path": str(fp.resolve()),
            }
        )

    filename_stem = video_path.name.strip()
    manifest_gt_like = [
        {
            "filename": filename_stem,
            "frame_timestamp": _frame_index_to_timestamp_ms(int(row["best_frame"]), fps),
            "x_min": _coord_gt_csv_style(float(row["bbox_xyxy"][0])),
            "y_min": _coord_gt_csv_style(float(row["bbox_xyxy"][1])),
            "x_max": _coord_gt_csv_style(float(row["bbox_xyxy"][2])),
            "y_max": _coord_gt_csv_style(float(row["bbox_xyxy"][3])),
        }
        for row in manifest
    ]
    meta_path.write_text(json.dumps(manifest_gt_like, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "track_id",
        "best_frame",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "laplacian_variance",
        "coco_class_id",
        "det_conf",
        "crop_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for row in manifest:
            bx = row["bbox_xyxy"]
            w.writerow(
                [
                    row["track_id"],
                    row["best_frame"],
                    bx[0],
                    bx[1],
                    bx[2],
                    bx[3],
                    row["laplacian_variance"],
                    row["coco_class_id"],
                    row["det_conf"],
                    row["crop_path"],
                ]
            )

    if cfg.log_yolo_ms:
        _yolo_track_timing_print_summary(timing_accum)

    return csv_path
