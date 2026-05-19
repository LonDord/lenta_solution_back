"""
Оценка детекций по CSV-разметке в координатах исходного видео.

Разметка (например ``43_15.csv``) задаёт bbox в системе координат оригинального кадра.
Пайплайн крутит кадр (по умолчанию ``ccw90``); предсказания из внешнего JSON timeline (``--timeline``)
или прямой инференс (``--detect``) приводятся через ``bbox_processed_to_original`` и сопоставляются с GT по IoU.

Запуск без аргументов (пути по умолчанию из ``lenta_cv.config.paths``)::

    python -m lenta_cv.eval --detect

Без ``--detect`` нужен файл timeline (поле ``entries`` с ``frame_index`` и ``bbox_xyxy``); базовый пайплайн его больше не пишет.

``frame_timestamp`` в CSV по умолчанию в миллисекундах (2472 → кадр ~49 при fps≈20).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from lenta_cv.config.paths import (
    DEFAULT_GT_CSV,
    DEFAULT_VIDEO,
    EVAL_REPORT_JSON,
    EVAL_VIZ_DIR,
    MODEL_DEFAULT,
    MODEL_HIGH,
    MODEL_LOW,
    OUTPUT_DIR,
    REPO_ROOT,
    TIMELINE_JSON,
)
from lenta_cv.config.settings import PipelineSettings
from lenta_cv.core.detect import detect_pricetag_boxes
from lenta_cv.core.pipeline import ensure_bgr
from lenta_cv.core.video_geometry import apply_frame_geometry, bbox_processed_to_original

CONFIG = {
    "GT": DEFAULT_GT_CSV,
    "VIDEO": DEFAULT_VIDEO,
    "OUTPUT_DIR": OUTPUT_DIR,
    "TIMELINE": TIMELINE_JSON,
    "REPORT": EVAL_REPORT_JSON,
    "VIZ_DIR": EVAL_VIZ_DIR,
    "WEIGHTS": MODEL_DEFAULT,
    "WEIGHTS_LOW": MODEL_LOW,
    "WEIGHTS_HIGH": MODEL_HIGH,
}


def _resolve_path(p: Path) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class GtRow:
    frame: int
    """Индекс кадра в видеофайле (после преобразования timestamp)."""
    timestamp_raw: int
    bbox: BBox
    product_name: str
    barcode: str


@dataclass
class MatchRow:
    frame: int
    gt_index: int
    pred_index: int
    iou: float
    gt: BBox
    pred: BBox
    product_name: str


def _parse_num(raw: str) -> float:
    s = str(raw).strip().replace(",", ".")
    if not s or s.lower() == "нет":
        return float("nan")
    return float(s)


def timestamp_to_frame_index(timestamp_raw: int, *, unit: str, fps: float) -> int:
    """
  ``ms`` — миллисекунды от начала ролика (как в ``43_15.csv``: 2472 → ~2.47 с);
    ``frame`` — прямой индекс кадра в файле.
    """
    if unit == "frame":
        return int(timestamp_raw)
    if unit == "ms":
        if fps <= 0.0:
            raise ValueError("fps=0: нельзя перевести timestamp из миллисекунд")
        return int(round(float(timestamp_raw) / 1000.0 * fps))
    raise ValueError(f"Неизвестный timestamp-unit={unit!r}, допустимо: ms | frame")


def load_gt_csv(path: Path, *, unit: str, fps: float) -> Dict[int, List[GtRow]]:
    by_frame: DefaultDict[int, List[GtRow]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_raw = int(row["frame_timestamp"])
            frame = timestamp_to_frame_index(ts_raw, unit=unit, fps=fps)
            bbox = BBox(
                _parse_num(row["x_min"]),
                _parse_num(row["y_min"]),
                _parse_num(row["x_max"]),
                _parse_num(row["y_max"]),
            )
            by_frame[frame].append(
                GtRow(
                    frame=frame,
                    timestamp_raw=ts_raw,
                    bbox=bbox,
                    product_name=str(row.get("product_name", "")).strip(),
                    barcode=str(row.get("barcode", "")).strip(),
                )
            )
    return dict(by_frame)


def load_timeline_predictions(
    path: Path,
    orig_wh: Tuple[int, int],
    *,
    rotate: str,
    flip: str,
    scale_x: float,
) -> Dict[int, List[BBox]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    # Новый timeline (после pipeline): bbox_xyxy уже в координатах оригинала (есть frame_geometry).
    coords_original = "frame_geometry" in doc
    by_frame: DefaultDict[int, List[BBox]] = defaultdict(list)
    for entry in doc.get("entries", []):
        frame = int(entry["frame_index"])
        if coords_original:
            xyxy = tuple(float(v) for v in entry["bbox_xyxy"])
        else:
            xyxy = bbox_processed_to_original(
                entry["bbox_xyxy"],
                orig_wh,
                rotate=rotate,
                flip=flip,
                scale_x=scale_x,
            )
        by_frame[frame].append(BBox(*xyxy))
    return dict(by_frame)


def detect_on_frames(
    video_path: Path,
    frame_ids: Sequence[int],
    weights: Path,
    settings: PipelineSettings,
) -> Dict[int, List[BBox]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_wh = (orig_w, orig_h)

    model = YOLO(str(weights))
    track_kwargs: dict = dict(
        conf=float(settings.conf),
        imgsz=int(settings.imgsz),
        verbose=False,
        persist=True,
        tracker=settings.tracker,
        stream=False,
    )
    if settings.device:
        track_kwargs["device"] = settings.device

    by_frame: Dict[int, List[BBox]] = {}
    for frame_idx in sorted(set(int(f) for f in frame_ids)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, raw = cap.read()
        if not ok:
            print(f"WARN: кадр {frame_idx} не прочитан из видео", file=sys.stderr)
            by_frame[frame_idx] = []
            continue

        img = ensure_bgr(raw)
        img = apply_frame_geometry(
            img,
            rotate=settings.frame_rotate,
            flip=settings.frame_flip,
            scale_x=settings.frame_scale_x,
        )
        det, _ = detect_pricetag_boxes(img, model=model, track_kwargs=track_kwargs, settings=settings)
        boxes: List[BBox] = []
        for i in range(det.xyxy.shape[0]):
            xyxy = det.xyxy[i].tolist()
            orig = bbox_processed_to_original(
                xyxy,
                orig_wh,
                rotate=settings.frame_rotate,
                flip=settings.frame_flip,
                scale_x=settings.frame_scale_x,
            )
            boxes.append(BBox(*orig))
        by_frame[frame_idx] = boxes

    cap.release()
    return by_frame


def iou_xyxy(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _fmt_xyxy(b: BBox) -> str:
    return f"[{b.x1:.1f}, {b.y1:.1f}, {b.x2:.1f}, {b.y2:.1f}]"


def _bbox_dict(b: BBox, **extra: object) -> dict:
    row = {"x_min": round(b.x1, 1), "y_min": round(b.y1, 1), "x_max": round(b.x2, 1), "y_max": round(b.y2, 1)}
    row.update(extra)
    return row


def match_boxes(
    gt_boxes: Sequence[BBox],
    pred_boxes: Sequence[BBox],
    iou_thresh: float,
) -> Tuple[List[MatchRow], List[int], List[int]]:
    if not gt_boxes and not pred_boxes:
        return [], [], []
    if not gt_boxes:
        return [], [], list(range(len(pred_boxes)))
    if not pred_boxes:
        return [], list(range(len(gt_boxes))), []

    n_g, n_p = len(gt_boxes), len(pred_boxes)
    iou_mat = np.zeros((n_g, n_p), dtype=np.float64)
    for gi, g in enumerate(gt_boxes):
        for pi, p in enumerate(pred_boxes):
            iou_mat[gi, pi] = iou_xyxy(g, p)

    matches: List[MatchRow] = []
    used_g: set[int] = set()
    used_p: set[int] = set()

    while True:
        best = (-1.0, -1, -1)
        for gi in range(n_g):
            if gi in used_g:
                continue
            for pi in range(n_p):
                if pi in used_p:
                    continue
                v = float(iou_mat[gi, pi])
                if v > best[0]:
                    best = (v, gi, pi)
        if best[0] < iou_thresh:
            break
        _, gi, pi = best
        used_g.add(gi)
        used_p.add(pi)
        matches.append(
            MatchRow(
                frame=-1,
                gt_index=gi,
                pred_index=pi,
                iou=best[0],
                gt=gt_boxes[gi],
                pred=pred_boxes[pi],
                product_name="",
            )
        )

    fn = [i for i in range(n_g) if i not in used_g]
    fp = [i for i in range(n_p) if i not in used_p]
    return matches, fn, fp


def evaluate(
    gt_by_frame: Dict[int, List[GtRow]],
    pred_by_frame: Dict[int, List[BBox]],
    iou_thresh: float,
) -> dict:
    all_matches: List[MatchRow] = []
    total_tp = total_fp = total_fn = 0
    ious: List[float] = []
    per_frame: List[dict] = []

    frames = sorted(gt_by_frame.keys())
    for frame in frames:
        gt_rows = gt_by_frame.get(frame, [])
        gt_boxes = [r.bbox for r in gt_rows]
        pred_boxes = pred_by_frame.get(frame, [])
        matches, fn_idx, fp_idx = match_boxes(gt_boxes, pred_boxes, iou_thresh)
        tp = len(matches)
        fp = len(fp_idx)
        fn = len(fn_idx)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        for m in matches:
            m.frame = frame
            if m.gt_index < len(gt_rows):
                m.product_name = gt_rows[m.gt_index].product_name
            all_matches.append(m)
            ious.append(m.iou)

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_frame.append(
            {
                "frame": frame,
                "coordinate_space": "original_video",
                "gt": len(gt_boxes),
                "pred": len(pred_boxes),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "mean_iou": round(float(np.mean([m.iou for m in matches])), 4) if matches else None,
                "gt_boxes": [
                    _bbox_dict(r.bbox, product_name=r.product_name, barcode=r.barcode)
                    for r in gt_rows
                ],
                "pred_boxes": [_bbox_dict(b) for b in pred_boxes],
                "matched": [
                    {
                        "gt_index": m.gt_index,
                        "pred_index": m.pred_index,
                        "iou": round(m.iou, 4),
                        "product_name": m.product_name,
                        "gt": _bbox_dict(m.gt),
                        "pred": _bbox_dict(m.pred),
                    }
                    for m in matches
                ],
                "false_negatives": [
                    _bbox_dict(gt_boxes[i], product_name=gt_rows[i].product_name, gt_index=i)
                    for i in fn_idx
                ],
                "false_positives": [
                    _bbox_dict(pred_boxes[i], pred_index=i) for i in fp_idx
                ],
            }
        )

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "iou_thresh": iou_thresh,
        "frames_evaluated": len(frames),
        "gt_boxes": sum(len(v) for v in gt_by_frame.values()),
        "pred_boxes": sum(len(pred_by_frame.get(f, [])) for f in frames),
        "frames_with_gt": frames,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mean_iou_matched": round(float(np.mean(ious)), 4) if ious else None,
        "per_frame": per_frame,
        "matches": [
            {
                "frame": m.frame,
                "gt_index": m.gt_index,
                "pred_index": m.pred_index,
                "iou": round(m.iou, 4),
                "product_name": m.product_name,
                "gt_xyxy": list(m.gt.xyxy()),
                "pred_xyxy": list(m.pred.xyxy()),
            }
            for m in all_matches
        ],
    }


_COLOR_GT = (50, 220, 50)  # BGR зелёный — разметка
_COLOR_PRED = (255, 160, 40)  # BGR голубой — детекции
_COLOR_MATCH_GT = (50, 220, 50)
_COLOR_MATCH_PRED = (40, 180, 255)
_COLOR_FN = (60, 60, 255)  # пропуск (GT без пары)
_COLOR_FP = (0, 140, 255)  # лишняя детекция


def _clip_bbox_to_image(b: BBox, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, int(round(b.x1))))
    y1 = max(0, min(h - 1, int(round(b.y1))))
    x2 = max(x1 + 1, min(w, int(round(b.x2))))
    y2 = max(y1 + 1, min(h, int(round(b.y2))))
    return x1, y1, x2, y2


def _draw_box(
    img: np.ndarray,
    b: BBox,
    color: Tuple[int, int, int],
    *,
    label: str = "",
    thickness: int = 2,
) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _clip_bbox_to_image(b, w, h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if not label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(0.9, w / 3840 * 0.75))
    thick = max(1, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    ty = max(th + 4, y1 - 4)
    cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty + 2), color, -1)
    cv2.putText(img, label, (x1 + 3, ty - 3), font, scale, (20, 20, 20), thick, cv2.LINE_AA)


def _legend_bar(img: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(0.8, img.shape[1] / 3840 * 0.7))
    thick = 2
    line_h = int(26 * scale / 0.6)
    bar_h = line_h * len(lines) + 14
    bar = np.full((bar_h, img.shape[1], 3), 32, dtype=np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(bar, text, (10, 20 + i * line_h), font, scale, (240, 240, 240), thick, cv2.LINE_AA)
    return np.vstack([bar, img])


def _read_video_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, raw = cap.read()
    cap.release()
    if not ok or raw is None:
        return None
    return ensure_bgr(raw)


def _bbox_from_dict(d: dict) -> BBox:
    return BBox(float(d["x_min"]), float(d["y_min"]), float(d["x_max"]), float(d["y_max"]))


def save_eval_visualizations(
    video_path: Path,
    report: dict,
    viz_dir: Path,
) -> List[Path]:
    """Сохраняет кадры оригинального видео с GT, предсказаниями и сводкой."""
    viz_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for row in report["per_frame"]:
        frame = int(row["frame"])
        frame_bgr = _read_video_frame(video_path, frame)
        if frame_bgr is None:
            print(f"WARN: не удалось прочитать кадр {frame} для визуализации", file=sys.stderr)
            continue

        matched_gt = {int(m["gt_index"]) for m in row["matched"]}
        matched_pred = {int(m["pred_index"]) for m in row["matched"]}

        img_gt = frame_bgr.copy()
        for i, g in enumerate(row["gt_boxes"]):
            b = _bbox_from_dict(g)
            if i in matched_gt:
                iou = next(
                    (m["iou"] for m in row["matched"] if m["gt_index"] == i),
                    None,
                )
                lbl = f"GT{i}" + (f" IoU={iou:.2f}" if iou is not None else "")
                _draw_box(img_gt, b, _COLOR_MATCH_GT, label=lbl, thickness=2)
            else:
                _draw_box(img_gt, b, _COLOR_FN, label=f"GT{i} FN", thickness=3)

        img_pred = frame_bgr.copy()
        for i, p in enumerate(row["pred_boxes"]):
            b = _bbox_from_dict(p)
            if i in matched_pred:
                _draw_box(img_pred, b, _COLOR_MATCH_PRED, label=f"P{i}", thickness=2)
            else:
                _draw_box(img_pred, b, _COLOR_FP, label=f"P{i} FP", thickness=3)

        img_both = frame_bgr.copy()
        for i, g in enumerate(row["gt_boxes"]):
            b = _bbox_from_dict(g)
            col = _COLOR_MATCH_GT if i in matched_gt else _COLOR_FN
            _draw_box(img_both, b, col, label=f"GT{i}", thickness=2)
        for i, p in enumerate(row["pred_boxes"]):
            b = _bbox_from_dict(p)
            col = _COLOR_MATCH_PRED if i in matched_pred else _COLOR_FP
            _draw_box(img_both, b, col, label=f"P{i}", thickness=2)
        for m in row["matched"]:
            gi, pi = int(m["gt_index"]), int(m["pred_index"])
            g = _bbox_from_dict(m["gt"])
            p = _bbox_from_dict(m["pred"])
            gc = ((g.x1 + g.x2) / 2, (g.y1 + g.y2) / 2)
            pc = ((p.x1 + p.x2) / 2, (p.y1 + p.y2) / 2)
            cv2.line(
                img_both,
                (int(gc[0]), int(gc[1])),
                (int(pc[0]), int(pc[1])),
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        stem = f"frame_{frame:06d}"
        legend = [
            "GT green | PRED cyan | FN red | FP orange | match: yellow line",
            f"frame={frame}  TP={row['tp']} FP={row['fp']} FN={row['fn']}",
        ]
        paths = {
            "gt": viz_dir / f"{stem}_gt.jpg",
            "pred": viz_dir / f"{stem}_pred.jpg",
            "compare": viz_dir / f"{stem}_compare.jpg",
        }
        cv2.imwrite(str(paths["gt"]), _legend_bar(img_gt, ["GT (razmetka)"] + [legend[1]]))
        cv2.imwrite(str(paths["pred"]), _legend_bar(img_pred, ["PRED (detekcija)"] + [legend[1]]))
        cv2.imwrite(str(paths["compare"]), _legend_bar(img_both, legend))
        saved.extend(paths.values())

    return saved


def _video_meta(video_path: Path) -> Tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return w, h, fps, nframes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Оценка bbox по CSV-разметке (координаты оригинала).")
    parser.add_argument("--gt", type=Path, default=CONFIG["GT"], help="CSV с разметкой")
    parser.add_argument("--video", type=Path, default=CONFIG["VIDEO"], help="Исходное видео")
    parser.add_argument(
        "--timeline",
        type=Path,
        default=CONFIG["TIMELINE"],
        help="JSON с bbox по кадрам (entries); базовый пайплайн его не создаёт — см. --detect",
    )
    parser.add_argument("--detect", action="store_true", help="Запустить YOLO на кадрах из GT (вместо timeline)")
    parser.add_argument(
        "--model",
        choices=("high", "low"),
        default="high",
        help="Веса для --detect: high/best.pt (X) или low/best.pt (nano)",
    )
    parser.add_argument("--weights", type=Path, default=None, help="Явный путь к .pt (перекрывает --model)")
    parser.add_argument("--iou", type=float, default=0.5, help="Порог IoU для TP")
    parser.add_argument(
        "--timestamp-unit",
        type=str,
        default="ms",
        choices=("ms", "frame"),
        help="Единица frame_timestamp в CSV: ms (по умолчанию) или frame",
    )
    parser.add_argument("--rotate", type=str, default="ccw90", help="Поворот как в пайплайне")
    parser.add_argument("--flip", type=str, default="none", help="Отражение как в пайплайне")
    parser.add_argument("--scale-x", type=float, default=1.0, help="scale_x как в пайплайне")
    parser.add_argument("--out", type=Path, default=CONFIG["REPORT"], help="JSON-отчёт")
    parser.add_argument("--viz-dir", type=Path, default=CONFIG["VIZ_DIR"], help="Папка для JPG с bbox")
    parser.add_argument("--no-viz", action="store_true", help="Не сохранять изображения")
    args = parser.parse_args(argv)

    args.gt = _resolve_path(args.gt)
    args.video = _resolve_path(args.video)
    args.timeline = _resolve_path(args.timeline)
    args.out = _resolve_path(args.out) if args.out is not None else None
    args.viz_dir = _resolve_path(args.viz_dir) if args.viz_dir is not None else None
    if args.weights is not None:
        args.weights = _resolve_path(args.weights)
    else:
        args.weights = _resolve_path(CONFIG["WEIGHTS_LOW" if args.model == "low" else "WEIGHTS_HIGH"])

    orig_w, orig_h, fps, nframes = _video_meta(args.video)
    orig_wh = (orig_w, orig_h)

    gt_by_frame = load_gt_csv(args.gt, unit=args.timestamp_unit, fps=fps)
    if not gt_by_frame:
        print("GT пустой", file=sys.stderr)
        return 1

    settings = PipelineSettings(
        frame_rotate=args.rotate,
        frame_flip=args.flip,
        frame_scale_x=args.scale_x,
    )

    print(f"Оригинал видео: {orig_w}×{orig_h} px, fps={fps:.3f}, кадров={nframes}")
    print(f"timestamp-unit={args.timestamp_unit}")
    ts_unique = sorted({r.timestamp_raw for rows in gt_by_frame.values() for r in rows})
    for ts in ts_unique:
        fi = timestamp_to_frame_index(ts, unit=args.timestamp_unit, fps=fps)
        print(f"  timestamp {ts} → frame_index {fi}")
    print(f"Кадры в разметке (после преобразования): {sorted(gt_by_frame)}")
    print(f"Всего bbox в GT: {sum(len(v) for v in gt_by_frame.values())}")
    print(f"Геометрия пайплайна: rotate={args.rotate}, flip={args.flip}, scale_x={args.scale_x}")

    if args.detect:
        if args.weights is None or not args.weights.is_file():
            print(f"Не найдены веса: {args.weights}", file=sys.stderr)
            return 1
        pred_by_frame = detect_on_frames(
            args.video,
            list(gt_by_frame.keys()),
            args.weights,
            settings,
        )
        print(f"Инференс на кадрах {sorted(pred_by_frame)}")
    else:
        if not args.timeline.is_file():
            print(f"Timeline не найден: {args.timeline}", file=sys.stderr)
            print("Сначала запустите pipeline или укажите --detect", file=sys.stderr)
            return 1
        pred_by_frame = load_timeline_predictions(
            args.timeline,
            orig_wh,
            rotate=args.rotate,
            flip=args.flip,
            scale_x=args.scale_x,
        )
        print(f"Предсказания из timeline: {args.timeline}")

    missing = [f for f in gt_by_frame if f not in pred_by_frame or not pred_by_frame.get(f)]
    if missing:
        print(f"WARN: нет предсказаний на кадрах: {missing}", file=sys.stderr)
    oob = [f for f in gt_by_frame if f < 0 or (nframes > 0 and f >= nframes)]
    if oob:
        print(f"WARN: кадры вне видео [0, {nframes - 1}]: {oob}", file=sys.stderr)

    report = evaluate(gt_by_frame, pred_by_frame, args.iou)
    print()
    print(
        f"Итого @ IoU≥{args.iou}: "
        f"TP={report['tp']} FP={report['fp']} FN={report['fn']} | "
        f"P={report['precision']:.2%} R={report['recall']:.2%} F1={report['f1']:.4f} | "
        f"mean IoU (matched)={report['mean_iou_matched']}"
    )
    for row in report["per_frame"]:
        print(
            f"  frame {row['frame']} (original coords): GT={row['gt']} pred={row['pred']} "
            f"TP={row['tp']} FP={row['fp']} FN={row['fn']} "
            f"P={row['precision']:.2%} R={row['recall']:.2%} "
            f"mean_iou={row['mean_iou']}"
        )
        for i, g in enumerate(row["gt_boxes"]):
            print(
                f"    GT[{i}] {_fmt_xyxy(BBox(g['x_min'], g['y_min'], g['x_max'], g['y_max']))}"
                + (f"  {g['product_name'][:50]}" if g.get("product_name") else "")
            )
        for i, p in enumerate(row["pred_boxes"]):
            print(f"    PRED[{i}] {_fmt_xyxy(BBox(p['x_min'], p['y_min'], p['x_max'], p['y_max']))}")
        for m in row["matched"]:
            print(
                f"    MATCH gt[{m['gt_index']}]↔pred[{m['pred_index']}] IoU={m['iou']:.3f}  "
                f"gt={_fmt_xyxy(BBox(m['gt']['x_min'], m['gt']['y_min'], m['gt']['x_max'], m['gt']['y_max']))}"
            )
        for fn in row["false_negatives"]:
            print(f"    FN gt[{fn['gt_index']}] {_fmt_xyxy(BBox(fn['x_min'], fn['y_min'], fn['x_max'], fn['y_max']))}")
        for fp in row["false_positives"]:
            print(f"    FP pred[{fp['pred_index']}] {_fmt_xyxy(BBox(fp['x_min'], fp['y_min'], fp['x_max'], fp['y_max']))}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nОтчёт: {args.out}")

    if not args.no_viz and args.viz_dir is not None:
        viz_paths = save_eval_visualizations(args.video, report, args.viz_dir)
        stems = sorted({p.name.rsplit("_", 1)[0] for p in viz_paths})
        print(f"\nВизуализация: {args.viz_dir} ({len(stems)} кадров, {len(viz_paths)} файлов)")
        for stem in stems:
            print(f"  {stem}_gt.jpg | {stem}_pred.jpg | {stem}_compare.jpg")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
