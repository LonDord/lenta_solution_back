"""
Рисует bbox из ``tracks_best_manifest.json`` на кадрах исходного видео.

Координаты в манифесте — в системе исходного ролика (как в GT CSV), frame_timestamp — мс от начала.

Запуск (из корня репозитория): python -m lenta_cv.visualize_manifest
С аргументами: --manifest, --video-dir, --out-dir, опционально --per-entry, --video-ext .mp4
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict

import cv2


def _parse_coord(s: str | float | int) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    return float(str(s).strip().replace(",", "."))


def _bbox_from_row(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        _parse_coord(row["x_min"]),
        _parse_coord(row["y_min"]),
        _parse_coord(row["x_max"]),
        _parse_coord(row["y_max"]),
    )


def _timestamp_to_frame_idx(ts_ms: int, fps: float) -> int:
    if fps <= 1e-6:
        return 0
    return int(round(float(ts_ms) / 1000.0 * fps))


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("manifest должен быть JSON-массивом объектов")
    return data


def _palette() -> list[tuple[int, int, int]]:
    return [
        (0, 165, 255),
        (0, 255, 0),
        (255, 128, 0),
        (203, 192, 255),
        (180, 105, 255),
        (147, 20, 255),
        (255, 0, 255),
        (255, 191, 0),
    ]


def _draw_boxes(
    frame_bgr: Any,
    boxes: list[tuple[float, float, float, float]],
    *,
    thickness: int = 2,
) -> Any:
    colors = _palette()
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        ix1 = int(max(0, min(w - 1, round(x1))))
        iy1 = int(max(0, min(h - 1, round(y1))))
        ix2 = int(max(0, min(w - 1, round(x2))))
        iy2 = int(max(0, min(h - 1, round(y2))))
        if ix2 < ix1:
            ix1, ix2 = ix2, ix1
        if iy2 < iy1:
            iy1, iy2 = iy2, iy1
        color = colors[i % len(colors)]
        cv2.rectangle(out, (ix1, iy1), (ix2, iy2), color, thickness, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            str(i + 1),
            (ix1 + 2, max(iy1 + 18, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def visualize_manifest(
    manifest_path: Path,
    video_dir: Path,
    out_dir: Path,
    *,
    per_entry: bool = False,
    video_ext: str = "",
) -> int:
    manifest_path = Path(manifest_path).resolve()
    video_dir = Path(video_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_manifest(manifest_path)
    if not rows:
        return 0

    caps: dict[str, cv2.VideoCapture] = {}
    fps_map: dict[str, float] = {}

    def video_path_for(fn: str) -> Path:
        p = video_dir / fn
        if p.is_file():
            return p
        if video_ext:
            stem = Path(fn).stem
            alt = video_dir / f"{stem}{video_ext}"
            if alt.is_file():
                return alt
        raise FileNotFoundError(
            f"Не найден файл видео для «{fn}» в {video_dir} (попробуйте --video-ext)"
        )

    def get_cap(name: str) -> tuple[cv2.VideoCapture, float]:
        if name not in caps:
            vp = video_path_for(name)
            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                cap.release()
                raise FileNotFoundError(f"Не удалось открыть: {vp}")
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            caps[name] = cap
            fps_map[name] = fps
        return caps[name], fps_map[name]

    n_written = 0
    try:
        if per_entry:
            for idx, row in enumerate(rows):
                vfn = str(row["filename"])
                ts_ms = int(row["frame_timestamp"])
                cap, fps = get_cap(vfn)
                frame_idx = _timestamp_to_frame_idx(ts_ms, fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                box = [_bbox_from_row(row)]
                vis = _draw_boxes(frame, box)
                stem = Path(vfn).stem
                slug = f"{stem}_t{ts_ms}_f{frame_idx:06d}_n{idx:04d}.png"
                cv2.imwrite(str(out_dir / slug), vis)
                n_written += 1
        else:
            grouped: DefaultDict[tuple[str, int], list[tuple[float, float, float, float]]] = (
                defaultdict(list)
            )
            order: list[tuple[str, int]] = []
            seen: set[tuple[str, int]] = set()
            for row in rows:
                vfn = str(row["filename"])
                ts_ms = int(row["frame_timestamp"])
                key = (vfn, ts_ms)
                grouped[key].append(_bbox_from_row(row))
                if key not in seen:
                    seen.add(key)
                    order.append(key)

            for vfn, ts_ms in order:
                cap, fps = get_cap(vfn)
                frame_idx = _timestamp_to_frame_idx(ts_ms, fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                boxes = grouped[(vfn, ts_ms)]
                vis = _draw_boxes(frame, boxes)
                stem = Path(vfn).stem
                slug = f"{stem}_t{ts_ms}_f{frame_idx:06d}_b{len(boxes)}.png"
                cv2.imwrite(str(out_dir / slug), vis)
                n_written += 1
    finally:
        for c in caps.values():
            c.release()

    return n_written


def main() -> None:
    p = argparse.ArgumentParser(description="Кадры видео с bbox из tracks_best_manifest.json")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/TEST_select/tracks_best_manifest.json"),
        help="Путь к JSON (формат как у пайплайна)",
    )
    p.add_argument(
        "--video-dir",
        type=Path,
        default=Path("VIDEO"),
        help="Каталог с видео; имя файла берётся из поля filename в манифесте",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/TEST_select/manifest_vis"),
        help="Куда сохранять PNG",
    )
    p.add_argument(
        "--per-entry",
        action="store_true",
        help="Отдельное изображение на каждую строку манифеста (один bbox)",
    )
    p.add_argument(
        "--video-ext",
        default="",
        help="Если файл не найден, искать <stem>+ext (например .mp4)",
    )
    args = p.parse_args()
    n = visualize_manifest(
        args.manifest,
        args.video_dir,
        args.out_dir,
        per_entry=args.per_entry,
        video_ext=args.video_ext,
    )
    print(f"Сохранено изображений: {n} → {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
