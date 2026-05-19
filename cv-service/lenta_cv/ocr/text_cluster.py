"""Кластеризация мелких текстовых bbox в крупные ROI «как ценник»."""

from __future__ import annotations

import numpy as np


def _y_overlap_frac(a: np.ndarray, b: np.ndarray) -> float:
    ay = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ha = max(1e-6, a[3] - a[1])
    hb = max(1e-6, b[3] - b[1])
    return float(ay / min(ha, hb))


def _h_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Горизонтальный зазор между двумя боксами (не перекрываются по x)."""
    if a[2] < b[0]:
        return float(b[0] - a[2])
    if b[2] < a[0]:
        return float(a[0] - b[2])
    return 0.0


def _segments_by_ordered_x_gap(
    idxs: list[int],
    boxes: np.ndarray,
    *,
    split_gap: float,
) -> list[list[int]]:
    """Режем список индексов слов там, где по оси X между соседними большой зазор."""
    order = sorted(idxs, key=lambda i: float(boxes[i, 0]))
    if len(order) <= 1:
        return [order]
    segs: list[list[int]] = []
    cur: list[int] = [order[0]]
    for k in range(1, len(order)):
        prev_i, next_i = order[k - 1], order[k]
        g = _h_gap(boxes[prev_i], boxes[next_i])
        if g > split_gap:
            segs.append(cur)
            cur = [next_i]
        else:
            cur.append(next_i)
    segs.append(cur)
    return segs


def cluster_text_boxes(
    xyxy: np.ndarray,
    *,
    gap_mul: float = 1.8,
    horizontal_gap_mul: float | None = None,
    min_y_overlap_frac: float = 0.35,
    pad_y_top_frac: float = 0.65,
    pad_y_bottom_frac: float = 0.25,
    pad_x_frac: float = 0.08,
    split_wide_if_width_gt_mul: float = 0.0,
    split_x_gap_mul: float = 0.0,
) -> np.ndarray:
    """
    Группирует близкие по вертикали и горизонтали слова в один прямоугольник оболочкой,
    затем расширяет bbox (по умолчанию сильнее вверх — под шапку/лого ценника).

    Слияние соседних по горизонтали карточек ограничивается условием: горизонтальный зазор
    между боксами ≤ ``horizontal_gap_mul * median_h слов``. Если ``horizontal_gap_mul`` не задан,
    используется ``gap_mul`` (историческая совместимость).

    Если ``split_wide_if_width_gt_mul > 0`` и ``split_x_gap_mul > 0``, то у компонентов, чья ширина
    оболочки слов превышает ``split_wide_if_width_gt_mul * med_h``, пробуем резать цепочку по X там,
    где зазор между соседними словами (в порядке слева направо) больше ``split_x_gap_mul * med_h`` —
    уменьшает «перекрытие» двух склеенных ценников.
    """
    if xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    boxes = xyxy.astype(np.float64).copy()
    h_all = boxes[:, 3] - boxes[:, 1]
    med_h = float(np.median(h_all[h_all > 1e-3])) if np.any(h_all > 1e-3) else 10.0
    h_mul = horizontal_gap_mul if horizontal_gap_mul is not None else gap_mul
    max_gap = float(h_mul) * max(med_h, 6.0)
    n = len(boxes)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = root(i), root(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if _y_overlap_frac(boxes[i], boxes[j]) < min_y_overlap_frac:
                continue
            if _h_gap(boxes[i], boxes[j]) <= max_gap:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)

    clusters: list[np.ndarray] = []
    split_enable = split_wide_if_width_gt_mul > 1e-6 and split_x_gap_mul > 1e-6
    split_gap_abs = split_x_gap_mul * max(med_h, 6.0)
    for ids in groups.values():
        seg_ids_batch: list[list[int]] = [ids]
        if split_enable:
            grp = boxes[ids]
            w_raw = float(grp[:, 2].max() - grp[:, 0].min())
            if w_raw >= split_wide_if_width_gt_mul * max(med_h, 6.0):
                seg_ids_batch = _segments_by_ordered_x_gap(list(ids), boxes, split_gap=split_gap_abs)

        for seg in seg_ids_batch:
            grp = boxes[seg]
            x1 = float(grp[:, 0].min())
            y1 = float(grp[:, 1].min())
            x2 = float(grp[:, 2].max())
            y2 = float(grp[:, 3].max())
            h_grp = grp[:, 3] - grp[:, 1]
            med_loc = (
                float(np.median(h_grp[h_grp > 1e-3]))
                if np.any(h_grp > 1e-3)
                else max(12.0, (y2 - y1) * 0.12)
            )
            h_env = max(1e-3, y2 - y1)
            w_env = max(1e-3, x2 - x1)
            # Базовый отступ по строке + доля высоты оболочки (ценник часто выше детектов текста)
            y1 -= med_loc * 0.35 + h_env * pad_y_top_frac
            y2 += med_loc * 0.2 + h_env * pad_y_bottom_frac
            x1 -= w_env * pad_x_frac
            x2 += w_env * pad_x_frac
            clusters.append(np.array([x1, y1, x2, y2]))

    out = np.array(clusters, dtype=np.float64) if clusters else np.zeros((0, 4))
    return out
