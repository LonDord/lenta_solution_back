"""Прокси-детекция текста без класса ценника: ONNX (RapidOCR) или опционально PaddleOCR."""

from __future__ import annotations

import re
from typing import Any, List, Literal, Sequence, Tuple

import cv2
import numpy as np

BackendName = Literal["rapid", "paddle"]
AnchorModeName = Literal["all_words", "digits"]

_engines: dict[str, Any] = {}

PROXY_CLASS_ID = -1


def ocr_line_has_digit(text: str) -> bool:
    """Строка OCR попадает в якорь, если в тексте есть хотя бы один символ-цифра (семантику цены не учитываем)."""
    if not text or not str(text).strip():
        return False
    return bool(re.search(r"\d", str(text)))

def _polygons_to_xyxy(polys: List[np.ndarray]) -> np.ndarray:
    if not polys:
        return np.zeros((0, 4), dtype=np.float64)
    out: List[List[float]] = []
    for p in polys:
        p = np.asarray(p, dtype=np.float64).reshape(-1, 2)
        x1, y1 = float(p[:, 0].min()), float(p[:, 1].min())
        x2, y2 = float(p[:, 0].max()), float(p[:, 1].max())
        out.append([x1, y1, x2, y2])
    return np.asarray(out, dtype=np.float64)


def _resize_long_side(img_bgr: np.ndarray, max_long_side: int) -> Tuple[np.ndarray, float]:
    h0, w0 = img_bgr.shape[:2]
    if max_long_side <= 0:
        return img_bgr, 1.0
    m = max(h0, w0)
    scale = 1.0 if m <= max_long_side else float(max_long_side) / float(m)
    if scale >= 1.0:
        return img_bgr, 1.0
    small = cv2.resize(
        img_bgr,
        (int(round(w0 * scale)), int(round(h0 * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return small, scale


def get_text_det_engine(
    backend: str = "rapid",
    *,
    paddle_lang: str = "ru",
    paddle_use_gpu: bool = False,
) -> Tuple[Any, BackendName]:
    """
    По умолчанию ``rapid``: ``pip install rapidocr-onnxruntime onnxruntime``.
    Paddle на Python 3.13 под Windows часто тянет сборки с Rust и падает.
    """
    b = (backend or "rapid").lower().strip()
    if b not in ("rapid", "paddle"):
        raise ValueError("text backend: rapid | paddle")

    if b == "rapid" and "rapid" not in _engines:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise ImportError(
                "Режим rapid: pip install rapidocr-onnxruntime onnxruntime"
            ) from e
        _engines["rapid"] = RapidOCR()

    if b == "paddle" and "paddle" not in _engines:
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError(
                "Режим paddle: pip install paddlepaddle paddleocr "
                "(на Py3.13 Win может не ставиться без Rust — используйте TEXT_DET_BACKEND=rapid)"
            ) from e
        _engines["paddle"] = PaddleOCR(
            use_angle_cls=True,
            lang=paddle_lang,
            show_log=False,
            use_gpu=paddle_use_gpu,
        )

    return _engines[b], b  # type: ignore[arg-type]


def _parse_paddle_polys(result: Any) -> List[np.ndarray]:
    polys: List[np.ndarray] = []
    if result is None:
        return polys
    lines = result[0] if isinstance(result, (list, tuple)) and len(result) > 0 else result
    if lines is None or not isinstance(lines, (list, tuple)):
        return polys
    for line in lines:
        if line is None:
            continue
        if isinstance(line, (list, tuple)) and len(line) >= 1:
            pts = line[0]
            arr = np.asarray(pts, dtype=np.float64)
            if arr.size >= 8:
                polys.append(arr)
    return polys


def _run_paddle_det(engine: Any, rgb: np.ndarray) -> List[np.ndarray]:
    try:
        result = engine.ocr(rgb, det=True, rec=False, cls=False)
    except TypeError:
        try:
            result = engine.ocr(rgb, cls=False)
        except TypeError:
            result = engine.ocr(rgb)
    return _parse_paddle_polys(result)


def _run_rapid_det(engine: Any, img_bgr: np.ndarray) -> List[np.ndarray]:
    out = engine(img_bgr)
    if isinstance(out, tuple):
        blocks = out[0]
    else:
        blocks = out
    polys: List[np.ndarray] = []
    if blocks is None:
        return polys
    if isinstance(blocks, np.ndarray):
        if blocks.size == 0:
            return polys
        if blocks.ndim == 3 and blocks.shape[-1] == 2:
            for i in range(blocks.shape[0]):
                polys.append(blocks[i].astype(np.float64))
            return polys
    if not isinstance(blocks, (list, tuple)):
        return polys
    for item in blocks:
        if item is None:
            continue
        box = None
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            box = item[0]
        elif isinstance(item, np.ndarray):
            box = item
        if box is None:
            continue
        arr = np.asarray(box, dtype=np.float64).reshape(-1, 2)
        if arr.size >= 8:
            polys.append(arr)
    return polys


def detect_word_boxes(
    img_bgr: np.ndarray,
    *,
    engine: Any,
    backend: BackendName | str,
    max_long_side: int = 1280,
) -> Tuple[np.ndarray, float]:
    """
    :return: (xyxy в координатах исходного img_bgr, scale относительно уменьшенного ввода)
    """
    small, scale = _resize_long_side(img_bgr, max_long_side)
    b = (backend or "rapid").lower().strip()

    if b == "rapid":
        polys = _run_rapid_det(engine, small)
    elif b == "paddle":
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        polys = _run_paddle_det(engine, rgb)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    xyxy_small = _polygons_to_xyxy(polys)
    if xyxy_small.size == 0:
        return xyxy_small, scale
    if scale < 1.0:
        return xyxy_small / scale, scale
    return xyxy_small, scale


def _parse_text_cell(cell: Any) -> Tuple[str, float]:
    if cell is None:
        return "", 0.0
    if isinstance(cell, str):
        return cell, 1.0
    if isinstance(cell, (list, tuple)):
        if len(cell) >= 2 and isinstance(cell[0], str):
            try:
                return str(cell[0]), float(cell[1])
            except (TypeError, ValueError):
                return str(cell[0]), 0.0
        if len(cell) == 1:
            return _parse_text_cell(cell[0])
    return str(cell), 0.0


def _iter_rapid_ocr_lines(blocks: Any) -> List[Tuple[np.ndarray, str, float]]:
    """Распознавание RapidOCR: элементы вида [box, text, score] или [box, (text, score)]."""
    rows: List[Tuple[np.ndarray, str, float]] = []
    if blocks is None:
        return rows
    if isinstance(blocks, np.ndarray):
        return rows
    if not isinstance(blocks, (list, tuple)):
        return rows
    for item in blocks:
        if item is None:
            continue
        if not isinstance(item, (list, tuple)) or len(item) < 1:
            continue
        box = item[0]
        arr = np.asarray(box, dtype=np.float64).reshape(-1, 2)
        if arr.size < 8:
            continue
        text, conf = "", 0.0
        if len(item) >= 3:
            text = str(item[1]) if item[1] is not None else ""
            try:
                conf = float(item[2])
            except (TypeError, ValueError):
                conf = 0.0
        elif len(item) >= 2:
            text, conf = _parse_text_cell(item[1])
        rows.append((arr, text, conf))
    return rows


def _paddle_ocr_full_lines(engine: Any, rgb: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
    out: List[Tuple[np.ndarray, str, float]] = []
    try:
        result = engine.ocr(rgb, cls=True)
    except TypeError:
        try:
            result = engine.ocr(rgb, cls=False)
        except TypeError:
            result = engine.ocr(rgb)
    lines = result[0] if isinstance(result, (list, tuple)) and len(result) > 0 else result
    if lines is None or not isinstance(lines, (list, tuple)):
        return out
    for line in lines:
        if line is None or not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        pts = line[0]
        arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if arr.size < 8:
            continue
        text, conf = _parse_text_cell(line[1])
        out.append((arr, text, conf))
    return out


def _ocr_lines_small(
    img_small_bgr: np.ndarray,
    engine: Any,
    backend: BackendName | str,
) -> List[Tuple[np.ndarray, str, float]]:
    b = (backend or "rapid").lower().strip()
    if b == "rapid":
        out = engine(img_small_bgr)
        if isinstance(out, tuple):
            blocks = out[0]
        else:
            blocks = out
        return _iter_rapid_ocr_lines(blocks)
    if b == "paddle":
        rgb = cv2.cvtColor(img_small_bgr, cv2.COLOR_BGR2RGB)
        return _paddle_ocr_full_lines(engine, rgb)
    raise ValueError(f"unknown backend {backend!r}")


def _digit_anchor_polys_from_lines(
    lines: Sequence[Tuple[np.ndarray, str, float]],
    *,
    rec_min_conf: float,
) -> List[np.ndarray]:
    polys: List[np.ndarray] = []
    for poly, text, conf in lines:
        if conf < rec_min_conf:
            continue
        if not ocr_line_has_digit(text):
            continue
        polys.append(poly)
    return polys


def detect_word_boxes_digit_anchored(
    img_bgr: np.ndarray,
    *,
    engine: Any,
    backend: BackendName | str,
    max_long_side: int = 1280,
    rec_min_conf: float = 0.0,
    fallback_all_if_empty: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    det+rec → только полигоны строк, где в распознанном тексте есть хотя бы одна цифра → xyxy в координатах исходного кадра.
    """
    small, scale = _resize_long_side(img_bgr, max_long_side)
    lines = _ocr_lines_small(small, engine, backend)
    polys = _digit_anchor_polys_from_lines(lines, rec_min_conf=rec_min_conf)
    xyxy_small = _polygons_to_xyxy(polys)
    if xyxy_small.size == 0 and fallback_all_if_empty:
        return detect_word_boxes(
            img_bgr,
            engine=engine,
            backend=backend,
            max_long_side=max_long_side,
        )
    if scale < 1.0 and xyxy_small.size > 0:
        return xyxy_small / scale, scale
    return xyxy_small, scale


def get_proxy_word_boxes(
    img_bgr: np.ndarray,
    *,
    engine: Any,
    backend: BackendName | str,
    max_long_side: int = 1280,
    anchor_mode: str | AnchorModeName = "all_words",
    rec_min_conf: float = 0.35,
    fallback_all_if_empty: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Прокси-слова для кластеризации ценника.

    * ``all_words`` — только детектор (как раньше).
    * ``digits`` — полный OCR, bbox только вокруг строк, где есть символы-цифры; при пустом — fallback на все слова.
    """
    am = (anchor_mode or "all_words").strip().lower()
    if am in ("all_words", "all", "words", "det"):
        return detect_word_boxes(
            img_bgr,
            engine=engine,
            backend=backend,
            max_long_side=max_long_side,
        )
    if am in ("digits", "digit", "price", "numbers"):
        return detect_word_boxes_digit_anchored(
            img_bgr,
            engine=engine,
            backend=backend,
            max_long_side=max_long_side,
            rec_min_conf=rec_min_conf,
            fallback_all_if_empty=fallback_all_if_empty,
        )
    raise ValueError(f"anchor_mode: all_words | digits (получено {anchor_mode!r})")
