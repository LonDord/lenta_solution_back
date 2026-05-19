"""Детекция ценников: YOLO + fallback на кластеризацию OCR."""

from lenta_cv.core.detect import detect_pricetag_boxes
from lenta_cv.core.pipeline import process_video

__version__ = "0.1.0"
__all__ = [
    "detect_pricetag_boxes",
    "process_video",
    "__version__",
]
