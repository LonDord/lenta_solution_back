"""
Пример запуска пайплайна детекции ценников.

Из корня репозитория (системный Python, после ``pip install -e .``)::

    python lenta_cv/example.py
"""

from __future__ import annotations

from pathlib import Path

from lenta_cv.config.paths import DEFAULT_VIDEO, MODEL_DEFAULT, OUTPUT_DIR
from lenta_cv.config.settings import PipelineSettings
from lenta_cv.core.pipeline import process_video


def run_example(
    video_path: Path | None = None,
    output_dir: Path | None = None,
    weights: Path | None = None,
) -> Path:
    video = video_path or DEFAULT_VIDEO
    out_dir = output_dir or OUTPUT_DIR
    w = weights or MODEL_DEFAULT
    settings = PipelineSettings(
        roi_dewarp_perspective=False,
    )
    return process_video(video, out_dir, weights=str(w), settings=settings)


if __name__ == "__main__":
    csv_path = run_example()
    print(f"Готово. Сводка по трекам: {csv_path}")
