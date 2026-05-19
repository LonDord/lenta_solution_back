"""Пути проекта после реорганизации (low=nano, high=X, VIDEO=ролики и разметка)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

VIDEO_DIR = REPO_ROOT / "VIDEO"
DATASET_YAML = REPO_ROOT / "dataset" / "pricetag_yolo" / "data.yaml"

MODEL_LOW = REPO_ROOT / "low" / "best.pt"
MODEL_HIGH = REPO_ROOT / "high" / "best.pt"
MODEL_DEFAULT = MODEL_LOW

DEFAULT_VIDEO = VIDEO_DIR / "43_15.mp4"
DEFAULT_GT_CSV = VIDEO_DIR / "43_15.csv"

OUTPUT_DIR = REPO_ROOT / "output" / "TEST_select"
TIMELINE_JSON = OUTPUT_DIR / "tracks_per_frame_timeline.json"
EVAL_REPORT_JSON = OUTPUT_DIR / "eval_43_15.json"
EVAL_VIZ_DIR = OUTPUT_DIR / "eval_viz"

TRAIN_LOW_PROJECT = REPO_ROOT / "low"
TRAIN_HIGH_PROJECT = REPO_ROOT / "high"
