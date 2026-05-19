"""
Дообучение YOLO на датасете ценников (``dataset/pricetag_yolo/data.yaml``, ``nc: 1``).

Структура датасета (YOLO):
  images/train, images/val, labels/train, labels/val, data.yaml

Запуск из корня репозитория::

    python -m lenta_cv.train
    python -m lenta_cv.train --profile high
    python -m lenta_cv.train --check-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO

from lenta_cv.config.paths import DATASET_YAML, MODEL_LOW, REPO_ROOT, TRAIN_HIGH_PROJECT, TRAIN_LOW_PROJECT

CONFIG_LOW = dict(
    BASE_WEIGHTS="yolo11n.pt",
    DATA_YAML=DATASET_YAML,
    IMGSZ=640,
    EPOCHS=150,
    PATIENCE=30,
    BATCH=24,
    DEVICE="0",
    PROJECT=TRAIN_LOW_PROJECT,
    NAME="train",
    WORKERS=4,
    CACHE="ram",
    AMP=True,
)

CONFIG_HIGH = dict(
    BASE_WEIGHTS=MODEL_LOW,
    DATA_YAML=DATASET_YAML,
    IMGSZ=960,
    EPOCHS=60,
    PATIENCE=12,
    BATCH=6,
    DEVICE="0",
    PROJECT=TRAIN_HIGH_PROJECT,
    NAME="train",
    WORKERS=4,
    CACHE="ram",
    AMP=True,
)

CONFIG = CONFIG_LOW


def _nvidia_gpu_listed_by_smi() -> bool:
    try:
        extra: dict = {}
        if sys.platform == "win32":
            extra["creationflags"] = subprocess.CREATE_NO_WINDOW
        r = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=15,
            **extra,
        )
        return bool(r.stdout and ("GPU" in r.stdout))
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return False


def _resolve_weights_spec(raw: str | Path) -> str:
    p = Path(raw)
    if p.is_absolute() and p.is_file():
        return str(p.resolve())
    cand = (REPO_ROOT / p).resolve()
    if cand.is_file():
        return str(cand)
    return str(raw)


def _resolve_paths(c: dict) -> tuple[str, Path, Path]:
    w_str = _resolve_weights_spec(str(c["BASE_WEIGHTS"]))
    dy = Path(c["DATA_YAML"])
    if not dy.is_absolute():
        dy = (REPO_ROOT / dy).resolve()
    proj = Path(c["PROJECT"])
    if not proj.is_absolute():
        proj = (REPO_ROOT / proj).resolve()
    return w_str, dy, proj


def _weights_local_path(weights_str: str) -> Path | None:
    p = Path(weights_str)
    if p.is_file():
        return p.resolve()
    return None


def _expects_torch_cuda(device_setting: object) -> bool:
    dv = str(device_setting or "").strip().lower()
    if not dv or dv == "cpu":
        return False
    if dv.isdigit():
        return True
    if dv.startswith("cuda:"):
        return "cpu" not in dv
    return dv == "cuda"


def run_preflight(c: dict) -> int:
    import torch

    bw_str, dy, _proj = _resolve_paths(c)
    lp = _weights_local_path(bw_str)
    if not dy.is_file():
        print(f"[FAIL] нет data.yaml: {dy}")
        return 1

    root = dy.parent
    train_img = root / "images" / "train"
    val_img = root / "images" / "val"
    lbl_train = root / "labels" / "train"
    lbl_val = root / "labels" / "val"
    img_ext = (".jpg", ".jpeg", ".png")

    def n_images(d: Path) -> int:
        if not d.is_dir():
            return 0
        return sum(1 for p in d.iterdir() if p.suffix.lower() in img_ext)

    def n_labels(d: Path) -> int:
        if not d.is_dir():
            return 0
        return sum(1 for p in d.iterdir() if p.suffix.lower() == ".txt")

    nt, nv = n_images(train_img), n_images(val_img)
    lt, lv = n_labels(lbl_train), n_labels(lbl_val)
    cuda = torch.cuda.is_available()
    print("--- проверка ---")
    if lp is not None:
        print(f"weights:     {lp}  ({lp.stat().st_size // (1024 * 1024)} MB)")
    else:
        print(f"weights:     {bw_str}  (локального файла нет — загрузка при старте train)")
    print(f"data.yaml:   {dy}")
    print(f"train images / labels: {nt} / {lt}  ({train_img})")
    print(f"val images / labels:   {nv} / {lv}  ({val_img})")
    print(f"torch cuda available: {cuda}", end="")
    if cuda:
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print("  (нет CUDA в сборке PyTorch)")
        cuda_hint = REPO_ROOT / "requirements-cuda.txt"
        if _nvidia_gpu_listed_by_smi():
            print(
                f"  Обнаружен NVIDIA-драйвер, но torch собран без GPU. Переустановите PyTorch под CUDA:"
                f"\n  см. {cuda_hint}"
            )
        else:
            print("  nvidia-smi не показал GPU — проверьте драйвер и видеокарту.")
    try:
        m = YOLO(bw_str)
        print(f"YOLO загрузился: task={getattr(m, 'task', '?')}")
    except Exception as e:
        print(f"[FAIL] не удалось загрузить веса в YOLO: {e}")
        return 1

    dv = str(c.get("DEVICE", "") or "").strip().lower()
    if _expects_torch_cuda(dv) and not cuda:
        print(
            "[FAIL] DEVICE задаёт CUDA, но torch.cuda.is_available() == False. "
            'Поставьте DEVICE="cpu" или установите torch с CUDA.'
        )
        return 2
    print("--- ok, можно train ---")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Дообучение YOLO на датасете ценников (1 класс)")
    ap.add_argument(
        "--profile",
        choices=("low", "high"),
        default="low",
        help="low: yolo11n → low/best.pt; high: low/best.pt → high/best.pt",
    )
    ap.add_argument("--base-weights", default=None, help="стартовые веса, напр. yolo11n.pt или low/best.pt")
    ap.add_argument("--data-yaml", default=None, help="путь к data.yaml")
    ap.add_argument("--project", default=None, help="каталог runs")
    ap.add_argument("--name", default=None, help="имя run")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--check-only", action="store_true", help="только проверка окружения")
    ap.add_argument("--smoke", action="store_true", help="одна эпоха")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    c = dict(CONFIG_LOW if args.profile == "low" else CONFIG_HIGH)
    if args.base_weights is not None:
        c["BASE_WEIGHTS"] = args.base_weights
    if args.data_yaml is not None:
        c["DATA_YAML"] = Path(args.data_yaml)
    if args.project is not None:
        c["PROJECT"] = Path(args.project)
    if args.name is not None:
        c["NAME"] = args.name
    if args.imgsz is not None:
        c["IMGSZ"] = args.imgsz
    if args.epochs is not None:
        c["EPOCHS"] = args.epochs
    if args.batch is not None:
        c["BATCH"] = args.batch
    if args.device is not None:
        c["DEVICE"] = args.device

    pre_rc = run_preflight(c)
    if pre_rc != 0:
        return pre_rc
    if args.check_only:
        return 0

    bw_str, dy, proj = _resolve_paths(c)
    kw: dict[str, object] = dict(
        data=str(dy),
        epochs=int(c["EPOCHS"]),
        imgsz=int(c["IMGSZ"]),
        patience=int(c["PATIENCE"]),
        batch=int(c["BATCH"]),
        project=str(proj),
        name=str(c["NAME"]),
        workers=int(c["WORKERS"]),
        amp=bool(c.get("AMP", True)),
    )
    cache = c.get("CACHE", False)
    if cache not in (False, None, ""):
        kw["cache"] = cache
    if args.smoke:
        kw["epochs"] = 1
        kw["patience"] = 999
        kw["name"] = f'{c["NAME"]}-smoke'
        print("(smoke) epochs=1, name=%s" % kw["name"])
    dv = c.get("DEVICE")
    if dv not in ("", None, False):
        kw["device"] = dv

    print(f"base: {bw_str}")
    print(f"data: {dy}")
    print(f"out:  {proj / c['NAME']} / weights/best.pt")
    model = YOLO(bw_str)
    model.train(**kw)
    print("Готово. Лучшие веса — weights/best.pt в каталоге run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
