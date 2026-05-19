"""
CV-сервис: YOLO-детекция ценников + декодирование QR/штрихкода.

Запуск:
    pip install fastapi "uvicorn[standard]" pyzbar pillow python-multipart
    python cv_service.py

Эндпоинт:
    POST /process?model=low|high
    multipart/form-data, поле "video" — видеофайл
    Ответ: JSON-массив детекций
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pyzbar.pyzbar import decode as pyzbar_decode

from lenta_cv import process_video
from lenta_cv.config.settings import PipelineSettings

app = FastAPI()

_REPO_ROOT = Path(__file__).parent
WEIGHTS = {
    "low": str(_REPO_ROOT / "low" / "best.pt"),
    "high": str(_REPO_ROOT / "high" / "best.pt"),
}


def _try_pyzbar(img_gray: np.ndarray) -> str | None:
    if img_gray.ndim == 3:
        img_gray = img_gray[:, :, 0]
    results = pyzbar_decode(Image.fromarray(img_gray))
    if results:
        return results[0].data.decode("utf-8", errors="ignore")
    return None


def _decode_crop(img_path: Path) -> str:
    """Каскад стратегий декодирования QR/штрихкода из кропа."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "нет"

    res = _try_pyzbar(img)
    if res:
        return res

    _, binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res = _try_pyzbar(binarized)
    if res:
        return res

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binarized, kernel, iterations=1)
    res = _try_pyzbar(eroded)
    if res:
        return res

    adaptive = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
    )
    res = _try_pyzbar(adaptive)
    if res:
        return res

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, blur_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res = _try_pyzbar(blur_otsu)
    if res:
        return res

    return "нет"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
async def process(
    video: UploadFile = File(...),
    model: str = Query("low", description="Модель: low (быстрее) или high (точнее)"),
):
    weights = WEIGHTS.get(model, WEIGHTS["low"])

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / (video.filename or "input.mp4")
        out_dir = tmp_path / "out"

        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        csv_path = process_video(
            video_path,
            out_dir,
            weights=weights,
            settings=PipelineSettings(),
        )

        import csv as csv_mod

        results = []
        crops_dir = out_dir / "crops_ocr_ready"

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv_mod.DictReader(f):
                track_id = int(row["track_id"])
                frame_id = int(row["best_frame"])
                crop_file = crops_dir / f"track_{track_id:06d}_frame_{frame_id:06d}.bmp"
                decoded = _decode_crop(crop_file) if crop_file.exists() else "нет"
                results.append(
                    {
                        "track_id": track_id,
                        "frame_id": frame_id,
                        "x1": float(row["bbox_x1"]),
                        "y1": float(row["bbox_y1"]),
                        "x2": float(row["bbox_x2"]),
                        "y2": float(row["bbox_y2"]),
                        "det_conf": float(row["det_conf"]),
                        "decoded_text": decoded,
                    }
                )

    return JSONResponse(content=results)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
