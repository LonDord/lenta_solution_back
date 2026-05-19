# Lenta — детекция ценников на полочном видео

Монорепозиторий для обработки вертикальных роликов с полками: обучение YOLO, инференс с трекингом, отбор лучших кадров для OCR и оценка качества. Пакет `lenta_cv` — основной код; данные и артефакты лежат рядом с ним в корне проекта.

## Структура репозитория

```
.
  VIDEO/                    # исходные mp4 и CSV-разметка (если есть)
  dataset/pricetag_yolo/    # датасет для YOLO (images/, labels/, data.yaml)
  output/                   # результаты прогонов (пример: output/TEST_select/)
  low/ high/                # веса моделей (best.pt)
  requirements.txt          # зависимости
  pyproject.toml
  lenta_cv/                 # Python-пакет
    example.py
    train.py / eval.py
    visualize_manifest.py   # визуализация bbox из JSON поверх видео
    config/                 # пути и PipelineSettings
    core/                   # пайплайн, детекция, геометрия, трекинг
    ocr/                    # fallback: детекция и кластеризация текста
```

Установка: из корня выполните `pip install -r requirements.txt` и `pip install -e .` (Python 3.10+). Для GPU-сборки PyTorch см. `requirements-cuda.txt`.

## Алгоритм пайплайна (`process_video`)

1. **Чтение кадра** из исходного видео в RGB/BGR.
2. **Геометрия до детекции** (`apply_frame_geometry`): опционально undistort по K/D, масштаб по ширине, поворот (по умолчанию `ccw90`, чтобы лист полки совпал с ожиданиями модели), отражение.
3. **Детекция и трекинг**: Ultralytics YOLO + ByteTrack (или иной трекер из конфига). Классы можно ограничить через `coco_class_ids`.
4. **Пустой кадр YOLO** → **fallback OCR**: RapidOCR (детекция текста), кластеризация строк в «псевдо-боксы», отдельный IoU-трекер для устойчивых id.
5. **Эвристики** по площади/пропорциям бокса (опционально `use_geom_filter`).
6. **По каждому track_id** накапливаются наблюдения; для сравнения кадров используется **резкость** ROI (лапласиан по вырезанному без паддинга участку). Кроп для OCR (`prepare_for_ocr`): прямоугольник по **bbox YOLO** (с паддингом), затем при `ocr_prep=True` — **CLAHE** в LAB; **перспективный dewarp по умолчанию выключен** (`roi_dewarp_perspective=False`) как более стабильная схема. Включить экспериментальный quad-dewarp можно, выставив `roi_dewarp_perspective=True`.
7. **Экспорт**: BMP-кропы, `tracks_best_summary.csv`, `tracks_best_manifest.json` (bbox в координатах **исходного** видео, `frame_timestamp` в миллисекундах, строки координат как в GT — с десятичной запятой).

CLI-обёртки: `python -m lenta_cv.train`, `python -m lenta_cv.eval`.

## Визуализация манифеста

PNG с нарисованными bbox на тех же таймкодах, что в `tracks_best_manifest.json` (по умолчанию все боксы одного таймкода — на одном кадре):

```bash
python -m lenta_cv.visualize_manifest --manifest output/TEST_select/tracks_best_manifest.json --video-dir VIDEO --out-dir output/TEST_select/manifest_vis
```

Флаг `--per-entry` сохраняет отдельный кадр на каждую строку JSON. `--video-ext .mp4` помогает, если в манифесте указано имя без расширения.

## Быстрый старт пайплайна

1. Положите видео в `VIDEO/` (например `43_15.mp4`).
2. Веса в `low/best.pt` (и при необходимости `high/best.pt`).
3. `python lenta_cv/example.py` или вызов `process_video(...)` из кода (см. `example.py`).

## Обучение и оценка

Датасет: `dataset/pricetag_yolo/`.

```bash
python -m lenta_cv.train --check-only
python -m lenta_cv.train --profile low    # yolo11n → low/best.pt
python -m lenta_cv.train --profile high   # дообучение → high/best.pt
```

Оценка относительно CSV-разметки в координатах **исходного** видео:

```bash
python -m lenta_cv.eval --detect
```

`frame_timestamp` в CSV по умолчанию в миллисекундах (`--timestamp-unit ms`).

