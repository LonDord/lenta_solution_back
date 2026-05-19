FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libzbar0 \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lenta_cv/ ./lenta_cv/
COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY cv_service.py .

# Weights are mounted as volumes: /app/low/best.pt, /app/high/best.pt
EXPOSE 8000

CMD ["python", "cv_service.py"]
