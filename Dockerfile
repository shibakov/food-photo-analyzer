FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Optional: if models/yolov8n.onnx is present in the build context, it will be copied via the line below.
# The app is resilient to missing YOLO model and will fall back to GPT-vision if the model is unavailable.
COPY . .

# Models are stored in the repository under /app/models (see models/ directory).
# No model download is performed at build time; the image relies on bundled models.

# Railway сам проставляет PORT, но можно задать дефолт локально
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --log-level info --workers ${UVICORN_WORKERS:-2}"]
