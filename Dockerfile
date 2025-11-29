FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        curl \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Optional: if models/yolov8n.onnx is present in the build context, it will be copied via the line below.
# The app is resilient to missing YOLO model and will fall back to GPT-vision if the model is unavailable.
COPY . .

# Download YOLOv8n ONNX model during build so it is always present in the container.
# Source: https://github.com/shoz-f/onnx_interp/releases/download/models/yolov8n.onnx
# Also create yolo_general.onnx as an alias, which is what Ensemble expects.
RUN mkdir -p /app/models && \
    curl -L "https://github.com/shoz-f/onnx_interp/releases/download/models/yolov8n.onnx" \
         -o /app/models/yolov8n.onnx && \
    cp /app/models/yolov8n.onnx /app/models/yolo_general.onnx

# Railway сам проставляет PORT, но можно задать дефолт локально
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --log-level info --workers ${UVICORN_WORKERS:-2}"]
