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

# Download all ensemble ONNX models during build so they are always present in the container.
# 1) YOLO-general (COCO, YOLOv8n)
#    Source (example): https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx
# 2) YOLO-food (YOLOv8-Food100)
#    Source: https://huggingface.co/dishfood/yolov8-food100/resolve/main/yolov8n_food100.onnx
# 3) Segmentor (YOLOv8n-Seg)
#    Source: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.onnx
# 4) Food classifier (MobileNetV3 FoodNet)
#    Source: https://huggingface.co/spaces/rice-project/FoodNet/resolve/main/food_classifier_mobilenet_v3_small.onnx
# NOTE: Ensemble expects the following filenames inside /app/models:
#   - yolo_general.onnx
#   - yolo_food.onnx
#   - segmentor.onnx
#   - classifier.onnx
RUN mkdir -p /app/models && \
    curl -L "https://github.com/shoz-f/onnx_interp/releases/download/models/yolov8n.onnx" \
         -o /app/models/yolov8n.onnx && \
    cp /app/models/yolov8n.onnx /app/models/yolo_general.onnx && \
    curl -L "https://huggingface.co/dishfood/yolov8-food100/resolve/main/yolov8n_food100.onnx" \
         -o /app/models/yolo_food.onnx && \
    curl -L "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.onnx" \
         -o /app/models/segmentor.onnx && \
    curl -L "https://huggingface.co/spaces/rice-project/FoodNet/resolve/main/food_classifier_mobilenet_v3_small.onnx" \
         -o /app/models/classifier.onnx

# Railway сам проставляет PORT, но можно задать дефолт локально
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --log-level info --workers ${UVICORN_WORKERS:-2}"]
