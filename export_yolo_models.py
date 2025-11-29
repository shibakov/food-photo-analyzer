import os
from ultralytics import YOLO
import onnx
from onnx import checker

MODELS = [
    {
        "source": "models/food_recognition/yolov8n-food-detection.pt",
        "target": "models/food_recognition/yolo_food.onnx",
        "imgsz": 640,
    },
    {
        "source": "models/segmetation/yolov8n-segmentation.pt",
        "target": "models/segmetation/segmentor.onnx",
        "imgsz": 640,
    },
]

def export_model(source, target, imgsz):
    print(f"\n🚀 Exporting {source} → {target}")
    if not os.path.exists(source):
        print(f"❌ ERROR: source file not found: {source}")
        return

    model = YOLO(source)
    model.export(format="onnx", imgsz=imgsz)

    # После экспорта Ultralytics кладёт onnx рядом с .pt, имя совпадает
    # Наша задача — переименовать
    generated = source.replace(".pt", ".onnx")
    if not os.path.exists(generated):
        print(f"❌ ONNX export file not found: {generated}")
        return

    os.replace(generated, target)
    print(f"✔ Saved as {target}")

    # Проверка валидности
    print(f"🔍 Checking ONNX {target} ...")
    m = onnx.load(target)
    checker.check_model(m)
    print(f"✔ ONNX validated: {target}")


if __name__ == "__main__":
    for m in MODELS:
        export_model(m["source"], m["target"], m["imgsz"])
