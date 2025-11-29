import torch
from transformers import AutoConfig, AutoModelForImageClassification

MODEL_DIR = "models/food_classifier"
ONNX_PATH = "models/food_classifier/classifier.onnx"
LABELS_PATH = "models/food_classifier/classifier_labels.json"

def main():
    # 1) Загружаем конфиг и модель
    print("Loading config & model...")
    config = AutoConfig.from_pretrained(MODEL_DIR)
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # 2) Определяем размер входа из конфига
    # В разных моделях поле может называться по-разному,
    # пробуем несколько вариантов.
    img_size = None

    if hasattr(config, "image_size"):
        img_size = config.image_size
    elif hasattr(config, "size"):
        img_size = config.size

    # image_size/size может быть int или dict
    if isinstance(img_size, dict):
        h = img_size.get("height") or img_size.get("shortest_edge") or list(img_size.values())[0]
        w = img_size.get("width") or img_size.get("shortest_edge") or list(img_size.values())[0]
    elif isinstance(img_size, int):
        h = w = img_size
    else:
        # запасной вариант, если в конфиге ничего нет
        h = w = 224

    print(f"Using input size: {h}x{w}")

    dummy = torch.randn(1, 3, h, w)

    # 3) Экспорт в ONNX
    print(f"Exporting to {ONNX_PATH} ...")
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    print("ONNX export done.")

    # 4) Сохраняем словарь id2label для маппинга классов
    import json

    id2label = getattr(config, "id2label", None)
    if id2label is not None:
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(id2label, f, ensure_ascii=False, indent=2)
        print(f"Saved labels mapping to {LABELS_PATH}")
    else:
        print("Warning: config.id2label not found, labels json not saved.")

if __name__ == "__main__":
    main()
