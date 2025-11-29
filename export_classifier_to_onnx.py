import torch
import json
import os
import onnx
from transformers import AutoConfig, AutoModelForImageClassification


MODEL_DIR = "models_dev/food_classifier"
ONNX_PATH = "models/food_classifier/classifier.onnx"
LABELS_PATH = "models/food_classifier/classifier_labels.json"


def main():
    print("Loading config & model...")
    config = AutoConfig.from_pretrained(MODEL_DIR)
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # --- determine input size ---
    img_size = getattr(config, "image_size", None) or getattr(config, "size", None)
    if isinstance(img_size, dict):
        h = img_size.get("height") or img_size.get("shortest_edge") or list(img_size.values())[0]
        w = img_size.get("width") or img_size.get("shortest_edge") or list(img_size.values())[0]
    elif isinstance(img_size, int):
        h = w = img_size
    else:
        h = w = 224

    print(f"Using input size: {h}x{w}")
    dummy = torch.randn(1, 3, h, w)

    # --- export raw (will create .onnx + .onnx.data) ---
    print(f"Exporting raw ONNX → {ONNX_PATH}")
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
        verbose=False,
        # IMPORTANT — DO NOT pass use_external_data_format here
    )
    print("Raw ONNX export complete.")

    # --- MERGE external weights into one file ---
    print("Merging external data into single ONNX file...")

    base_dir = os.path.dirname(ONNX_PATH)
    model_proto = onnx.load(ONNX_PATH, load_external_data=False, format=None)

    onnx.load_external_data_for_model(model_proto, base_dir)

    # save without external data
    onnx.save_model(model_proto, ONNX_PATH, save_as_external_data=False)
    print("Single-file ONNX successfully created!")

    # cleanup leftover .data file if exists
    data_path = ONNX_PATH + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
        print("Removed external weight file:", data_path)

    # save labels
    id2label = getattr(config, "id2label", None)
    if id2label:
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(id2label, f, ensure_ascii=False, indent=2)
        print("Saved:", LABELS_PATH)

    print("\n🎉 DONE — classifier.onnx is now a SINGLE FILE.")


if __name__ == "__main__":
    main()
