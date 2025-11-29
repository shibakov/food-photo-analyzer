import onnxruntime as ort
import numpy as np
import json
import os

MODEL_PATH = "models/food_classifier/classifier.onnx"
LABELS_PATH = "models/food_classifier/classifier_labels.json"


def test_classifier_loads():
    assert os.path.exists(MODEL_PATH), "classifier.onnx not found"
    assert os.path.exists(MODEL_PATH + ".data"), "classifier.onnx.data not found"

    # ONNX Runtime must load the model without crashes
    sess = ort.InferenceSession(MODEL_PATH)
    print("Model loaded OK:", MODEL_PATH)

    # Dummy input (BGR not required, ONNX model uses float32 RGB)
    dummy = np.random.rand(1, 3, 240, 240).astype(np.float32)

    # Find input/output names dynamically
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    result = sess.run([output_name], {input_name: dummy})
    logits = result[0]

    print("Logits shape:", logits.shape)
    assert logits.shape[1] > 0, "Model returned empty logits"

    # Load labels
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    print("Loaded labels:", len(labels), "classes")

    # Show TOP-5 predictions
    top5 = np.argsort(logits[0])[::-1][:5]
    print("Top-5 class IDs:", top5)
    print("Top-5 labels:", [labels[str(i)] for i in top5])
