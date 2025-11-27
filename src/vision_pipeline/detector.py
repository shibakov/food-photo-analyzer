import logging
from typing import List, Dict

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class FoodDetector:
    """
    Lightweight YOLOv8n ONNX detector wrapper.

    Expects model at models/yolov8n.onnx by default.
    """

    def __init__(self, model_path: str = "models/yolov8n.onnx"):
        self.model_path = model_path
        logger.info("Initializing FoodDetector with model: %s", model_path)
        # CPU-only for maximum portability
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        # Cache input / output names for faster calls
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Run YOLOv8n ONNX model on an RGB image.

        Returns list of detections:
        {
            "label": str,          # raw class id as string for now
            "confidence": float,   # score
            "bbox": [x1, y1, x2, y2],
            "size": float,         # relative bbox area vs model input area
        }
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image passed to detector")

        h, w = image.shape[:2]

        # Preprocess to YOLO resolution (640x640), normalize to [0, 1]
        img = cv2.resize(image, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]  # (1, 3, 640, 640)

        logger.debug(
            "Running YOLOv8n ONNX inference: original_size=%sx%s, input_shape=%s",
            w,
            h,
            img.shape,
        )

        outputs = self.session.run([self.output_name], {self.input_name: img})[0]

        # Many exported YOLOv8 ONNX models return shape (N, 6): [x1, y1, x2, y2, score, cls]
        # If batch dimension is present, squeeze it.
        if outputs.ndim == 3 and outputs.shape[0] == 1:
            outputs = outputs[0]

        detections: List[Dict] = []
        model_input_area = 640 * 640

        for det in outputs:
            # Defensive unpack: skip invalid rows
            if det.shape[0] < 6:
                continue

            x1, y1, x2, y2, score, cls_id = det[:6].tolist()
            if score < 0.25:
                continue

            # Clamp coordinates to model input space
            x1 = max(0.0, min(640.0, x1))
            y1 = max(0.0, min(640.0, y1))
            x2 = max(0.0, min(640.0, x2))
            y2 = max(0.0, min(640.0, y2))

            bbox_area = max(0.0, (x2 - x1) * (y2 - y1))
            relative_size = bbox_area / model_input_area if model_input_area > 0 else 0.0

            detections.append(
                {
                    "label": str(int(cls_id)),  # временно числовой класс
                    "confidence": float(score),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "size": float(relative_size),
                }
            )

        logger.info("YOLOv8n detected %d objects", len(detections))
        return detections
