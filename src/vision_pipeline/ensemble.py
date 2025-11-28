import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)

YOLO_GENERAL_PATH = os.path.join(MODEL_DIR, "yolo_general.onnx")
YOLO_FOOD_PATH = os.path.join(MODEL_DIR, "yolo_food.onnx")
SEGMENTOR_PATH = os.path.join(MODEL_DIR, "segmentor.onnx")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.onnx")

# Grams per full-plate normalized area (1.0) as requested
PLATE_GRAMS = 350.0


class Ensemble:
    """
    Core ensemble for multi-stage food analysis:

    - general YOLO (COCO) detection
    - food-specific YOLO (optional)
    - segmentation / mask area (currently bbox-based)
    - classifier for refined dish name
    - heuristic grams estimation
    """

    def __init__(
        self,
        general_model_path: str = YOLO_GENERAL_PATH,
        food_model_path: str = YOLO_FOOD_PATH,
        segmentor_model_path: str = SEGMENTOR_PATH,
        classifier_model_path: str = CLASSIFIER_PATH,
    ):
        # Import onnxruntime lazily so that missing dependency does not break import
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            logger.error("Failed to import onnxruntime in Ensemble: %s", e)
            raise

        self._ort = ort

        # General YOLO (COCO)
        if not os.path.exists(general_model_path):
            raise FileNotFoundError(
                f"YOLO general model not found at {general_model_path}"
            )
        logger.info("Initializing Ensemble general YOLO from %s", general_model_path)
        self.general_session = ort.InferenceSession(
            general_model_path, providers=["CPUExecutionProvider"]
        )
        self.general_input = self.general_session.get_inputs()[0].name
        self.general_output = self.general_session.get_outputs()[0].name

        # Food YOLO (optional: may be missing if private / 401)
        if os.path.exists(food_model_path):
            try:
                logger.info("Initializing Ensemble food YOLO from %s", food_model_path)
                self.food_session = ort.InferenceSession(
                    food_model_path, providers=["CPUExecutionProvider"]
                )
                self.food_input = self.food_session.get_inputs()[0].name
                self.food_output = self.food_session.get_outputs()[0].name
            except Exception as e:
                logger.error("Failed to initialize YOLO-food model: %s", e)
                self.food_session = None
                self.food_input = None
                self.food_output = None
        else:
            logger.warning(
                "YOLO-food model not found at %s. "
                "Food-specific classification will be skipped.",
                food_model_path,
            )
            self.food_session = None
            self.food_input = None
            self.food_output = None

        # Segmentor (currently unused in logic, kept for future upgrade)
        if os.path.exists(segmentor_model_path):
            try:
                logger.info(
                    "Segmentor model found at %s (not yet used for postprocessing)",
                    segmentor_model_path,
                )
                self.segmentor_session = ort.InferenceSession(
                    segmentor_model_path, providers=["CPUExecutionProvider"]
                )
                self.segmentor_input = self.segmentor_session.get_inputs()[0].name
                self.segmentor_output = self.segmentor_session.get_outputs()[0].name
            except Exception as e:
                logger.error("Failed to initialize segmentor model: %s", e)
                self.segmentor_session = None
                self.segmentor_input = None
                self.segmentor_output = None
        else:
            logger.warning(
                "Segmentor model not found at %s. "
                "Fallback to bbox-based mask area only.",
                segmentor_model_path,
            )
            self.segmentor_session = None
            self.segmentor_input = None
            self.segmentor_output = None

        # Classifier
        if os.path.exists(classifier_model_path):
            try:
                logger.info(
                    "Initializing Ensemble classifier from %s", classifier_model_path
                )
                self.classifier_session = ort.InferenceSession(
                    classifier_model_path, providers=["CPUExecutionProvider"]
                )
                self.classifier_input = self.classifier_session.get_inputs()[0].name
                self.classifier_output = self.classifier_session.get_outputs()[0].name
                # Try to infer input spatial size (e.g. 224x224)
                in_shape = self.classifier_session.get_inputs()[0].shape
                # Typical shapes: [1, 3, H, W] or [None, 3, H, W]
                if len(in_shape) == 4:
                    self.classifier_h = int(in_shape[2] or 224)
                    self.classifier_w = int(in_shape[3] or 224)
                else:
                    self.classifier_h = 224
                    self.classifier_w = 224
            except Exception as e:
                logger.error("Failed to initialize classifier model: %s", e)
                self.classifier_session = None
                self.classifier_input = None
                self.classifier_output = None
                self.classifier_h = 224
                self.classifier_w = 224
        else:
            logger.warning(
                "Classifier model not found at %s. "
                "Dish classification will be marked as ambiguous.",
                classifier_model_path,
            )
            self.classifier_session = None
            self.classifier_input = None
            self.classifier_output = None
            self.classifier_h = 224
            self.classifier_w = 224

    # -------------------------------------------------------------------------
    # 1) General YOLO detection (COCO)
    # -------------------------------------------------------------------------
    def detect_general(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run general YOLO (COCO) detector on RGB image.

        Returns list of detections:
        {
            "label": str,          # numeric class id as string
            "confidence": float,   # score
            "bbox": [x1, y1, x2, y2],  # in 640x640 model space
            "size": float,         # relative bbox area vs model input area (0..1)
        }
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image passed to detect_general")

        import cv2  # type: ignore

        h, w = image.shape[:2]

        # Preprocess to YOLO resolution (640x640), normalize to [0, 1]
        img_resized = cv2.resize(image, (640, 640))
        img = img_resized.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]  # (1, 3, 640, 640)

        logger.debug(
            "Running general YOLO ONNX inference: original_size=%sx%s, input_shape=%s",
            w,
            h,
            img.shape,
        )

        outputs = self.general_session.run(
            [self.general_output], {self.general_input: img}
        )[0]

        # Many exported YOLOv8 ONNX models return shape (N, 6): [x1, y1, x2, y2, score, cls]
        # If batch dimension is present, squeeze it.
        if outputs.ndim == 3 and outputs.shape[0] == 1:
            outputs = outputs[0]

        detections: List[Dict[str, Any]] = []
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
                    "label": str(int(cls_id)),
                    "confidence": float(score),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "size": float(relative_size),
                }
            )

        logger.info("General YOLO detected %d objects", len(detections))
        return detections

    # -------------------------------------------------------------------------
    # 2) YOLO-food classification (optional refinement)
    # -------------------------------------------------------------------------
    def detect_food(
        self,
        image: np.ndarray,
        general_detections: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Optionally refine each general detection with a food-specific YOLO model.

        If YOLO-food model is unavailable, returns general_detections unchanged but
        still logs stage timing in the pipeline.

        For now this function simply echoes back detections and adds a placeholder
        'food_cls_id' field when YOLO-food is missing. When the food model is
        accessible, it can be extended to run detection on each crop.
        """
        if not general_detections:
            return []

        if self.food_session is None:
            # Mark that food refinement was not performed
            for det in general_detections:
                det.setdefault("food_cls_id", None)
            return general_detections

        # Placeholder implementation: we *could* re-run YOLO on the full image
        # or per-crop, but without class-name mapping this is mainly for logging /
        # experimentation. For safety, we skip heavy logic here.
        for det in general_detections:
            det.setdefault("food_cls_id", None)

        return general_detections

    # -------------------------------------------------------------------------
    # 3) Segmentation mask (currently bbox-based)
    # -------------------------------------------------------------------------
    def segment_mask(
        self,
        image: np.ndarray,
        bbox: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Compute a binary mask for the object.

        Current implementation is bbox-based (no learned segmentation yet):
        - bbox is in 640x640 model coordinates
        - we map it back to the preprocessed image size (H, W)
        - mask has shape (H, W), dtype=uint8, values {0, 1}
        """
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("Empty image for segment_mask")

        x1_640, y1_640, x2_640, y2_640 = bbox

        # Map from 640x640 model space back to current image resolution
        scale_x = w / 640.0
        scale_y = h / 640.0

        x1 = max(0, min(w, int(round(x1_640 * scale_x))))
        y1 = max(0, min(h, int(round(y1_640 * scale_y))))
        x2 = max(0, min(w, int(round(x2_640 * scale_x))))
        y2 = max(0, min(h, int(round(y2_640 * scale_y))))

        if x2 <= x1 or y2 <= y1:
            # Degenerate bbox: return empty mask
            return np.zeros((h, w), dtype=np.uint8)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask

    # -------------------------------------------------------------------------
    # 4) Classifier for refined dish name
    # -------------------------------------------------------------------------
    def classify_crop(
        self,
        image: np.ndarray,
        bbox: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float]:
        """
        Classify cropped region with MobileNetV3/EfficientNet-lite food classifier.

        Returns:
            (label, confidence)

        If classifier model is unavailable, returns ("ambiguous", 0.0) which
        will trigger GPT fallback in the pipeline.
        """
        if self.classifier_session is None:
            return "ambiguous", 0.0

        import cv2  # type: ignore

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return "ambiguous", 0.0

        x1_640, y1_640, x2_640, y2_640 = bbox

        # Map bbox from 640x640 to current image resolution
        scale_x = w / 640.0
        scale_y = h / 640.0

        x1 = max(0, min(w, int(round(x1_640 * scale_x))))
        y1 = max(0, min(h, int(round(y1_640 * scale_y))))
        x2 = max(0, min(w, int(round(x2_640 * scale_x))))
        y2 = max(0, min(h, int(round(y2_640 * scale_y))))

        if x2 <= x1 or y2 <= y1:
            return "ambiguous", 0.0

        crop = image[y1:y2, x1:x2, :]

        # Resize to classifier input size and normalize to [0, 1]
        crop_resized = cv2.resize(crop, (self.classifier_w, self.classifier_h))
        crop_resized = crop_resized.astype(np.float32) / 255.0
        crop_resized = crop_resized.transpose(2, 0, 1)[None]  # (1, 3, H, W)

        outputs = self.classifier_session.run(
            [self.classifier_output],
            {self.classifier_input: crop_resized},
        )[0]

        # Assume outputs shape is (1, num_classes) or (num_classes,)
        logits = outputs[0] if outputs.ndim == 2 else outputs
        logits = logits.astype(np.float32)

        # Softmax to probabilities
        e_x = np.exp(logits - np.max(logits))
        probs = e_x / e_x.sum()

        cls_idx = int(np.argmax(probs))
        conf = float(probs[cls_idx])

        # We do not have the textual class names for the classifier yet,
        # so we encode the class index into the label.
        label = f"class_{cls_idx}"

        return label, conf

    # -------------------------------------------------------------------------
    # 5) Heuristic grams estimation
    # -------------------------------------------------------------------------
    def estimate_grams(
        self,
        mask: np.ndarray,
        plate_grams: float = PLATE_GRAMS,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Estimate grams based on normalized mask area.

        grams = (mask_pixels / image_pixels) * plate_grams
        """
        if mask is None or mask.size == 0:
            return 0.0

        # Ensure boolean or 0/1 mask
        mask_pixels = float(np.count_nonzero(mask))
        image_pixels = float(mask.size)
        if image_pixels <= 0.0:
            return 0.0

        area_normalized = mask_pixels / image_pixels
        grams = area_normalized * float(plate_grams)
        return grams


# Singleton pattern to avoid reloading all ONNX models for each request
_ENSEMBLE_SINGLETON: Optional[Ensemble] = None


def get_ensemble_singleton() -> Optional[Ensemble]:
    """
    Lazily initialize Ensemble singleton.

    Returns:
        Ensemble instance, or None if initialization failed.
    """
    global _ENSEMBLE_SINGLETON
    if _ENSEMBLE_SINGLETON is not None:
        return _ENSEMBLE_SINGLETON

    try:
        _ENSEMBLE_SINGLETON = Ensemble()
    except Exception as e:
        logger.error("Failed to initialize Ensemble: %s", e)
        _ENSEMBLE_SINGLETON = None

    return _ENSEMBLE_SINGLETON
