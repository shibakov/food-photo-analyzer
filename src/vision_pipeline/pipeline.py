import logging

from .preprocess import preprocess_image
from .detector import FoodDetector
from .refiner import GPTRefiner
from .food_schema import FOOD_NUTRITION

logger = logging.getLogger(__name__)

# Singleton detector instance so that YOLO ONNX model is loaded once per process.
DETECTOR_SINGLETON = None


def _get_detector_singleton():
    """
    Lazily initialize YOLO ONNX detector.

    If model is missing or fails to load, log the error and mark detector
    as unavailable so that the pipeline can immediately fall back to GPT-vision
    instead of crashing the whole app.
    """
    global DETECTOR_SINGLETON
    if DETECTOR_SINGLETON is None:
        try:
            DETECTOR_SINGLETON = FoodDetector("models/yolov8n.onnx")
        except Exception as e:
            logger.error(
                "Failed to initialize FoodDetector (models/yolov8n.onnx): %s",
                e,
            )
            # Use explicit False sentinel to distinguish "never tried" vs "unavailable"
            DETECTOR_SINGLETON = False
    return DETECTOR_SINGLETON


class VisionPipeline:
    """
    High-level vision pipeline:

    - lightweight RGB preprocessing (resize + normalization)
    - local YOLOv8n ONNX detection
    - GPT-4o-mini refinement using FOOD_NUTRITION schema
    - fallback to GPT-vision if detector finds nothing
    """

    def __init__(self):
        # Reuse global singleton to avoid reloading ONNX model every request
        self.detector = _get_detector_singleton()
        self.refiner = GPTRefiner()

    def analyze(self, image_path: str):
        """
        Full analysis entrypoint.

        Args:
            image_path: path to image on disk.

        Returns:
            Dict with keys:
                - products: list[...] (each with product_name, quantity_g, macros)
                - totals: {...}
            or fallback JSON from src.gpt_vision.analyze_food().
        """
        logger.info("VisionPipeline: starting analysis for %s", image_path)

        # 1) Lightweight detector-focused preprocessing
        img = preprocess_image(image_path)

        # 2) Local YOLO detection (with graceful degradation)
        detector = self.detector or _get_detector_singleton()
        if not detector:
            logger.warning(
                "VisionPipeline: detector unavailable, falling back to GPT vision for %s",
                image_path,
            )
            from src.gpt_vision import analyze_food as fallback

            return fallback(image_path)

        try:
            detections = detector.detect(img)
        except Exception as e:
            logger.error(
                "VisionPipeline: detector error for %s: %s; falling back to GPT vision",
                image_path,
                e,
            )
            from src.gpt_vision import analyze_food as fallback

            return fallback(image_path)

        logger.info("VisionPipeline: detector returned %d objects", len(detections))

        # 3) Fallback: if detector didn't find anything, call GPT-vision directly
        if len(detections) == 0:
            logger.warning(
                "VisionPipeline: no detections, falling back to GPT vision for %s",
                image_path,
            )
            from src.gpt_vision import analyze_food as fallback

            return fallback(image_path)

        # 4) GPT-mini refinement using nutrition schema
        return self.refiner.refine(detections, FOOD_NUTRITION)
