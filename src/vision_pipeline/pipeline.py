import logging
import time

from src.gpt_vision import analyze_food as fallback_analyze_food
from .preprocess import preprocess_image
from .detector import FoodDetector

logger = logging.getLogger(__name__)

# Singleton detector instance so that YOLO ONNX model is loaded once per process.
DETECTOR_SINGLETON = None

# Base grams per 1.0 relative bbox area from YOLO
BASE_WEIGHT = 350  # грамм на 1.0 относительной площади


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
            # Use internal MODEL_PATH from FoodDetector by default so path resolution is robust
            DETECTOR_SINGLETON = FoodDetector()
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
    - simple Python-only gram estimation
    - GPT-vision fallback when YOLO is unavailable or finds nothing
    """

    def __init__(self):
        # Reuse global singleton to avoid reloading ONNX model every request
        self.detector = _get_detector_singleton()

    def analyze(self, image_path: str):
        """
        YOLO-first analysis entrypoint.

        Args:
            image_path: path to image on disk.

        Returns:
            Dict with keys:
                - products: list[...] (each with product_name, quantity_g, macros)
                - totals: {...}
                - meta: {pipeline, yolo_model, fallback_used}
        """
        logger.info("VisionPipeline: starting analysis for %s", image_path)

        # Global start timer
        t0 = time.perf_counter()

        # 1) Lightweight detector-focused preprocessing
        img = preprocess_image(image_path)
        t_preprocessed = time.perf_counter()

        # 2) Local YOLO detection (with graceful degradation)
        detector = self.detector or _get_detector_singleton()
        if not detector:
            # Detector never initialized or failed to load -> direct GPT-vision fallback
            logger.warning(
                "[PIPELINE] YOLO detector unavailable \u2192 using GPT fallback"
            )
            return self._fallback_gpt(image_path, t_preprocessed)

        try:
            detections = detector.detect(img)
            t_detected = time.perf_counter()

            logger.info(
                "[PIPELINE] YOLO detected %s objects in %.3f sec: %s",
                len(detections),
                t_detected - t_preprocessed,
                detections,
            )

        except Exception as e:
            logger.error("[PIPELINE] YOLO detection FAILED, error=%s", e)
            return self._fallback_gpt(image_path, t_preprocessed)

        # 3) If YOLO found nothing -> GPT fallback + log
        if not detections:
            logger.warning(
                "[PIPELINE] YOLO returned NO detections \u2192 using GPT fallback"
            )
            return self._fallback_gpt(image_path, t_preprocessed)

        # 4) Python-only gram estimation (no GPT, KБЖУ=0)
        products = []
        for det in detections:
            grams = det.get("size", 0.0) * BASE_WEIGHT
            products.append(
                {
                    "product_name": det.get("label", "unknown"),
                    "quantity_g": round(grams, 1),
                    "kcal": 0,
                    "protein": 0,
                    "fat": 0,
                    "carbs": 0,
                }
            )

        # 5) Totals = zeros
        totals = {"kcal": 0, "protein": 0, "fat": 0, "carbs": 0}

        # 6) Final YOLO-path timing log
        t_done = time.perf_counter()

        logger.info(
            "[PIPELINE] Completed via YOLO-only path: preprocess=%.3fs, detect=%.3fs, python_refine=%.3fs, total=%.3fs",
            t_preprocessed - t0,
            t_detected - t_preprocessed,
            t_done - t_detected,
            t_done - t0,
        )

        # 10) Final JSON response
        return {
            "products": products,
            "totals": totals,
            "meta": {
                "pipeline": "yolo_only",
                "yolo_model": "yolov8n.onnx",
                "fallback_used": False,
            },
        }

    def _fallback_gpt(self, image_path: str, t_preprocessed: float):
        """
        Fallback path that uses GPT-vision-based analysis with detailed timing logs.
        """
        import time

        t_before_gpt = time.perf_counter()

        # 7) Call GPT-vision fallback
        result = fallback_analyze_food(image_path)

        t_after_gpt = time.perf_counter()

        logger.warning(
            "[PIPELINE] GPT fallback used: gpt_time=%.3fs, from_preprocess_to_gpt=%.3fs",
            t_after_gpt - t_before_gpt,
            t_after_gpt - t_preprocessed,
        )

        # Ensure meta block with fallback flag set
        meta = result.get("meta") or {}
        meta.setdefault("pipeline", "yolo_only")
        meta.setdefault("yolo_model", "yolov8n.onnx")
        meta["fallback_used"] = True
        result["meta"] = meta

        return result
