import logging
import time

from src.gpt_vision import analyze_food as gpt_vision_fallback, _get_vision_model_name
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

        t_total_start = time.perf_counter()

        # 1) Lightweight detector-focused preprocessing
        t_pre_start = time.perf_counter()
        img = preprocess_image(image_path)
        t_pre_end = time.perf_counter()
        preproc_s = t_pre_end - t_pre_start
        logger.info("VisionPipeline: preprocess done in %.3fs", preproc_s)

        # 2) Local YOLO detection (with graceful degradation)
        detector = self.detector or _get_detector_singleton()
        if not detector:
            # Detector never initialized or failed to load -> direct GPT-vision fallback
            vision_model = _get_vision_model_name()
            logger.warning(
                "VisionPipeline: detector unavailable, falling back to GPT vision "
                "(model=%s) for %s",
                vision_model,
                image_path,
            )
            t_fb_start = time.perf_counter()
            result = gpt_vision_fallback(image_path)
            t_fb_end = time.perf_counter()
            fb_s = t_fb_end - t_fb_start
            total_s = t_fb_end - t_total_start
            logger.info(
                "VisionPipeline: completed via GPT-vision model=%s; "
                "timings: preprocess=%.3fs, gpt_vision=%.3fs, total=%.3fs",
                vision_model,
                preproc_s,
                fb_s,
                total_s,
            )
            return result

        # Try local detector
        try:
            t_det_start = time.perf_counter()
            detections = detector.detect(img)
            t_det_end = time.perf_counter()
            detect_s = t_det_end - t_det_start
        except Exception as e:
            logger.error(
                "VisionPipeline: detector error for %s: %s; falling back to GPT vision",
                image_path,
                e,
            )
            vision_model = _get_vision_model_name()
            t_fb_start = time.perf_counter()
            result = gpt_vision_fallback(image_path)
            t_fb_end = time.perf_counter()
            fb_s = t_fb_end - t_fb_start
            total_s = t_fb_end - t_total_start
            logger.info(
                "VisionPipeline: completed via GPT-vision model=%s after detector error; "
                "timings: preprocess=%.3fs, gpt_vision=%.3fs, total=%.3fs",
                vision_model,
                preproc_s,
                fb_s,
                total_s,
            )
            return result

        logger.info(
            "VisionPipeline: detector returned %d objects in %.3fs",
            len(detections),
            detect_s,
        )

        # 3) Fallback: if detector didn't find anything, call GPT-vision directly
        if len(detections) == 0:
            vision_model = _get_vision_model_name()
            logger.warning(
                "VisionPipeline: no detections, falling back to GPT vision (model=%s) "
                "for %s",
                vision_model,
                image_path,
            )
            t_fb_start = time.perf_counter()
            result = gpt_vision_fallback(image_path)
            t_fb_end = time.perf_counter()
            fb_s = t_fb_end - t_fb_start
            total_s = t_fb_end - t_total_start
            logger.info(
                "VisionPipeline: completed via GPT-vision model=%s; "
                "timings: preprocess=%.3fs, detect=%.3fs, gpt_vision=%.3fs, total=%.3fs",
                vision_model,
                preproc_s,
                detect_s,
                fb_s,
                total_s,
            )
            return result

        # 4) GPT-mini refinement using nutrition schema
        t_ref_start = time.perf_counter()
        result = self.refiner.refine(detections, FOOD_NUTRITION)
        t_ref_end = time.perf_counter()
        refine_s = t_ref_end - t_ref_start
        total_s = t_ref_end - t_total_start

        detector_model = getattr(detector, "model_path", "models/yolov8n.onnx")
        refiner_model = getattr(self.refiner, "model", "gpt-4o-mini")

        logger.info(
            "VisionPipeline: completed via local detector model=%s + GPT refiner "
            "model=%s; timings: preprocess=%.3fs, detect=%.3fs, refine=%.3fs, "
            "total=%.3fs",
            detector_model,
            refiner_model,
            preproc_s,
            detect_s,
            refine_s,
            total_s,
        )

        return result
