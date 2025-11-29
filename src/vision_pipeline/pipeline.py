import logging
import time
from typing import Any, Dict

import numpy as np  # type: ignore

from src.gpt_vision import analyze_food as fallback_analyze_food
from .preprocess import preprocess_image
from .ensemble import get_ensemble_singleton, PLATE_GRAMS

logger = logging.getLogger(__name__)


class VisionPipeline:
    """
    High-level ensemble vision pipeline:

    Original Image
      ↓
    preprocess (resize, normalize)
      ↓
    YOLO-general (обнаружение еды)
      ↓
    YOLO-food (классификация конкретного продукта, опционально)
      ↓
    Segmentation (оценка площади по маске/bbox)
      ↓
    Classifier (уточнение названия блюда)
      ↓
    Heuristic grams (по площади)
      ↓
    Macros (пока нули)
      ↓
    GPT-mini fallback (по условиям)
    """

    def __init__(self):
        # Ensemble singleton will be lazily initialized on first use
        self.ensemble = get_ensemble_singleton()

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Ensemble-based analysis entrypoint.

        Returns:
            {
              "products": [...],
              "totals": {...},
              "meta": {
                  "pipeline": "ensemble_v1",
                  "general_yolo_time": ...,
                  "food_yolo_time": ...,
                  "segmentation_time": ...,
                  "classifier_time": ...,
                  "grams_time": ...,
                  "fallback_used": bool,
                  "fallback_time": ...,
                  "total_time": ...
              }
            }
        """
        logger.info("VisionPipeline: starting ensemble_v1 analysis for %s", image_path)

        # Global start timer
        t0 = time.perf_counter()

        # 1) Preprocess (resize, normalize)
        img = preprocess_image(image_path)
        t_preprocessed = time.perf_counter()

        # Ensure ensemble is available
        ensemble = self.ensemble or get_ensemble_singleton()
        if not ensemble:
            logger.warning(
                "[ENSEMBLE] Ensemble unavailable \u2192 using GPT fallback"
            )
            result = self._fallback_gpt(
                image_path=image_path,
                t_preprocessed=t_preprocessed,
                pipeline_name="ensemble_v1",
            )
            t_done = time.perf_counter()
            meta = result.get("meta") or {}
            meta.setdefault("pipeline", "ensemble_v1")
            meta.setdefault("general_yolo_time", 0.0)
            meta.setdefault("food_yolo_time", 0.0)
            meta.setdefault("segmentation_time", 0.0)
            meta.setdefault("classifier_time", 0.0)
            meta.setdefault("grams_time", 0.0)
            meta["total_time"] = t_done - t0
            result["meta"] = meta
            logger.info(
                "[ENSEMBLE] Total pipeline time %.3fs",
                t_done - t0,
            )
            return result

        # Log per-request model availability snapshot
        logger.info(
            "[ENSEMBLE] Models on request: general=ENABLED, food=%s, segmentor=%s, classifier=%s",
            "ENABLED" if getattr(ensemble, "food_session", None) is not None else "DISABLED",
            "ENABLED" if getattr(ensemble, "segmentor_session", None) is not None else "DISABLED",
            "ENABLED" if getattr(ensemble, "classifier_session", None) is not None else "DISABLED",
        )

        # --- Stage timings (initialized to 0 so they are always present in meta) ---
        general_yolo_time = 0.0
        food_yolo_time = 0.0
        segmentation_time = 0.0
        classifier_time = 0.0
        grams_time = 0.0
        fallback_used = False
        fallback_time = 0.0

        # 2) YOLO-general detection
        try:
            t_gen_start = time.perf_counter()
            detections_general = ensemble.detect_general(img)
            t_gen_end = time.perf_counter()
            general_yolo_time = t_gen_end - t_gen_start

            logger.info(
                "[ENSEMBLE] YOLO-general detected %s objs in %.3fs: %s",
                len(detections_general),
                general_yolo_time,
                detections_general,
            )
        except Exception as e:
            logger.error("[ENSEMBLE] YOLO-general FAILED, error=%s", e)
            result = self._fallback_gpt(
                image_path=image_path,
                t_preprocessed=t_preprocessed,
                pipeline_name="ensemble_v1",
            )
            t_done = time.perf_counter()
            meta = result.get("meta") or {}
            meta.setdefault("pipeline", "ensemble_v1")
            meta.setdefault("general_yolo_time", 0.0)
            meta.setdefault("food_yolo_time", 0.0)
            meta.setdefault("segmentation_time", 0.0)
            meta.setdefault("classifier_time", 0.0)
            meta.setdefault("grams_time", 0.0)
            meta["total_time"] = t_done - t0
            result["meta"] = meta
            logger.info(
                "[ENSEMBLE] Total pipeline time %.3fs",
                t_done - t0,
            )
            return result

        # 3) Filter only "food-like" detections (here: score > 0.3)
        food_candidates = [
            d for d in detections_general if d.get("confidence", 0.0) > 0.3
        ]

        if not food_candidates:
            logger.warning(
                "[ENSEMBLE] No high-confidence detections (score > 0.3) \u2192 GPT fallback"
            )
            result = self._fallback_gpt(
                image_path=image_path,
                t_preprocessed=t_preprocessed,
                pipeline_name="ensemble_v1",
            )
            t_done = time.perf_counter()
            meta = result.get("meta") or {}
            meta.setdefault("pipeline", "ensemble_v1")
            meta.setdefault("general_yolo_time", general_yolo_time)
            meta.setdefault("food_yolo_time", 0.0)
            meta.setdefault("segmentation_time", 0.0)
            meta.setdefault("classifier_time", 0.0)
            meta.setdefault("grams_time", 0.0)
            meta["total_time"] = t_done - t0
            result["meta"] = meta
            logger.info(
                "[ENSEMBLE] Total pipeline time %.3fs",
                t_done - t0,
            )
            return result

        # For now, take the top-1 detection by confidence as primary dish
        food_candidates.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
        primary_det = food_candidates[0]

        # 4) YOLO-food refinement (if available)
        try:
            t_food_start = time.perf_counter()
            refined_detections = ensemble.detect_food(img, food_candidates)
            t_food_end = time.perf_counter()
            food_yolo_time = t_food_end - t_food_start

            logger.info(
                "[ENSEMBLE] YOLO-food classified: %s in %.3fs",
                refined_detections,
                food_yolo_time,
            )
        except Exception as e:
            logger.error("[ENSEMBLE] YOLO-food stage FAILED, error=%s", e)
            refined_detections = food_candidates
            t_food_end = time.perf_counter()
            food_yolo_time = t_food_end - t_food_start

        primary_refined = refined_detections[0] if refined_detections else primary_det
        bbox = primary_refined.get("bbox") or primary_det.get("bbox")
        if not bbox:
            logger.warning(
                "[ENSEMBLE] Primary detection has no bbox \u2192 GPT fallback"
            )
            result = self._fallback_gpt(
                image_path=image_path,
                t_preprocessed=t_preprocessed,
                pipeline_name="ensemble_v1",
            )
            t_done = time.perf_counter()
            meta = result.get("meta") or {}
            meta.setdefault("pipeline", "ensemble_v1")
            meta.setdefault("general_yolo_time", general_yolo_time)
            meta.setdefault("food_yolo_time", food_yolo_time)
            meta.setdefault("segmentation_time", 0.0)
            meta.setdefault("classifier_time", 0.0)
            meta.setdefault("grams_time", 0.0)
            meta["total_time"] = t_done - t0
            result["meta"] = meta
            logger.info(
                "[ENSEMBLE] Total pipeline time %.3fs",
                t_done - t0,
            )
            return result

        # 5) Segmentation for bbox (current impl: bbox-mask)
        logger.info(
            "[ENSEMBLE] Segmentation stage: segmentor_session=%s (logic=bbox-only)",
            "PRESENT" if getattr(ensemble, "segmentor_session", None) is not None else "ABSENT",
        )
        try:
            t_seg_start = time.perf_counter()
            mask = ensemble.segment_mask(img, bbox)
            t_seg_end = time.perf_counter()
            segmentation_time = t_seg_end - t_seg_start

            # Compute normalized area from mask
            mask_pixels = float(np.count_nonzero(mask))
            image_pixels = float(mask.size) if mask is not None else 0.0
            area_normalized = (mask_pixels / image_pixels) if image_pixels > 0 else 0.0

            logger.info(
                "[ENSEMBLE] Segmentation area=%.4f in %.3fs",
                area_normalized,
                segmentation_time,
            )
        except Exception as e:
            logger.error("[ENSEMBLE] Segmentation FAILED, error=%s", e)
            result = self._fallback_gpt(
                image_path=image_path,
                t_preprocessed=t_preprocessed,
                pipeline_name="ensemble_v1",
            )
            t_done = time.perf_counter()
            meta = result.get("meta") or {}
            meta.setdefault("pipeline", "ensemble_v1")
            meta.setdefault("general_yolo_time", general_yolo_time)
            meta.setdefault("food_yolo_time", food_yolo_time)
            meta.setdefault("segmentation_time", 0.0)
            meta.setdefault("classifier_time", 0.0)
            meta.setdefault("grams_time", 0.0)
            meta["total_time"] = t_done - t0
            result["meta"] = meta
            logger.info(
                "[ENSEMBLE] Total pipeline time %.3fs",
                t_done - t0,
            )
            return result

        # 6) Classifier refine
        logger.info(
            "[ENSEMBLE] Classifier stage: session=%s",
            "PRESENT" if getattr(ensemble, "classifier_session", None) is not None else "ABSENT",
        )
        try:
            t_clf_start = time.perf_counter()
            if getattr(ensemble, "classifier_session", None) is not None:
                logger.info(
                    "[ENSEMBLE] Classifier: loaded OK (onnx=True, external_data=True)"
                )
            else:
                logger.info(
                    "[ENSEMBLE] Classifier: DISABLED or unavailable (fallback may be used)"
                )

            food_name, classifier_conf = ensemble.classify_crop(img, bbox)
            t_clf_end = time.perf_counter()
            classifier_time = t_clf_end - t_clf_start

            logger.info(
                '[ENSEMBLE] Classifier inference: class="%s", conf=%.3f',
                food_name,
                classifier_conf,
            )
        except Exception as e:
            logger.error("[ENSEMBLE] Classifier FAILED, error=%s", e)
            food_name, classifier_conf = "ambiguous", 0.0
            t_clf_end = time.perf_counter()
            classifier_time = t_clf_end - t_clf_start

        # 7) Grams estimation (size-based)
        t_grams_start = time.perf_counter()
        grams = area_normalized * PLATE_GRAMS
        t_grams_end = time.perf_counter()
        grams_time = t_grams_end - t_grams_start

        logger.info("[ENSEMBLE] Grams=%s in %.3fs", grams, grams_time)

        # 8) Fallback condition based on classifier
        if classifier_conf < 0.40 or food_name == "ambiguous":
            # Low confidence / ambiguous dish: use GPT fallback
            t_fb_start = time.perf_counter()
            result = self._fallback_gpt(
                image_path=image_path,
                t_preprocessed=t_preprocessed,
                pipeline_name="ensemble_v1",
            )
            t_fb_end = time.perf_counter()
            fallback_used = True
            fallback_time = t_fb_end - t_fb_start

            logger.info(
                "[ENSEMBLE] Fallback GPT used in %.3fs",
                fallback_time,
            )

            t_done = time.perf_counter()
            meta = result.get("meta") or {}
            meta.setdefault("pipeline", "ensemble_v1")
            meta.setdefault("general_yolo_time", general_yolo_time)
            meta.setdefault("food_yolo_time", food_yolo_time)
            meta.setdefault("segmentation_time", segmentation_time)
            meta.setdefault("classifier_time", classifier_time)
            meta.setdefault("grams_time", grams_time)
            meta["fallback_used"] = True
            meta["fallback_time"] = fallback_time
            meta["total_time"] = t_done - t0
            result["meta"] = meta

            logger.info(
                "[ENSEMBLE] Total pipeline time %.3fs",
                t_done - t0,
            )
            return result

        # 9) No fallback: build YOLO-only ensemble result
        products = [
            {
                "product_name": food_name,
                "quantity_g": round(grams, 1),
                "kcal": 0,
                "protein": 0,
                "fat": 0,
                "carbs": 0,
            }
        ]

        totals = {"kcal": 0, "protein": 0, "fat": 0, "carbs": 0}

        t_done = time.perf_counter()

        logger.info(
            "[ENSEMBLE] Total pipeline time %.3fs",
            t_done - t0,
        )

        meta = {
            "pipeline": "ensemble_v1",
            "general_yolo_time": general_yolo_time,
            "food_yolo_time": food_yolo_time,
            "segmentation_time": segmentation_time,
            "classifier_time": classifier_time,
            "grams_time": grams_time,
            "fallback_used": fallback_used,
            "fallback_time": fallback_time,
            "total_time": t_done - t0,
        }

        return {
            "products": products,
            "totals": totals,
            "meta": meta,
        }

    def _fallback_gpt(
        self, image_path: str, t_preprocessed: float, pipeline_name: str = "ensemble_v1"
    ) -> Dict[str, Any]:
        """
        Fallback path that uses GPT-vision-based analysis with detailed timing logs.

        This is used when:
        - ensemble is unavailable
        - YOLO-general / segmentation / classifier fail
        - classifier is low-confidence or ambiguous
        """
        t_before_gpt = time.perf_counter()

        # Call GPT-vision fallback
        result = fallback_analyze_food(image_path)

        t_after_gpt = time.perf_counter()
        gpt_time = t_after_gpt - t_before_gpt

        logger.info(
            "[ENSEMBLE] Fallback GPT used in %.3fs (from preprocess=%.3fs)",
            gpt_time,
            t_after_gpt - t_preprocessed,
        )

        # Ensure meta block with ensemble fields
        meta = result.get("meta") or {}
        meta.setdefault("pipeline", pipeline_name)
        meta.setdefault("general_yolo_time", 0.0)
        meta.setdefault("food_yolo_time", 0.0)
        meta.setdefault("segmentation_time", 0.0)
        meta.setdefault("classifier_time", 0.0)
        meta.setdefault("grams_time", 0.0)
        meta["fallback_used"] = True
        meta["fallback_time"] = gpt_time
        result["meta"] = meta

        return result
