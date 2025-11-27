"""Image preprocessing functions for food photo analysis.

Goals:
- Support multiple strategies via PREPROCESS_STRATEGY:
  - "rembg" (default): background removal via rembg
  - "no_bg_removal": no background removal, resize + optional plate crop only
  - "grabcut": fast GrabCut-based foreground extraction
- Unified resize stage controlled by USE_BACKEND_RESIZE / BACKEND_MAX_SIDE_PX
- Optional rembg preloading as singleton via PRELOAD_REMBG_MODEL
- Basic timeout + fallback strategy handling via PREPROCESS_TIMEOUT_MS /
  PREPROCESS_FALLBACK_STRATEGY
- Detailed latency & resolution logging for metrics.
"""

import logging
import os
import shutil
import time
from io import BytesIO
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove

from src.config import (
    REMBG_MODEL,
    ENABLE_PLATE_CROP,
    PREPROCESS_STRATEGY,
    PREPROCESS_FALLBACK_STRATEGY,
    USE_BACKEND_RESIZE,
    BACKEND_MAX_SIDE_PX,
    PRELOAD_REMBG_MODEL,
    PREPROCESS_TIMEOUT_MS,
)
from src.preprocess.grabcut import grabcut_foreground

logger = logging.getLogger(__name__)

# -----------------------------------
# rembg session handling (singleton)
# -----------------------------------

_rembg_session = None
if PRELOAD_REMBG_MODEL:
    _rembg_session = new_session(model_name=REMBG_MODEL)
    logger.info(
        "Initialized rembg session with model %s (preload=%s)",
        REMBG_MODEL,
        PRELOAD_REMBG_MODEL,
    )
else:
    logger.info("Rembg preload disabled (PRELOAD_REMBG_MODEL=%s)", PRELOAD_REMBG_MODEL)


def _get_rembg_session():
    """Get (lazy-init) singleton rembg session."""
    global _rembg_session
    if _rembg_session is None:
        logger.info("Lazy-loading rembg model %s", REMBG_MODEL)
        _rembg_session = new_session(model_name=REMBG_MODEL)
    return _rembg_session


# -----------------------------------
# Low-level helpers
# -----------------------------------


def safe_imread(path: str):
    """Safe wrapper around cv2.imread with strong validation."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"cv2.imread() failed to read file: {path}")
    return img


def get_image_resolution(path: str) -> Tuple[int, int]:
    """Return (width, height) of image."""
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        # Fallback to OpenCV if PIL fails
        img = safe_imread(path)
        h, w = img.shape[:2]
        return w, h


def resize_image(input_path: str, output_path: str, max_side: int = BACKEND_MAX_SIDE_PX):
    """Resize image so that the longer side == max_side, keep aspect ratio, force JPEG."""
    img = Image.open(input_path)
    img.thumbnail((max_side, max_side))
    img = img.convert("RGB")  # PNG -> JPEG
    img.save(output_path, format="JPEG")  # ALWAYS JPEG


def crop_to_plate(input_path, output_path):
    """Existing expensive HoughCircles-based plate crop (optional)."""
    img = safe_imread(input_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=200,
        param1=100,
        param2=30,
        minRadius=80,
        maxRadius=600,
    )

    if circles is None:
        return False

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]

    crop = img[y - r : y + r, x - r : x + r]

    cv2.imwrite(output_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def remove_background(input_path, output_path):
    """Background removal via rembg singleton session."""
    session = _get_rembg_session()
    with open(input_path, "rb") as i:
        data = remove(i.read(), session=session)

    # force JPEG output
    img = Image.open(BytesIO(data)).convert("RGB")
    img.save(output_path, format="JPEG")

    return output_path


# -----------------------------------
# Strategy implementations
# -----------------------------------


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _strategy_rembg(
    src_path: str,
    workdir: str,
    timings: Dict[str, float],
    total_start: float,
) -> str:
    """Original pipeline: resize → optional plate crop → rembg → final resize."""
    resized = os.path.join(workdir, "resized.jpg")
    cropped = os.path.join(workdir, "cropped.jpg")
    no_bg = os.path.join(workdir, "no_bg.jpg")
    final = os.path.join(workdir, "final.jpg")

    # Unified initial resize (can be disabled)
    if USE_BACKEND_RESIZE:
        t = time.time()
        resize_image(src_path, resized, max_side=BACKEND_MAX_SIDE_PX)
        timings["resize_ms"] = round((time.time() - t) * 1000, 2)
        src_for_crop = resized
        logger.info("Resized image saved: %s", resized)
    else:
        shutil.copyfile(src_path, resized)
        timings["resize_ms"] = 0.0
        src_for_crop = resized
        logger.info("Backend resize disabled, copied input to: %s", resized)

    # Optional crop to plate
    if ENABLE_PLATE_CROP:
        t = time.time()
        try:
            cropped_ok = crop_to_plate(src_for_crop, cropped)
        except Exception as e:
            logger.warning("Cropping failed: %s", e)
            cropped_ok = False
        timings["crop_ms"] = round((time.time() - t) * 1000, 2)
        to_bg = cropped if cropped_ok else src_for_crop
        logger.info("Cropping result: %s", "success" if cropped_ok else "skipped")
    else:
        timings["crop_ms"] = 0.0
        to_bg = src_for_crop
        logger.info("Cropping is disabled via ENABLE_PLATE_CROP")

    # Background removal
    t = time.time()
    try:
        remove_background(to_bg, no_bg)
        bg_source = no_bg
    except Exception as e:
        logger.warning("Background removal failed: %s", e)
        bg_source = to_bg  # fallback
    timings["rembg_ms"] = round((time.time() - t) * 1000, 2)

    # Final resize to keep bounded resolution
    t = time.time()
    resize_image(bg_source, final, max_side=BACKEND_MAX_SIDE_PX)
    timings["final_resize_ms"] = round((time.time() - t) * 1000, 2)
    logger.info("Final image saved: %s", final)

    return final


def _strategy_no_bg_removal(
    src_path: str,
    workdir: str,
    timings: Dict[str, float],
    total_start: float,
) -> str:
    """
    Fast path without any background removal.

    Pipeline:
    - Unified resize
    - Optional plate detection crop (if ENABLE_PLATE_CROP)
    - Final resize (to clean up and normalize)
    """
    resized = os.path.join(workdir, "resized_no_bg.jpg")
    cropped = os.path.join(workdir, "cropped_no_bg.jpg")
    final = os.path.join(workdir, "final_no_bg.jpg")

    # Unified resize
    t = time.time()
    resize_image(src_path, resized, max_side=BACKEND_MAX_SIDE_PX)
    timings["resize_ms"] = round((time.time() - t) * 1000, 2)
    src_for_crop = resized
    logger.info("[no_bg_removal] Resized image saved: %s", resized)

    # Optional plate crop
    if ENABLE_PLATE_CROP:
        t = time.time()
        try:
            cropped_ok = crop_to_plate(src_for_crop, cropped)
        except Exception as e:
            logger.warning("[no_bg_removal] Cropping failed: %s", e)
            cropped_ok = False
        timings["crop_ms"] = round((time.time() - t) * 1000, 2)
        src_for_final = cropped if cropped_ok else src_for_crop
        logger.info(
            "[no_bg_removal] Cropping result: %s",
            "success" if cropped_ok else "skipped",
        )
    else:
        timings["crop_ms"] = 0.0
        src_for_final = src_for_crop
        logger.info("[no_bg_removal] Cropping disabled via ENABLE_PLATE_CROP")

    # Final resize for normalization
    t = time.time()
    resize_image(src_for_final, final, max_side=BACKEND_MAX_SIDE_PX)
    timings["final_resize_ms"] = round((time.time() - t) * 1000, 2)
    logger.info("[no_bg_removal] Final image saved: %s", final)

    return final


def _strategy_grabcut(
    src_path: str,
    workdir: str,
    timings: Dict[str, float],
    total_start: float,
) -> str:
    """
    GrabCut-based lightweight alternative to rembg.

    Pipeline:
    - Unified resize
    - GrabCut foreground extraction
    """
    resized = os.path.join(workdir, "resized_grabcut.jpg")
    grabcut_out = os.path.join(workdir, "grabcut.jpg")

    # Unified resize
    t = time.time()
    resize_image(src_path, resized, max_side=BACKEND_MAX_SIDE_PX)
    timings["resize_ms"] = round((time.time() - t) * 1000, 2)
    logger.info("[grabcut] Resized image saved: %s", resized)

    # GrabCut
    grabcut_path, grabcut_timings = grabcut_foreground(resized, grabcut_out)
    timings.update(grabcut_timings)
    logger.info("[grabcut] Output image saved: %s", grabcut_path)

    return grabcut_path


def _run_strategy(
    strategy: str,
    src_path: str,
    workdir: str,
    timings: Dict[str, float],
    total_start: float,
) -> str:
    """Dispatch to specific strategy implementation."""
    strategy = (strategy or "rembg").lower()
    timings["strategy_used"] = strategy

    if strategy == "no_bg_removal":
        return _strategy_no_bg_removal(src_path, workdir, timings, total_start)
    if strategy == "grabcut":
        return _strategy_grabcut(src_path, workdir, timings, total_start)
    # default: rembg
    return _strategy_rembg(src_path, workdir, timings, total_start)


# -----------------------------------
# Public API
# -----------------------------------


def preprocess_image(path: str):
    """
    Full preprocessing pipeline with detailed timing and strategy support.

    Returns:
        (final_path, timings_dict)
    Timings dict includes (where applicable):
        - preprocess_total_ms / total_ms
        - resize_ms
        - crop_ms
        - rembg_ms
        - final_resize_ms
        - grabcut_ms / grabcut_total_ms
        - strategy_used
        - fallback_strategy_used
        - timed_out (bool)
        - image_input_resolution
        - image_output_resolution
    """
    total_start = time.time()
    timings: Dict[str, float] = {}
    base = "/tmp/preprocess"
    _ensure_dir(base)

    input_w, input_h = get_image_resolution(path)
    timings["image_input_resolution"] = {"width": input_w, "height": input_h}
    timings["fallback_strategy_used"] = None
    timings["timed_out"] = False

    logger.info(
        "Starting preprocessing for: %s (strategy=%s, rembg_model=%s, plate_crop_enabled=%s, timeout_ms=%s)",
        path,
        PREPROCESS_STRATEGY,
        REMBG_MODEL,
        ENABLE_PLATE_CROP,
        PREPROCESS_TIMEOUT_MS,
    )

    try:
        # Primary strategy
        final_path = _run_strategy(
            PREPROCESS_STRATEGY,
            path,
            base,
            timings,
            total_start,
        )

        # Timeout-based fallback (best-effort, since Python cannot hard-kill work easily)
        elapsed_ms = round((time.time() - total_start) * 1000, 2)
        timings["preprocess_total_ms"] = elapsed_ms
        timings["total_ms"] = elapsed_ms

        if (
            PREPROCESS_TIMEOUT_MS > 0
            and elapsed_ms > PREPROCESS_TIMEOUT_MS
            and PREPROCESS_FALLBACK_STRATEGY != "none"
            and PREPROCESS_FALLBACK_STRATEGY.lower()
            != (PREPROCESS_STRATEGY or "rembg").lower()
        ):
            logger.warning(
                "Preprocessing exceeded timeout (elapsed=%sms, limit=%sms). "
                "Falling back to strategy=%s.",
                elapsed_ms,
                PREPROCESS_TIMEOUT_MS,
                PREPROCESS_FALLBACK_STRATEGY,
            )
            timings["timed_out"] = True
            timings["fallback_strategy_used"] = PREPROCESS_FALLBACK_STRATEGY

            # Run lightweight fallback strategy from original input
            final_path = _run_strategy(
                PREPROCESS_FALLBACK_STRATEGY,
                path,
                base,
                timings,
                total_start,
            )
            # Recompute total time including fallback
            elapsed_ms = round((time.time() - total_start) * 1000, 2)
            timings["preprocess_total_ms"] = elapsed_ms
            timings["total_ms"] = elapsed_ms

        # Output resolution
        out_w, out_h = get_image_resolution(final_path)
        timings["image_output_resolution"] = {"width": out_w, "height": out_h}

        logger.info(
            "Preprocessing completed (strategy=%s, fallback=%s, total=%sms)",
            timings.get("strategy_used"),
            timings.get("fallback_strategy_used"),
            timings.get("preprocess_total_ms"),
        )
        return final_path, timings

    except Exception as e:
        logger.error("Preprocessing failed: %s", e)
        timings["error"] = str(e)
        elapsed_ms = round((time.time() - total_start) * 1000, 2)
        timings["total_ms"] = elapsed_ms
        timings["preprocess_total_ms"] = elapsed_ms
        # Fallback: return original image untouched
        return path, timings
