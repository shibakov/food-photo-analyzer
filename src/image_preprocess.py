"""Image preprocessing functions for food photo analysis."""

import os
import time
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from rembg import remove
import logging

logger = logging.getLogger(__name__)


def safe_imread(path):
    """Safe wrapper around cv2.imread with strong validation."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"cv2.imread() failed to read file: {path}")
    return img


def resize_image(input_path, output_path, max_side=800):
    img = Image.open(input_path)
    img.thumbnail((max_side, max_side))
    img = img.convert("RGB")  # PNG -> JPEG
    img.save(output_path, format="JPEG")  # ALWAYS JPEG


def crop_to_plate(input_path, output_path):
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
    with open(input_path, "rb") as i:
        data = remove(i.read())

    # force JPEG output
    img = Image.open(BytesIO(data)).convert("RGB")
    img.save(output_path, format="JPEG")

    return output_path


def preprocess_image(path):
    """
    Full preprocessing pipeline with detailed timing.

    Returns:
        (final_path, timings_dict)
    """
    logger.info(f"Starting preprocessing for: {path}")

    base = "/tmp/preprocess"
    os.makedirs(base, exist_ok=True)

    resized = f"{base}/resized.jpg"
    cropped = f"{base}/cropped.jpg"
    no_bg = f"{base}/no_bg.jpg"
    final = f"{base}/final.jpg"

    timings: dict[str, float] = {}
    total_start = time.time()

    try:
        # Resize
        t = time.time()
        resize_image(path, resized)
        timings["resize_ms"] = round((time.time() - t) * 1000, 2)
        logger.info(f"Resized image saved: {resized}")

        # Crop
        t = time.time()
        try:
            cropped_ok = crop_to_plate(resized, cropped)
        except Exception as e:
            logger.warning(f"Cropping failed: {e}")
            cropped_ok = False
        timings["crop_ms"] = round((time.time() - t) * 1000, 2)

        to_bg = cropped if cropped_ok else resized
        logger.info(f"Cropping result: {'success' if cropped_ok else 'skipped'}")

        # Background removal
        t = time.time()
        try:
            remove_background(to_bg, no_bg)
            bg_source = no_bg
        except Exception as e:
            logger.warning(f"Background removal failed: {e}")
            bg_source = to_bg  # fallback
        timings["remove_bg_ms"] = round((time.time() - t) * 1000, 2)

        # Final resize
        t = time.time()
        resize_image(bg_source, final)
        timings["final_resize_ms"] = round((time.time() - t) * 1000, 2)
        logger.info(f"Final image saved: {final}")

        timings["total_ms"] = round((time.time() - total_start) * 1000, 2)
        return final, timings

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        timings["error"] = str(e)
        timings["total_ms"] = round((time.time() - total_start) * 1000, 2)
        return path, timings  # Fallback: return original image
