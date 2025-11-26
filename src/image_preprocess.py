"""Image preprocessing functions for food photo analysis."""

import os
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
    img = img.convert("RGB")                   # <- PNG -> JPEG
    img.save(output_path, format="JPEG")       # <- ALWAYS JPEG


def crop_to_plate(input_path, output_path):
    img = safe_imread(input_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
        param1=100, param2=30, minRadius=80, maxRadius=600
    )

    if circles is None:
        return False

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]

    crop = img[y - r:y + r, x - r:x + r]

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
    logger.info(f"Starting preprocessing for: {path}")

    base = "/tmp/preprocess"
    os.makedirs(base, exist_ok=True)

    resized = f"{base}/resized.jpg"
    cropped = f"{base}/cropped.jpg"
    no_bg = f"{base}/no_bg.jpg"
    final = f"{base}/final.jpg"

    try:
        # Resize
        resize_image(path, resized)
        logger.info(f"Resized image saved: {resized}")

        # Crop
        try:
            cropped_ok = crop_to_plate(resized, cropped)
        except Exception as e:
            logger.warning(f"Cropping failed: {e}")
            cropped_ok = False

        to_bg = cropped if cropped_ok else resized
        logger.info(f"Cropping result: {'success' if cropped_ok else 'skipped'}")

        # Background
        try:
            remove_background(to_bg, no_bg)
        except Exception as e:
            logger.warning(f"Background removal failed: {e}")
            no_bg = to_bg  # fallback

        # Final resize
        resize_image(no_bg, final)
        logger.info(f"Final image saved: {final}")

        return final

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return path  # Fallback: return original image
