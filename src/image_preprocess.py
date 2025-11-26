"""Image preprocessing functions for food photo analysis."""

import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import logging

logger = logging.getLogger(__name__)


def resize_image(input_path, output_path, max_side=800):
    """Resize image to fit within max_side x max_side, keeping aspect ratio."""
    img = Image.open(input_path)
    img.thumbnail((max_side, max_side))
    img.save(output_path, format="PNG")


def crop_to_plate(input_path, output_path):
    """Crop to circular plate using HoughCircles. Returns True if cropped, False if not."""
    img = cv2.imread(input_path)
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
    crop = img[y-r:y+r, x-r:x+r]
    cv2.imwrite(output_path, crop)
    return True


def remove_background(input_path, output_path):
    """Remove background using rembg."""
    with open(input_path, "rb") as i:
        with open(output_path, "wb") as o:
            o.write(remove(i.read()))
    return output_path


def preprocess_image(path):
    """Full preprocessing pipeline: resize -> crop -> remove_bg -> final resize."""
    import os

    base = "/tmp/preprocess"
    os.makedirs(base, exist_ok=True)

    resized = f"{base}/resized.png"
    cropped = f"{base}/cropped.png"
    no_bg = f"{base}/no_bg.png"
    final = f"{base}/final.png"

    resize_image(path, resized)

    cropped_ok = crop_to_plate(resized, cropped)
    to_bg = cropped if cropped_ok else resized

    remove_background(to_bg, no_bg)

    # финальный ресайз
    resize_image(no_bg, final)

    return final
