import cv2
import numpy as np


def preprocess_image(path: str, max_size: int = 768) -> np.ndarray:
    """
    Load image → resize long side to max_size → normalize lighting → return RGB array.

    This is a lightweight, detector-focused preprocessing step:
    - ensures bounded resolution for fast YOLO inference
    - normalizes brightness/contrast
    - returns an in-memory RGB array (no disk I/O here)
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize while preserving aspect ratio
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Light normalization to [0, 255]
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    return img
