import logging
from typing import Optional

logger = logging.getLogger(__name__)


def preprocess_image(path: str, max_size: int = 768):
    """
    Load image → resize long side to max_size → normalize lighting → return RGB array.

    This is a lightweight, detector-focused preprocessing step:
    - ensures bounded resolution for fast YOLO inference
    - normalizes brightness/contrast
    - returns an in-memory RGB array (no disk I/O here)

    Implementation is defensive:
    - tries to use OpenCV if available (fast path)
    - if cv2 is not installed, falls back to PIL-based implementation
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        logger.warning(
            "OpenCV not available in vision_pipeline.preprocess (error=%s); "
            "falling back to PIL-based preprocessing",
            e,
        )
        # PIL-based fallback: keeps the same contract (returns RGB ndarray)
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = max_size / max(w, h)
        if scale < 1:
            img = img.resize((int(w * scale), int(h * scale)))

        arr = np.asarray(img).astype("float32")

        # Light normalization to [0, 255]
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) * (255.0 / (arr_max - arr_min))
        return arr.astype("uint8")

    # --- OpenCV fast path ---

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
