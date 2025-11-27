"""GrabCut-based lightweight foreground extraction for food photos."""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _safe_imread(path: str):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"cv2.imread() failed to read file: {path}")
    return img


def grabcut_foreground(
    input_path: str,
    output_path: str,
    rect_ratio: float = 0.8,
    iterations: int = 5,
) -> Tuple[str, dict]:
    """
    Fast GrabCut-based foreground segmentation.

    Strategy:
    - Read image
    - Define central rectangle covering `rect_ratio` of width/height as probable foreground
    - Run cv2.grabCut for a few iterations
    - Mask background to white and keep foreground region
    - Save result as JPEG

    Returns:
        (output_path, timings_dict)
    """
    import time

    timings: dict[str, float] = {}
    t0 = time.time()

    img = _safe_imread(input_path)
    h, w = img.shape[:2]
    timings["grabcut_input_resolution"] = {"width": w, "height": h}

    # Central rectangle
    rw = int(w * rect_ratio)
    rh = int(h * rect_ratio)
    x = (w - rw) // 2
    y = (h - rh) // 2
    rect = (x, y, rw, rh)

    t = time.time()
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            img,
            mask,
            rect,
            bgd_model,
            fgd_model,
            iterations,
            cv2.GC_INIT_WITH_RECT,
        )
    except Exception as e:
        logger.warning("GrabCut failed: %s", e)
        # Fallback: just copy original image
        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        timings["grabcut_ms"] = round((time.time() - t) * 1000, 2)
        timings["grabcut_failed"] = True
        timings["grabcut_total_ms"] = round((time.time() - t0) * 1000, 2)
        timings["grabcut_output_resolution"] = {"width": w, "height": h}
        return output_path, timings

    # Build mask: 1 for foreground, 0 for background
    mask2 = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        1,
        0,
    ).astype("uint8")

    fg = img * mask2[:, :, np.newaxis]

    # Put white background where mask is 0
    white_bg = np.full(img.shape, 255, dtype=np.uint8)
    result = np.where(mask2[:, :, np.newaxis] == 1, fg, white_bg)

    timings["grabcut_ms"] = round((time.time() - t) * 1000, 2)

    # Save JPEG
    t_save = time.time()
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    timings["grabcut_save_ms"] = round((time.time() - t_save) * 1000, 2)

    h_out, w_out = result.shape[:2]
    timings["grabcut_output_resolution"] = {"width": w_out, "height": h_out}
    timings["grabcut_total_ms"] = round((time.time() - t0) * 1000, 2)

    logger.info(
        "GrabCut foreground extraction done in %sms (input=%sx%s, output=%sx%s)",
        timings["grabcut_total_ms"],
        w,
        h,
        w_out,
        h_out,
    )

    return output_path, timings
