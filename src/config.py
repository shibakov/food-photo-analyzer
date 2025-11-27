import os


def _parse_cors_origins(raw: str) -> list[str]:
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CORS_ORIGINS = _parse_cors_origins(os.getenv("CORS_ORIGINS", "*"))
ALLOW_ALL_ORIGINS = CORS_ORIGINS == ["*"]

# -----------------------------------
# Preprocessing / rembg configuration
# -----------------------------------

# REMBG_MODEL: which rembg model to use (e.g. "u2net", "u2netp", "u2net_human_seg")
REMBG_MODEL = os.getenv("REMBG_MODEL", "u2netp")

# ENABLE_PLATE_CROP: whether to run expensive HoughCircles-based crop_to_plate
ENABLE_PLATE_CROP = os.getenv("ENABLE_PLATE_CROP", "false").lower() == "true"

# PREPROCESS_STRATEGY:
# - "rembg"          — current default pipeline with background removal
# - "no_bg_removal"  — resize + region detection only, no background removal step
# - "grabcut"        — fast GrabCut-based crop of main foreground region
PREPROCESS_STRATEGY = os.getenv("PREPROCESS_STRATEGY", "rembg").lower()

# PREPROCESS_FALLBACK_STRATEGY:
# - "no_bg_removal" (default)
# - "grabcut"
# - "none"          — do not change strategy on timeout/failures
PREPROCESS_FALLBACK_STRATEGY = os.getenv("PREPROCESS_FALLBACK_STRATEGY", "no_bg_removal").lower()

# USE_BACKEND_RESIZE: enable unified backend-side resize before any preprocessing
USE_BACKEND_RESIZE = os.getenv("USE_BACKEND_RESIZE", "true").lower() == "true"

# BACKEND_MAX_SIDE_PX: longer side of image after initial resize (for all strategies)
BACKEND_MAX_SIDE_PX = int(os.getenv("BACKEND_MAX_SIDE_PX", "600"))

# PRELOAD_REMBG_MODEL: load rembg model once on service startup (singleton)
PRELOAD_REMBG_MODEL = os.getenv("PRELOAD_REMBG_MODEL", "true").lower() == "true"

# PREPROCESS_TIMEOUT_MS: hard timeout for full preprocessing pipeline (0 = disabled)
PREPROCESS_TIMEOUT_MS = int(os.getenv("PREPROCESS_TIMEOUT_MS", "1000"))

# -----------------------------------
# GPT / models configuration
# -----------------------------------

# GPT_MODEL: main vision model for food analysis (used in /recognize pipeline)
# Expected values: "gpt-4o" (default) or "gpt-4o-mini"
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")

# USE_LOCAL_FAST_MODEL: enable local offline fallback (CNN + OCR) when OpenAI fails
USE_LOCAL_FAST_MODEL = os.getenv("USE_LOCAL_FAST_MODEL", "false").lower() == "true"
