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
# Default: enabled ("true") to help GPT focus on the plate region.
ENABLE_PLATE_CROP = os.getenv("ENABLE_PLATE_CROP", "true").lower() == "true"

# PREPROCESS_STRATEGY:
# - "rembg"           — rembg-based background removal
# - "no_bg"           — alias for "no_bg_removal"
# - "no_bg_removal"   — resize + optional plate crop only, no background removal
# - "grabcut"         — fast GrabCut-based crop of main foreground region (recommended default)
PREPROCESS_STRATEGY = os.getenv("PREPROCESS_STRATEGY", "grabcut").lower()

# PREPROCESS_FALLBACK_STRATEGY:
# - "no_bg_removal" (default)
# - "grabcut"
# - "none"          — do not change strategy on timeout/failures
PREPROCESS_FALLBACK_STRATEGY = os.getenv("PREPROCESS_FALLBACK_STRATEGY", "no_bg_removal").lower()

# USE_BACKEND_RESIZE: enable unified backend-side resize before any preprocessing
USE_BACKEND_RESIZE = os.getenv("USE_BACKEND_RESIZE", "true").lower() == "true"

# BACKEND_MAX_SIDE_PX: longer side of image after initial resize (for all strategies)
# Reduced default to 400px to speed up GPT vision latency.
BACKEND_MAX_SIDE_PX = int(os.getenv("BACKEND_MAX_SIDE_PX", "400"))

# PRELOAD_REMBG_MODEL: load rembg model once on service startup (singleton)
# Default: disabled to avoid rembg overhead unless explicitly requested.
PRELOAD_REMBG_MODEL = os.getenv("PRELOAD_REMBG_MODEL", "false").lower() == "true"

# PREPROCESS_TIMEOUT_MS: hard timeout for full preprocessing pipeline (0 = disabled)
# Reduced to 500ms so that heavy strategies (like rembg) quickly fall back.
PREPROCESS_TIMEOUT_MS = int(os.getenv("PREPROCESS_TIMEOUT_MS", "500"))

# -----------------------------------
# GPT / models configuration
# -----------------------------------

# GPT_MODEL: main vision model for food analysis (used in /recognize pipeline)
# Expected values: "gpt-4o-mini" (default) or "gpt-4o"
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

# USE_LOCAL_FAST_MODEL: enable local offline fallback (CNN + OCR) when OpenAI fails
USE_LOCAL_FAST_MODEL = os.getenv("USE_LOCAL_FAST_MODEL", "false").lower() == "true"
