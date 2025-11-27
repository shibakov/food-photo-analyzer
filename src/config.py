import os


def _parse_cors_origins(raw: str) -> list[str]:
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CORS_ORIGINS = _parse_cors_origins(os.getenv("CORS_ORIGINS", "*"))
ALLOW_ALL_ORIGINS = CORS_ORIGINS == ["*"]

# Rembg / preprocessing configuration
# REMBG_MODEL: which rembg model to use (e.g. "u2net", "u2netp", "u2net_human_seg")
REMBG_MODEL = os.getenv("REMBG_MODEL", "u2netp")

# ENABLE_PLATE_CROP: whether to run expensive HoughCircles-based crop_to_plate
ENABLE_PLATE_CROP = os.getenv("ENABLE_PLATE_CROP", "false").lower() == "true"
