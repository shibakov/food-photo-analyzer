import os


def _parse_cors_origins(raw: str) -> list[str]:
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CORS_ORIGINS = _parse_cors_origins(os.getenv("CORS_ORIGINS", "*"))
ALLOW_ALL_ORIGINS = CORS_ORIGINS == ["*"]
