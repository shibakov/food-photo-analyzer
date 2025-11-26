import logging
from functools import lru_cache

from openai import OpenAI

from src.config import OPENAI_API_KEY

logger = logging.getLogger(__name__)


@lru_cache
def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    logger.info("Initializing OpenAI client")
    return OpenAI(api_key=OPENAI_API_KEY)
