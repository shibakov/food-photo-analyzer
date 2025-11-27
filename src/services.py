"""Services for OpenAI interactions."""

import json
import logging
from typing import Any, Dict

from src.openai_client import get_openai_client
from src.prompts import VISION_PROMPT, REFINE_PROMPT
from src.utils import extract_json
from src.config import GPT_MODEL

logger = logging.getLogger(__name__)


def _client():
    return get_openai_client()


def analyze_image_with_vision(img_b64: str, content_type: str) -> Dict[str, Any]:
    """Analyze image using vision model."""
    vision_model = (GPT_MODEL or "gpt-4o-mini").strip() or "gpt-4o-mini"
    logger.info(
        "Analyzing image with content_type=%s, b64_len=%s, model_used=%s",
        content_type,
        len(img_b64),
        vision_model,
    )
    response = _client().chat.completions.create(
        model=vision_model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{content_type};base64,{img_b64}"},
                    },
                ],
            }
        ],
    )
    vision_text = response.choices[0].message.content
    logger.info("Vision raw response: %s", vision_text)
    vision_json = extract_json(vision_text)
    logger.info("Vision parsed JSON: %s", vision_json)

    if "products" not in vision_json:
        raise ValueError("No products in vision response")

    return vision_json


def refine_products(products: list) -> Dict[str, Any]:
    """Refine products with nutritionist model."""
    products_list = json.dumps(products, ensure_ascii=False)
    logger.info("Refining products with model=gpt-4o-mini: %s", products_list)
    response = _client().chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": REFINE_PROMPT.replace("{products_list}", products_list),
            }
        ],
    )
    refine_text = response.choices[0].message.content
    logger.info("Refine raw response: %s", refine_text)
    refine_json = extract_json(refine_text)
    logger.info("Refine parsed JSON: %s", refine_json)

    if "products" not in refine_json or "totals" not in refine_json:
        raise ValueError("Missing products or totals in refine response")

    return refine_json
