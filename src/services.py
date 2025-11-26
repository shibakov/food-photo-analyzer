"""Services for OpenAI interactions."""

import json
import logging
from typing import Dict, Any

from openai import OpenAI

from src.config import OPENAI_API_KEY
from src.prompts import VISION_PROMPT, REFINE_PROMPT
from src.utils import extract_json

logger = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)


def analyze_image_with_vision(img_b64: str, content_type: str) -> Dict[str, Any]:
    """Analyze image using vision model."""
    logger.info(f"Analyzing image with content_type: {content_type}, b64 len: {len(img_b64)}")
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{img_b64}"}}
                ]
            }
        ]
    )
    vision_text = response.choices[0].message.content
    logger.info(f"Vision raw response: {vision_text}")
    vision_json = extract_json(vision_text)
    logger.info(f"Vision parsed JSON: {vision_json}")

    if "products" not in vision_json:
        raise ValueError("No products in vision response")

    return vision_json


def refine_products(products: list) -> Dict[str, Any]:
    """Refine products with nutritionist model."""
    products_list = json.dumps(products, ensure_ascii=False)
    logger.info(f"Refining products: {products_list}")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": REFINE_PROMPT.replace("{products_list}", products_list)
            }
        ]
    )
    refine_text = response.choices[0].message.content
    logger.info(f"Refine raw response: {refine_text}")
    refine_json = extract_json(refine_text)
    logger.info(f"Refine parsed JSON: {refine_json}")

    if "products" not in refine_json or "totals" not in refine_json:
        raise ValueError("Missing products or totals in refine response")

    return refine_json
