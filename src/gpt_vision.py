"""Single-call GPT-4o vision service for food analysis."""

import base64
import json
import logging
import re
from typing import Any, Dict

from src.openai_client import get_openai_client
from src.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _client():
    return get_openai_client()


def analyze_food(image_path: str) -> Dict[str, Any]:
    """
    Analyze food photo with GPT-4o using single vision call.
    Preprocessed image should be PNG/JPEG with transparent background preferred.

    Returns: JSON with products and totals including individual macros.
    """
    try:
        with open(image_path, "rb") as f:
            img_data = f.read()

        b64_img = base64.b64encode(img_data).decode("utf-8")

        prompt_text = (
            "Проанализируй фото и верни JSON строго по структуре:\n\n"
            '{\n  "products": [\n    {\n      "product_name": "",\n'
            '      "quantity_g": 0,\n      "kcal": 0,\n      "protein": 0,\n'
            '      "fat": 0,\n      "carbs": 0\n    }\n  ],\n'
            '  "totals": {\n    "kcal": 0,\n    "protein": 0,\n'
            '    "fat": 0,\n    "carbs": 0\n  }\n}\n\n'
            "Не добавляй никакого текста кроме JSON."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}  # type: ignore
                ],
            },
        ]

        logger.info("Sending %.1fkb image to GPT-4o", len(b64_img) / 1024)

        response = _client().chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1500,
            temperature=0,
        )

        result_text = response.choices[0].message.content or ""
        logger.info("GPT response received, length: %s", len(result_text))

        clean = re.sub(r"```.*?```", "", result_text, flags=re.DOTALL)
        start_idx = clean.find("{")
        end_idx = clean.rfind("}") + 1

        if start_idx == -1 or end_idx <= start_idx:
            raise ValueError("No JSON found in response")

        parsed = json.loads(clean[start_idx:end_idx])
        logger.info("Successfully parsed JSON with %s products", len(parsed.get("products", [])))
        return parsed

    except Exception as e:
        logger.error("Food analysis failed: %s", e)
        raise
