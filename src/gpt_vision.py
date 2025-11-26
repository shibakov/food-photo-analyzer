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


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    """
    Try several strategies to extract JSON from model text output.
    Raises ValueError if nothing valid is found.
    """
    if not text:
        raise ValueError("Empty response from model")

    # 1) Прямая попытка распарсить как JSON целиком
    try:
        parsed = json.loads(text.strip())
        return parsed
    except json.JSONDecodeError:
        pass

    # 2) Убрать ```...``` и снова попытаться
    cleaned = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError:
        pass

    # 3) Вырезать первый JSON-блок по { ... }
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}") + 1
    if start_idx != -1 and end_idx > start_idx:
        snippet = cleaned[start_idx:end_idx]
        try:
            parsed = json.loads(snippet)
            return parsed
        except json.JSONDecodeError:
            pass

    # 4) Регекс по всему тексту
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        snippet = json_match.group(0)
        try:
            parsed = json.loads(snippet)
            return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in model response")


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
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                    },
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

        parsed = _parse_json_from_text(result_text)
        logger.info("Parsed JSON with %s products", len(parsed.get("products", [])))
        return parsed

    except Exception as e:
        logger.error("Food analysis failed: %s", e)
        raise
