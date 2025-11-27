"""Fallback GPT vision service for food analysis.

This module is now used ONLY as a backup when the local YOLO ONNX detector
in src.vision_pipeline fails to find any objects.

Design goals:
- super short prompt (minimal tokens)
- single vision call
- always return stable JSON (response_format=json_object)
"""

import base64
import json
import logging
from typing import Any, Dict

from src.openai_client import get_openai_client
from src.prompts import SYSTEM_PROMPT
from src.config import GPT_MODEL

logger = logging.getLogger(__name__)


def _client():
    return get_openai_client()


def _get_vision_model_name() -> str:
    """
    Resolve which GPT vision model to use for fallback analysis.

    Controlled via GPT_MODEL env:
    - "gpt-4o-mini" (default)
    - "gpt-4o"
    """
    model = (GPT_MODEL or "gpt-4o-mini").strip()
    return model or "gpt-4o-mini"


def analyze_food(image_path: str) -> Dict[str, Any]:
    """
    Lightweight fallback: analyze food photo with GPT vision using a single call.

    Used ONLY when the fast local detector-based pipeline fails
    to find any objects.

    Returns: JSON with products and totals.
    """
    with open(image_path, "rb") as f:
        img_data = f.read()

    b64_img = base64.b64encode(img_data).decode("utf-8")

    # Минимальный пользовательский промпт: только задача и целевая JSON-структура.
    prompt_text = (
        'Проанализируй еду на изображении и верни ТОЛЬКО JSON вида:\n'
        '{\n'
        '  "products": [\n'
        '    {\n'
        '      "product_name": "",\n'
        '      "quantity_g": 0,\n'
        '      "kcal": 0,\n'
        '      "protein": 0,\n'
        '      "fat": 0,\n'
        '      "carbs": 0\n'
        '    }\n'
        '  ],\n'
        '  "totals": {\n'
        '    "kcal": 0,\n'
        '    "protein": 0,\n'
        '    "fat": 0,\n'
        '    "carbs": 0\n'
        '  }\n'
        '}\n'
        "Никакого текста вне JSON."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                },
            ],
        },
    ]

    model_name = _get_vision_model_name()
    logger.info(
        "Fallback GPT-vision: sending image to model=%s (size_kb=%.1f)",
        model_name,
        len(b64_img) / 1024,
    )

    response = _client().chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=500,
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    result = json.loads(content)

    # Basic shape validation
    if "products" not in result:
        result["products"] = []
    if "totals" not in result:
        result["totals"] = {"kcal": 0, "protein": 0, "fat": 0, "carbs": 0}

    logger.info(
        "Fallback GPT-vision result: products=%s",
        len(result.get("products", [])),
    )
    return result
