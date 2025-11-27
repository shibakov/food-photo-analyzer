"""Single-call GPT-4o vision service for food analysis."""

import base64
import json
import logging
import re
from typing import Any, Dict

from src.openai_client import get_openai_client
from src.prompts import SYSTEM_PROMPT
from src.config import GPT_MODEL, USE_LOCAL_FAST_MODEL

logger = logging.getLogger(__name__)


def _client():
    return get_openai_client()


def _get_vision_model_name() -> str:
    """
    Resolve which GPT vision model to use for analysis.

    Controlled via GPT_MODEL env:
    - "gpt-4o" (default)
    - "gpt-4o-mini"
    """
    model = (GPT_MODEL or "gpt-4o").strip()
    return model or "gpt-4o"


def _run_local_fast_model(image_path: str) -> Dict[str, Any]:
    """
    Local fast-model fallback placeholder.

    TODO:
    - plug MobileNet/EfficientNet-Lite classifier (via ONNX)
    - plug Tesseract OCR for package text
    - combine signals into structured products/totals

    For now this returns a structured stub so that the API shape is stable.
    """
    logger.warning(
        "Local fast model fallback is enabled, but offline pipeline is not "
        "implemented yet. Returning empty stub result."
    )
    return {
        "products": [],
        "totals": {"kcal": 0, "protein": 0, "fat": 0, "carbs": 0},
        "meta": {
            "local_fast_model_used": True,
            "offline_model": "not_implemented",
        },
    }


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
    Analyze food photo with GPT vision model using single call.

    Preprocessed image should be PNG/JPEG with background already optimized
    (cropped region only / GrabCut / rembg).

    Returns: JSON with products and totals including individual macros.
    """
    with open(image_path, "rb") as f:
        img_data = f.read()

    b64_img = base64.b64encode(img_data).decode("utf-8")

    # Minimal task description: structure + KБЖУ, no extra text.
    prompt_text = (
        "Проанализируй фото блюда и верни JSON со списком продуктов и суммарными КБЖУ.\n\n"
        '{\n  "products": [\n    {\n      "product_name": "",\n'
        '      "quantity_g": 0,\n      "kcal": 0,\n      "protein": 0,\n'
        '      "fat": 0,\n      "carbs": 0\n    }\n  ],\n'
        '  "totals": {\n    "kcal": 0,\n    "protein": 0,\n'
        '    "fat": 0,\n    "carbs": 0\n  }\n}\n\n'
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
                    "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                },
            ],
        },
    ]

    model_name = _get_vision_model_name()
    logger.info(
        "Sending %.1fkb image to model=%s",
        len(b64_img) / 1024,
        model_name,
    )

    try:
        response = _client().chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1500,
            temperature=0,
            response_format={"type": "json_object"},
        )

        result_text = response.choices[0].message.content or ""
        logger.info("GPT response received, length: %s", len(result_text))

        parsed = _parse_json_from_text(result_text)
        logger.info("Parsed JSON with %s products", len(parsed.get("products", [])))
        return parsed

    except Exception as e:
        logger.error("Food analysis via OpenAI failed: %s", e)
        if USE_LOCAL_FAST_MODEL:
            logger.info(
                "Falling back to local fast model (USE_LOCAL_FAST_MODEL=true). "
                "Image path: %s",
                image_path,
            )
            return _run_local_fast_model(image_path)
        raise
