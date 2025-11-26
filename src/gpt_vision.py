"""Single-call GPT-4o vision service for food analysis."""

import base64
from typing import Dict, Any
from openai import OpenAI
import logging

from src.config import OPENAI_API_KEY
from src.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

def analyze_food(image_path: str) -> Dict[str, Any]:
    """
    Analyze food photo with GPT-4o using single vision call.
    Preprocessed image should be PNG with transparent background.

    Returns: JSON with products and totals including individual macros.
    """
    try:
        # Read and encode processed image
        with open(image_path, 'rb') as f:
            img_data = f.read()

        b64_img = base64.b64encode(img_data).decode('utf-8')

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Проанализируй фото и верни JSON строго по структуре:\n\n{\n  \"products\": [\n    {\n      \"product_name\": \"\",\n      \"quantity_g\": 0,\n      \"kcal\": 0,\n      \"protein\": 0,\n      \"fat\": 0,\n      \"carbs\": 0\n    }\n  ],\n  \"totals\": {\n    \"kcal\": 0,\n    \"protein\": 0,\n    \"fat\": 0,\n    \"carbs\": 0\n  }\n}\n\nНе добавляй никакого текста кроме JSON."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}"}
                    }
                ]
            }
        ]

        logger.info(f"Sending {len(b64_img)/1024:.1f}kb image to GPT-4o")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1500,
            temperature=0,
        )

        result_text = response.choices[0].message.content
        logger.info(f"GPT response received, length: {len(result_text)}")

        # Try to extract JSON from response
        import json
        import re

        # Clean response (remove markdown if present)
        result_text = re.sub(r'```.*?```', '', result_text, flags=re.DOTALL)

        # Find JSON object
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in response")

        clean_json = result_text[start_idx:end_idx]

        try:
            parsed = json.loads(clean_json)
            logger.info(f"Successfully parsed JSON with {len(parsed.get('products', []))} products")
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}, clean response: {clean_json[:500]}...")
            raise ValueError(f"Invalid JSON: {str(e)}")

    except Exception as e:
        logger.error(f"Food analysis failed: {str(e)}")
        raise
