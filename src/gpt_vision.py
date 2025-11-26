"""Single-call GPT-4o-mini vision service for food analysis."""

import base64
from typing import Dict, Any
from openai import OpenAI
import logging

from src.config import OPENAI_API_KEY

logger = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
Ты — эксперт по распознаванию еды на фотографии и расчёту КБЖУ.
Не объясняй reasoning.
Не описывай шаги.
Не используй Chain-of-Thought.
Верни только JSON строго по структуре.
Если не уверен — дай реалистичную оценку.

Структура ответа:
{
  "products": [
    {
      "product_name": "название",
      "quantity_g": 123,
      "confidence": 0.85,
      "kcal": 45.5,
      "protein": 5.2,
      "fat": 3.1,
      "carbs": 2.0
    }
  ],
  "totals": {
    "kcal": 552.25,
    "protein": 37.4,
    "fat": 29.7,
    "carbs": 39.4
  }
}

Все числовые значения должны быть числами, не строками.
Не добавляй extra поля.
Только JSON.
"""

def analyze_food(image_path: str) -> Dict[str, Any]:
    """
    Analyze food photo with GPT-4o-mini using single vision call.
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
                        "text": "Проанализируй эту фотографию еды. Определи все ингредиенты с точностью до граммов. Для каждого ингредиента посчитай kcal, protein(g), fat(g), carbs(g). Верни JSON строго по указанной структуре."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                    }
                ]
            }
        ]

        logger.info(f"Sending {len(b64_img)/1024:.1f}kb image to GPT-4o-mini")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=800,
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
