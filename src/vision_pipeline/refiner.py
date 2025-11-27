import json

from src.openai_client import get_openai_client


class GPTRefiner:
    """
    GPT-based refinement of raw detections into concrete products and macros.

    Input:
        detections: list of objects with fields like
            - label (class id as string)
            - confidence
            - size (relative bbox area)
        food_schema: dict with nutrition per 100g for known products.

    Output (strict JSON, enforced via response_format=json_object):
        {
          "products": [...],
          "totals": {...}
        }
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = get_openai_client()
        self.model = model

    def refine(self, detections, food_schema):
        prompt = (
            "У тебя есть список объектов еды с относительными размерами.\n"
            "Определи продукт, оцени его вес в граммах, рассчитай БЖУ.\n"
            "Используй справочник нутриентов (значения на 100 г):\n"
            f"{json.dumps(food_schema, ensure_ascii=False)}\n\n"
            "Данные детектора (label, confidence, size):\n"
            f"{json.dumps(detections, ensure_ascii=False)}\n\n"
            "Верни строго JSON:\n"
            "{products: [...], totals: {...}}"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0,
        )

        # response_format=json_object гарантирует валидный JSON-объект
        return json.loads(resp.choices[0].message.content)
