from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import base64
import os
from openai import OpenAI
import json

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ProductItem(BaseModel):
    product_name: str
    quantity_g: float
    confidence: float

class AnalyzeResponse(BaseModel):
    products: List[ProductItem]
    total_kcal: Optional[float] = None
    total_protein: Optional[float] = None
    total_fat: Optional[float] = None
    total_carbs: Optional[float] = None

@app.post("/analyze_photo", response_model=AnalyzeResponse)
async def analyze_photo(
    image: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    meal_type: Optional[str] = Form(None),  # Breakfast / Lunch / etc
):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    # читаем файл в память
    image_bytes = await image.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # вызываем GPT Vision (модель назови ту, что используешь — 4.1/4o/5)
    prompt = build_prompt(user_id=user_id, meal_type=meal_type)

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Ты — ассистент-нутрициолог, отвечай только JSON."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "name": image.filename,
                            "mime_type": image.content_type,
                            "data": image_bytes
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        temperature=0.2,
    )





    raw = completion.choices[0].message.content
    # здесь нужно распарсить JSON из raw
    products, totals = parse_model_output(raw)

    return AnalyzeResponse(
        products=products,
        total_kcal=totals.get("kcal"),
        total_protein=totals.get("protein"),
        total_fat=totals.get("fat"),
        total_carbs=totals.get("carbs"),
    )

def build_prompt(user_id: Optional[str], meal_type: Optional[str]) -> str:
    return f"""
Распознай еду на фото.

Требования:
- Определи все видимые компоненты блюда (даже если это салат, рагу или набор продуктов).
- Для каждого компонента определи:
  - краткое название продукта на русском
  - примерный вес в граммах (целое число)
  - уверенность от 0 до 1
- Точность по составу и весу — около 80–90%. Лучше дай приблизительную оценку, чем пропусти компонент.
- Если сомневаешься между похожими продуктами, выбери самый типичный вариант для домашней еды.

Верни строгий JSON такого вида:

{{
  "products": [
    {{
      "product_name": "курица отварная",
      "quantity_g": 150,
      "confidence": 0.87
    }}
  ],
  "totals": {{
    "kcal": 500,
    "protein": 40,
    "fat": 15,
    "carbs": 45
  }}
}}

Не добавляй никаких пояснений, комментариев или текста вне JSON.
"""

def parse_model_output(raw: str):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # можно попробовать вычистить мусор/обрезать до фигурных скобок, но на MVP просто бросаем ошибку
        raise ValueError("Model returned non-JSON response")

    products = [
        ProductItem(
            product_name=item["product_name"],
            quantity_g=float(item["quantity_g"]),
            confidence=float(item.get("confidence", 0.5))
        )
        for item in data.get("products", [])
    ]

    totals = data.get("totals", {}) or {}

    return products, totals
