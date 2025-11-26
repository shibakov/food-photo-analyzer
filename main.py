from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List, Optional
from pydantic import BaseModel
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
    meal_type: Optional[str] = Form(None),
):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    # читаем файл в память
    image_bytes = await image.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # 🔥 1) Загружаем файл в OpenAI (единственный поддерживаемый Vision-путь)
    uploaded = client.files.create(
        file=image_bytes,
        purpose="vision"
    )
    file_id = uploaded.id

    # 🔥 2) Формируем промт
    prompt = build_prompt(user_id=user_id, meal_type=meal_type)

    # 🔥 3) Делаем Vision запрос с file_id
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
                        "file_id": file_id
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

    # 🔥 4) Парсим JSON
    products, totals = parse_model_output(raw)

    # 🔥 5) (опционально) удаляем файл из OpenAI
    try:
        client.files.delete(file_id)
    except:
        pass  # неважно, пусть живёт

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
- Определи все видимые компоненты блюда.
- Для каждого компонента определи:
  - название продукта на русском
  - примерный вес (целое число граммов)
  - уверенность (0–1)
- Верни точный JSON.

Пример формата:

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

Не добавляй ничего вне JSON.
"""


def parse_model_output(raw: str):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
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
