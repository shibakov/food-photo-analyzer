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

    # читаем файл
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # prompt
    prompt = build_prompt(user_id, meal_type)

    # 🔥 НОВЫЙ Vision вызов через Responses API (работает всегда)
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"image": image_bytes},
            {"text": prompt}
        ]
    )

    raw = response.output_text

    # парсим модель
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
- Определи каждый компонент блюда.
- Верни JSON строго в формате:

{{
  "products": [
    {{
      "product_name": "...",
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

Только JSON. Без текста.
"""


def parse_model_output(raw: str):
    data = json.loads(raw)

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
