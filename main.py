from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from openai import OpenAI
import os
import json

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/health")
async def health():
    return {"status": "ok"}


# ================== MODELS ======================

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


# ================== ENDPOINT ======================

@app.post("/analyze_photo", response_model=AnalyzeResponse)
async def analyze_photo(image: UploadFile = File(...)):

    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "Unsupported image format")

    # 1) Читаем файл
    img_bytes = await image.read()

    # 2) Загружаем в OpenAI CDN
    uploaded_file = client.files.create(
        file=img_bytes,
        purpose="vision"
    )

    file_id = uploaded_file.id

    # 3) Промт
    prompt = """
Распознай еду на фото.

Верни строгий JSON вида:

{
  "products": [
    {
      "product_name": "курица отварная",
      "quantity_g": 150,
      "confidence": 0.87
    }
  ],
  "totals": {
    "kcal": 500,
    "protein": 40,
    "fat": 15,
    "carbs": 45
  }
}

Без текста вне JSON.
"""

    # 4) GPT-4o vision через /responses
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "system",
                "content": "Ты — ассистент-нутрициолог. Отвечай только JSON."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_file_id": file_id
                    }
                ]
            }
        ]
    )

    raw = response.output_text

    # 5) JSON parse
    try:
        data = json.loads(raw)
    except Exception:
        raise HTTPException(500, f"Invalid JSON from model: {raw[:200]}")

    # 6) Продукты
    products = [
        ProductItem(
            product_name=item["product_name"],
            quantity_g=float(item["quantity_g"]),
            confidence=float(item.get("confidence", 0.5)),
        )
        for item in data.get("products", [])
    ]

    totals = data.get("totals", {}) or {}

    return AnalyzeResponse(
        products=products,
        total_kcal=totals.get("kcal"),
        total_protein=totals.get("protein"),
        total_fat=totals.get("fat"),
        total_carbs=totals.get("carbs"),
    )
