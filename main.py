from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from openai import OpenAI
import os, json, requests, base64

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
IMGUR_CLIENT_ID = os.getenv("IMGUR_CLIENT_ID")


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

    # читаем байты
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # 1️⃣ Загружаем на Imgur → получаем URL
    image_url = upload_to_imgur(image_bytes)

    # 2️⃣ Промт
    prompt = build_prompt(user_id, meal_type)

    # 3️⃣ Вызов Vision
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"image_url": image_url},
            {"text": prompt}
        ]
    )

    raw = response.output_text
    products, totals = parse_model_output(raw)

    return AnalyzeResponse(
        products=products,
        total_kcal=totals.get("kcal"),
        total_protein=totals.get("protein"),
        total_fat=totals.get("fat"),
        total_carbs=totals.get("carbs"),
    )


def upload_to_imgur(file_bytes: bytes) -> str:
    response = requests.post(
        "https://api.imgur.com/3/image",
        headers={"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"},
        data={"image": base64.b64encode(file_bytes)}
    )
    data = response.json()
    return data["data"]["link"]


def build_prompt(user_id: Optional[str], meal_type: Optional[str]) -> str:
    return """
Верни JSON:
{
  "products": [...],
  "totals": {...}
}
"""


def parse_model_output(raw: str):
    data = json.loads(raw)

    products = [
        ProductItem(
            product_name=p["product_name"],
            quantity_g=float(p["quantity_g"]),
            confidence=float(p.get("confidence", 0.5))
        )
        for p in data.get("products", [])
    ]

    return products, data.get("totals", {})
