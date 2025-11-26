from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI
import base64
import os
import json

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze_photo")
async def analyze_photo(image: UploadFile = File(...)):
    # проверяем тип файла
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    # читаем картинку → base64
    img_bytes = await image.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # улучшенный prompt
    prompt = """
Ты — профессиональный нутрициолог, анализирующий еду по фото.

Используй этот скрытый план (НЕ выводи его):
1) Определи сцену и контекст.
2) Определи объекты, относящиеся к еде.
3) Если блюдо комплексное (бургер, салат, рагу) — выдели ингредиенты.
4) Используй размеры руки, тарелки, упаковок как масштаб.
5) Определи примерный вес каждого ингредиента.
6) Определи уверенность от 0 до 1.
7) Определи суммарные КБЖУ.
8) Проверь JSON перед выводом.

Видимые или скрытые ингредиенты НЕ пропускай.
Если часть блюда внутри булки, под соусом или скрыта — оцени типичную структуру блюда.

Выводи СТРОГО JSON:
{
  "products": [
    {
      "product_name": "string",
      "quantity_g": number,
      "confidence": number
    }
  ],
  "totals": {
    "kcal": number,
    "protein": number,
    "fat": number,
    "carbs": number
  }
}
"""

    # корректный вызов OpenAI Vision
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image.content_type};base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
    )

    raw = response.choices[0].message.content

    # преобразуем в объект
    try:
        data = json.loads(raw)
    except Exception:
        return {"error": "Модель вернула не-JSON", "raw": raw}

    return data
