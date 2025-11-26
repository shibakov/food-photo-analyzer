from fastapi import FastAPI, UploadFile, File
from openai import OpenAI
import base64
import os

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze_photo")
async def analyze_photo(image: UploadFile = File(...)):
    # читаем картинку в байты
    img_bytes = await image.read()

    prompt = """
Распознай еду на фото.
Верни строгий JSON:
{
  "products": [...],
  "totals": {...}
}
"""

    # ====== Главный корректный вызов Responses API ======
    response = client.responses.create(
        model="gpt-4o-mini",     # можешь заменить
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image": img_bytes   # <<<<< ВОТ ЭТО ПРАВИЛЬНО
                    }
                ]
            }
        ]
    )

    # безопасное извлечение текста
    result_text = getattr(response, "output_text", None)

    if not result_text:
        return {"error": "model returned no text", "raw": response.model_dump()}

    return {"result": result_text}
