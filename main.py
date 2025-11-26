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
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    prompt = """
Распознай еду на фото.
Верни строгий JSON:
{
  "products": [...],
  "totals": {...}
}
"""

    # Правильный вызов OpenAI Vision API
    response = client.chat.completions.create(
        model="gpt-4o",  # Используем GPT-4o для поддержки изображений
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ]
    )

    # Извлекаем текст из ответа
    result_text = response.choices[0].message.content

    if not result_text:
        return {"error": "model returned no text"}

    return {"result": result_text}
