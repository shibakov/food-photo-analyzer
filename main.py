from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
import base64
import os
import json
import re

app = FastAPI()

# Add CORS middleware to handle preflight OPTIONS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Utility: clean JSON -----------
def extract_json(text: str):
    """
    Чистит Markdown, ```json, комментарии.
    Возвращает только JSON-объект.
    """
    if not text:
        raise ValueError("Empty model output")

    # убираем ```json ... ```
    text = re.sub(r"```.*?```", "", text, flags=re.S)

    # ищем первую { и последнюю }
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object detected")

    cleaned = text[start:end+1]

    return json.loads(cleaned)


# ---------- PROMPTS ----------------------

VISION_PROMPT = """
You are a food analysis expert by photo.

1) Identify ALL ingredients (even small ones: sauce, cheese, lettuce leaves).
2) If there is a HAND in the photo — use it as a scale.
   - Average hand width ≈ 9.5 cm.
   - Calibrate weight more accurately.
3) If no hand — estimate based on standard sizes.

Return CLEAN JSON strictly in this format:

{
  "products": [
    {"product_name": "...", "quantity_g": 123, "confidence": 0.85}
  ],
  "meta": {
    "hand_detected": true/false
  }
}

⚠️ No text, no markdown, no comments.
"""

REFINE_PROMPT = """
You are a nutritionist. Refine weights and calculate nutrition facts (kcal, protein, fat, carbs).

Input — list of products:
{products_list}

Return JSON strictly in this format:

{
  "products": [
    {"product_name": "...", "quantity_g": 123, "confidence": 0.9}
  ],
  "totals": {
    "kcal": 0,
    "protein": 0,
    "fat": 0,
    "carbs": 0
  }
}

⚠️ No text outside JSON.
"""


# ---------- Main endpoint -----------------

@app.post("/analyze")
async def analyze_photo(image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "Unsupported format (use jpeg/png)")

    img_bytes = await image.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # ====== STEP 1 — VISION RECOGNITION ======
    try:
        vision = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:{image.content_type};base64,{img_b64}"}}
                    ]
                }
            ]
        )
    except Exception as e:
        return {"error": "vision_api_error", "details": str(e)}

    try:
        vision_json = extract_json(vision.choices[0].message.content)
    except Exception as e:
        return {"error": "vision_json_parse_error", "details": str(e)}

    # Check required keys
    if "products" not in vision_json:
        return {"error": "vision_no_products", "details": "No products in vision response"}

    # ====== STEP 2 — REFINEMENT (weights + macros) ======
    products_list = json.dumps(vision_json["products"], ensure_ascii=False)

    try:
        refine = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": REFINE_PROMPT.replace("{products_list}", products_list)
                }
            ]
        )
    except Exception as e:
        return {"error": "refine_api_error", "details": str(e)}

    try:
        refine_json = extract_json(refine.choices[0].message.content)
    except Exception as e:
        return {"error": "refine_json_parse_error", "details": str(e)}

    # Check required keys
    if "products" not in refine_json or "totals" not in refine_json:
        return {"error": "refine_invalid_response", "details": "Missing products or totals"}

    # ====== DONE ======
    return refine_json
