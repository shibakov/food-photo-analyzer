"""Main FastAPI application."""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
import time
import logging
import os

from src.services import analyze_image_with_vision, refine_products
from src.image_preprocess import preprocess_image
from src.gpt_vision import analyze_food

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- CORS: максимально разрешённый вариант для отладки фронта ---
app.add_middleware(
     CORSMiddleware,
     allow_origins=[
         "http://localhost:5173",
         "https://web.telegram.org",
         "https://*.web.telegram.org",
         "https://t.me",
          "my-miniapp-production.up.railway.app",
         "https://food-photo-analyzer-production.up.railway.app"
     ],
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
     expose_headers=["*"],
)

@app.options("/{path:path}")
async def preflight_handler(path: str):
    return {}

# ----------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_photo(image: UploadFile = File(None)):
    if not image:
        raise HTTPException(422, "Image field is required")

    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(422, "Unsupported format (use jpeg/png)")
    
    start_time = time.time()
    
    img_bytes = await image.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    try:
        # STEP 1 — VISION RECOGNITION
        vision_json = analyze_image_with_vision(img_b64, image.content_type)
        products = vision_json["products"]

        # STEP 2 — REFINEMENT
        refine_json = refine_products(products)

        # Add processing time
        end_time = time.time()
        refine_json["processing_time_ms"] = round((end_time - start_time) * 1000, 2)

        return refine_json
    except Exception as e:
        logging.exception("Error in /analyze")
        raise HTTPException(422, f"Analysis error: {str(e)}")


@app.post("/recognize")
async def recognize_food(image: UploadFile = File(None)):
    """Optimized endpoint: preprocess → single GPT-4o-mini vision call."""
    if not image:
        raise HTTPException(422, "Image field is required")

    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(422, "Unsupported format (use jpeg/png)")

    total_start = time.time()

    try:
        # Save uploaded image temporarily
        temp_input = f"/tmp/preprocess/input_{time.time()}.jpg"
        content = await image.read()
        with open(temp_input, 'wb') as f:
            f.write(content)

        # Preprocess image
        preprocess_start = time.time()
        processed_path = preprocess_image(temp_input)
        preprocess_time = time.time() - preprocess_start

        # GPT analysis
        gpt_start = time.time()
        result_json = analyze_food(processed_path)
        gpt_time = time.time() - gpt_start

        # Add timing info
        total_time = time.time() - total_start
        result_json["processing_times"] = {
            "preprocessing_ms": round(preprocess_time * 1000, 2),
            "gpt_ms": round(gpt_time * 1000, 2),
            "total_ms": round(total_time * 1000, 2),
        }

        # Cleanup temp file
        try:
            os.remove(temp_input)
        except Exception:
            pass

        return result_json

    except Exception as e:
        logging.exception("Error in /recognize")
        raise HTTPException(422, f"Recognition error: {str(e)}")
