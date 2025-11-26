"""Main FastAPI application."""

import base64
import logging
import os
import sys
import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.services import analyze_image_with_vision, refine_products
from src.image_preprocess import preprocess_image
from src.gpt_vision import analyze_food

# -----------------------------------
# Инициализация приложения
# -----------------------------------

app = FastAPI()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# -----------------------------------
# CORS
# -----------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # локальный фронт
        "http://localhost:5173",
        # прод-миниапп
        "https://my-miniapp-production.up.railway.app",
        # телега (на будущее)
        "https://web.telegram.org",
        "https://app.telegram.org",
        "https://t.me",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Если хочешь, можно вообще открыть для всех:
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# -----------------------------------
# Тех. эндпоинты
# -----------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------------
# /analyze — старый пайплайн
# -----------------------------------

@app.post("/analyze")
async def analyze_photo(image: UploadFile = File(None)):
    if not image:
        raise HTTPException(422, "Image field is required")

    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(422, "Unsupported format (use jpeg/png)")

    start_time = time.time()
    logging.info(f"[PIPELINE] Starting /analyze endpoint for file: {image.filename}")

    img_bytes = await image.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    try:
        # STEP 1 — VISION RECOGNITION
        logging.info("[PIPELINE] Step 1: Starting vision recognition")
        vision_json = analyze_image_with_vision(img_b64, image.content_type)
        products = vision_json["products"]
        logging.info(f"[PIPELINE] Step 1: Vision completed, found {len(products)} products")

        # STEP 2 — REFINEMENT
        logging.info("[PIPELINE] Step 2: Starting refinement")
        refine_json = refine_products(products)
        logging.info("[PIPELINE] Step 2: Refinement completed")

        # Add processing time
        end_time = time.time()
        refine_json["processing_time_ms"] = round((end_time - start_time) * 1000, 2)
        logging.info(f"[PIPELINE] /analyze completed successfully, time: {refine_json['processing_time_ms']}ms")

        return refine_json
    except Exception as e:
        logging.exception("Error in /analyze")
        raise HTTPException(422, f"Analysis error: {str(e)}")


# -----------------------------------
# /recognize — новый быстрый пайплайн
# -----------------------------------

@app.post("/recognize")
async def recognize_food(image: UploadFile = File(None)):
    """
    Optimized endpoint: preprocess → single GPT-4o-mini vision call.
    """
    if not image:
        raise HTTPException(422, "Image field is required")

    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(422, "Unsupported format (use jpeg/png)")

    total_start = time.time()
    logging.info(f"[PIPELINE] Starting /recognize endpoint for file: {image.filename}")

    try:
        # Готовим временную папку
        tmp_dir = "/tmp/preprocess"
        os.makedirs(tmp_dir, exist_ok=True)

        temp_input = os.path.join(tmp_dir, f"input_{time.time()}.jpg")

        # Сохраняем файл
        content = await image.read()
        with open(temp_input, "wb") as f:
            f.write(content)
        logging.info(f"[PIPELINE] Saved input image: {temp_input}")

        # Препроцесс
        preprocess_start = time.time()
        logging.info("[PIPELINE] Step 1: Starting image preprocessing")
        processed_path = preprocess_image(temp_input)
        preprocess_time = time.time() - preprocess_start
        logging.info(f"[PIPELINE] Step 1: Preprocessing completed in {round(preprocess_time * 1000, 2)}ms")

        # GPT-анализ
        gpt_start = time.time()
        logging.info("[PIPELINE] Step 2: Starting GPT analysis")
        result_json = analyze_food(processed_path)
        gpt_time = time.time() - gpt_start
        logging.info(f"[PIPELINE] Step 2: GPT analysis completed in {round(gpt_time * 1000, 2)}ms, found {len(result_json.get('products', []))} products")

        # Тайминги
        total_time = time.time() - total_start
        result_json["processing_times"] = {
            "preprocessing_ms": round(preprocess_time * 1000, 2),
            "gpt_ms": round(gpt_time * 1000, 2),
            "total_ms": round(total_time * 1000, 2),
        }

        logging.info(f"[PIPELINE] /recognize completed successfully, total time: {round(total_time * 1000, 2)}ms")

        # Чистим временный файл
        try:
            os.remove(temp_input)
        except Exception:
            pass

        return result_json

    except Exception as e:
        logging.exception("Error in /recognize")
        raise HTTPException(422, f"Recognition error: {str(e)}")
