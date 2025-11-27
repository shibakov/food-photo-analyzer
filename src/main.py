"""Main FastAPI application."""

import base64
import os
import time
import asyncio

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import CORS_ORIGINS, ALLOW_ALL_ORIGINS, REMBG_MODEL, ENABLE_PLATE_CROP
from src.services import analyze_image_with_vision, refine_products
from src.image_preprocess import preprocess_image
from src.gpt_vision import analyze_food

# -----------------------------------
# Инициализация приложения
# -----------------------------------

app = FastAPI()

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# -----------------------------------
# CORS
# -----------------------------------

if ALLOW_ALL_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    total_start = time.time()
    logging.info(f"[PIPELINE] Starting /analyze endpoint for file: {image.filename}")

    img_bytes = await image.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    try:
        # STEP 1 — VISION RECOGNITION
        vision_start = time.time()
        logging.info("[PIPELINE] Step 1: Starting vision recognition")
        vision_json = analyze_image_with_vision(img_b64, image.content_type)
        products = vision_json["products"]
        vision_time = time.time() - vision_start
        logging.info(f"[PIPELINE] Step 1: Vision completed in {round(vision_time * 1000, 2)}ms, found {len(products)} products")

        # STEP 2 — REFINEMENT
        refine_start = time.time()
        logging.info("[PIPELINE] Step 2: Starting refinement")
        refine_json = refine_products(products)
        refine_time = time.time() - refine_start
        logging.info(f"[PIPELINE] Step 2: Refinement completed in {round(refine_time * 1000, 2)}ms")

        # Total timing
        total_time = time.time() - total_start
        processing_times = {
            "vision_ms": round(vision_time * 1000, 2),
            "refine_ms": round(refine_time * 1000, 2),
            "total_ms": round(total_time * 1000, 2),
        }

        # Для обратной совместимости сохраняем старое поле как total_ms
        refine_json["processing_time_ms"] = processing_times["total_ms"]
        refine_json["processing_times"] = processing_times

        logging.info(f"[PIPELINE] /analyze completed successfully, total time: {processing_times['total_ms']}ms")

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
    logging.debug("📸 /recognize called — starting pipeline")

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
        logging.info("[PIPELINE] Step 1: Starting image preprocessing")
        processed_path, preprocess_timings = await asyncio.to_thread(
            preprocess_image,
            temp_input,
        )
        logging.info(
            "[PIPELINE] Step 1: Preprocessing completed in %sms (resize=%sms, crop=%sms, remove_bg=%sms, final_resize=%sms)",
            preprocess_timings.get("total_ms"),
            preprocess_timings.get("resize_ms"),
            preprocess_timings.get("crop_ms"),
            preprocess_timings.get("remove_bg_ms"),
            preprocess_timings.get("final_resize_ms"),
        )

        # GPT-анализ
        gpt_start = time.time()
        logging.info("[PIPELINE] Step 2: Starting GPT analysis")
        result_json = await asyncio.to_thread(
            analyze_food,
            processed_path,
        )
        gpt_time = time.time() - gpt_start
        logging.info(
            "[PIPELINE] Step 2: GPT analysis completed in %sms, found %s products",
            round(gpt_time * 1000, 2),
            len(result_json.get("products", [])),
        )

        # Тайминги
        total_time = time.time() - total_start
        result_json["processing_times"] = {
            "preprocessing_ms": preprocess_timings.get("total_ms"),
            "gpt_ms": round(gpt_time * 1000, 2),
            "total_ms": round(total_time * 1000, 2),
            "preprocess_breakdown": preprocess_timings,
            "rembg_model": REMBG_MODEL,
            "plate_crop_enabled": ENABLE_PLATE_CROP,
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
