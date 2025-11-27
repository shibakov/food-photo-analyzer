"""Main FastAPI application."""

import base64
import os
import time
import asyncio

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import CORS_ORIGINS, ALLOW_ALL_ORIGINS
from src.services import analyze_image_with_vision, refine_products
from src.vision_pipeline.pipeline import VisionPipeline

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

# Global vision pipeline instance (reuses YOLO ONNX singleton internally)
vision_pipeline = VisionPipeline()

# -----------------------------------
# CORS
# -----------------------------------
# Разрешаем все origins для упрощения интеграции с фронтендом (в т.ч. Railway)
# Если понадобится ограничить домены — можно вернуть проверку CORS_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# Тех. эндпоинты
# -----------------------------------

@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok"}


# -----------------------------------
# /analyze — старый пайплайн
# -----------------------------------

@app.post("/analyze")
@app.post("/api/analyze")
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

        # Логируем сводку по этапам в ms
        logging.info("[PIPELINE] /analyze timings_ms=%s", processing_times)
        logging.info(f"[PIPELINE] /analyze completed successfully, total time: {processing_times['total_ms']}ms")

        return refine_json
    except Exception as e:
        logging.exception("Error in /analyze")
        raise HTTPException(422, f"Analysis error: {str(e)}")


# -----------------------------------
# /recognize — новый быстрый пайплайн
# -----------------------------------

@app.post("/recognize")
@app.post("/api/recognize")
async def recognize_food(image: UploadFile = File(None)):
    """
    Optimized endpoint: local YOLOv8n ONNX + GPT-4o-mini refiner.
    """
    logging.debug("📸 /recognize called — starting fast vision pipeline")

    if not image:
        raise HTTPException(422, "Image field is required")

    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(422, "Unsupported format (use jpeg/png)")

    total_start = time.time()
    logging.info(f"[PIPELINE] Starting /recognize endpoint for file: {image.filename}")

    try:
        # Temporary directory for raw upload
        tmp_dir = "/tmp/vision_pipeline"
        os.makedirs(tmp_dir, exist_ok=True)

        temp_input = os.path.join(tmp_dir, f"input_{time.time()}.jpg")

        # Save uploaded file
        content = await image.read()
        with open(temp_input, "wb") as f:
            f.write(content)
        logging.info(f"[PIPELINE] Saved input image for vision pipeline: {temp_input}")

        # Run fast local vision pipeline in thread to avoid blocking event loop
        pipeline_start = time.time()
        result_json = await asyncio.to_thread(
            vision_pipeline.analyze,
            temp_input,
        )
        pipeline_time = time.time() - pipeline_start

        # Timings
        total_time = time.time() - total_start
        processing_times = {
            "pipeline_ms": round(pipeline_time * 1000, 2),
            "total_ms": round(total_time * 1000, 2),
        }

        result_json["processing_times"] = processing_times

        logging.info(
            "[PIPELINE] /recognize completed successfully, pipeline_ms=%s, total_ms=%s, products=%s",
            processing_times.get("pipeline_ms"),
            processing_times.get("total_ms"),
            len(result_json.get("products", [])),
        )

        # Clean up temp file
        try:
            os.remove(temp_input)
        except Exception:
            pass

        return result_json

    except Exception as e:
        logging.exception("Error in /recognize")
        raise HTTPException(422, f"Recognition error: {str(e)}")
