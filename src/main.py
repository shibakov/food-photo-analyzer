"""Main FastAPI application (updated CORS)."""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
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

# --- CORS configuration ---
# During development and production, specify allowed origins explicitly.
# Wildcards are not permitted when allow_credentials is True, so here we set
# allow_credentials to False and list concrete origins. Adjust this list based
# on your deployment domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://web.telegram.org",
        "https://t.me",
        "https://my-miniapp-production.up.railway.app",
        "https://food-photo-analyzer-production.up.railway.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Provide an OPTIONS handler to satisfy CORS preflight requests for any path.
@app.options("/{path:path}")
async def preflight_handler(path: str):
    return {}


@app.get("/health")
def health():
    """Health endpoint to check service status."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_photo(image: UploadFile = File(None)):
    """
    Full analysis endpoint. Performs computer vision recognition and then
    refinement of detected products.
    """
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
        products = vision_json.get("products", [])

        # STEP 2 — REFINEMENT
        refine_json = refine_products(products)

        # Add processing time
        end_time = time.time()
        refine_json["processing_time_ms"] = round((end_time - start_time) * 1000, 2)

        return refine_json
    except Exception as e:
        logging.exception("Error in /analyze")
        raise HTTPException(422, f"Analysis error: {str(e)}")


@app.head("/recognize")
async def recognize_head():
    return Response(status_code=204)


@app.post("/recognize")
async def recognize_food(image: UploadFile = File(None)):
    """
    Optimized endpoint: preprocess → single GPT-4o vision call.
    """
    logging.info(f"Starting /recognize request, content_type: {image.content_type if image else None}")

    if not image:
        logging.error("No image provided in request")
        raise HTTPException(422, "Image field is required")

    if image.content_type not in ["image/jpeg", "image/png"]:
        logging.error(f"Unsupported content_type: {image.content_type}")
        raise HTTPException(422, "Unsupported format (use jpeg/png)")

    total_start = time.time()
    temp_input = None

    try:
        # Save uploaded image temporarily
        temp_input = f"/tmp/input_{time.time()}.jpg"
        logging.info(f"Saving temp file: {temp_input}")
        content = await image.read()
        logging.info(f"Image content length: {len(content)} bytes")

        with open(temp_input, 'wb') as f:
            f.write(content)

        # Preprocess image
        logging.info("Starting image preprocessing")
        preprocess_start = time.time()
        processed_path = preprocess_image(temp_input)
        preprocess_time = time.time() - preprocess_start
        logging.info(f"Preprocessing completed in {preprocess_time:.2f}s, output: {processed_path}")

        # Check if processed file exists
        if not os.path.exists(processed_path):
            raise ValueError(f"Processed file not found: {processed_path}")

        # GPT analysis
        logging.info("Starting GPT analysis")
        gpt_start = time.time()
        result_json = analyze_food(processed_path)
        gpt_time = time.time() - gpt_start
        logging.info(f"GPT analysis completed in {gpt_time:.2f}s with {len(result_json.get('products', []))} products")

        # Add timing info
        total_time = time.time() - total_start
        result_json["processing_times"] = {
            "preprocessing_ms": round(preprocess_time * 1000, 2),
            "gpt_ms": round(gpt_time * 1000, 2),
            "total_ms": round(total_time * 1000, 2),
        }

        logging.info(f"Recognize completed successfully in {total_time:.2f}s")
        return result_json

    except Exception as e:
        logging.exception(f"Error in /recognize: {str(e)}")
        logging.error(f"Total time before error: {time.time() - total_start:.2f}s")
        raise HTTPException(422, f"Recognition error: {str(e)}")
    finally:
        # Cleanup temp files
        for path in [temp_input, processed_path if 'processed_path' in locals() else None]:
            if path and path.startswith("/tmp/") and os.path.exists(path):
                try:
                    os.remove(path)
                    logging.info(f"Cleaned up temp file: {path}")
                except Exception as cleanup_error:
                    logging.warning(f"Failed to cleanup {path}: {cleanup_error}")
