"""Main FastAPI application."""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
import time

from src.services import analyze_image_with_vision, refine_products


app = FastAPI()

# Add CORS middleware to handle preflight OPTIONS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origins as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        raise HTTPException(422, f"Analysis error: {str(e)}")
