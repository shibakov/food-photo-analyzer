"""Main FastAPI application."""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64

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
        raise HTTPException(400, "Image field is required")

    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "Unsupported format (use jpeg/png)")

    img_bytes = await image.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    try:
        # STEP 1 — VISION RECOGNITION
        vision_json = analyze_image_with_vision(img_b64, image.content_type)
        products = vision_json["products"]

        # STEP 2 — REFINEMENT
        refine_json = refine_products(products)

        return refine_json
    except Exception as e:
        return {"error": "analysis_error", "details": str(e)}
