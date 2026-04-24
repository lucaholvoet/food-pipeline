from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import time

from pipeline import FoodPipeline
from api.models import CVPipelineOutput

app = FastAPI(
    title="Food Calorie & Nutrient Estimator",
    description="CV pipeline for food detection, classification, portion estimation and nutrition lookup.",
    version="1.0.0"
)

pipeline = None

@app.on_event("startup")
async def load_pipeline():
    global pipeline
    print("Loading pipeline...")
    pipeline = FoodPipeline()
    print("Pipeline ready.")

@app.get("/health")
def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image, got {file.content_type}"
        )

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image too large. Maximum size is 10MB."
        )

    try:
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open image: {e}")

    result = pipeline.run(img)
    return result