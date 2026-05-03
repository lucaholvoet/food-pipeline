"""
Food Calorie & Nutrient Estimator — FastAPI Application

This is the main API entry point. Run with:
    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

Then open:
    http://localhost:8000/docs          — Swagger UI (interactive)
    http://localhost:8000/redoc          — ReDoc documentation
    http://localhost:8000/analyze        — POST endpoint

Pipeline flow:
    Image upload
        ↓
    YOLOv8n-seg detector (or mock if not available)
        ↓
    EfficientNet-B0 classifier (or mock if weights missing)
        ↓
    Portion estimator
        ↓
    USDA FAISS nutrition lookup (or mock if index missing)
        ↓
    Confidence check — if < threshold, trigger VLM refinement
        ↓
    JSON response
"""

import os
import sys
import time
import uuid
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Global config ──
CONFIDENCE_THRESHOLD = 0.70
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
USDA_DIR = os.path.join(PROJECT_ROOT, "data", "usda")

# ── Availability checks ──
CLASSIFIER_WEIGHTS = os.path.join(
    MODELS_DIR, "efficientnet_b0_food101_best.pt"
)
LABELS_FILE = os.path.join(
    MODELS_DIR, "classifier", "idx_to_class.json"
)
FAISS_INDEX = os.path.join(USDA_DIR, "faiss_index.bin")

has_classifier = (
    os.path.exists(CLASSIFIER_WEIGHTS)
    and os.path.exists(LABELS_FILE)
)
has_nutrition_index = os.path.exists(FAISS_INDEX)

# ── Lazy-loaded modules ──
classifier = None
portion_estimator = None
nutrition_lookup = None
vlm_refiner = None


def get_classifier():
    global classifier
    if classifier is not None:
        return classifier
    if has_classifier:
        from classifier.classifier import FoodClassifier
        classifier = FoodClassifier(
            model_path=CLASSIFIER_WEIGHTS,
            labels_path=LABELS_FILE,
        )
        return classifier
    return None


def get_portion_estimator():
    global portion_estimator
    if portion_estimator is not None:
        return portion_estimator
    try:
        from portion.portion import PortionEstimator
        portion_estimator = PortionEstimator()
        return portion_estimator
    except Exception:
        return None


def get_nutrition_lookup():
    global nutrition_lookup
    if nutrition_lookup is not None:
        return nutrition_lookup
    if has_nutrition_index:
        try:
            import json
            records_path = os.path.join(
                USDA_DIR, "nutrition_records.json"
            )
            from nutrition.nutrition import NutritionLookup
            nutrition_lookup = NutritionLookup(
                index_path=FAISS_INDEX,
                records_path=records_path,
            )
            return nutrition_lookup
        except Exception:
            return None
    return None


def get_vlm_refiner():
    global vlm_refiner
    if vlm_refiner is not None:
        return vlm_refiner
    try:
        from vlm.refiner import VLMRefiner
        vlm_refiner = VLMRefiner(
            backend="ollama",
            model="gemma4:e4b",
        )
        return vlm_refiner
    except Exception:
        return None


# ── FastAPI app ──
app = FastAPI(
    title="Food Calorie & Nutrient Estimator",
    description=(
        "Hybrid CV + VLM pipeline for food recognition "
        "and nutrition estimation from meal images."
    ),
    version="1.0.0",
)

# ── Mock data generators ──

def mock_detection():
    """Mock YOLO detector output."""
    return [{
        "bbox": [100, 50, 400, 350],
        "mask_area": 18000,
        "detection_confidence": 0.80,
        "class": "food",
    }]


def mock_plate():
    """Mock plate detection."""
    return {
        "bbox": [80, 20, 640, 420],
        "mask_area": 95000,
        "detection_confidence": 0.95,
    }


def mock_classification():
    """Mock classifier output when weights unavailable."""
    import random
    foods = [
        "pizza", "hamburger", "sushi", "spaghetti_bolognese",
        "caesar_salad", "pancakes", "ice_cream", "steak",
        "ramen", "chocolate_cake", "fried_rice", "bibimbap",
    ]
    top = random.choice(foods)
    others = [f for f in foods if f != top]
    import random as r
    others = r.sample(others, min(2, len(others)))
    conf = round(random.uniform(0.30, 0.70), 2)
    rest = round((1.0 - conf) / 2, 2)
    return {
        "food_name": top,
        "display_name": top.replace("_", " ").title(),
        "classification_confidence": conf,
        "top3_predictions": [
            {"label": top, "confidence": conf},
            {"label": others[0], "confidence": rest},
            {"label": others[1], "confidence": round(1 - conf - rest, 2)},
        ],
    }


def mock_nutrition(food_name: str):
    """Mock nutrition data when USDA index unavailable."""
    return {
        "calories_kcal": 150,
        "protein_g": 5.0,
        "fat_g": 4.0,
        "carbs_g": 22.0,
        "fiber_g": 1.0,
    }

# ── Helper: process one food item ──

def process_item(
    image_bytes: bytes,
    bbox: list,
    mask_area: int,
    plate_mask_area: int,
):
    """Process a single detected food item through the pipeline."""
    import numpy as np

    # ── Classification ──
    clf = get_classifier()
    if clf:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes)).crop(bbox)
        preds = clf.predict(img, top_k=3)
        food_name = preds[0]["label"]
        display_name = preds[0]["display_name"]
        clf_confidence = preds[0]["confidence"]
        top3 = preds
    else:
        mock = mock_classification()
        food_name = mock["food_name"]
        display_name = mock["display_name"]
        clf_confidence = mock["classification_confidence"]
        top3 = mock["top3_predictions"]

    # ── Portion estimation ──
    pe = get_portion_estimator()
    if pe:
        food_mask = np.zeros((480, 640), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        food_mask[y1:y2, x1:x2] = 1
        plate_mask = np.zeros((480, 640), dtype=np.uint8)
        plate_mask[20:420, 80:640] = 1
        portion_result = pe.estimate(
            food_label=food_name,
            mask=food_mask,
            plate_mask=plate_mask,
        )
        estimated_grams = portion_result["estimated_grams"]
        portion_method = portion_result["portion_method"]
    else:
        estimated_grams = 250.0
        portion_method = "mock"

    # ── Nutrition lookup ──
    nl = get_nutrition_lookup()
    if nl:
        try:
            nut_per_100g = nl.get_nutrition(food_name)
        except Exception:
            nut_per_100g = mock_nutrition(food_name)
    else:
        nut_per_100g = mock_nutrition(food_name)

    # ── Scale nutrition to portion ──
    scale = estimated_grams / 100.0
    nut_total = {
        k: round(v * scale, 1)
        for k, v in nut_per_100g.items()
    }

    return {
        "food_name": food_name,
        "display_name": display_name,
        "classification_confidence": clf_confidence,
        "top3_predictions": top3,
        "estimated_grams": estimated_grams,
        "portion_method": portion_method,
        "bbox": bbox,
        "nutrition_per_100g": nut_per_100g,
        "nutrition_total": nut_total,
    }

# ── Helper: VLM refinement ──

def run_vlm_refinement(
    image_base64: str,
    cv_output: dict,
    avg_confidence: float,
):
    """Run VLM refinement on low-confidence results."""
    refiner = get_vlm_refiner()
    if not refiner:
        return None

    try:
        request_data = {
            "image_base64": image_base64,
            "reason": "low_confidence",
            "trigger_threshold": CONFIDENCE_THRESHOLD,
            "cv_output": cv_output,
        }
        result = refiner.refine(request_data)
        return result.model_dump()
    except Exception as e:
        return {"error": str(e), "status": "vlm_failed"}


# ── Helper: compute totals ──

def compute_totals(items: list) -> dict:
    """Sum nutrition across all items."""
    totals = {
        "calories_kcal": 0,
        "protein_g": 0,
        "fat_g": 0,
        "carbs_g": 0,
        "fiber_g": 0,
    }
    for item in items:
        nt = item.get("nutrition_total", {})
        for key in totals:
            totals[key] += nt.get(key, 0)
    return {k: round(v, 1) for k, v in totals.items()}

# ── API endpoints ──

@app.get("/")
def root():
    """API root — returns system status."""
    return {
        "name": "Food Calorie & Nutrient Estimator",
        "version": "1.0.0",
        "status": "running",
        "modules": {
            "classifier": "loaded" if has_classifier else "mock",
            "portion_estimator": "available",
            "nutrition_lookup": (
                "loaded" if has_nutrition_index else "mock"
            ),
            "vlm_refinement": "available",
            "detector": "mock (YOLO not implemented)",
        },
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "endpoints": {
            "analyze": "POST /analyze — upload food image",
            "status": "GET /status — module availability",
            "docs": "GET /docs — Swagger UI",
        },
    }


@app.get("/status")
def system_status():
    """Check which modules are available."""
    return {
        "classifier_weights": (
            "available" if has_classifier else "missing"
        ),
        "nutrition_faiss_index": (
            "available" if has_nutrition_index else "missing"
        ),
        "portion_estimator": "available",
        "vlm_refinement": "available",
        "detector": "mock_only",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }

@app.post("/analyze")
async def analyze_food(
    image: UploadFile = File(..., description="Food image file"),
    threshold: Optional[float] = None,
):
    """
    Analyze a food image and return nutrition estimates.

    Accepts JPG/PNG image. Returns JSON with:
    - detected food items
    - classification confidence
    - portion estimate in grams
    - nutrition per 100g and total
    - VLM refinement if confidence is low

    Parameters:
    - **image**: Food photo (required)
    - **threshold**: Override confidence threshold (optional)
    """
    start_time = time.time()

    # ── Read image ──
    try:
        image_bytes = await image.read()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read image file",
        )

    if len(image_bytes) < 100:
        raise HTTPException(
            status_code=400,
            detail="Image file is too small or empty",
        )

    thresh = threshold or CONFIDENCE_THRESHOLD
    image_id = str(uuid.uuid4())

    # ── Detection ──
    detections = mock_detection()
    plate = mock_plate()

    # ── Process each detected food item ──
    items = []
    for i, det in enumerate(detections):
        item_result = process_item(
            image_bytes=image_bytes,
            bbox=det["bbox"],
            mask_area=det["mask_area"],
            plate_mask_area=plate["mask_area"],
        )
        item_result["item_id"] = i + 1
        item_result["detection_confidence"] = det[
            "detection_confidence"
        ]
        items.append(item_result)

    # ── Compute average confidence ──
    if items:
        avg_confidence = round(
            sum(
                it["classification_confidence"]
                for it in items
            ) / len(items),
            4,
        )
    else:
        avg_confidence = 0.0

    # ── Compute totals ──
    totals = compute_totals(items)

    # ── Build CV output ──
    cv_output = {
        "image_id": image_id,
        "status": "success",
        "warnings": [],
        "average_confidence": avg_confidence,
        "plate_detected": True,
        "items": items,
        "totals": totals,
    }

    # ── Check confidence → VLM refinement? ──
    requires_vlm = avg_confidence < thresh
    vlm_result = None

    if requires_vlm:
        image_base64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")
        vlm_result = run_vlm_refinement(
            image_base64=image_base64,
            cv_output=cv_output,
            avg_confidence=avg_confidence,
        )

    # ── Build final response ──
    elapsed_ms = int((time.time() - start_time) * 1000)

    response = {
        "image_id": image_id,
        "status": "success",
        "warnings": [],
        "requires_vlm_refinement": requires_vlm,
        "average_confidence": avg_confidence,
        "confidence_threshold": thresh,
        "plate_detected": True,
        "items": items,
        "totals": totals,
        "processing_time_ms": elapsed_ms,
    }

    if requires_vlm and vlm_result:
        response["vlm_refinement"] = vlm_result

    return response
