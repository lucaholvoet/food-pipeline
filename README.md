# Food Calorie & Nutrient Estimator — CV Pipeline

AI-powered food detection and nutrition estimation pipeline.
Part of a larger hybrid CV + VLM system for meal logging and calorie tracking.

## Project Structure

```
food-pipeline/
├── classifier/        # EfficientNet-B0 food classifier (Food-101)
├── detector/          # YOLOv8n-seg food detection & segmentation
├── portion/           # Portion size estimator (geometry + density lookup)
├── nutrition/         # USDA FAISS nutrition lookup
├── api/               # FastAPI endpoint
├── tests/             # Test suite
├── scripts/           # Utility scripts (build index, tests)
├── data/              # Local data (not tracked by git)
├── models/            # Model weights (not tracked by git)
└── notebooks/         # Colab training notebooks
```

## Setup

```bash
# Clone the repo
git clone https://github.com/lucaholvoet/food-pipeline.git
cd food-pipeline

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Pipeline Overview

```
Image Input
    ↓
YOLOv8n-seg          → detects food regions + plate, outputs masks + bboxes
    ↓
EfficientNet-B0      → classifies each food crop (Food-101, 101 classes)
    ↓
Portion Estimator    → mask area + plate reference → estimated grams
    ↓
USDA FAISS Lookup    → food name → calories + protein + fat + carbs + fiber
    ↓
JSON Output          → per-item + totals, confidence scores, VLM trigger flag
```

## JSON Output Schema

```json
{
  "image_id": "uuid-string",
  "status": "success",
  "warnings": [],
  "requires_vlm_refinement": false,
  "average_confidence": 0.84,
  "plate_detected": true,
  "scale_cm_per_px": 0.043,
  "items": [
    {
      "item_id": 1,
      "food_name": "fried_rice",
      "display_name": "Fried rice",
      "classification_confidence": 0.91,
      "detection_confidence": 0.87,
      "top3_predictions": [
        {"label": "fried_rice", "confidence": 0.91},
        {"label": "risotto", "confidence": 0.06},
        {"label": "pilaf", "confidence": 0.03}
      ],
      "estimated_grams": 185,
      "portion_method": "plate_reference",
      "bbox": [120, 45, 380, 310],
      "nutrition_per_100g": {
        "calories_kcal": 163,
        "protein_g": 3.5,
        "fat_g": 4.9,
        "carbs_g": 26.6,
        "fiber_g": 0.6
      },
      "nutrition_total": {
        "calories_kcal": 302,
        "protein_g": 6.5,
        "fat_g": 9.1,
        "carbs_g": 49.2,
        "fiber_g": 1.1
      }
    }
  ],
  "totals": {
    "calories_kcal": 302,
    "protein_g": 6.5,
    "fat_g": 9.1,
    "carbs_g": 49.2,
    "fiber_g": 1.1
  },
  "processing_time_ms": 1840
}
```

## Rebuilding the USDA Index

The FAISS index is not tracked by git. After cloning, rebuild it:

```bash
# Download USDA FoodData Central CSVs from:
# https://fdc.nal.usda.gov/download-datasets.html
# Place Foundation Foods in: data/usda/foundation/
# Place SR Legacy in:        data/usda/sr_legacy/

python scripts/build_usda_index.py
```

## Running Tests

```bash
python scripts/test_nutrition.py
python scripts/test_portion.py
python scripts/test_full_nutrition_portion.py
```

## Model Weights

Model weights are not tracked by git.
After training completes on Colab, download from Google Drive and place in:

```
models/efficientnet_b0_food101_best.pt
models/classifier/idx_to_class.json
```

## Team

- **Emiel & Luca** — CV pipeline (this repo)
- **Mahesh & Furaha** — VLM refinement stage

The CV pipeline outputs a JSON that the VLM stage receives.
`requires_vlm_refinement: true` is set when average confidence < threshold (TBD with VLM team).

## Progress

- [x] Phase 1 — Environment & project setup
- [x] Phase 4A — Portion estimator
- [x] Phase 4B — USDA FAISS nutrition lookup
- [ ] Phase 2 — EfficientNet-B0 classifier (training in progress on Colab)
- [ ] Phase 3 — YOLOv8n-seg detector
- [ ] Phase 5 — Pipeline integration & FastAPI
