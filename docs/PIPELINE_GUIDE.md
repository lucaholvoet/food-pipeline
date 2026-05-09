# Food Calorie & Nutrient Estimator — Full Pipeline Guide

**Last Updated:** May 2026

This document provides a comprehensive technical reference for every component of the food analysis pipeline — from raw data sources to final output.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Data Sources](#2-data-sources)
3. [CV Pipeline — Detailed](#3-cv-pipeline--detailed)
4. [VLM Pipeline — Detailed](#4-vlm-pipeline--detailed)
5. [Pipeline Orchestration](#5-pipeline-orchestration)
6. [API Layer](#6-api-layer)
7. [Application Layer](#7-application-layer)
8. [Output Format Reference](#8-output-format-reference)

---

## 1. Pipeline Overview

The system follows a **two-stage hybrid architecture** that balances speed and accuracy:

```
📸 Input Image
     │
     ▼
┌─────────────────────────────────────────────┐
│              STAGE 1: CV Pipeline           │
│                                             │
│  YOLOv8n-seg ──► EfficientNet-B0 ──► MiDaS  │
│  (Detection)     (Classification)   (Depth) │
│       │                │               │     │
│       └────────────────┼───────────────┘     │
│                        ▼                     │
│              Portion Estimator               │
│                        │                     │
│                        ▼                     │
│             USDA FAISS Nutrition             │
│              Lookup (per 100g)               │
│                        │                     │
│                        ▼                     │
│              Confidence Check                │
│           avg_confidence ≥ 0.70?             │
└────────────┬─────────────────────┬───────────┘
             │ YES                 │ NO
             ▼                     ▼
      ┌──────────────┐   ┌─────────────────────┐
      │  Return CV   │   │  STAGE 2: VLM       │
      │  Result      │   │  Refinement          │
      │  (Fast)      │   │  (Gemini Flash)      │
      └──────────────┘   │                      │
                         │ Corrects labels,     │
                         │ re-estimates portion │
                         │ recalculates nutrition│
                         └──────────┬───────────┘
                                    ▼
                           ┌──────────────┐
                           │ Final JSON   │
                           │ Response     │
                           └──────────────┘
```

**Design Principle:** Use the fast CV pipeline for most images (≈87% correct). Only invoke the slower VLM when CV confidence is low, achieving better accuracy without the cost of running a large model on every image.

---

## 2. Data Sources

### 2.1 Food-101 Dataset (Classification)

| Property | Value |
|----------|-------|
| **Purpose** | Training and evaluation of the food classifier |
| **Total images** | 101,000 |
| **Number of classes** | 101 |
| **Train split** | 75,750 images (750 per class) |
| **Test split** | 25,250 images (250 per class) |
| **Image resolution** | Variable (resized to 224×224 for training) |
| **Source** | [ETH Zurich / Bossard et al., 2014](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) |

**Class categories** include appetizers, main courses, desserts, and side dishes from diverse cuisines. Examples: `pizza`, `sushi`, `ramen`, `hamburger`, `chocolate_cake`, `caesar_salad`, `pad_thai`, `bibimbap`.

**Challenge:** Many classes are visually similar (e.g., `filet_mignon` vs `prime_rib`, `spaghetti_bolognese` vs `spaghetti_carbonara`, `ramen` vs `pho`), which drives the need for VLM refinement.

### 2.2 FoodSeg103 Dataset (Detection)

| Property | Value |
|----------|-------|
| **Purpose** | Training the YOLOv8 food/plate segmentation detector |
| **Total images** | ~7,000+ |
| **Classes** | 104 food categories + background |
| **Annotations** | Instance-level segmentation masks |
| **Source** | [FoodSeg103 — Wu et al., 2021](https://xiongweiwu.github.io/foodseg103.html) |

The detector was trained to identify **two high-level classes** — `food` (class 1) and `plate` (class 0) — using a subset of FoodSeg103.

### 2.3 USDA FoodData Central (Nutrition)

| Property | Value |
|----------|-------|
| **Purpose** | Nutritional reference database |
| **Datasets used** | Foundation Foods + SR Legacy |
| **Total records indexed** | 9,013 unique food entries |
| **Fields per record** | Description, calories, protein, fat, carbs, fiber |
| **Source** | [USDA FoodData Central](https://fdc.nal.usda.gov/) |

**How it's built:**
1. Raw CSV files are downloaded from USDA
2. `scripts/build_usda_index.py` processes them into a clean JSON records file
3. Each food description is encoded using `all-MiniLM-L6-v2` sentence embeddings
4. Embeddings are stored in a FAISS index for fast similarity search

### 2.4 Density Table (Portion Estimation)

A hand-crafted lookup table mapping each of the 101 Food-101 classes to:

| Field | Description |
|-------|-------------|
| `height_cm` | Typical height of the food item (e.g., pizza=2.5cm, hamburger=8.0cm) |
| `density_g_cm3` | Approximate density (e.g., salad=0.3, steak=1.0, ice_cream=0.55) |

Used in conjunction with MiDaS depth to convert pixel area + estimated height → volume → mass (grams).

---

## 3. CV Pipeline — Detailed

### 3.1 Food Detection (`detector/detector.py`)

**Model:** YOLOv8n-seg (Ultralytics) — the nano variant of YOLOv8 with instance segmentation.

**Training:**
- Trained on FoodSeg103 dataset
- Configured for 2 classes: `plate` (class 0) and `food` (class 1)
- Achieved **mAP50: 0.934** on validation set

**Inference flow:**
```
Input: PIL Image
    │
    ▼
YOLOv8 inference (conf=0.25, iou=0.45)
    │
    ├── For each detected instance:
    │   ├── class 0 → plate_mask (binary mask, resized to original image size)
    │   └── class 1 → food item with:
    │       ├── bbox: [x1, y1, x2, y2]
    │       ├── mask: binary segmentation mask
    │       ├── crop: PIL Image crop with 10px padding
    │       └── detection_confidence: float
    │
    ▼
Output: { food_items: list, plate_mask: ndarray, plate_detected: bool, image_size: tuple }
```

**Key implementation details:**
- Masks are resized to original image dimensions using nearest-neighbor interpolation
- Food crops include 10px padding around the bounding box for better classification
- If no masks are produced, returns empty results gracefully

### 3.2 Food Classification (`classifier/classifier.py`)

**Model:** EfficientNet-B0 with custom classification head.

**Architecture:**
```
EfficientNet-B0 backbone (pretrained on ImageNet)
    │
    ▼  Global Average Pooling → 1280 features
    │
    Dropout(p=0.2)
    Linear(1280 → 512)
    SiLU activation
    Dropout(p=0.2)
    Linear(512 → 101)    ← 101 Food-101 classes
```

**Performance:** 87.37% top-1 accuracy on Food-101 test set.

**Preprocessing pipeline:**
```
Resize to 256px
CenterCrop to 224×224
Convert to tensor
Normalize with ImageNet stats:
  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]
```

**Inference:**
```python
classifier.predict(crop_image, top_k=3)
# Returns: [
#   {"label": "pizza", "display_name": "Pizza", "confidence": 0.92},
#   {"label": "hamburger", "display_name": "Hamburger", "confidence": 0.04},
#   {"label": "hot_dog", "display_name": "Hot Dog", "confidence": 0.02}
# ]
```

### 3.3 Depth Estimation (`portion/depth_estimator.py`)

**Model:** MiDaS_small from Intel ISL (loaded via `torch.hub`).

**Purpose:** Generates a monocular depth map of the image to estimate the relative height of food items above the plate surface.

**Inference flow:**
```
Input: PIL Image (RGB)
    │
    ▼
MiDaS_small transform
    │
    ▼
Model inference (no grad)
    │
    ▼
Bicubic interpolation to original size
    │
    ▼
Normalize to [0, 1] range
    │
    ▼
Output: 2D depth map (same H×W as input)
```

**Integration with portion estimation:** For each food item, the depth values within the food mask are extracted. The relative height is calculated as the difference between the mean depth over the food region and the minimum depth. This relative height is scaled to centimeters using a calibration multiplier of 8.0.

### 3.4 Portion Estimation (`portion/portion.py`)

**Approach:** Geometry-based estimation using pixel area, real-world scale, and food-specific density.

**Two scaling modes:**

| Mode | When Used | How It Works |
|------|-----------|-------------|
| **plate_reference** | Plate detected | `cm_per_px = plate_radius_cm / plate_radius_px` where plate diameter is assumed to be 26cm |
| **fallback_scale** | No plate detected | Assumes food occupies 40% of a standard plate footprint, derives `cm_per_px` from image dimensions |

**Grams calculation:**
```
food_area_cm2 = food_mask_pixels × (cm_per_px)²
height_cm = MiDaS relative height (or density table default)
volume_cm3 = food_area_cm2 × height_cm
grams = volume_cm3 × density_g_cm3
grams = clamp(grams, 10.0, 1500.0)
```

**Portion method naming:**
- `plate_reference` — plate detected, density table height
- `plate_reference_midas` — plate detected, MiDaS depth used for height
- `fallback_scale` — no plate, assumed scale
- `vlm_visual_estimate` — VLM provided the estimate

### 3.5 Nutrition Lookup (`nutrition/nutrition.py`)

**Approach:** Semantic search over USDA food records using FAISS.

**Components:**
- **Embedding model:** `all-MiniLM-L6-v2` (384-dimensional sentence embeddings)
- **Index:** FAISS L2-normalized inner product index
- **Records:** JSON file with 9,013 food entries

**Lookup flow:**
```
Input: "fried_rice" (predicted food label)
    │
    ▼
Replace underscores: "fried rice"
    │
    ▼
Encode with SentenceTransformer → 384-dim embedding
    │
    ▼
L2 normalize embedding
    │
    ▼
FAISS search (top-1 nearest neighbor)
    │
    ▼
Output: {
    "usda_description": "Restaurant, Chinese, fried rice",
    "similarity": 0.82,
    "nutrition_per_100g": {
        "calories_kcal": 163,
        "protein_g": 4.1,
        "fat_g": 5.3,
        "carbs_g": 24.8,
        "fiber_g": 1.0
    }
}
```

**Scaling to portion:**
```
nutrition_total = nutrition_per_100g × (estimated_grams / 100)
```

---

## 4. VLM Pipeline — Detailed

### 4.1 Overview

The VLM (Vision-Language Model) pipeline activates when the CV pipeline's average confidence falls below the threshold (0.70) or when no plate is detected. It receives the original image and the CV output, then produces corrected or confirmed predictions.

**Trigger conditions:**
- `average_confidence < 0.70`
- `plate_detected == False`

### 4.2 Data Schemas (`vlm/schemas.py`)

All input/output is validated using Pydantic models.

**Input schema (`VLMRequest`):**
```python
class VLMRequest(BaseModel):
    image_base64: str           # Base64-encoded original image
    reason: RefinementReason    # Why VLM was triggered
    trigger_threshold: float    # Confidence threshold (default: 0.70)
    cv_output: CVOutput         # Full CV pipeline output
```

**Refinement reasons:**
- `low_confidence` — CV predictions below threshold
- `multiple_similar_predictions` — Top-3 predictions are close
- `no_food_detected` — Detector found nothing
- `portion_estimate_suspect` — Unreasonable portion size

**Output schema (`VLMResponse`):**
```python
class VLMResponse(BaseModel):
    image_id: str
    refinement_status: str          # "completed"
    vlm_model: str                  # e.g., "gemini-flash-latest"
    confidence_threshold_used: float
    items: list[RefinedItem]        # Per-item results
    totals: CVNutrition             # Recalculated totals
    notes: list[str]                # Observations/warnings
    processing_time_ms: int
```

**Per-item result (`RefinedItem`):**
```python
class RefinedItem(BaseModel):
    item_id: int
    action: RefinementAction        # "confirmed", "corrected", or "unknown"
    original: OriginalResult        # What CV said
    refined: RefinedResult          # What VLM says
    portion: RefinedPortion         # VLM's gram estimate
    nutrition_per_100g: CVNutrition
    nutrition_total: CVNutrition
```

### 4.3 Prompt Engineering (`vlm/prompts.py`)

The prompt system is critical to VLM output quality. Two prompt variants exist:

**Full system prompt** (for production refinement):
- Identifies the VLM as a "food identification and nutrition refinement expert"
- Includes the complete list of 101 valid Food-101 class names
- Provides common mapping guidance (e.g., "burger → hamburger")
- Specifies confidence calibration guidelines:
  - 0.90–1.00: visually obvious match
  - 0.75–0.89: strong guess
  - 0.50–0.74: plausible but uncertain
  - 0.30–0.49: weak guess
- Enforces JSON-only output

**User prompt template:**
- Includes trigger reason, threshold, average CV confidence
- Embeds the full CV output JSON
- Shows the exact expected JSON response structure with an example

**Fast prompt** (for evaluation/testing):
- Shorter prompt with minimal context
- Used in `eval_vlm.py` for faster iteration

### 4.4 VLM Client (`vlm/client.py`)

Supports two backends:

| Backend | Model | When Used |
|---------|-------|-----------|
| **Google AI Studio** | `gemini-flash-latest` | Production (deployed server) |
| **Ollama** | `gemma4:e4b` | Local development/testing |

**JSON extraction logic:**
1. Try direct `json.loads()` on the raw response
2. If wrapped in ` ```json ... ``` `, extract the code block
3. If wrapped in generic ` ``` ... ``` `, extract the code block
4. Search for `{` ... `}` boundaries as last resort

### 4.5 VLM Refiner (`vlm/refiner.py`)

The refiner orchestrates the entire VLM refinement process:

```
VLMRequest
    │
    ▼
Validate input with Pydantic
    │
    ▼
Build user prompt (reason + threshold + CV output JSON)
    │
    ▼
Send to VLM client (image + system prompt + user prompt)
    │
    ▼
Parse raw response into structured items
    │
    ├── Normalize food labels to Food-101 classes
    ├── Extract portion estimates
    ├── Determine action (confirmed/corrected)
    └── Build nutrition per 100g and total
    │
    ▼
Recalculate totals across all items
    │
    ▼
Return VLMResponse
```

**Label normalization** (`vlm/normalization.py`):
- Converts VLM free-text food names to the closest Food-101 class
- Uses string similarity matching
- Falls back to the CV prediction if no good match found

---

## 5. Pipeline Orchestration (`pipeline.py`)

The `FoodPipeline` class in `pipeline.py` ties all modules together.

### Initialization
```python
pipeline = FoodPipeline(
    detector_path="models/yolov8n_food_best.pt",
    classifier_path="models/efficientnet_b0_food101_best.pt",
    labels_path="models/idx_to_class.json",
    index_path="nutrition/usda.index",
    records_path="nutrition/usda_records.json",
    device="cpu",
    use_vlm=True
)
```

**Graceful degradation:** Each module has a try/except during initialization. If any component fails to load (e.g., MiDaS not available), the pipeline continues without it and logs a warning.

### Processing Flow

```
pipeline.run(pil_image)
    │
    ▼
Step 1: DETECT
    detector.detect(image)
    → food_items_raw, plate_mask, plate_detected
    │
    ▼
Step 1.5: DEPTH (optional)
    depth_estimator.get_depth_map(image) → depth_map
    │
    ▼
Step 1.6: SCALE
    if plate_detected:
        scale_cm_per_px = 13.0 / plate_radius_px
    │
    ▼
Step 2: CLASSIFY + PORTION + NUTRITION (per item)
    for each food_item:
        classifier.predict(crop, top_k=3) → predictions
        portion.estimate(label, mask, plate_mask, depth_map) → grams
        nutrition.get_nutrition(label) → nutrition_per_100g
        nutrition_total = nutrition_per_100g × (grams / 100)
    │
    ▼
Step 3: CONFIDENCE CHECK
    avg_confidence = mean(all classification + detection confidences)
    requires_vlm = (avg_conf < 0.70) OR (not plate_detected)
    │
    ▼
Step 4: BUILD OUTPUT
    CVPipelineOutput with items, totals, warnings
    │
    ▼
Step 5: VLM REFINEMENT (if needed)
    Convert CVOutput → VLMRequest
    refiner.refine(vlm_request) → VLMResponse
    Return VLMResponse
```

---

## 6. API Layer (`api/app.py`)

**Framework:** FastAPI with Uvicorn server

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System status (module availability, version) |
| `/status` | GET | Detailed module availability check |
| `/analyze` | POST | Main analysis endpoint — accepts image file |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc documentation |

**Lazy loading:** All ML models are loaded on first use, not at startup. This prevents long boot times and memory pressure.

**Mock fallbacks:** If model weights or the FAISS index are not available, the API returns mock data so the UI remains functional for development and demonstration.

### `/analyze` Request

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@meal.jpg" \
  -F "threshold=0.70"
```

Parameters:
- `file` (required): JPG/PNG image
- `threshold` (optional): Override confidence threshold

---

## 7. Application Layer (`ui/`)

### 7.1 Frontend (`ui/app.py`)

**Framework:** Gradio 5.29.0

**Tabs:**

| Tab | Functionality |
|-----|-------------|
| **📷 Analyze** | Upload/capture meal image, view results, log meal, AI correction chat |
| **📊 Dashboard** | Daily calorie tracker, 7-day history bar chart, meal list, manual logging |
| **👤 Profile** | BMR/TDEE calculator, goal setting, activity level, personalized targets |

**Features:**
- Annotated image with color-coded bounding boxes and confidence badges
- VLM refinement panel with CV → VLM comparison
- AI chat for correcting results via natural language
- Manual meal logging via Gemini 2.5 Flash Lite
- Dark theme with custom CSS

### 7.2 Authentication (`ui/auth.py`)

- SQLite-backed user accounts
- SHA256 + random salt password hashing
- Username and password validation (min 3 and 6 chars)

### 7.3 Meal Database (`ui/db.py`)

- SQLite database for meal logging
- Per-user meal history with timestamps
- Daily and weekly aggregation queries
- Sources tracked: `cv`, `vlm`, `llm_correction`, `manual`

### 7.4 Profile System (`ui/profile.py`)

- BMR calculation using Mifflin-St Jeor equation
- TDEE = BMR × activity multiplier
- Daily calorie target based on goal weight and goal date
- Macro split: 1.8g protein/kg, 25% calories from fat, rest from carbs
- 5 activity levels from sedentary to extremely active

### 7.5 AI Chat (`ui/llm_chat.py`)

Two chat modes using **Gemini 2.5 Flash Lite**:

| Mode | Purpose | Trigger |
|------|---------|---------|
| **Correction chat** | Disagree with pipeline result, correct it via conversation | "Disagree? Adjust with AI" button |
| **Manual logging** | Describe what you ate in natural language | "Log food manually" button |

Both modes output a structured JSON block when ready to log, which the UI renders as a confirmation card.

---

## 8. Output Format Reference

### 8.1 CV-Only Response (high confidence)

```json
{
  "image_id": "uuid-string",
  "status": "success",
  "warnings": [],
  "requires_vlm_refinement": false,
  "average_confidence": 0.87,
  "plate_detected": true,
  "scale_cm_per_px": 0.04832,
  "items": [
    {
      "item_id": 1,
      "food_name": "pizza",
      "display_name": "Pizza",
      "classification_confidence": 0.92,
      "detection_confidence": 0.88,
      "top3_predictions": [
        {"label": "pizza", "confidence": 0.92},
        {"label": "hamburger", "confidence": 0.03},
        {"label": "hot_dog", "confidence": 0.02}
      ],
      "estimated_grams": 350,
      "portion_method": "plate_reference_midas",
      "bbox": [100, 50, 400, 350],
      "nutrition_per_100g": {
        "calories_kcal": 266.0,
        "protein_g": 11.0,
        "fat_g": 10.0,
        "carbs_g": 33.0,
        "fiber_g": 2.3
      },
      "nutrition_total": {
        "calories_kcal": 931.0,
        "protein_g": 38.5,
        "fat_g": 35.0,
        "carbs_g": 115.5,
        "fiber_g": 8.1
      }
    }
  ],
  "totals": {
    "calories_kcal": 931.0,
    "protein_g": 38.5,
    "fat_g": 35.0,
    "carbs_g": 115.5,
    "fiber_g": 8.1
  },
  "processing_time_ms": 1243
}
```

### 8.2 VLM-Refined Response (low confidence)

```json
{
  "image_id": "uuid-string",
  "refinement_status": "completed",
  "vlm_model": "gemini-flash-latest",
  "confidence_threshold_used": 0.70,
  "items": [
    {
      "item_id": 1,
      "action": "corrected",
      "original": {
        "food_name": "fried_rice",
        "classification_confidence": 0.45
      },
      "refined": {
        "food_name": "bibimbap",
        "display_name": "Bibimbap",
        "vlm_confidence": 0.88,
        "food_description": "Korean rice bowl with vegetables and egg"
      },
      "portion": {
        "estimated_grams": 320,
        "portion_method": "vlm_visual_estimate",
        "vlm_confidence": 0.75
      },
      "nutrition_per_100g": { "..." : "..." },
      "nutrition_total": { "..." : "..." }
    }
  ],
  "totals": { "..." : "..." },
  "notes": ["Corrected from fried_rice to bibimbap based on visual appearance"],
  "processing_time_ms": 8500
}
```

---

## Tech Stack Summary

| Component | Technology | Version |
|-----------|-----------|---------|
| Detection | YOLOv8n-seg (Ultralytics) | 8.4.41 |
| Classification | EfficientNet-B0 (timm) | — |
| Depth | MiDaS_small (torch.hub) | — |
| Nutrition | FAISS + SentenceTransformers | 1.8.0 / 3.0.0 |
| VLM (production) | Gemini Flash (Google AI Studio) | latest |
| VLM (local dev) | Gemma 4 E4B (Ollama) | e4b |
| Chat | Gemini 2.5 Flash Lite | latest |
| API | FastAPI + Uvicorn | 0.111.0 |
| Frontend | Gradio | 5.29.0 |
| Database | SQLite | — |
| Deployment | Docker Compose + Hetzner CX23 | — |
| Deep Learning | PyTorch + torchvision | 2.2.2 / 0.17.2 |
| Image Processing | OpenCV + Pillow | 4.9.0 / 10.3.0 |
