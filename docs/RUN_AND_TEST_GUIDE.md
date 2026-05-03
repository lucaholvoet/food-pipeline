# Food Calorie & Nutrient Estimator — Setup, Run & Test Guide

**Last Updated:** May 2026

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Docker Deployment (Recommended)](#2-docker-deployment-recommended)
3. [Local Development Setup](#3-local-development-setup)
4. [Running the Pipeline](#4-running-the-pipeline)
5. [Testing & Evaluation Scripts](#5-testing--evaluation-scripts)
6. [Visualization Scripts](#6-visualization-scripts)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | How to Check |
|-------------|---------|--------------|
| Python | 3.11+ | `python --version` |
| Git | Any | `git --version` |
| Docker | 20+ (for deployment) | `docker --version` |

**API Key:**
- Google AI Studio API key for VLM refinement — [get one here](https://ai.google.dev/gemini-api/docs/api-key)

---

## 2. Docker Deployment (Recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/lucaholvoet/food-pipeline.git
cd food-pipeline
```

### Step 2: Prepare Model Files

Place these files in the project root:

```
models/yolov8n_food_best.pt
models/efficientnet_b0_food101_best.pt
models/classifier/idx_to_class.json
```

And the nutrition index:

```
data/usda/faiss_index.bin
data/usda/nutrition_records.json
```

> Contact the team for download links if needed.

### Step 3: Configure Environment

```bash
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### Step 4: Launch

```bash
docker-compose up --build
```

**Services started:**

| Service | Port | URL |
|---------|------|-----|
| CV Pipeline API | 8000 | http://localhost:8000/docs |
| Gradio UI | 7860 | http://localhost:7860 |

### Step 5: Verify

```bash
# Check API health
curl http://localhost:8000/

# Test with an image
curl -X POST http://localhost:8000/analyze -F "file=@data/test_food.jpg"
```

---

## 3. Local Development Setup

### Step 1: Clone & Enter

```bash
git clone https://github.com/lucaholvoet/food-pipeline.git
cd food-pipeline
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

**Activate:**

Windows (PowerShell):
```powershell
venv\Scripts\Activate.ps1
```

Windows (CMD):
```cmd
venv\Scripts\activate.bat
```

Linux/macOS:
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# PyTorch (CPU version)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# All project dependencies
pip install -r requirements.txt
```

### Step 4: Configure API Key

```bash
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### Step 5: Prepare Model & Data Files

Ensure the following files exist:

```
models/yolov8n_food_best.pt
models/efficientnet_b0_food101_best.pt
models/classifier/idx_to_class.json
data/usda/faiss_index.bin
data/usda/nutrition_records.json
```

---

## 4. Running the Pipeline

### Option A: Full Pipeline (API + UI)

**Terminal 1 — Start API:**
```bash
venv\Scripts\activate
python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Start UI:**
```bash
venv\Scripts\activate
python ui/app.py
```

Open http://localhost:7860

### Option B: Pipeline as Python Module

```python
from PIL import Image
from pipeline import FoodPipeline

pipeline = FoodPipeline(
    detector_path="models/yolov8n_food_best.pt",
    classifier_path="models/efficientnet_b0_food101_best.pt",
    labels_path="models/classifier/idx_to_class.json",
    index_path="data/usda/faiss_index.bin",
    records_path="data/usda/nutrition_records.json",
    use_vlm=True
)

image = Image.open("data/test_food.jpg")
result = pipeline.run(image)

print(f"Status: {result.status}")
print(f"Items: {len(result.items)}")
for item in result.items:
    print(f"  {item.display_name}: {item.estimated_grams}g, {item.nutrition_total.calories_kcal} kcal")
```

---

## 5. Testing & Evaluation Scripts

All scripts assume you are in the project root with venv activated.

### Quick Reference

| Script | What It Does | Time | Needs API Key? |
|--------|-------------|------|----------------|
| `playground.py` | Walks through VLM data flow | Instant | No |
| `scripts/test_vlm.py` | Offline schema/prompt tests | Instant | No |
| `scripts/test_pipeline_real.py` | Full pipeline on test image | 5–15s | Yes |
| `scripts/test_api.py` | Tests the API endpoint | 5–15s | Yes |
| `scripts/eval_vlm.py` | 10-image VLM evaluation | 5–20 min | Yes |

### Playground (Data Flow Walkthrough)

```bash
python -u playground.py
```

Shows: input JSON → Pydantic validation → prompt → expected VLM output.

### Offline VLM Tests

```bash
python scripts/test_vlm.py
```

Validates schemas, prompt builder, and client setup without calling any model.

### Full Pipeline Test

```bash
python -u scripts/test_pipeline_real.py
```

Runs the complete pipeline (detection → classification → portion → nutrition → VLM) on `data/test_food.jpg`.

### API Test

```bash
# Start the API first in another terminal
python -m uvicorn api.app:app --port 8000

# Run the test
python scripts/test_api.py
```

### 10-Image VLM Evaluation

```bash
python -u scripts/eval_vlm.py
```

Tests the VLM on 10 food images from `data/eval/` with deliberately wrong CV predictions. Results are saved to `data/eval_results.json`.

---

## 6. Visualization Scripts

### Embedding Visualization (t-SNE + Similarity)

```bash
python -u scripts/embedding_vis.py
```

**Generates:**
- `output/tsne_food101.png` — 2D embedding plot of 101 food classes
- `output/similarity_heatmap.png` — cosine similarity matrix
- `output/confusion_pairs.png` — most similar food pairs

**Time:** 2–3 minutes.

### Grad-CAM Visualization

```bash
python -u scripts/grad_cam.py
```

**Generates:** `output/gradcam_*.png` — Original → Heatmap → Overlay for each eval image.

**Time:** 1–2 minutes. Falls back to demo mode if classifier weights are missing.

### Build USDA Index

```bash
python scripts/build_usda_index.py
```

Rebuilds the FAISS nutrition index from USDA CSV files. Only needed if updating nutrition data.

---

## 7. Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|---------|
| `ModuleNotFoundError: No module named 'vlm'` | Running from wrong directory | `cd food-pipeline` first |
| `Cannot connect to backend` | API not running | Start with `uvicorn api.app:app` |
| `CUDA out of memory` | GPU too small for models | Use CPU mode (default) |
| `GOOGLE_API_KEY not set` | Missing `.env` file | Create `.env` with your key |
| `FAISS index not found` | Missing nutrition data | Run `scripts/build_usda_index.py` |
| `pip install timeout` | Slow internet | Re-run; pip resumes downloads |

### Checking Module Status

```bash
# API status
curl http://localhost:8000/status

# Docker status
docker-compose ps

# Docker logs
docker-compose logs cv-pipeline
docker-compose logs gradio-ui
```

---

## Project File Structure

```
food-pipeline/
├── pipeline.py                  # Main orchestrator
├── detector/
│   └── detector.py              # YOLOv8n-seg wrapper
├── classifier/
│   └── classifier.py            # EfficientNet-B0 wrapper
├── portion/
│   ├── portion.py               # Portion estimator
│   └── depth_estimator.py       # MiDaS_small wrapper
├── nutrition/
│   └── nutrition.py             # FAISS USDA lookup
├── vlm/
│   ├── refiner.py               # VLM refiner orchestrator
│   ├── schemas.py               # Pydantic input/output models
│   ├── prompts.py               # Prompt templates
│   ├── client.py                # Gemini/Gemma API client
│   └── normalization.py         # Food label normalization
├── api/
│   ├── app.py                   # FastAPI application
│   ├── models.py                # Pydantic response models
│   └── run.py                   # Uvicorn launcher
├── ui/
│   ├── app.py                   # Gradio frontend
│   ├── auth.py                  # User authentication
│   ├── db.py                    # SQLite meal logging
│   ├── profile.py               # BMR/TDEE calculator
│   └── llm_chat.py              # AI correction/manual chat
├── scripts/
│   ├── build_usda_index.py      # USDA index builder
│   ├── eval_vlm.py              # VLM evaluation
│   ├── embedding_vis.py         # t-SNE + similarity plots
│   ├── grad_cam.py              # Grad-CAM heatmaps
│   ├── test_pipeline_real.py    # Full pipeline test
│   ├── test_api.py              # API endpoint test
│   └── test_vlm.py              # Offline VLM tests
├── data/
│   ├── eval/                    # 10 evaluation images
│   ├── eval_results.json        # Evaluation results
│   ├── test_food.jpg            # Test image
│   └── usda/                    # USDA data & FAISS index
├── output/                      # Generated visualizations
├── docs/                        # Documentation
├── models/                      # Trained model weights
├── Dockerfile                   # CV pipeline container
├── docker-compose.yml           # Docker orchestration
└── requirements.txt             # Python dependencies
```
