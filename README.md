# 🍽️ Food Calorie & Nutrient Estimator

A full-stack AI-powered food calorie estimation system. Take a photo of your meal and get instant per-item nutritional analysis — calories, protein, fat, carbs, and fiber. Built with computer vision, deep learning, and LLM refinement.

**Live demo:** [http://65.109.133.173:7860](http://65.109.133.173:7860)

---

## Pipeline Overview

```
Photo → YOLOv8 Detection → EfficientNet-B0 Classification → MiDaS Depth
      → Portion Estimation → USDA FAISS Lookup → Gemini Flash Refinement
      → Nutrition Output
```

---

## Features

### CV Pipeline
- **Food detection** — YOLOv8n-seg trained on FoodSeg103 (mAP50: 0.934)
- **Food classification** — EfficientNet-B0 fine-tuned on Food-101 (87.37% accuracy, 101 classes)
- **Depth estimation** — MiDaS_small for height-based volume estimation
- **Portion estimation** — plate reference scale + MiDaS depth + per-food density table
- **Nutrition lookup** — FAISS semantic search over 9,013 USDA food database entries

### VLM Refinement
- **Gemini Flash** via Google AI Studio
- Triggers when CV confidence < 70% or no plate detected
- Corrects food labels, re-estimates portions, recalculates nutrition
- Shows CV → VLM confidence comparison per item

### App Features
- 🔐 **User accounts** — register/login with hashed passwords, per-user data
- 📋 **Meal logging** — log from CV result, VLM result, AI correction, or manual entry
- 💬 **AI correction chat** — disagree with the pipeline result and correct it via chat
- ✏️ **Manual logging** — describe what you ate in natural language, Gemini estimates nutrition
- 📊 **Dashboard** — daily calories vs target, 7-day bar chart with dates, meals list
- 👤 **Profile** — BMR/TDEE calculation, goal date picker, personalized calorie targets
- 🌐 **Deployed** — accessible from any device, anywhere

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection | YOLOv8n-seg (Ultralytics) |
| Classification | EfficientNet-B0 (timm) |
| Depth estimation | MiDaS_small (torch.hub) |
| Nutrition lookup | FAISS + sentence-transformers |
| VLM refinement | Gemini Flash (Google AI Studio) |
| Chat logging | Gemini 2.5 Flash Lite |
| API | FastAPI |
| Frontend | Gradio 5.29.0 |
| Database | SQLite |
| Deployment | Docker + Hetzner CX23 |

---

## Project Structure

```
food-pipeline/
├── pipeline.py                  # Main orchestrator
├── api/
│   ├── main.py                  # FastAPI POST /analyze endpoint
│   └── models.py                # CVPipelineOutput Pydantic model
├── detector/
│   └── detector.py              # YOLOv8n-seg wrapper
├── classifier/
│   └── classifier.py            # EfficientNet-B0 wrapper
├── portion/
│   ├── portion.py               # Portion estimator (plate + MiDaS + density)
│   └── depth_estimator.py       # MiDaS_small wrapper
├── nutrition/
│   ├── nutrition.py             # FAISS USDA lookup
│   ├── usda.index               # FAISS index (9,013 entries)
│   └── usda_records.json        # USDA nutrition records
├── vlm/
│   ├── refiner.py               # VLMRefiner — calls Gemini Flash
│   ├── schemas.py               # Pydantic schemas (VLMRequest, VLMResponse)
│   └── client.py                # google-genai SDK client
├── ui/
│   ├── app.py                   # Gradio frontend
│   ├── db.py                    # SQLite meal logging
│   ├── profile.py               # BMR/TDEE/calorie target calculation
│   ├── auth.py                  # Login/register with SHA256+salt
│   ├── llm_chat.py              # Gemini chat for correction + manual logging
│   ├── Dockerfile
│   └── requirements.txt
├── models/
│   ├── yolov8n_food_best.pt           # YOLOv8 weights (6.4MB)
│   ├── efficientnet_b0_food101_best.pt  # EfficientNet weights (18.3MB)
│   └── idx_to_class.json              # Food-101 label map
├── Dockerfile                   # CV pipeline container
└── docker-compose.yml           # Docker orchestration
```

---

## Setup

### Prerequisites
- Docker Desktop
- Google AI Studio API key ([get one here](https://ai.google.dev/gemini-api/docs/api-key))

### 1. Clone the repo
```bash
git clone https://github.com/lucaholvoet/food-pipeline.git
cd food-pipeline
```

All model weights and nutrition data are included in the repo — no separate download needed.

### 2. Create `.env` file
```bash
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### 3. Run
```bash
docker-compose up --build
```

Open [http://localhost:7860](http://localhost:7860)

---

## API

### `POST /analyze`
Send a food image, get back nutrition analysis.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@meal.jpg"
```

**Response** (CV result):
```json
{
  "image_id": "uuid",
  "status": "success",
  "average_confidence": 0.87,
  "plate_detected": true,
  "items": [
    {
      "item_id": 1,
      "display_name": "Hamburger",
      "estimated_grams": 250,
      "portion_method": "plate_reference_midas",
      "classification_confidence": 0.87,
      "nutrition_total": {
        "calories_kcal": 625,
        "protein_g": 32.5,
        "fat_g": 30.0,
        "carbs_g": 60.0,
        "fiber_g": 4.8
      }
    }
  ],
  "totals": { "calories_kcal": 625, "protein_g": 32.5, "fat_g": 30.0, "carbs_g": 60.0, "fiber_g": 4.8 },
  "processing_time_ms": 1243
}
```

When VLM refinement triggers, response includes `refinement_status` and per-item `action` (corrected/confirmed).

---

## Team

| Name | Role |
|------|------|
| Emiel | CV pipeline — YOLOv8, EfficientNet-B0, portion estimation |
| Luca | CV pipeline — MiDaS, FAISS, FastAPI, UI, deployment |
| Mahesh | VLM refinement — Gemini Flash integration, schema design |
| Furaha | VLM refinement — prompt engineering, evaluation |

---

## Known Limitations

- **Portion accuracy** — MiDaS calibration in progress using Nutrition5k dataset. Current multiplier (8.0) is a rough estimate.
- **101 food classes** — EfficientNet-B0 is limited to Food-101 classes. Many home-cooked foods not covered.
- **No GPU** — inference runs on CPU (~1-2 seconds). Server has no GPU.
- **USDA coverage** — 9,013 entries. Branded and specialty foods often not found.
- **Login session** — Gradio `gr.State` resets on page reload. Users must log in again after closing the browser.

---

## License

For academic use only.