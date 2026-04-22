# Food Calorie & Nutrient Estimator — Run & Test Guide

**Authors:** Mahesh & Furaha (VLM Refinement)  
**Last Updated:** April 22, 2026

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Install Dependencies](#2-install-dependencies)
3. [Run & Test Commands](#3-run-and-test-commands)
4. [Expected Outcomes](#4-expected-outcomes)
5. [Test Results](#5-test-results)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Environment Setup

### Prerequisites

| Requirement | Version | How to Check |
|-------------|---------|--------------|
| Python | 3.11+ | `python --version` |
| Git | Any | `git --version` |
| Ollama | 0.21+ | `ollama --version` |

### Step 1: Clone Repository

```bash
git clone https://github.com/lucaholvoet/food-pipeline.git
cd food-pipeline
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**Windows (PowerShell):**
```powershell
# If you get execution policy error, run this first:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemotePaid

# Then activate:
venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

You should see `(venv)` at the start of your prompt.

### Step 4: Install Ollama

1. Download from [ollama.com](https://ollama.com)
2. Install and restart terminal
3. Verify: `ollama --version`
4. Download the VLM model:
```bash
ollama pull gemma4:e4b
```
This downloads ~9.6 GB. One-time only.

---

## 2. Install Dependencies

### Core Dependencies

```bash
# PyTorch (CPU version)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# Project dependencies
pip install -r requirements.txt
```

### Additional Dependencies (for VLM + visualization)

```bash
# VLM and validation
pip install pydantic google-generativeai python-dotenv

# Visualization (for embedding_vis.py and grad_cam.py)
pip install scikit-learn matplotlib sentence-transformers
```

### Verify All Packages

```bash
pip list | findstr /i "torch pydantic google scikit matplotlib sentence pillow requests fastapi ultralytics faiss numpy"
```

**Expected output — all of these should appear:**

| Package | Version |
|---------|---------|
| torch | 2.2.2 |
| pydantic | 2.x |
| google-generativeai | 0.8.x |
| scikit-learn | 1.x |
| matplotlib | 3.x |
| sentence-transformers | 3.x |
| Pillow | 10.x |
| requests | 2.x |
| ultralytics | 8.2.x |
| faiss-cpu | 1.8.x |
| numpy | 1.26.x |

---

## 3. Run and Test Commands

All commands assume you are in the project root with venv activated:

```bash
cd food-pipeline
venv\Scripts\activate   # Windows
```

### Quick Reference Table

| # | Command | What It Does | Time | Needs Ollama? |
|---|---------|-------------|------|---------------|
| 1 | `python -u playground.py` | Walks through the data flow | Instant | No |
| 2 | `python scripts/test_vlm.py` | Offline tests (schemas, prompts) | Instant | No |
| 3 | `python -u scripts/test_vlm_fast.py` | Live VLM test with food image | 2-4 min | Yes |
| 4 | `python -u scripts/test_vlm_fast.py path\photo.jpg` | Test with your own image | 2-4 min | Yes |
| 5 | `python -u scripts/eval_vlm.py` | 10-image evaluation | 25-40 min | Yes |
| 6 | `python -u scripts/embedding_vis.py` | t-SNE + similarity plots | 2-3 min | No |
| 7 | `python -u scripts/grad_cam.py` | Grad-CAM heatmaps | 1-2 min | No |

---

### Command 1: Playground (Data Flow Walkthrough)

```bash
python -u playground.py
```

**What it does:** Reads example JSON, validates with Pydantic, shows the prompt that gets sent to Gemma 4, displays the expected output.

**Expected output:**
```
==================================================
STEP 1: What the VLM receives from CV pipeline
==================================================
{
  "image_base64": "<base64-encoded-image-data>",
  "reason": "low_confidence",
  "trigger_threshold": 0.7,
  "cv_output": { ... }
}

==================================================
STEP 2: Validate with Pydantic schemas
==================================================
Valid! Image ID: 550e8400-e29b-41d4-a716-446655440000
Items: 1
Avg confidence: 0.62
Food: fried_rice

==================================================
STEP 3: The prompt sent to Gemma 4
==================================================
System prompt (1000 chars): ...
User prompt (2339 chars): ...

==================================================
STEP 4: What the VLM returns
==================================================
Status: completed
Action: corrected
Original: fried_rice
Corrected to: bibimbap
VLM confidence: 0.91
Portion: 280.0g

==================================================
STEP 5: Full output JSON
==================================================
{ ... full VLMResponse JSON ... }
```

---

### Command 2: Offline Tests

```bash
python scripts/test_vlm.py
```

**What it does:** Validates all Pydantic schemas, tests prompt builder, checks client setup.

**Expected output:**
```
==================================================
VLM Module — Offline Tests
==================================================
📋 Testing schemas...
  ✅ Input schema valid — image_id: 550e8400-...
  ✅ Output schema valid — 1 items

📝 Testing prompt builder...
  ✅ System prompt: 1000 chars
  ✅ User prompt: 2339 chars

🔌 Testing client setup...
  ✅ Ollama client — model: gemma4:e4b
  ✅ Google client — model: gemma-4-27b-it

==================================================
All tests passed! ✅
==================================================
```

---

### Command 3: Live VLM Test

**Before running:** Make sure Ollama is running (system tray icon or `ollama serve`).

```bash
python -u scripts/test_vlm_fast.py
```

**What it does:** Sends a real food image to Gemma 4 via Ollama with a mock CV prediction.

**Expected output:**
```
=======================================================
Fast VLM Test - gemma4:e4b
=======================================================

Image: ...\data\test_food.jpg
Image size: 71968 chars
CV prediction: spaghetti_bolognese (55%)

Calling gemma4:e4b...

=======================================================
Response (142573ms / 142.6s)
=======================================================
{
  "items": [{
    "item_id": 1,
    "action": "corrected",
    "food_name": "Cheese Pancakes",
    "vlm_confidence": 0.95,
    "estimated_grams": 280
  }]
}

  Result: CORRECTED
  Food:   Cheese Pancakes (confidence: 0.95)
  Portion: 280g, 220 kcal/100g
```

**Time:** 2-4 minutes (first run is slower due to model loading).

---

### Command 4: Test With Your Own Image

```bash
python -u scripts/test_vlm_fast.py "C:\path\to\your\food_photo.jpg"
```

**Expected output:** Same format as Command 3, but with your image.

---

### Command 5: 10-Image Evaluation

```bash
python -u scripts/eval_vlm.py
```

**What it does:** Tests Gemma 4 on 10 food images with deliberately wrong CV predictions.

**Expected output:**
```
============================================================
VLM Evaluation - Gemma 4 on 10 Food Images
============================================================

[1/10] pizza.jpg
  Actual:    pizza
  CV said:   pizza (45%)
  Calling gemma4:e4b... 128.7s
  VLM said:  pizza (95%) [EXACT]
  Action:    confirmed | 600g

[2/10] burger.jpg
  Actual:    hamburger
  CV said:   hot_dog (38%)
  Calling gemma4:e4b... 129.8s
  VLM said:  burger (95%) [PARTIAL]
  Action:    corrected | 300g

... (8 more images) ...

============================================================
EVALUATION SUMMARY
============================================================
  Total images:    10
  Exact matches:   5 (50%)
  Partial matches: 2 (20%)
  Wrong:           3 (30%)
  Overall accuracy: 70%
  Corrections made:  8
  Corrections right: 5
  Avg time/image:    146.0s

Results saved to: data/eval_results.json
```

**Time:** 25-40 minutes total (~2.5 min per image).

**Results also saved to:** `data/eval_results.json`

---

### Command 6: Embedding Visualization (Lab 3)

```bash
python -u scripts/embedding_vis.py
```

**What it does:** Generates t-SNE plots, similarity heatmaps, and confusion analysis.

**Expected output:**
```
============================================================
Food-101 Embedding Visualization
============================================================

[1/4] Loading sentence-transformer model...
[1/4] Encoding 101 food names...
Batches: 100%|██████████| 4/4 [00:02<00:00]
  Embedding shape: (101, 384)

[2/4] Computing cosine similarity matrix...
  Top 10 most similar food pairs:
    filet_mignon            <-> prime_rib              sim=0.89
    spaghetti_bolognese     <-> spaghetti_carbonara     sim=0.88
    chocolate_cake          <-> red_velvet_cake         sim=0.87
    ...

[3/4] Running t-SNE...
[4/4] Generating plots...
  Saved: output/tsne_food101.png
  Saved: output/similarity_heatmap.png
  Saved: output/confusion_pairs.png
```

**Files generated in `output/`:**
- `tsne_food101.png` — 101 foods plotted in 2D, colored by category
- `similarity_heatmap.png` — cosine similarity matrix (30×30 foods)
- `confusion_pairs.png` — bar chart of similar food pairs

**Time:** 2-3 minutes.

---

### Command 7: Grad-CAM Visualization (Lab 2)

```bash
python -u scripts/grad_cam.py
```

**What it does:** Shows which pixels the CNN focuses on for each food prediction.

**Expected output (with model weights):**
```
============================================================
Grad-CAM: CNN Interpretation
============================================================

Loading trained model...
  Model loaded!

Processing 10 images...
  pizza.jpg: predicted as Pizza (92.1%)
  Saved: output/gradcam_pizza.png
  burger.jpg: predicted as Hamburger (88.3%)
  Saved: output/gradcam_burger.png
  ...
```

**Expected output (without model weights — demo mode):**
```
============================================================
Grad-CAM: CNN Interpretation
============================================================

Model weights not found: models/efficientnet_b0_food101_best.pt
Generating demo with random weights...

Processing 10 images...
  pizza.jpg: predicted as ... (DEMO - random weights)
```

**Files generated in `output/`:**
- `gradcam_pizza.png` — Original → Heatmap → Overlay
- `gradcam_burger.png` — same
- (one per eval image)

**Time:** 1-2 minutes.

---

## 4. Expected Outcomes

### VLM Module Outcomes

| Test | Expected Result | Success Criteria |
|------|----------------|------------------|
| Offline tests | All 3 tests pass | ✅ schemas, prompts, client |
| Live test | Returns JSON with food name, confidence, grams | ✅ valid JSON, action = confirmed/corrected |
| 10-image eval | 50-70% accuracy (exact + partial) | ✅ improves over CV alone |

### Evaluation Benchmarks

| Metric | CV Pipeline (before) | After VLM (expected) |
|--------|---------------------|---------------------|
| Exact Match | 30% | 50-60% |
| Exact + Partial | 40% | 70-80% |
| Wrong | 60% | 20-30% |
| Avg Confidence | ~41% | ~90% |
| Time per image | ~50ms | ~150s |

### Visualization Outcomes

| Visualization | What You Should See |
|--------------|-------------------|
| t-SNE plot | Desserts cluster together, Asian foods cluster together, meats cluster together |
| Similarity heatmap | Green diagonal (self=1.0), green blocks for similar foods, red for different foods |
| Confusion pairs | ramen/pho > 0.80 similarity, explaining CV confusion |
| Grad-CAM | Red overlay on food region, showing CNN attention |

---

## 5. Test Results

### Actual Results — Recorded April 22, 2026

**System:** Intel i7-1165G7, 32 GB RAM, NVIDIA T500 (4 GB), Windows 11  
**Model:** gemma4:e4b (9.6 GB, Ollama, CPU/GPU split 67/33)

#### VLM Evaluation — 10 Images

| # | Image | CV Prediction | VLM Prediction | Actual | Match |
|---|-------|--------------|----------------|--------|-------|
| 1 | pizza.jpg | pizza (45%) | pizza | pizza | EXACT ✅ |
| 2 | burger.jpg | hot_dog (38%) | burger | hamburger | PARTIAL 🟡 |
| 3 | sushi.jpg | sushi (50%) | Sushi Rolls | sushi | PARTIAL 🟡 |
| 4 | pasta.jpg | ramen (42%) | pasta | spaghetti_bolognese | WRONG ❌ |
| 5 | salad.jpg | caesar_salad (35%) | Salad Bowl | greek_salad | WRONG ❌ |
| 6 | pancakes.jpg | french_toast (40%) | pancakes | pancakes | EXACT ✅ |
| 7 | icecream.jpg | frozen_yogurt (33%) | ice cream | ice_cream | EXACT ✅ |
| 8 | steak.jpg | filet_mignon (48%) | steak | steak | EXACT ✅ |
| 9 | ramen.jpg | pho (36%) | Shrimp Noodle Soup | ramen | WRONG ❌ |
| 10 | cake.jpg | red_velvet_cake (41%) | chocolate_cake | chocolate_cake | EXACT ✅ |

#### Summary

| Metric | Value |
|--------|-------|
| Total images | 10 |
| Exact matches | 5 (50%) |
| Partial matches | 2 (20%) |
| Wrong | 3 (30%) |
| Overall accuracy (exact + partial) | **70%** |
| Corrections made | 8 |
| Corrections correct | 5 (63%) |
| Avg VLM confidence | 95% |
| Avg CV confidence | 41% |
| Avg time per image | 146 seconds |
| Total evaluation time | ~24 minutes |
| Cost | **$0.00** (fully local) |

#### Improvement Over CV Pipeline

| Metric | CV Only | After VLM | Change |
|--------|---------|-----------|--------|
| Exact match rate | 3/10 (30%) | 5/10 (50%) | **+67%** |
| Correct rate (exact+partial) | 4/10 (40%) | 7/10 (70%) | **+75%** |
| Error rate | 6/10 (60%) | 3/10 (30%) | **−50%** |

#### Known Issues

| Issue | Cause | Fix Applied |
|-------|-------|------------|
| VLM says "pasta" not "spaghetti_bolognese" | Prompt didn't include class list | ✅ Added Food-101 names to prompt (v2) |
| VLM says "Salad Bowl" not "greek_salad" | Same as above | ✅ Fixed in prompt v2 |
| VLM always returns 95% confidence | No calibration instruction | ✅ Added "0.3-1.0 honest" instruction |
| Slow inference (~2.5 min/image) | Small GPU (4 GB T500) | Acceptable for demo; faster on Colab |

---

## 6. Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|---------|
| `ModuleNotFoundError: No module named 'vlm'` | Running from wrong directory | `cd food-pipeline` then run |
| `ConnectionError: Ollama not responding` | Ollama not running | Run `ollama serve` in a separate terminal |
| `NameError: name '__file__' is not defined` | Running with `exec()` | Run with `python script.py` instead |
| `SyntaxWarning: "\S" is an invalid escape` | Windows path in docstring | Safe to ignore, doesn't affect execution |
| `FutureWarning: google.generativeai deprecated` | Old SDK version | Safe to ignore for now |
| `model not found` | gemma4:e4b not downloaded | Run `ollama pull gemma4:e4b` |
| `pip install timeout` | Slow internet | Re-run the command, it will continue |

### How to Check Ollama Status

```bash
# Is Ollama running?
ollama ps

# What models are downloaded?
ollama list

# Expected output:
# NAME            ID              SIZE
# gemma4:e4b      c6eb396dbd59    9.6 GB
```

---

## File Locations

### Project Files

```
food-pipeline/
├── vlm/                          # Your VLM module
│   ├── __init__.py               # Module header
│   ├── schemas.py                # Pydantic models
│   ├── prompts.py                # Prompt templates (v2)
│   ├── client.py                 # Gemma 4 client
│   ├── refiner.py                # Orchestrator
│   ├── example_input.json        # Sample CV output
│   └── example_output.json       # Sample VLM output
├── scripts/
│   ├── test_vlm.py               # Offline tests
│   ├── test_vlm_fast.py          # Live VLM test
│   ├── test_vlm_local.py         # Full pipeline test
│   ├── eval_vlm.py               # 10-image evaluation
│   ├── embedding_vis.py          # Lab 3: t-SNE/UMAP
│   ├── grad_cam.py               # Lab 2: Grad-CAM
│   ├── test_nutrition.py         # Luca's nutrition test
│   ├── test_portion.py           # Luca's portion test
│   └── build_usda_index.py      # USDA index builder
├── data/
│   ├── test_food.jpg             # Test image
│   ├── eval/                     # 10 eval images
│   └── eval_results.json         # Saved eval results
├── output/                       # Generated plots
│   ├── tsne_food101.png
│   ├── similarity_heatmap.png
│   ├── confusion_pairs.png
│   └── gradcam_*.png
├── playground.py                 # Step-by-step walkthrough
├── vlm_refinement_colab.ipynb    # Colab notebook
├── .env                          # API key config
└── requirements.txt              # Dependencies
```
