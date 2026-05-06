# Data Setup Checklist

## ✅ Auto-Created (Done)
- [x] `output/` — for generated plots
- [x] `models/` — for model weights
- [x] `models/classifier/` — for labels file
- [x] `data/usda/foundation/` — for USDA Foundation Foods CSV
- [x] `data/usda/sr_legacy/` — for USDA SR Legacy CSV

## ❌ Manual Downloads Needed

### 1. Classifier Model Weights (from Luca's Google Drive)

Ask Luca for the Google Drive link, then download these 2 files:

```
models/efficientnet_b0_food101_best.pt    (~30 MB)
models/classifier/idx_to_class.json       (~2 KB)
```

Place them in:
```
C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\models\
C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\models\classifier\
```

**Why needed:** Grad-CAM visualization, classifier unit tests

---

### 2. USDA Nutrition Data (free download)

Step 1: Go to https://fdc.nal.usda.gov/download-datasets.html

Step 2: Download **Foundation Foods** CSV
- Click "Foundation Foods" → download CSV zip
- Extract and place CSV files in:
  `data/usda/foundation/`

Step 3: Download **SR Legacy** CSV
- Click "SR Legacy" → download CSV zip
- Extract and place CSV files in:
  `data/usda/sr_legacy/`

**Why needed:** The nutrition lookup module needs this to build the FAISS index

---

### 3. Build the FAISS Index (after downloading USDA data)

Once USDA CSVs are in place, run:

```bash
venv\Scripts\activate
python scripts/build_usda_index.py
```

This creates:
- `data/usda/faiss_index.bin` (~50 MB)
- `data/usda/nutrition_records.json`

**Why needed:** Nutrition lookup works only with the FAISS index

---

## Quick Check — Verify Everything

After all downloads, run this to verify:

```bash
# Check model weights exist
dir models\efficientnet_b0_food101_best.pt
dir models\classifier\idx_to_class.json

# Check USDA data exists
dir data\usda\foundation\*.csv
dir data\usda\sr_legacy\*.csv

# Check FAISS index (only after building)
dir data\usda\faiss_index.bin

# Run full verification
python scripts/test_vlm.py
python -u scripts/embedding_vis.py
python -u scripts/grad_cam.py
```

---

## Current Folder Structure (with all data)

```
food-pipeline/
├── models/
│   ├── efficientnet_b0_food101_best.pt   ← FROM GOOGLE DRIVE
│   └── classifier/
│       └── idx_to_class.json              ← FROM GOOGLE DRIVE
├── data/
│   ├── test_food.jpg                      ✅ Already have
│   ├── eval/                              ✅ 10 images already have
│   │   ├── pizza.jpg
│   │   ├── burger.jpg
│   │   └── ... (8 more)
│   ├── eval_results.json                  ✅ Already have
│   └── usda/
│       ├── foundation/                    ← NEED USDA DOWNLOAD
│       │   └── *.csv
│       ├── sr_legacy/                     ← NEED USDA DOWNLOAD
│       │   └── *.csv
│       ├── faiss_index.bin                ← BUILT BY SCRIPT
│       └── nutrition_records.json         ← BUILT BY SCRIPT
├── output/                                ✅ Created (empty, for plots)
└── ...
```
