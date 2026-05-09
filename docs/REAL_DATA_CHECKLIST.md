# Real Data Checklist

This checklist shows exactly what is already available in the project, what is missing, where each file should go, and what to run after each step.

---

## 1. What is already available

### Code modules already present
- [classifier/classifier.py](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\classifier\classifier.py)
- [portion/portion.py](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\portion\portion.py)
- [nutrition/nutrition.py](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\nutrition\nutrition.py)
- [vlm/schemas.py](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\vlm\schemas.py)
- [vlm/prompts.py](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\vlm\prompts.py)
- [vlm/client.py](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\vlm\client.py)
- [vlm/refiner.py](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\vlm\refiner.py)

### Scripts already available
- [scripts/test_vlm.py](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\scripts\test_vlm.py)
- [scripts/test_vlm_fast.py](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\scripts\test_vlm_fast.py)
- [scripts/eval_vlm.py](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\scripts\eval_vlm.py)
- [scripts/embedding_vis.py](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\scripts\embedding_vis.py)
- [scripts/grad_cam.py](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\scripts\grad_cam.py)
- [scripts/build_usda_index.py](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\scripts\build_usda_index.py)

### Test data already available
- [data/test_food.jpg](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\data\test_food.jpg)
- [data/eval](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\data\eval) — contains 10 food images
- [data/eval_results.json](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\data\eval_results.json)

### Output already generated
- [output/tsne_food101.png](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\output\tsne_food101.png)
- [output/similarity_heatmap.png](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\output\similarity_heatmap.png)
- [output/confusion_pairs.png](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\output\confusion_pairs.png)
- [output/gradcam_pizza.png](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\output\gradcam_pizza.png)

### Environment already available
- Python virtual environment exists
- Ollama installed
- Gemma 4 e4b model downloaded
- Required VLM/visualization packages installed

---

## 2. What is missing

### Missing item A — classifier weights from Luca

These files are required for **real classifier inference** and **real Grad-CAM**:

- [models/efficientnet_b0_food101_best.pt](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\models\efficientnet_b0_food101_best.pt)
- [models/classifier/idx_to_class.json](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\models\classifier\idx_to_class.json)

#### Action
Ask Luca for the Google Drive / Colab export files and place them exactly here:

```text
C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\models\efficientnet_b0_food101_best.pt
C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\models\classifier\idx_to_class.json
```

#### After downloading, run
```bash
venv\Scripts\activate
python -u scripts\grad_cam.py
```

#### Expected result
- Grad-CAM stops running in demo mode
- classifier predictions become meaningful
- `output/gradcam_*.png` become real interpretation images

---

### Missing item B — USDA food CSV datasets

These are required for **real nutrition lookup**.

#### Download source
USDA FoodData Central:
- https://fdc.nal.usda.gov/download-datasets.html

#### Required datasets
1. **Foundation Foods** CSV
2. **SR Legacy** CSV

#### Where to place them
Foundation Foods:
- [data/usda/foundation](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\data\usda\foundation)

SR Legacy:
- [data/usda/sr_legacy](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\data\usda\sr_legacy)

#### Action
- Download ZIPs from USDA site
- Extract the CSV files
- Copy the CSV files into the folders above

#### After downloading, run
```bash
venv\Scripts\activate
python scripts\build_usda_index.py
```

#### Expected result
New files should appear, for example:
- `data/usda/faiss_index.bin`
- `data/usda/nutrition_records.json`

Then run:
```bash
python scripts\test_nutrition.py
python scripts\test_full_nutrition_portion.py
```

---

### Missing item C — detector implementation

This is not a download; it is currently missing code.

Missing file:
- [detector/__init__.py](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\detector\__init__.py)

#### Why needed
Without detector output, the project does **not** yet have:
- real food bounding boxes
- real segmentation masks
- real plate masks
- real end-to-end CV JSON

#### Current effect
Right now the VLM is tested using **real images + mock CV predictions**, not real detector output.

---

### Missing item D — API integration

Also not a download; missing implementation.

Missing file:
- [api/__init__.py](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\api\__init__.py)

#### Why needed
Without API integration, the project is not yet exposed as a final app/service endpoint.

---

## 3. Exact order to move toward real data

### Step 1 — Add classifier weights
- [ ] Get `efficientnet_b0_food101_best.pt` from Luca
- [ ] Get `idx_to_class.json` from Luca
- [ ] Put them in the correct `models/` folders

Then test:
```bash
python -u scripts\grad_cam.py
```

### Step 2 — Add USDA data
- [ ] Download Foundation Foods CSV
- [ ] Download SR Legacy CSV
- [ ] Extract into `data/usda/foundation/` and `data/usda/sr_legacy/`

Then build index:
```bash
python scripts\build_usda_index.py
```

Then test:
```bash
python scripts\test_nutrition.py
python scripts\test_full_nutrition_portion.py
```

### Step 3 — Real classifier + nutrition workflow
Once steps 1 and 2 are done, you can test:
- real classifier outputs
- real nutrition retrieval
- real Grad-CAM

### Step 4 — Detector
- [ ] Implement or get Luca’s YOLO detector code

### Step 5 — End-to-end pipeline
Once detector exists, test:
- image → detector → classifier → portion → nutrition → VLM

---

## 4. What you can already do now

### Already working now
- [x] Run VLM offline tests
- [x] Run Gemma 4 local test on food images
- [x] Run 10-image VLM evaluation
- [x] Generate embedding visualizations
- [x] Generate Grad-CAM in demo mode
- [x] Write documentation and report

### Not yet fully real
- [ ] Real classifier inference from trained weights
- [ ] Real nutrition lookup from USDA index
- [ ] Real detector masks and plate reference
- [ ] Full end-to-end pipeline

---

## 5. Verification commands after each stage

### Verify classifier assets
```bash
dir models\efficientnet_b0_food101_best.pt
dir models\classifier\idx_to_class.json
```

### Verify USDA assets
```bash
dir data\usda\foundation
dir data\usda\sr_legacy
```

### Verify built nutrition index
```bash
dir data\usda
```
Look for generated index/records files.

### Verify VLM still works
```bash
python scripts\test_vlm.py
python -u scripts\test_vlm_fast.py
```

### Verify visualizations
```bash
python -u scripts\embedding_vis.py
python -u scripts\grad_cam.py
```

---

## 6. Final target state

You will know the project is using real data when all of the following are true:

- [ ] real classifier weights are present
- [ ] USDA CSV data is present
- [ ] FAISS nutrition index is built
- [ ] detector produces real masks/bboxes
- [ ] classifier can run on detector crops
- [ ] portion estimator uses real masks
- [ ] nutrition lookup uses real USDA index
- [ ] VLM receives real CV output instead of mock JSON

At that point, the system becomes a **true end-to-end real-data food pipeline**.

---

## 7. Recommended next action

**Best next step:** get Luca’s 2 classifier files first.

That unlocks:
- real Grad-CAM
- real classifier testing
- more meaningful report screenshots

After that, download USDA CSVs.

---

## 8. Quick summary

### Already available
- code
- scripts
- local VLM
- evaluation images
- visualizations

### Must be downloaded
- Luca’s classifier weights
- USDA Foundation Foods
- USDA SR Legacy

### Must be implemented
- detector
- API
