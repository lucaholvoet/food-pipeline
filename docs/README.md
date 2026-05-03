# Project Documentation

Documentation for the **Food Calorie & Nutrient Estimator** — a hybrid CV + VLM food recognition and nutrition estimation system.

---

## Documents

| Document | Description |
|----------|-------------|
| [**Pipeline Guide**](PIPELINE_GUIDE.md) | Complete technical reference: data sources, CV pipeline, VLM pipeline, API, output formats |
| [**Project Report**](report.md) | Academic report: objectives, methodology, experiments, results, discussion |
| [**User Manual**](USER_MANUAL.md) | How to use the app: analyzing meals, logging, dashboard, profile, troubleshooting |
| [**Run & Test Guide**](RUN_AND_TEST_GUIDE.md) | Setup, installation, running scripts, evaluation, and troubleshooting |

## Suggested Reading Order

1. [**Pipeline Guide**](PIPELINE_GUIDE.md) — understand how everything works
2. [**Project Report**](report.md) — academic context and experimental results
3. [**User Manual**](USER_MANUAL.md) — how to use the application
4. [**Run & Test Guide**](RUN_AND_TEST_GUIDE.md) — how to set up and run the system

## Generated Outputs

Visualization outputs are in the project `output/` folder:

| File | Description |
|------|-------------|
| `tsne_food101.png` | 2D t-SNE embedding plot of 101 Food-101 classes |
| `similarity_heatmap.png` | Cosine similarity matrix between food classes |
| `confusion_pairs.png` | Bar chart of most similar food pairs |
| `gradcam_*.png` | Grad-CAM heatmaps per evaluation image |

Evaluation data:
- `data/eval/` — 10 evaluation food images
- `data/eval_results.json` — VLM evaluation results
