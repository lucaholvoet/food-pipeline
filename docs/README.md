# Project Documentation Index

This folder contains the main documentation for the **Food Calorie & Nutrient Estimator** project.

## Documents

- [report.md](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\docs\report.md) — Main project report with objectives, methodology, progress, experiments, results, discussion, and conclusion.
- [RUN_AND_TEST_GUIDE.md](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\docs\RUN_AND_TEST_GUIDE.md) — Step-by-step handbook for setup, installation, running scripts, expected outcomes, and troubleshooting.
- [DATA_SETUP.md](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\docs\DATA_SETUP.md) — Data and model download/setup checklist, including USDA data and classifier weights.
- [Full_Pipeline_Guide.pdf](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\docs\Full_Pipeline_Guide.pdf) — PDF guide explaining the full architecture, pipeline, team roles, and technical design.

## Suggested Reading Order

1. [report.md](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\docs\report.md)
2. [Full_Pipeline_Guide.pdf](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\docs\Full_Pipeline_Guide.pdf)
3. [RUN_AND_TEST_GUIDE.md](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\docs\RUN_AND_TEST_GUIDE.md)
4. [DATA_SETUP.md](C:\Users\mahes\OneDrive\Desktop\SEMESTER 02\Deep Learning\GitRepo_Project\Project\food-pipeline\docs\DATA_SETUP.md)

## Notes

- The VLM refinement module is implemented and tested locally.
- The detector and FastAPI integration are still pending.
- Some visualizations (Grad-CAM) currently run in demo mode until the trained classifier weights are added.
- Evaluation results are available in `data/eval_results.json`.

## Related Outputs

Generated plots are in the project `output/` folder:
- `tsne_food101.png`
- `similarity_heatmap.png`
- `confusion_pairs.png`
- `gradcam_*.png`

Evaluation data is in:
- `data/eval/`
- `data/eval_results.json`
