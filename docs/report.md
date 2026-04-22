# Food Calorie & Nutrient Estimator — Project Report

**Course:** Deep Learning  
**Semester:** 02  
**Project Title:** Food Calorie & Nutrient Estimator using CV + VLM  
**Team Members:** Emiel, Luca, Mahesh, Furaha  
**Date:** April 2026

---

## Abstract

This project develops a hybrid AI system for automatic food recognition and nutrition estimation from meal images. The system combines a **computer vision (CV) pipeline** with a **vision language model (VLM)** refinement stage. The CV pipeline is responsible for food detection, food classification, portion estimation, and nutrition lookup. The VLM stage is used only when the CV pipeline is uncertain, enabling the system to correct low-confidence predictions while keeping computation manageable.

The classifier is based on **EfficientNet-B0** trained on the **Food-101** dataset and achieved **87.37% accuracy**. The nutrition module uses **sentence-transformer embeddings + FAISS** to retrieve the closest USDA food record. The VLM refinement stage uses **Gemma 4 (open-source)** through **Ollama** for local inference. Initial local evaluation on 10 food images showed that the VLM improved the final exact-or-partial match rate from approximately **40% to 70%**, demonstrating that a hybrid CV + VLM design is practical for ambiguous food recognition tasks.

---

## 1. Introduction

Estimating calories and nutrition from food images is an important problem in health tracking, diet analysis, and meal logging applications. However, this task is difficult because:

- many foods look visually similar,
- food portion sizes vary significantly,
- a single image may contain multiple items,
- nutrition databases use different naming conventions than image classifiers.

A single CV model is often fast but may fail on ambiguous foods. On the other hand, modern multimodal VLMs can reason better from visual context, but they are slower and more computationally expensive. Therefore, this project uses a **two-stage hybrid pipeline**:

1. First, a CV pipeline processes the image quickly.
2. If the confidence is high enough, the system accepts the result.
3. If confidence is low, the result is passed to a VLM refinement stage.

This design balances **speed**, **accuracy**, and **interpretability**.

---

## 2. Project Objectives

The main objectives of the project are:

- detect food items in an input image,
- classify each item into one of the Food-101 food classes,
- estimate portion size in grams,
- retrieve nutrition values from USDA food data,
- refine uncertain predictions using a VLM,
- output structured JSON suitable for integration into a meal logging system.

---

## 3. Overall System Architecture

The system is designed as a modular pipeline:

1. **YOLOv8n-seg detector** — detects food items and plate regions.
2. **EfficientNet-B0 classifier** — classifies food crops into Food-101 classes.
3. **Portion estimator** — estimates grams from segmentation masks and plate reference.
4. **Nutrition lookup** — retrieves USDA nutrition per 100g using semantic search.
5. **Confidence gate** — checks average confidence.
6. **Gemma 4 VLM refinement** — corrects or confirms low-confidence cases.
7. **Final JSON output** — item-level and total nutrition.

At the current stage of the project, the detector and FastAPI integration are not yet implemented, but the classifier, portion estimator, nutrition lookup, and VLM refinement modules are available.

---

## 4. Dataset and Models

### 4.1 Food Classification Dataset

The classifier uses the **Food-101** dataset, which contains **101,000 food images** across **101 food classes**. The data is split into:

- **75,750 training images**
- **25,250 validation/test images**

This dataset is well-suited for food recognition because it contains a broad variety of meal categories such as pizza, ramen, sushi, salad, burgers, and cakes.

### 4.2 Classifier Model

The classification module is implemented in `classifier/classifier.py`. It uses **EfficientNet-B0** with a custom classification head:

- Dropout(0.2)
- Linear(1280 → 512)
- SiLU activation
- Dropout(0.2)
- Linear(512 → 101)

The model achieved **87.37% accuracy** after training and testing.

### 4.3 Portion Estimation Model

The portion estimator is implemented in `portion/portion.py`. It does not use a learned neural network, but instead a **geometry + density lookup approach**:

- segmentation mask area provides food size in pixels,
- detected plate provides real-world scaling,
- food-specific density table provides approximate volume-to-mass conversion.

This is a practical engineering solution that allows portion estimation without requiring a separate 3D estimation model.

### 4.4 Nutrition Retrieval Model

The nutrition module in `nutrition/nutrition.py` uses:

- **SentenceTransformer** (`all-MiniLM-L6-v2`) for text embeddings,
- **FAISS** for nearest-neighbor search,
- USDA records as the nutrition source.

This allows the system to map predicted food labels to the closest USDA food description and retrieve per-100g nutritional information.

### 4.5 VLM Refinement Model

The VLM refinement stage uses **Gemma 4 E4B** through **Ollama**. This model was selected because:

- it is open-source,
- it supports image + text input,
- it can run locally on the available hardware,
- it avoids cloud cost.

The VLM receives the original image plus the CV pipeline output and produces corrected or confirmed predictions in structured JSON format.

---

## 5. Methodology

### 5.1 CV Pipeline Workflow

The intended CV workflow is as follows:

1. Input image is passed to the YOLO detector.
2. Each detected food region is cropped.
3. Each crop is classified by EfficientNet-B0.
4. Portion size is estimated from the segmentation mask.
5. Nutrition values are retrieved from USDA FAISS lookup.
6. Average confidence is computed.
7. If the confidence is lower than a threshold, VLM refinement is triggered.

### 5.2 Why a Confidence Threshold is Needed

The VLM is slower than the CV pipeline, so it is not efficient to use it on every image. Instead, the system uses a threshold to decide when the CV result is too uncertain.

At the current project stage, **0.70** is used as a **provisional threshold**. This is not claimed to be a universally optimal value. Instead, it is a practical initial cutoff motivated by research on confidence-based routing, selective classification, and human-in-the-loop systems. The exact threshold should ideally be tuned using validation data.

### 5.3 VLM Refinement Workflow

The VLM stage was developed by Mahesh and Furaha. It follows this flow:

1. Validate input JSON using Pydantic schemas.
2. Build a prompt containing:
   - reason for refinement,
   - confidence threshold,
   - full CV output,
   - explicit Food-101 class constraints.
3. Send image + prompt to Gemma 4.
4. Parse the returned JSON.
5. Recalculate totals and return a structured `VLMResponse`.

### 5.4 Prompt Engineering

During development, prompt quality turned out to be very important. The first version of the prompt often returned:

- generic names like `pasta`,
- free-form descriptions instead of Food-101 labels,
- overconfident outputs (e.g. always 95%).

To address this, the prompt was updated to:

- include the **full Food-101 class list**,
- require exact Food-101 names when possible,
- enforce JSON-only responses,
- encourage more realistic confidence scores.

---

## 6. Implementation Progress

### 6.1 Completed Components

The following components are complete:

- **EfficientNet-B0 classifier**
- **Portion estimator**
- **USDA nutrition lookup**
- **VLM schemas**
- **VLM prompts**
- **Gemma 4 client**
- **VLM refiner module**
- **Offline tests**
- **Local live VLM test**
- **10-image evaluation framework**
- **Embedding visualization**
- **Grad-CAM visualization (demo mode)**
- **Colab notebook for VLM refinement**

### 6.2 Incomplete / Pending Components

The following components are not yet complete:

- **YOLOv8 detector** (`detector/__init__.py` is empty)
- **FastAPI integration** (`api/__init__.py` is empty)
- **Real end-to-end integration with actual detector outputs**
- **Classifier weight sharing / final deployment packaging**

This means that the current VLM stage is tested using **mock CV outputs**, not yet with the fully integrated detector.

---

## 7. Experimental Results

### 7.1 Offline Tests

The offline test script validated:

- input schema (`VLMRequest`),
- output schema (`VLMResponse`),
- prompt builder,
- client setup.

All offline tests passed successfully.

### 7.2 Live Single-Image Test

A local live test was run with Gemma 4 through Ollama. In that test:

- the mock CV prediction was: `spaghetti_bolognese (55%)`,
- the image actually showed a cheese pancake / toast dish,
- the VLM corrected the prediction to a pancake-based food,
- the output was returned as valid JSON.

This demonstrated that the VLM routing mechanism and prompt structure worked correctly.

### 7.3 Multi-Image Evaluation

A 10-image evaluation was performed on local hardware using Gemma 4 E4B. The recorded results were:

| Metric | Result |
|--------|--------|
| Total images | 10 |
| Exact matches | 5 |
| Partial matches | 2 |
| Wrong | 3 |
| Overall exact+partial accuracy | **70%** |
| Corrections made | 8 |
| Corrections right | 5 |
| Average inference time | **146s per image** |

### 7.4 Detailed Evaluation Table

| Image | Actual Food | CV Said | VLM Said | Match |
|------|-------------|---------|----------|-------|
| pizza.jpg | pizza | pizza | pizza | exact |
| burger.jpg | hamburger | hot_dog | burger | partial |
| sushi.jpg | sushi | sushi | Sushi Rolls | partial |
| pasta.jpg | spaghetti_bolognese | ramen | pasta | wrong |
| salad.jpg | greek_salad | caesar_salad | Salad Bowl | wrong |
| pancakes.jpg | pancakes | french_toast | pancakes | exact |
| icecream.jpg | ice_cream | frozen_yogurt | ice cream | exact |
| steak.jpg | steak | filet_mignon | steak | exact |
| ramen.jpg | ramen | pho | Shrimp Noodle Soup | wrong |
| cake.jpg | chocolate_cake | red_velvet_cake | chocolate_cake | exact |

### 7.5 Interpretation of Results

These results show that the VLM is useful, but not perfect.

**Positive findings:**
- it successfully corrected several wrong CV predictions,
- it improved the final exact-or-partial match rate,
- it handled obvious foods like pizza, pancakes, steak, ice cream, and cake well.

**Observed limitations:**
- it sometimes returned names that were semantically close but not exact Food-101 class names,
- it struggled with ambiguous dishes like ramen vs pho,
- generic categories like `pasta` or `salad bowl` reduced exact-match performance,
- inference time was high on local hardware.

---

## 8. Visualization and Analysis

To strengthen the interpretability aspect of the project, two visualization directions were developed.

### 8.1 Embedding Visualization (Lab 3 Alignment)

A t-SNE plot and cosine similarity heatmap were generated using sentence-transformer embeddings of Food-101 class names.

These visualizations showed that:

- dessert foods cluster together,
- salad classes are close to each other,
- pasta-related dishes show moderate similarity,
- confusion-prone foods (e.g. caesar_salad vs greek_salad, chocolate_cake vs red_velvet_cake) are semantically close in embedding space.

This helps explain why some foods are harder to distinguish for a classifier.

### 8.2 Grad-CAM Visualization (Lab 2 Alignment)

Grad-CAM plots were also generated. Since the real classifier weights were not yet placed in the project, Grad-CAM currently runs in **demo mode** using random weights. Therefore:

- the pipeline for visualization is working,
- but the visual explanations are not yet scientifically meaningful,
- final Grad-CAM analysis requires the real trained EfficientNet model weights from Luca.

---

## 9. Discussion

The project demonstrates the usefulness of a **hybrid CV + VLM system** for food understanding. Pure CV is efficient but makes mistakes on visually similar foods. A VLM can improve these low-confidence cases by incorporating broader visual reasoning.

However, several important lessons were learned:

1. **Prompt engineering matters a lot.** Small changes in prompt wording significantly changed result quality.
2. **Confidence thresholding is necessary.** Running a VLM on every image is too slow.
3. **Open-source local models are practical but slower.** Gemma 4 worked locally without cost, but latency was high.
4. **Exact label constraints are important.** Without explicit class constraints, VLMs tend to produce free-form names.
5. **Evaluation needs both exact and partial scoring.** In food recognition, semantically close labels are common.

---

## 10. Limitations

The current project has several limitations:

- no completed detector, so no fully integrated end-to-end food pipeline,
- no final API layer,
- no calibrated threshold tuning experiment yet,
- VLM evaluation used mock CV predictions,
- Grad-CAM currently lacks real trained weights,
- local inference speed is slow for production use.

---

## 11. Future Work

Future work should include:

- implementing the YOLO detector,
- integrating all modules into the FastAPI layer,
- tuning the confidence threshold empirically,
- evaluating prompt version 2 against prompt version 1,
- running more systematic experiments on a larger food image set,
- using real classifier weights for Grad-CAM,
- comparing local Gemma 4 against cloud-based multimodal models,
- testing calibration metrics such as ECE or reliability diagrams.

---

## 12. Conclusion

This project successfully developed the core parts of a hybrid food recognition pipeline. The CV components for **classification**, **portion estimation**, and **nutrition retrieval** are in place, and the VLM refinement stage has been designed, implemented, and tested locally using **Gemma 4**.

The classifier achieved **87.37% accuracy** on Food-101, and the VLM stage improved multi-image evaluation performance to **70% exact-or-partial accuracy** on a local 10-image benchmark. Although the project is not yet fully integrated end-to-end due to missing detector and API components, the work completed so far demonstrates that a selective CV + VLM design is both technically feasible and educationally valuable.

The system aligns well with core deep learning concepts from the course, including:

- CNN-based image classification,
- embeddings and semantic similarity,
- interpretability through Grad-CAM,
- generative AI / multimodal reasoning through VLMs.

Overall, the project provides a strong foundation for a future production-ready meal analysis system.

---

## References

1. Food-101 dataset  
2. EfficientNet architecture  
3. YOLOv8 segmentation  
4. SentenceTransformers (all-MiniLM-L6-v2)  
5. FAISS similarity search  
6. Gemma 4 multimodal model  
7. USDA FoodData Central database
