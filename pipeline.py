import time
import uuid
import base64
import numpy as np
from PIL import Image
from io import BytesIO

from detector.detector import FoodDetector
from classifier.classifier import FoodClassifier
from portion.portion import PortionEstimator
from nutrition.nutrition import NutritionLookup
from api.models import CVPipelineOutput, FoodItem, NutritionValues, TopPrediction

from vlm.refiner import VLMRefiner
from vlm.schemas import VLMRequest, RefinementReason

VLM_CONFIDENCE_THRESHOLD = 0.70

class FoodPipeline:
    def __init__(
        self,
        detector_path: str = "models/yolov8n_food_best.pt",
        classifier_path: str = "models/efficientnet_b0_food101_best.pt",
        labels_path: str = "models/idx_to_class.json",
        index_path: str = "nutrition/usda.index",
        records_path: str = "nutrition/usda_records.json",
        device: str = "cpu",
        use_vlm: bool = True
    ):
        print("Loading detector...")
        self.detector = FoodDetector(detector_path, device=device)
        print("Loading classifier...")
        self.classifier = FoodClassifier(classifier_path, labels_path, device=device)
        print("Loading portion estimator...")
        self.portion = PortionEstimator()
        print("Loading nutrition lookup...")
        self.nutrition = NutritionLookup(index_path, records_path, device=device)

        if use_vlm:
            try:
                self.refiner = VLMRefiner(backend="ollama", model="gemma4:26b")
                print("VLM refiner ready.")
            except Exception as e:
                print(f"VLM not available: {e}. Running CV-only.")
                self.refiner = None
        else:
            self.refiner = None

        print("Pipeline ready.")

    def _image_to_base64(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _compute_totals(self, items: list[FoodItem]) -> NutritionValues:
        return NutritionValues(
            calories_kcal=round(sum(i.nutrition_total.calories_kcal for i in items), 1),
            protein_g=round(sum(i.nutrition_total.protein_g for i in items), 1),
            fat_g=round(sum(i.nutrition_total.fat_g for i in items), 1),
            carbs_g=round(sum(i.nutrition_total.carbs_g for i in items), 1),
            fiber_g=round(sum(i.nutrition_total.fiber_g for i in items), 1),
        )

    def run(self, image: Image.Image) -> CVPipelineOutput:
        start_time = time.time()
        image_id = str(uuid.uuid4())
        warnings = []

        # Step 1 — detect
        detection = self.detector.detect(image)
        food_items_raw = detection["food_items"]
        plate_mask = detection["plate_mask"]
        plate_detected = detection["plate_detected"]
        img_w, img_h = detection["image_size"]

        if not plate_detected:
            warnings.append("no_plate_detected")

        if not food_items_raw:
            warnings.append("no_food_detected")
            return CVPipelineOutput(
                image_id=image_id,
                image_base64=self._image_to_base64(image),
                status="no_food_detected",
                warnings=warnings,
                requires_vlm_refinement=True,
                plate_detected=plate_detected,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

        # Compute scale
        scale_cm_per_px = 0.0
        if plate_detected and plate_mask is not None:
            plate_px = float(plate_mask.sum())
            if plate_px > 0:
                plate_radius_px = np.sqrt(plate_px / np.pi)
                scale_cm_per_px = round(13.0 / plate_radius_px, 5)

        # Step 2 — classify + portion + nutrition per item
        items = []
        confidence_scores = []

        for idx, raw in enumerate(food_items_raw):
            crop = raw["crop"]
            mask = raw["mask"]
            bbox = raw["bbox"]
            detection_conf = raw["detection_confidence"]

            # Classify
            predictions = self.classifier.predict(crop, top_k=3)
            top_label = predictions[0]["label"]
            class_conf = predictions[0]["confidence"]
            confidence_scores.append(class_conf)
            confidence_scores.append(detection_conf)

            # Portion
            portion_result = self.portion.estimate(top_label, mask, plate_mask)
            grams = portion_result["estimated_grams"]
            portion_method = portion_result["portion_method"]

            # Nutrition
            nutrition_100g = self.nutrition.get_nutrition(
                top_label.replace("_", " ")
            )

            if not nutrition_100g:
                warnings.append(f"no_nutrition_found_for_{top_label}")
                nutrition_100g = {
                    "calories_kcal": 0, "protein_g": 0,
                    "fat_g": 0, "carbs_g": 0, "fiber_g": 0
                }

            factor = grams / 100.0
            nutrition_total = {
                k: round(v * factor, 1)
                for k, v in nutrition_100g.items()
            }

            items.append(FoodItem(
                item_id=idx + 1,
                food_name=top_label,
                display_name=predictions[0]["display_name"],
                classification_confidence=round(class_conf, 4),
                detection_confidence=round(detection_conf, 4),
                top3_predictions=[
                    TopPrediction(
                        label=p["label"],
                        confidence=round(p["confidence"], 4)
                    ) for p in predictions
                ],
                estimated_grams=grams,
                portion_method=portion_method,
                bbox=bbox,
                nutrition_per_100g=NutritionValues(**{
                    k: round(v, 2) for k, v in nutrition_100g.items()
                }),
                nutrition_total=NutritionValues(**nutrition_total)
            ))

        # Step 3 — compute average confidence and VLM trigger
        avg_confidence = round(
            sum(confidence_scores) / len(confidence_scores), 4
        ) if confidence_scores else 0.0

        requires_vlm = avg_confidence < VLM_CONFIDENCE_THRESHOLD

        if requires_vlm:
            warnings.append("low_confidence_vlm_triggered")

        # Step 4 — build output
        totals = self._compute_totals(items)
        processing_ms = int((time.time() - start_time) * 1000)

        result = CVPipelineOutput(
            image_id=image_id,
            image_base64=self._image_to_base64(image),
            status="success",
            warnings=warnings,
            requires_vlm_refinement=requires_vlm,
            average_confidence=avg_confidence,
            plate_detected=plate_detected,
            scale_cm_per_px=scale_cm_per_px,
            items=items,
            totals=totals,
            processing_time_ms=processing_ms
        )

        if result.requires_vlm_refinement and self.refiner is not None:
            try:
                vlm_request = VLMRequest(
                    image_base64=result.image_base64,
                    reason=RefinementReason.LOW_CONFIDENCE,
                    trigger_threshold=VLM_CONFIDENCE_THRESHOLD,
                    cv_output=result
                )
                return self.refiner.refine(vlm_request)
            except Exception as e:
                result.warnings.append(f"vlm_failed: {str(e)}")

        return result