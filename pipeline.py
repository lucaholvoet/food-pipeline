"""
Food Calorie & Nutrient Estimator — Main Pipeline

This is the main pipeline orchestrator. It ties together:
- Detection (YOLOv8)
- Classification (EfficientNet-B0)
- Portion estimation (MiDaS + density table)
- Nutrition lookup (FAISS)
- VLM refinement (Gemini/Gemma)

Usage:
    from pipeline import FoodPipeline
    
    pipeline = FoodPipeline()
    result = pipeline.analyze_image(image_path)
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union
import uuid
import time
import base64
import numpy as np
from PIL import Image
import io

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import all modules
from detector.detector import FoodDetector
from classifier.classifier import FoodClassifier
from portion.portion import PortionEstimator
from portion.depth_estimator import DepthEstimator
from nutrition.nutrition import NutritionLookup
from vlm.refiner import VLMRefiner

from pipeline.schemas import (
    CVPipelineOutput,
    CVItem,
    CVNutrition,
)
from vlm.schemas import VLMRequest, RefinementReason

# Constants
CONFIDENCE_THRESHOLD = 0.70
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
USDA_DIR = os.path.join(PROJECT_ROOT, "data", "usda")


class FoodPipeline:
    """Complete food analysis pipeline."""

    def __init__(
        self,
        detector: Optional[FoodDetector] = None,
        classifier: Optional[FoodClassifier] = None,
        portion_estimator: Optional[PortionEstimator] = None,
        depth_estimator: Optional[DepthEstimator] = None,
        nutrition_lookup: Optional[NutritionLookup] = None,
        vlm_refiner: Optional[VLMRefiner] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        """Initialize the pipeline with optional components."""
        self.detector = detector or FoodDetector()
        self.classifier = classifier or FoodClassifier()
        self.portion_estimator = portion_estimator or PortionEstimator()
        self.depth_estimator = depth_estimator or DepthEstimator()
        self.nutrition_lookup = nutrition_lookup or NutritionLookup()
        self.vlm_refiner = vlm_refiner
        self.confidence_threshold = confidence_threshold

    def analyze_image(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> CVPipelineOutput:
        """
        Analyze a food image through the complete pipeline.
        
        Args:
            image_input: Image as path, bytes, numpy array, or PIL Image
            
        Returns:
            CVPipelineOutput with all analysis results
        """
        start_time = time.time()
        
        # ── Load image ──
        if isinstance(image_input, (str, bytes)):
            image = Image.open(io.BytesIO(image_input) if isinstance(image_input, bytes) else image_input)
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        else:
            image = image_input
            
        image_id = str(uuid.uuid4())
        
        # ── Detection ──
        try:
            detection_results = self.detector.detect(image)
            food_detections = [d for d in detection_results if d["class"] == "food"]
            plate_detections = [d for d in detection_results if d["class"] == "plate"]
            plate_detected = len(plate_detections) > 0
        except Exception:
            # Fallback to single item at center
            food_detections = [{
                "bbox": [image.width * 0.25, image.height * 0.25, 
                         image.width * 0.75, image.height * 0.75],
                "mask_area": image.width * image.height * 0.5,
                "detection_confidence": 0.8,
                "class": "food"
            }]
            plate_detected = False
            
        # ── Depth estimation (if available) ──
        try:
            depth_map = self.depth_estimator.estimate_depth(image)
            depth_available = True
        except Exception:
            depth_map = None
            depth_available = False
            
        # ── Process each food item ──
        items = []
        for i, detection in enumerate(food_detections):
            item_result = self._process_food_item(
                image=image,
                detection=detection,
                plate_detections=plate_detections,
                depth_map=depth_map,
                item_id=i + 1
            )
            items.append(item_result)
            
        # ── Calculate totals ──
        totals = self._calculate_totals(items)
        
        # ── Calculate average confidence ──
        avg_confidence = np.mean([item.classification_confidence for item in items]) if items else 0.0
        
        # ── Check for VLM refinement ──
        requires_vlm = (avg_confidence < self.confidence_threshold) or (not plate_detected)
        vlm_result = None
        
        if requires_vlm and self.vlm_refiner:
            try:
                # Convert image to base64 for VLM
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # Build CV output for VLM
                cv_output = CVPipelineOutput(
                    image_id=image_id,
                    status="success",
                    warnings=[],
                    average_confidence=avg_confidence,
                    plate_detected=plate_detected,
                    items=items,
                    totals=totals,
                )
                
                # Create VLM request
                vlm_request = VLMRequest(
                    image_base64=image_base64,
                    reason=RefinementReason.low_confidence if avg_confidence < self.confidence_threshold else RefinementReason.no_plate,
                    trigger_threshold=self.confidence_threshold,
                    cv_output=cv_output,
                )
                
                # Run VLM refinement
                vlm_result = self.vlm_refiner.refine(vlm_request)
                
            except Exception as e:
                print(f"VLM refinement failed: {e}")
        
        # ── Build final output ──
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        output = CVPipelineOutput(
            image_id=image_id,
            status="success",
            warnings=[] if not requires_vlm else ["Low confidence - VLM refinement recommended"],
            average_confidence=avg_confidence,
            plate_detected=plate_detected,
            items=items,
            totals=totals,
            processing_time_ms=elapsed_ms,
            vlm_refinement=vlm_result,
        )
        
        return output

    def _process_food_item(
        self,
        image: Image.Image,
        detection: dict,
        plate_detections: list,
        depth_map: Optional[np.ndarray],
        item_id: int
    ) -> CVItem:
        """Process a single detected food item."""
        
        # ── Extract bbox and crop ──
        bbox = detection["bbox"]
        food_crop = image.crop(bbox).convert("RGB")
        
        # ── Classification ──
        try:
            preds = self.classifier.predict(food_crop, top_k=3)
            food_name = preds[0]["label"]
            display_name = preds[0]["display_name"]
            confidence = preds[0]["confidence"]
            top3 = preds
        except Exception:
            # Fallback
            food_name = "unknown"
            display_name = "Unknown"
            confidence = 0.5
            top3 = [{"label": "unknown", "confidence": 0.5}]
            
        # ── Portion estimation ──
        try:
            # Create food mask (simplified)
            food_mask = np.zeros((image.height, image.width), dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            food_mask[y1:y2, x1:x2] = 255
            
            # Create plate mask if available
            plate_mask = np.zeros((image.height, image.width), dtype=np.uint8)
            if plate_detections:
                plate_bbox = plate_detections[0]["bbox"]
                px1, py1, px2, py2 = plate_bbox
                plate_mask[py1:py2, px1:px2] = 255
            
            # Estimate portion
            portion_result = self.portion_estimator.estimate(
                food_label=food_name,
                mask=food_mask,
                plate_mask=plate_mask if plate_detected else None,
                depth_map=depth_map,
            )
            estimated_grams = portion_result["estimated_grams"]
            portion_method = portion_result["portion_method"]
        except Exception:
            # Fallback
            estimated_grams = 250.0
            portion_method = "fallback"
            
        # ── Nutrition lookup ──
        try:
            nutrition_per_100g = self.nutrition_lookup.get_nutrition(food_name)
        except Exception:
            # Fallback nutrition
            nutrition_per_100g = CVNutrition(
                calories_kcal=150,
                protein_g=5.0,
                fat_g=4.0,
                carbs_g=22.0,
                fiber_g=1.0,
            )
            
        # ── Scale nutrition to portion ──
        scale = estimated_grams / 100.0
        nutrition_total = CVNutrition(
            calories_kcal=round(nutrition_per_100g.calories_kcal * scale, 1),
            protein_g=round(nutrition_per_100g.protein_g * scale, 1),
            fat_g=round(nutrition_per_100g.fat_g * scale, 1),
            carbs_g=round(nutrition_per_100g.carbs_g * scale, 1),
            fiber_g=round(nutrition_per_100g.fiber_g * scale, 1),
        )
        
        return CVItem(
            item_id=item_id,
            food_name=food_name,
            display_name=display_name,
            classification_confidence=confidence,
            top3_predictions=top3,
            bbox=bbox,
            detection_confidence=detection["detection_confidence"],
            estimated_grams=estimated_grams,
            portion_method=portion_method,
            nutrition_per_100g=nutrition_per_100g,
            nutrition_total=nutrition_total,
        )

    def _calculate_totals(self, items: list[CVItem]) -> CVNutrition:
        """Sum nutrition across all items."""
        totals = CVNutrition(
            calories_kcal=0,
            protein_g=0,
            fat_g=0,
            carbs_g=0,
            fiber_g=0,
        )
        
        for item in items:
            nt = item.nutrition_total
            totals.calories_kcal += nt.calories_kcal
            totals.protein_g += nt.protein_g
            totals.fat_g += nt.fat_g
            totals.carbs_g += nt.carbs_g
            totals.fiber_g += nt.fiber_g
            
        # Round totals
        totals.calories_kcal = round(totals.calories_kcal, 1)
        totals.protein_g = round(totals.protein_g, 1)
        totals.fat_g = round(totals.fat_g, 1)
        totals.carbs_g = round(totals.carbs_g, 1)
        totals.fiber_g = round(totals.fiber_g, 1)
        
        return totals


def create_pipeline() -> FoodPipeline:
    """Create a pipeline instance with all components."""
    return FoodPipeline(
        detector=FoodDetector(),
        classifier=FoodClassifier(),
        portion_estimator=PortionEstimator(),
        depth_estimator=DepthEstimator(),
        nutrition_lookup=NutritionLookup(),
        vlm_refiner=VLMRefiner(),
    )


if __name__ == "__main__":
    # Quick test
    pipeline = create_pipeline()
    print("Pipeline created successfully!")
    
    # Test with a sample image if available
    test_image_path = os.path.join(PROJECT_ROOT, "data", "sample", "pizza.jpg")
    if os.path.exists(test_image_path):
        result = pipeline.analyze_image(test_image_path)
        print(f"Analyzed {result.image_id}: {len(result.items)} items found")
    else:
        print("No test image found at data/sample/pizza.jpg")