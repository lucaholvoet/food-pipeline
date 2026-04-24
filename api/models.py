from pydantic import BaseModel, Field
from typing import Optional
import uuid


class TopPrediction(BaseModel):
    label: str
    confidence: float


class NutritionValues(BaseModel):
    calories_kcal: float
    protein_g: float
    fat_g: float
    carbs_g: float
    fiber_g: float


class FoodItem(BaseModel):
    item_id: int
    food_name: str
    display_name: str
    classification_confidence: float
    detection_confidence: float
    top3_predictions: list[TopPrediction]
    estimated_grams: float
    portion_method: str
    bbox: list[int]
    nutrition_per_100g: NutritionValues
    nutrition_total: NutritionValues


class CVPipelineOutput(BaseModel):
    image_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    image_base64: str = ""
    status: str = "success"
    warnings: list[str] = []
    requires_vlm_refinement: bool = False
    average_confidence: float = 0.0
    plate_detected: bool = False
    scale_cm_per_px: float = 0.0
    items: list[FoodItem] = []
    totals: NutritionValues = NutritionValues(
        calories_kcal=0, protein_g=0,
        fat_g=0, carbs_g=0, fiber_g=0
    )
    processing_time_ms: int = 0