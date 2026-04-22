"""
VLM Refinement — JSON Schemas (Pydantic models)

These define the exact input/output format for the VLM stage.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ──────────────────────────────────────────────
#  Enums
# ──────────────────────────────────────────────

class RefinementAction(str, Enum):
    CONFIRMED = "confirmed"   # CV pipeline was correct
    CORRECTED = "corrected"   # VLM changed the result
    UNKNOWN = "unknown"       # VLM couldn't determine


class RefinementReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    MULTIPLE_SIMILAR = "multiple_similar_predictions"
    NO_FOOD_DETECTED = "no_food_detected"
    PORTION_SUSPECT = "portion_estimate_suspect"


class PortionMethod(str, Enum):
    PLATE_REFERENCE = "plate_reference"
    FALLBACK_SCALE = "fallback_scale"
    VLM_VISUAL_ESTIMATE = "vlm_visual_estimate"


# ──────────────────────────────────────────────
#  Input Schema (what VLM receives from CV pipeline)
# ──────────────────────────────────────────────

class CVTopPrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class CVNutrition(BaseModel):
    calories_kcal: float
    protein_g: float
    fat_g: float
    carbs_g: float
    fiber_g: float


class CVItem(BaseModel):
    item_id: int
    food_name: str
    display_name: str
    classification_confidence: float = Field(ge=0.0, le=1.0)
    detection_confidence: Optional[float] = None
    top3_predictions: list[CVTopPrediction]
    estimated_grams: float
    portion_method: str
    bbox: list[int]  # [x1, y1, x2, y2]
    nutrition_per_100g: CVNutrition
    nutrition_total: CVNutrition


class CVOutput(BaseModel):
    image_id: str
    status: str = "success"
    average_confidence: float = Field(ge=0.0, le=1.0)
    plate_detected: bool
    items: list[CVItem]
    totals: CVNutrition


class VLMRequest(BaseModel):
    """Full input sent to the VLM refinement module."""
    image_base64: str = Field(description="Base64-encoded original image")
    reason: RefinementReason = Field(
        description="Why the image was flagged for VLM refinement"
    )
    trigger_threshold: float = Field(
        default=0.70,
        description="Confidence threshold that triggered VLM"
    )
    cv_output: CVOutput = Field(
        description="Full CV pipeline output"
    )


# ──────────────────────────────────────────────
#  Output Schema (what VLM returns)
# ──────────────────────────────────────────────

class OriginalResult(BaseModel):
    """Preserves CV pipeline's answer for comparison."""
    food_name: str
    classification_confidence: float


class RefinedResult(BaseModel):
    """VLM's corrected/confirmed food identification."""
    food_name: str
    display_name: str
    vlm_confidence: float = Field(ge=0.0, le=1.0)
    food_description: Optional[str] = Field(
        default=None,
        description="Short text description of the food item"
    )


class RefinedPortion(BaseModel):
    """VLM's portion estimate."""
    estimated_grams: float
    portion_method: PortionMethod = PortionMethod.VLM_VISUAL_ESTIMATE
    vlm_confidence: float = Field(ge=0.0, le=1.0)


class RefinedItem(BaseModel):
    """One refined food item from VLM."""
    item_id: int
    action: RefinementAction
    original: OriginalResult
    refined: RefinedResult
    portion: RefinedPortion
    nutrition_per_100g: CVNutrition
    nutrition_total: CVNutrition


class VLMResponse(BaseModel):
    """Full output returned by the VLM refinement module."""
    image_id: str
    refinement_status: str = "completed"
    vlm_model: str = Field(
        description="Name of VLM used, e.g. 'gpt-4-vision'"
    )
    confidence_threshold_used: float
    items: list[RefinedItem]
    totals: CVNutrition
    notes: list[str] = Field(
        default_factory=list,
        description="Any warnings or observations"
    )
    processing_time_ms: int
