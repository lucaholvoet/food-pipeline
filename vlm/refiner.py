"""
VLM Refiner — Orchestrates the refinement pipeline

Flow:
  1. Load VLMRequest (image + CV output)
  2. Build prompt using CV data
  3. Call Gemma 4 via client
  4. Parse & validate response against schema
  5. Return VLMResponse

Usage:
    from vlm.refiner import VLMRefiner

    refiner = VLMRefiner(backend="ollama", model="gemma4:26b")
    result = refiner.refine(request_dict)
"""

import json
import time
from typing import Union

from vlm.client import Gemma4Client
from vlm.prompts import SYSTEM_PROMPT, build_user_prompt
from vlm.schemas import (
    VLMRequest,
    VLMResponse,
    RefinedItem,
    OriginalResult,
    RefinedResult,
    RefinedPortion,
    CVNutrition,
    RefinementAction,
    PortionMethod,
)


class VLMRefiner:
    """Orchestrates the VLM refinement of CV pipeline results."""

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "gemma4:26b",
        api_key: str = None,
        ollama_url: str = "http://localhost:11434",
    ):
        self.client = Gemma4Client(
            backend=backend,
            model=model,
            api_key=api_key,
            ollama_url=ollama_url,
        )
        self.model_name = model

    def refine(
        self, request: Union[dict, VLMRequest]
    ) -> VLMResponse:
        """
        Run VLM refinement on a CV pipeline result.

        Args:
            request: VLMRequest dict or Pydantic model

        Returns:
            VLMResponse with refined items
        """
        # ── Step 1: Validate input ──
        if isinstance(request, dict):
            request = VLMRequest(**request)

        # ── Step 2: Build prompt ──
        cv_json = request.cv_output.model_dump_json(indent=2)
        user_prompt = build_user_prompt(
            reason=request.reason.value,
            threshold=request.trigger_threshold,
            avg_confidence=request.cv_output.average_confidence,
            cv_output_json=cv_json,
        )

        # ── Step 3: Call Gemma 4 ──
        raw_response = self.client.analyze_image(
            image_base64=request.image_base64,
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        processing_time = raw_response.pop(
            "_processing_time_ms", 0
        )

        # ── Step 4: Parse response into schema ──
        refined_items = self._parse_items(
            raw_response, request.cv_output
        )

        # ── Step 5: Calculate totals ──
        totals = self._calculate_totals(refined_items)

        # ── Step 6: Build final response ──
        return VLMResponse(
            image_id=request.cv_output.image_id,
            refinement_status="completed",
            vlm_model=self.model_name,
            confidence_threshold_used=request.trigger_threshold,
            items=refined_items,
            totals=totals,
            notes=self._extract_notes(raw_response),
            processing_time_ms=processing_time,
        )

    def _parse_items(
        self, raw: dict, cv_output
    ) -> list[RefinedItem]:
        """Parse raw VLM response into RefinedItem models."""
        items = []
        raw_items = raw.get("items", [])

        for raw_item in raw_items:
            # Get original CV data for this item
            cv_item = None
            for ci in cv_output.items:
                if ci.item_id == raw_item.get("item_id"):
                    cv_item = ci
                    break

            if cv_item is None:
                continue

            # Build nutrition from response
            nutrition_100g = self._make_nutrition(
                raw_item.get("nutrition_per_100g", {})
            )
            portion = raw_item.get("portion", {})
            grams = portion.get(
                "estimated_grams",
                cv_item.estimated_grams,
            )
            nutrition_total = self._scale_nutrition(
                nutrition_100g, grams
            )

            item = RefinedItem(
                item_id=raw_item.get("item_id", cv_item.item_id),
                action=RefinementAction(
                    raw_item.get("action", "unknown")
                ),
                original=OriginalResult(
                    food_name=cv_item.food_name,
                    classification_confidence=(
                        cv_item.classification_confidence
                    ),
                ),
                refined=RefinedResult(
                    food_name=raw_item.get("refined", {}).get(
                        "food_name", cv_item.food_name
                    ),
                    display_name=raw_item.get("refined", {}).get(
                        "display_name",
                        cv_item.display_name,
                    ),
                    vlm_confidence=raw_item.get("refined", {}).get(
                        "vlm_confidence", 0.5
                    ),
                    food_description=raw_item.get(
                        "refined", {}
                    ).get("food_description"),
                ),
                portion=RefinedPortion(
                    estimated_grams=grams,
                    portion_method=PortionMethod(
                        portion.get(
                            "portion_method",
                            "vlm_visual_estimate",
                        )
                    ),
                    vlm_confidence=portion.get(
                        "vlm_confidence", 0.5
                    ),
                ),
                nutrition_per_100g=nutrition_100g,
                nutrition_total=nutrition_total,
            )
            items.append(item)

        return items

    @staticmethod
    def _make_nutrition(data: dict) -> CVNutrition:
        """Build CVNutrition from dict with defaults."""
        return CVNutrition(
            calories_kcal=data.get("calories_kcal", 0),
            protein_g=data.get("protein_g", 0),
            fat_g=data.get("fat_g", 0),
            carbs_g=data.get("carbs_g", 0),
            fiber_g=data.get("fiber_g", 0),
        )

    @staticmethod
    def _scale_nutrition(
        per_100g: CVNutrition, grams: float
    ) -> CVNutrition:
        """Scale nutrition from per-100g to actual portion."""
        scale = grams / 100.0
        return CVNutrition(
            calories_kcal=round(
                per_100g.calories_kcal * scale, 1
            ),
            protein_g=round(per_100g.protein_g * scale, 1),
            fat_g=round(per_100g.fat_g * scale, 1),
            carbs_g=round(per_100g.carbs_g * scale, 1),
            fiber_g=round(per_100g.fiber_g * scale, 1),
        )

    @staticmethod
    def _calculate_totals(
        items: list[RefinedItem],
    ) -> CVNutrition:
        """Sum nutrition across all refined items."""
        totals = CVNutrition(
            calories_kcal=0, protein_g=0,
            fat_g=0, carbs_g=0, fiber_g=0,
        )
        for item in items:
            t = item.nutrition_total
            totals.calories_kcal += t.calories_kcal
            totals.protein_g += t.protein_g
            totals.fat_g += t.fat_g
            totals.carbs_g += t.carbs_g
            totals.fiber_g += t.fiber_g

        # Round totals
        totals.calories_kcal = round(totals.calories_kcal, 1)
        totals.protein_g = round(totals.protein_g, 1)
        totals.fat_g = round(totals.fat_g, 1)
        totals.carbs_g = round(totals.carbs_g, 1)
        totals.fiber_g = round(totals.fiber_g, 1)

        return totals

    @staticmethod
    def _extract_notes(raw: dict) -> list[str]:
        """Extract any notes from raw response."""
        notes = raw.get("notes", [])
        if isinstance(notes, str):
            notes = [notes]
        return notes
