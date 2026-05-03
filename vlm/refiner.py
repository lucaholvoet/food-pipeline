"""Orchestrates VLM refinement of CV pipeline results."""

from typing import Union

from vlm.client import Gemma4Client
from vlm.normalization import display_name_from_label, normalize_food101_label
from vlm.prompts import SYSTEM_PROMPT, build_user_prompt
from vlm.schemas import (
    CVNutrition,
    OriginalResult,
    PortionMethod,
    RefinedItem,
    RefinedPortion,
    RefinedResult,
    RefinementAction,
    VLMRequest,
    VLMResponse,
)


class VLMRefiner:
    """Runs prompt building, model inference, and response normalization."""

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

    def refine(self, request: Union[dict, VLMRequest]) -> VLMResponse:
        if isinstance(request, dict):
            request = VLMRequest(**request)

        cv_json = request.cv_output.model_dump_json(indent=2)
        user_prompt = build_user_prompt(
            reason=request.reason.value,
            threshold=request.trigger_threshold,
            avg_confidence=request.cv_output.average_confidence,
            cv_output_json=cv_json,
        )

        raw_response = self.client.analyze_image(
            image_base64=request.image_base64,
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
        )
        processing_time = raw_response.pop("_processing_time_ms", 0)

        refined_items = self._parse_items(raw_response, request)
        totals = self._calculate_totals(refined_items)

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

    def _parse_items(self, raw: dict, request: VLMRequest) -> list[RefinedItem]:
        raw_items = self._coerce_items(raw)
        items: list[RefinedItem] = []

        for cv_item, raw_item in zip(request.cv_output.items, raw_items):
            refined_block = raw_item.get("refined", {})
            portion_block = raw_item.get("portion", {})
            normalized_food = normalize_food101_label(
                refined_block.get("food_name") or raw_item.get("food_name") or cv_item.food_name,
                fallback=cv_item.food_name,
            )

            grams = float(
                portion_block.get("estimated_grams")
                or raw_item.get("estimated_grams")
                or cv_item.estimated_grams
            )
            vlm_confidence = float(
                refined_block.get("vlm_confidence")
                or raw_item.get("vlm_confidence")
                or cv_item.classification_confidence
            )
            portion_confidence = float(
                portion_block.get("vlm_confidence")
                or raw_item.get("portion_confidence")
                or vlm_confidence
            )

            nutrition_100g = self._make_nutrition(
                raw_item.get("nutrition_per_100g", {}),
                raw_item,
                fallback=cv_item.nutrition_per_100g,
            )
            nutrition_total = self._make_total_nutrition(
                raw_item=raw_item,
                per_100g=nutrition_100g,
                grams=grams,
            )

            action = raw_item.get("action", "unknown").lower()
            if normalized_food == cv_item.food_name and action == "corrected":
                action = "confirmed"
            if action not in {"confirmed", "corrected", "unknown"}:
                action = "corrected" if normalized_food != cv_item.food_name else "confirmed"

            items.append(
                RefinedItem(
                    item_id=raw_item.get("item_id", cv_item.item_id),
                    action=RefinementAction(action),
                    original=OriginalResult(
                        food_name=cv_item.food_name,
                        classification_confidence=cv_item.classification_confidence,
                    ),
                    refined=RefinedResult(
                        food_name=normalized_food,
                        display_name=refined_block.get("display_name")
                        or raw_item.get("display_name")
                        or display_name_from_label(normalized_food),
                        vlm_confidence=max(0.0, min(1.0, vlm_confidence)),
                        food_description=refined_block.get("food_description")
                        or raw_item.get("food_description"),
                    ),
                    portion=RefinedPortion(
                        estimated_grams=grams,
                        portion_method=PortionMethod(
                            portion_block.get("portion_method", "vlm_visual_estimate")
                        ),
                        vlm_confidence=max(0.0, min(1.0, portion_confidence)),
                    ),
                    nutrition_per_100g=nutrition_100g,
                    nutrition_total=nutrition_total,
                )
            )

        return items

    @staticmethod
    def _coerce_items(raw: dict) -> list[dict]:
        if isinstance(raw.get("items"), list):
            return raw["items"]
        if "food_name" in raw:
            return [{"item_id": 1, **raw}]
        return []

    @staticmethod
    def _make_nutrition(data: dict, raw_item: dict, fallback: CVNutrition) -> CVNutrition:
        return CVNutrition(
            calories_kcal=float(data.get("calories_kcal", raw_item.get("calories_per_100g", fallback.calories_kcal))),
            protein_g=float(data.get("protein_g", fallback.protein_g)),
            fat_g=float(data.get("fat_g", fallback.fat_g)),
            carbs_g=float(data.get("carbs_g", fallback.carbs_g)),
            fiber_g=float(data.get("fiber_g", fallback.fiber_g)),
        )

    @staticmethod
    def _make_total_nutrition(raw_item: dict, per_100g: CVNutrition, grams: float) -> CVNutrition:
        total = raw_item.get("nutrition_total")
        if isinstance(total, dict):
            return CVNutrition(
                calories_kcal=float(total.get("calories_kcal", 0)),
                protein_g=float(total.get("protein_g", 0)),
                fat_g=float(total.get("fat_g", 0)),
                carbs_g=float(total.get("carbs_g", 0)),
                fiber_g=float(total.get("fiber_g", 0)),
            )
        scale = grams / 100.0
        return CVNutrition(
            calories_kcal=round(per_100g.calories_kcal * scale, 1),
            protein_g=round(per_100g.protein_g * scale, 1),
            fat_g=round(per_100g.fat_g * scale, 1),
            carbs_g=round(per_100g.carbs_g * scale, 1),
            fiber_g=round(per_100g.fiber_g * scale, 1),
        )

    @staticmethod
    def _calculate_totals(items: list[RefinedItem]) -> CVNutrition:
        totals = CVNutrition(
            calories_kcal=0,
            protein_g=0,
            fat_g=0,
            carbs_g=0,
            fiber_g=0,
        )
        for item in items:
            totals.calories_kcal += item.nutrition_total.calories_kcal
            totals.protein_g += item.nutrition_total.protein_g
            totals.fat_g += item.nutrition_total.fat_g
            totals.carbs_g += item.nutrition_total.carbs_g
            totals.fiber_g += item.nutrition_total.fiber_g

        totals.calories_kcal = round(totals.calories_kcal, 1)
        totals.protein_g = round(totals.protein_g, 1)
        totals.fat_g = round(totals.fat_g, 1)
        totals.carbs_g = round(totals.carbs_g, 1)
        totals.fiber_g = round(totals.fiber_g, 1)
        return totals

    @staticmethod
    def _extract_notes(raw: dict) -> list[str]:
        notes = raw.get("notes", [])
        if isinstance(notes, str):
            return [notes]
        if isinstance(notes, list):
            return [str(note) for note in notes]
        return []
