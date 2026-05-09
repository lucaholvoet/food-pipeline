"""
VLM Refinement - Prompt Templates (v2)

Improved based on evaluation results:
- Added full Food-101 class list so VLM uses exact names
- Added confidence calibration (was stuck at 95%)
- Added FAST prompts for quick inference
"""

FOOD101_CLASSES = sorted([
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio",
    "beef_tartare", "beet_salad", "beignets", "bibimbap",
    "bread_pudding", "breakfast_burrito", "bruschetta",
    "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry",
    "chicken_quesadilla", "chicken_wings", "chocolate_cake",
    "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee",
    "croque_madame", "cup_cakes", "deviled_eggs", "donuts",
    "dumplings", "edamame", "eggs_benedict", "escargots",
    "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast",
    "fried_calamari", "fried_rice", "frozen_yogurt",
    "garlic_bread", "gnocchi", "greek_salad",
    "grilled_cheese_sandwich", "grilled_salmon", "guacamole",
    "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog",
    "huevos_rancheros", "hummus", "ice_cream",
    "lobster_bisque", "lobster_roll_sandwich",
    "macaroni_and_cheese", "macarons", "miso_soup", "mussels",
    "nachos", "omelette", "onion_rings", "oysters", "pad_thai",
    "paella", "pancakes", "panna_cotta", "peking_duck", "pho",
    "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli",
    "red_velvet_cake", "risotto", "samosa", "sashimi",
    "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls",
    "steak", "strawberry_shortcake", "sushi", "tacos",
    "takoyaki", "tiramisu", "tuna_tartare", "waffles",
])

FOOD101_LIST_STR = ", ".join(FOOD101_CLASSES)

SYSTEM_PROMPT = (
    "You are a food identification expert.\n"
    "The CV pipeline classifies food into exactly 101 "
    "categories (Food-101).\n"
    "CRITICAL: You MUST use one of these EXACT class "
    "names (lowercase, underscores):\n"
    f"{FOOD101_LIST_STR}\n\n"
    "If unsure, pick the CLOSEST match from the list.\n"
    "Examples: 'burger' -> hamburger, "
    "'cheese pancakes' -> pancakes, "
    "'noodle soup' -> ramen.\n\n"
    "Set vlm_confidence honestly (0.3-1.0), "
    "NOT always 0.95.\n"
    "Respond ONLY with JSON."
)


SYSTEM_PROMPT_FULL = (
    "You are a food identification and nutrition expert.\n"
    "The CV pipeline classifies food into exactly 101 "
    "categories (Food-101).\n"
    "CRITICAL: You MUST use one of these EXACT class "
    "names (lowercase, underscores):\n"
    f"{FOOD101_LIST_STR}\n\n"
    "If unsure, pick the CLOSEST match.\n\n"
    "Rules:\n"
    "- food_name MUST be a Food-101 class name\n"
    "- Be honest about confidence (0.3-1.0)\n"
    "- nutrition_total = (estimated_grams/100) "
    "* nutrition_per_100g\n"
    "Respond ONLY with valid JSON."
)

USER_PROMPT_TEMPLATE = (
    "Refine the food analysis for this image.\n\n"
    "Reason: {reason} (threshold: {threshold}, "
    "CV confidence: {avg_confidence})\n\n"
    "CV Pipeline Output:\n"
    "```json\n{cv_output}\n```\n\n"
    "For each item, respond with JSON:\n"
    "{{\n  "
    '"items": [\n    {{\n      '
    '"item_id": <same>,\n      '
    '"action": "confirmed" | "corrected",\n      '
    '"original": {{"food_name": "<cv>", '
    '"classification_confidence": <val>}},\n      '
    '"refined": {{"food_name": '
    '"<MUST be Food-101 class name>", '
    '"display_name": "<readable>", '
    '"vlm_confidence": <0.3-1.0>, '
    '"food_description": "<desc>"}},\n      '
    '"portion": {{"estimated_grams": <val>, '
    '"portion_method": "vlm_visual_estimate", '
    '"vlm_confidence": <0.3-1.0>}},\n      '
    '"nutrition_per_100g": '
    '{{"calories_kcal":<v>, "protein_g":<v>, '
    '"fat_g":<v>, "carbs_g":<v>, "fiber_g":<v>}},\n      '
    '"nutrition_total": '
    '{{"calories_kcal":<v>, "protein_g":<v>, '
    '"fat_g":<v>, "carbs_g":<v>, "fiber_g":<v>}}\n    '
    '}}\n  ]\n}}\n\n'
    "IMPORTANT: food_name MUST be from the "
    "Food-101 list. Respond ONLY JSON."
)


FAST_SYSTEM_PROMPT = (
    "You are a food identification expert.\n"
    "The CV pipeline classifies food into exactly 101 "
    "categories (Food-101).\n"
    "You MUST use one of these EXACT class names "
    "(lowercase, underscores):\n"
    f"{FOOD101_LIST_STR}\n\n"
    "If unsure, pick the closest match.\n"
    "Example: 'burger' -> hamburger, "
    "'cheese pancakes' -> pancakes, "
    "'noodle soup' -> ramen.\n\n"
    'Respond ONLY with JSON: '
    '{{"food_name": "exact_class_name", '
    '"display_name": "Readable Name", '
    '"action": "confirmed or corrected", '
    '"vlm_confidence": <0.3-1.0 honest value>, '
    '"food_description": "short desc", '
    '"estimated_grams": <grams>}}'
)

FAST_USER_PROMPT = (
    "The CV pipeline predicted this food is "
    "'{cv_prediction}' with {confidence:.0%} confidence. "
    "Look at the image. If wrong, correct it to the best "
    "Food-101 class name. "
    "Respond with JSON only."
)


def build_user_prompt(
    reason: str,
    threshold: float,
    avg_confidence: float,
    cv_output_json: str,
) -> str:
    """Build the full user prompt with CV pipeline data."""
    return USER_PROMPT_TEMPLATE.format(
        reason=reason,
        threshold=threshold,
        avg_confidence=avg_confidence,
        cv_output=cv_output_json,
    )


def build_fast_prompt(
    cv_prediction: str,
    confidence: float,
) -> str:
    """Build a short prompt for fast inference."""
    return FAST_USER_PROMPT.format(
        cv_prediction=cv_prediction,
        confidence=confidence,
    )
