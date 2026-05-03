"""
VLM prompt templates tuned for local open-source models.

Design goals:
- Keep prompts short enough for local multimodal models.
- Force exact Food-101 labels whenever possible.
- Reduce generic outputs like "pasta", "burger", or "salad bowl".
- Keep a fast path for evaluation and a full path for the refiner.
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

LABEL_GUIDANCE = """
Common mappings to valid Food-101 labels:
- burger -> hamburger
- cheeseburger -> hamburger
- cheese pancakes -> pancakes
- pancake stack -> pancakes
- pasta with meat sauce -> spaghetti_bolognese
- creamy pasta -> spaghetti_carbonara
- noodle soup -> ramen
- sushi rolls -> sushi
- ice cream -> ice_cream
- fries -> french_fries
- salad bowl with feta/olives/tomato -> greek_salad
- leafy salad with croutons/parmesan -> caesar_salad
- salmon fillet -> grilled_salmon
""".strip()

FAST_SYSTEM_PROMPT = (
    "You are a food recognition assistant correcting a weak CV prediction. "
    "Use one exact Food-101 class name when possible. "
    f"Valid classes: {FOOD101_LIST_STR}.\n\n"
    "Rules:\n"
    "- Return JSON only.\n"
    "- food_name should be an exact Food-101 label.\n"
    "- If the image does not perfectly match, choose the closest Food-101 label.\n"
    "- Avoid generic names like burger, pasta, salad, noodle soup, cake, or ice cream.\n"
    f"{LABEL_GUIDANCE}\n\n"
    "Return exactly this shape:\n"
    '{"items":[{"item_id":1,"action":"confirmed or corrected","food_name":"exact_label","display_name":"Readable Name","vlm_confidence":0.78,"food_description":"short description","estimated_grams":220,"calories_per_100g":180,"protein_g":7,"fat_g":6,"carbs_g":24,"fiber_g":2}]}'
)

SYSTEM_PROMPT = (
    "You are a food identification and nutrition refinement expert. "
    "A CV pipeline has already produced detections, candidate labels, and portion estimates. "
    "Your job is to confirm or correct each item.\n\n"
    "Use Food-101 labels whenever possible. "
    f"Valid Food-101 labels: {FOOD101_LIST_STR}.\n\n"
    "Decision process:\n"
    "1. Look at the food appearance.\n"
    "2. Compare it with the CV prediction and top-3 candidates.\n"
    "3. Choose the best Food-101 label.\n"
    "4. Estimate a realistic portion size in grams.\n"
    "5. Return only valid JSON.\n\n"
    "Confidence guide:\n"
    "- 0.90 to 1.00: visually obvious match\n"
    "- 0.75 to 0.89: strong guess\n"
    "- 0.50 to 0.74: plausible but uncertain\n"
    "- 0.30 to 0.49: weak guess\n\n"
    "Strict rules:\n"
    "- Do not output prose, markdown, or explanations outside JSON.\n"
    "- Do not invent labels outside Food-101 unless absolutely forced.\n"
    "- Prefer exact Food-101 labels over generic names.\n"
    f"{LABEL_GUIDANCE}"
)

USER_PROMPT_TEMPLATE = """Refine this CV pipeline result.

Trigger reason: {reason}
Threshold: {threshold}
Average CV confidence: {avg_confidence}

CV output:
{cv_output}

Return JSON with this exact top-level structure:
{{
  "items": [
    {{
      "item_id": 1,
      "action": "confirmed",
      "original": {{
        "food_name": "fried_rice",
        "classification_confidence": 0.58
      }},
      "refined": {{
        "food_name": "bibimbap",
        "display_name": "Bibimbap",
        "vlm_confidence": 0.86,
        "food_description": "Rice bowl with vegetables and egg"
      }},
      "portion": {{
        "estimated_grams": 240,
        "portion_method": "vlm_visual_estimate",
        "vlm_confidence": 0.74
      }},
      "nutrition_per_100g": {{
        "calories_kcal": 150,
        "protein_g": 5,
        "fat_g": 4,
        "carbs_g": 24,
        "fiber_g": 2
      }},
      "nutrition_total": {{
        "calories_kcal": 360,
        "protein_g": 12,
        "fat_g": 9.6,
        "carbs_g": 57.6,
        "fiber_g": 4.8
      }}
    }}
  ],
  "notes": ["optional note"]
}}

Return JSON only.
"""

FAST_USER_PROMPT = (
    "The CV pipeline prediction is '{cv_prediction}' at {confidence}. "
    "Look at the image and correct it if needed. "
    "Return JSON only."
)


def build_user_prompt(
    reason: str,
    threshold: float,
    avg_confidence: float,
    cv_output_json: str,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        reason=reason,
        threshold=threshold,
        avg_confidence=avg_confidence,
        cv_output=cv_output_json,
    )


def build_fast_prompt(cv_prediction: str, confidence: float) -> str:
    return FAST_USER_PROMPT.format(
        cv_prediction=cv_prediction,
        confidence=f"{confidence:.0%}",
    )
