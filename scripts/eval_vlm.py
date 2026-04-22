"""
VLM Evaluation - Test Gemma 4 on 10 food images

Run:
    venv\\Scripts\\activate
    python -u scripts/eval_vlm.py
"""

import sys
import os
import json
import base64
import time

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))
sys.path.insert(0, PROJECT)

import requests as http_requests

EVAL_DIR = os.path.join(PROJECT, "data", "eval")
RESULTS_FILE = os.path.join(PROJECT, "data", "eval_results.json")

SYSTEM_PROMPT = (
    "You are a food identification expert.\n"
    "The CV pipeline classifies food into exactly 101 categories "
    "(Food-101).\n"
    "You MUST use one of these EXACT class names "
    "(lowercase, underscores):\n"
    "apple_pie, baby_back_ribs, baklava, beef_carpaccio, "
    "beef_tartare, beet_salad, beignets, bibimbap, bread_pudding, "
    "breakfast_burrito, bruschetta, caesar_salad, cannoli, "
    "caprese_salad, carrot_cake, ceviche, cheesecake, cheese_plate, "
    "chicken_curry, chicken_quesadilla, chicken_wings, "
    "chocolate_cake, chocolate_mousse, churros, clam_chowder, "
    "club_sandwich, crab_cakes, creme_brulee, croque_madame, "
    "cup_cakes, deviled_eggs, donuts, dumplings, edamame, "
    "eggs_benedict, escargots, falafel, filet_mignon, "
    "fish_and_chips, foie_gras, french_fries, french_onion_soup, "
    "french_toast, fried_calamari, fried_rice, frozen_yogurt, "
    "garlic_bread, gnocchi, greek_salad, grilled_cheese_sandwich, "
    "grilled_salmon, guacamole, gyoza, hamburger, "
    "hot_and_sour_soup, hot_dog, huevos_rancheros, hummus, "
    "ice_cream, lobster_bisque, lobster_roll_sandwich, "
    "macaroni_and_cheese, macarons, miso_soup, mussels, nachos, "
    "omelette, onion_rings, oysters, pad_thai, paella, pancakes, "
    "panna_cotta, peking_duck, pho, pizza, pork_chop, poutine, "
    "prime_rib, pulled_pork_sandwich, ramen, ravioli, "
    "red_velvet_cake, risotto, samosa, sashimi, scallops, "
    "seaweed_salad, shrimp_and_grits, spaghetti_bolognese, "
    "spaghetti_carbonara, spring_rolls, steak, "
    "strawberry_shortcake, sushi, tacos, takoyaki, tiramisu, "
    "tuna_tartare, waffles\n\n"
    "If unsure, pick the CLOSEST match from the list.\n"
    "Example: 'burger' -> hamburger, 'cheese pancakes' -> pancakes, "
    "'noodle soup' -> ramen.\n"
    'Respond ONLY with JSON: '
    '{"food_name": "exact_class_name", "display_name": "Name", '
    '"action": "confirmed or corrected", '
    '"vlm_confidence": <0.3-1.0 honest value>, '
    '"food_description": "short description", '
    '"estimated_grams": <grams>}'
)


def build_prompt(cv_prediction, confidence):
    return (
        f"The CV pipeline predicted this food is "
        f"'{cv_prediction}' with {confidence:.0%} confidence. "
        f"Look at the image and correct if needed. "
        f"Respond with JSON only."
    )


def call_gemma4(image_base64, prompt, model="gemma4:e4b"):
    r = http_requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt,
                 "images": [image_base64]},
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.95,
                "top_k": 64,
            },
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def parse_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "{" in text:
        s = text.index("{")
        e = text.rindex("}") + 1
        return json.loads(text[s:e])
    return None

def evaluate_match(vlm_food, actual_food):
    vlm = vlm_food.lower().replace(" ", "_")
    actual = actual_food.lower().replace(" ", "_")
    if vlm == actual:
        return "exact"
    if actual.replace("_", "") in vlm.replace("_", ""):
        return "partial"
    if vlm.replace("_", "") in actual.replace("_", ""):
        return "partial"
    return "wrong"


TEST_CASES = [
    {"file": "pizza.jpg", "actual_food": "pizza",
     "cv_prediction": "pizza", "confidence": 0.45,
     "should_correct": False},
    {"file": "burger.jpg", "actual_food": "hamburger",
     "cv_prediction": "hot_dog", "confidence": 0.38,
     "should_correct": True},
    {"file": "sushi.jpg", "actual_food": "sushi",
     "cv_prediction": "sushi", "confidence": 0.50,
     "should_correct": False},
    {"file": "pasta.jpg", "actual_food": "spaghetti_bolognese",
     "cv_prediction": "ramen", "confidence": 0.42,
     "should_correct": True},
    {"file": "salad.jpg", "actual_food": "greek_salad",
     "cv_prediction": "caesar_salad", "confidence": 0.35,
     "should_correct": True},
    {"file": "pancakes.jpg", "actual_food": "pancakes",
     "cv_prediction": "french_toast", "confidence": 0.40,
     "should_correct": True},
    {"file": "icecream.jpg", "actual_food": "ice_cream",
     "cv_prediction": "frozen_yogurt", "confidence": 0.33,
     "should_correct": True},
    {"file": "steak.jpg", "actual_food": "steak",
     "cv_prediction": "filet_mignon", "confidence": 0.48,
     "should_correct": False},
    {"file": "ramen.jpg", "actual_food": "ramen",
     "cv_prediction": "pho", "confidence": 0.36,
     "should_correct": True},
    {"file": "cake.jpg", "actual_food": "chocolate_cake",
     "cv_prediction": "red_velvet_cake", "confidence": 0.41,
     "should_correct": True},
]

def run_evaluation():
    print("=" * 60)
    print("VLM Evaluation - Gemma 4 on 10 Food Images")
    print("=" * 60)

    results = []
    correct = 0
    partial = 0
    wrong = 0
    total_time = 0
    corrections_made = 0
    corrections_right = 0

    for i, tc in enumerate(TEST_CASES):
        img_path = os.path.join(EVAL_DIR, tc["file"])
        if not os.path.exists(img_path):
            print(f"\n[{i+1}/10] SKIP {tc['file']} - not found")
            continue

        print(f"\n[{i+1}/10] {tc['file']}", flush=True)
        print(f"  Actual:    {tc['actual_food']}", flush=True)
        print(f"  CV said:   {tc['cv_prediction']} "
              f"({tc['confidence']:.0%})", flush=True)
        print(f"  Calling Gemma 4...", end="", flush=True)

        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        prompt = build_prompt(
            tc["cv_prediction"], tc["confidence"]
        )

        start = time.time()
        try:
            raw = call_gemma4(img_b64, prompt)
            elapsed = int((time.time() - start) * 1000)
            total_time += elapsed
            print(f" {elapsed/1000:.1f}s", flush=True)

            parsed = parse_json(raw)
            if parsed is None:
                print("  PARSE ERROR")
                wrong += 1
                results.append({
                    "file": tc["file"],
                    "actual": tc["actual_food"],
                    "cv_prediction": tc["cv_prediction"],
                    "vlm_food": "parse_error",
                    "match": "wrong",
                    "time_ms": elapsed,
                })
                continue

            vlm_food = parsed.get("food_name", "?")
            vlm_conf = parsed.get("vlm_confidence", 0)
            vlm_action = parsed.get("action", "?")
            vlm_grams = parsed.get("estimated_grams", 0)
            match = evaluate_match(
                vlm_food, tc["actual_food"]
            )

            if match == "exact":
                correct += 1
                icon = "EXACT"
            elif match == "partial":
                partial += 1
                icon = "PARTIAL"
            else:
                wrong += 1
                icon = "WRONG"

            if vlm_action == "corrected":
                corrections_made += 1
                if match in ("exact", "partial"):
                    corrections_right += 1

            print(f"  VLM said:  {vlm_food} "
                  f"({vlm_conf:.0%}) [{icon}]", flush=True)
            print(f"  Action:    {vlm_action} | "
                  f"{vlm_grams}g", flush=True)

            results.append({
                "file": tc["file"],
                "actual": tc["actual_food"],
                "cv_prediction": tc["cv_prediction"],
                "vlm_food": vlm_food,
                "vlm_confidence": vlm_conf,
                "match": match,
                "action": vlm_action,
                "estimated_grams": vlm_grams,
                "time_ms": elapsed,
            })

        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            total_time += elapsed
            print(f" ERROR: {e}")
            wrong += 1
            results.append({
                "file": tc["file"],
                "actual": tc["actual_food"],
                "cv_prediction": tc["cv_prediction"],
                "vlm_food": "error",
                "match": "wrong",
                "time_ms": elapsed,
                "error": str(e),
            })

    # Summary
    total = len(results)
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total images:    {total}")
    print(f"  Exact matches:   {correct} "
          f"({correct/max(total,1)*100:.0f}%)")
    print(f"  Partial matches: {partial} "
          f"({partial/max(total,1)*100:.0f}%)")
    print(f"  Wrong:           {wrong} "
          f"({wrong/max(total,1)*100:.0f}%)")
    print(f"  Overall accuracy: "
          f"{(correct+partial)/max(total,1)*100:.0f}%")
    print(f"  Corrections made:  {corrections_made}")
    print(f"  Corrections right: {corrections_right}")
    print(f"  Avg time/image:    "
          f"{total_time/max(total,1)/1000:.1f}s")

    # Results table
    print(f"\n{'=' * 60}")
    print("DETAILED RESULTS")
    print(f"{'=' * 60}")
    print(f"{'Image':<14} {'Actual':<22} "
          f"{'CV Said':<22} {'VLM Said':<20} "
          f"{'Match':<8}")
    print("-" * 86)
    for r in results:
        print(f"{r['file']:<14} "
              f"{r['actual']:<22} "
              f"{r.get('cv_prediction','?'):<22} "
              f"{r.get('vlm_food','?'):<20} "
              f"{r['match']:<8}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: data/eval_results.json")


if __name__ == "__main__":
    run_evaluation()
