"""
Fast VLM Local Test — Simplified prompt for speed

Uses a lightweight prompt that gets results in ~2 minutes
instead of 10+ minutes with the full prompt.

Run:
    venv\\Scripts\\activate
    python scripts/test_vlm_fast.py [image_path] [model]
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


FAST_SYSTEM = """You are a food identification expert. The user will show you a food image and a CV pipeline prediction. Respond ONLY with a JSON object matching this exact format:
{"items": [{"item_id": 1, "action": "confirmed or corrected", "food_name": "name", "display_name": "Name", "vlm_confidence": 0.9, "food_description": "short description", "estimated_grams": 250, "calories_per_100g": 150, "protein_g": 5, "fat_g": 3, "carbs_g": 20, "fiber_g": 1}]}"""


def build_fast_prompt(cv_prediction, confidence):
    return (
        f"The CV pipeline predicted this food is '{cv_prediction}' "
        f"with {confidence:.0%} confidence. "
        f"Look at the image and correct if needed. "
        f"Respond with JSON only."
    )


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_fast_test(image_path, model="gemma4:e4b"):
    print("=" * 55)
    print(f"Fast VLM Test - {model}")
    print("=" * 55)

    print(f"\nImage: {image_path}")
    img_b64 = encode_image(image_path)
    print(f"Image size: {len(img_b64)} chars")

    cv_prediction = "spaghetti_bolognese"
    cv_confidence = 0.55

    prompt = build_fast_prompt(cv_prediction, cv_confidence)
    print(f"CV prediction: {cv_prediction} ({cv_confidence:.0%})")
    print(f"\nCalling {model}...")

    start = time.time()
    r = http_requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": FAST_SYSTEM},
                {"role": "user", "content": prompt,
                 "images": [img_b64]}
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
    elapsed = int((time.time() - start) * 1000)

    raw = r.json()["message"]["content"]

    print(f"\n{'=' * 55}")
    print(f"Response ({elapsed}ms / {elapsed/1000:.1f}s)")
    print(f"{'=' * 55}")
    print(raw)

    # Parse JSON
    try:
        if "{" in raw:
            s = raw.index("{")
            e = raw.rindex("}") + 1
            parsed = json.loads(raw[s:e])
            print(f"\n{'=' * 55}")
            print("Parsed JSON")
            print(f"{'=' * 55}")
            print(json.dumps(parsed, indent=2))

            # Summary
            for item in parsed.get("items", []):
                action = item.get("action", "?")
                name = item.get("food_name", "?")
                conf = item.get("vlm_confidence", 0)
                grams = item.get("estimated_grams", 0)
                cal = item.get("calories_per_100g", 0)
                print(f"\n  Result: {action.upper()}")
                print(f"  Food:   {name} "
                      f"(confidence: {conf})")
                print(f"  Portion: {grams}g, "
                      f"{cal} kcal/100g")
    except Exception as e:
        print(f"\nParse error: {e}")


if __name__ == "__main__":
    data_dir = os.path.join(PROJECT, "data")
    image_path = None
    for name in ["test_food.jpg", "test_food.png",
                  "test.jpg", "food.jpg"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            image_path = p
            break

    model = "gemma4:e4b"
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if os.path.exists(arg):
                image_path = arg
            elif arg.startswith("gemma"):
                model = arg

    if image_path:
        run_fast_test(image_path, model)
    else:
        print(f"No test image in {data_dir}/")
