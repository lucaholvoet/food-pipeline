"""Evaluate the local VLM on the 10-image benchmark set."""

import base64
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

import requests as http_requests
from vlm.normalization import normalize_food101_label
from vlm.prompts import FAST_SYSTEM_PROMPT, build_fast_prompt

EVAL_DIR = os.path.join(PROJECT, "data", "eval")
RESULTS_FILE = os.path.join(PROJECT, "data", "eval_results.json")
MODEL = "gemma4:e4b"

TEST_CASES = [
    {"file": "pizza.jpg", "actual_food": "pizza", "cv_prediction": "pizza", "confidence": 0.45, "should_correct": False},
    {"file": "burger.jpg", "actual_food": "hamburger", "cv_prediction": "hot_dog", "confidence": 0.38, "should_correct": True},
    {"file": "sushi.jpg", "actual_food": "sushi", "cv_prediction": "sushi", "confidence": 0.50, "should_correct": False},
    {"file": "pasta.jpg", "actual_food": "spaghetti_bolognese", "cv_prediction": "ramen", "confidence": 0.42, "should_correct": True},
    {"file": "salad.jpg", "actual_food": "greek_salad", "cv_prediction": "caesar_salad", "confidence": 0.35, "should_correct": True},
    {"file": "pancakes.jpg", "actual_food": "pancakes", "cv_prediction": "french_toast", "confidence": 0.40, "should_correct": True},
    {"file": "icecream.jpg", "actual_food": "ice_cream", "cv_prediction": "frozen_yogurt", "confidence": 0.33, "should_correct": True},
    {"file": "steak.jpg", "actual_food": "steak", "cv_prediction": "filet_mignon", "confidence": 0.48, "should_correct": False},
    {"file": "ramen.jpg", "actual_food": "ramen", "cv_prediction": "pho", "confidence": 0.36, "should_correct": True},
    {"file": "cake.jpg", "actual_food": "chocolate_cake", "cv_prediction": "red_velvet_cake", "confidence": 0.41, "should_correct": True},
]


def call_gemma4(image_base64: str, prompt: str, model: str = MODEL) -> str:
    response = http_requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": FAST_SYSTEM_PROMPT},
                {"role": "user", "content": prompt, "images": [image_base64]},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "top_p": 0.9, "top_k": 40},
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def parse_response(text: str) -> dict | None:
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if parsed is None and "{" in text and "}" in text:
        start = text.index("{")
        end = text.rindex("}") + 1
        parsed = json.loads(text[start:end])

    if parsed is None:
        return None

    if isinstance(parsed.get("items"), list) and parsed["items"]:
        return parsed["items"][0]
    return parsed


def evaluate_match(vlm_food: str, actual_food: str) -> str:
    vlm = normalize_food101_label(vlm_food, fallback=vlm_food)
    actual = normalize_food101_label(actual_food, fallback=actual_food)
    if vlm == actual:
        return "exact"
    if actual.replace("_", "") in vlm.replace("_", ""):
        return "partial"
    if vlm.replace("_", "") in actual.replace("_", ""):
        return "partial"
    return "wrong"


def run_evaluation() -> None:
    print("=" * 60)
    print("Optimized VLM Evaluation - Gemma 4 on 10 Food Images")
    print("=" * 60)

    results = []
    exact = 0
    partial = 0
    wrong = 0
    total_time = 0

    for index, case in enumerate(TEST_CASES, start=1):
        image_path = os.path.join(EVAL_DIR, case["file"])
        if not os.path.exists(image_path):
            print(f"\n[{index}/10] SKIP {case['file']} - not found")
            continue

        print(f"\n[{index}/10] {case['file']}")
        print(f"  Actual:  {case['actual_food']}")
        print(f"  CV:      {case['cv_prediction']} ({case['confidence']:.0%})")

        with open(image_path, "rb") as handle:
            image_base64 = base64.b64encode(handle.read()).decode()

        prompt = build_fast_prompt(case["cv_prediction"], case["confidence"])

        start = time.time()
        try:
            raw = call_gemma4(image_base64, prompt)
            elapsed = int((time.time() - start) * 1000)
            total_time += elapsed
            parsed = parse_response(raw)
            if parsed is None:
                raise ValueError("Could not parse model JSON")

            predicted = normalize_food101_label(
                parsed.get("food_name", case["cv_prediction"]),
                fallback=case["cv_prediction"],
            )
            match = evaluate_match(predicted, case["actual_food"])

            if match == "exact":
                exact += 1
            elif match == "partial":
                partial += 1
            else:
                wrong += 1

            print(f"  VLM:     {predicted} ({parsed.get('vlm_confidence', 0):.0%}) [{match.upper()}]")
            print(f"  Action:  {parsed.get('action', '?')} | {parsed.get('estimated_grams', '?')}g | {elapsed/1000:.1f}s")

            results.append(
                {
                    "file": case["file"],
                    "actual": case["actual_food"],
                    "cv_prediction": case["cv_prediction"],
                    "vlm_food": predicted,
                    "vlm_confidence": parsed.get("vlm_confidence", 0),
                    "match": match,
                    "action": parsed.get("action", "unknown"),
                    "estimated_grams": parsed.get("estimated_grams", 0),
                    "time_ms": elapsed,
                }
            )
        except Exception as exc:
            elapsed = int((time.time() - start) * 1000)
            total_time += elapsed
            wrong += 1
            print(f"  ERROR: {exc}")
            results.append(
                {
                    "file": case["file"],
                    "actual": case["actual_food"],
                    "cv_prediction": case["cv_prediction"],
                    "vlm_food": "error",
                    "match": "wrong",
                    "time_ms": elapsed,
                    "error": str(exc),
                }
            )

    total = len(results)
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Exact:           {exact}/{total}")
    print(f"  Partial:         {partial}/{total}")
    print(f"  Wrong:           {wrong}/{total}")
    print(f"  Exact+Partial:   {(exact + partial) / max(total, 1) * 100:.0f}%")
    print(f"  Avg time/image:  {total_time / max(total, 1) / 1000:.1f}s")

    with open(RESULTS_FILE, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_evaluation()
