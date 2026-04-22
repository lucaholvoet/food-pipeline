"""
Test VLM with REAL Gemma 4 - Local (Ollama)

Fully open-source, fully local, zero cost.
Uses Ollama to run Gemma 4 on your machine.

Run:
    venv\\Scripts\\activate
    python scripts/test_vlm_local.py
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
from vlm.schemas import VLMRequest
from vlm.prompts import SYSTEM_PROMPT, build_user_prompt


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_test_request(image_path: str) -> dict:
    return {
        "image_base64": encode_image(image_path),
        "reason": "low_confidence",
        "trigger_threshold": 0.70,
        "cv_output": {
            "image_id": "test-local-001",
            "status": "success",
            "average_confidence": 0.55,
            "plate_detected": True,
            "items": [
                {
                    "item_id": 1,
                    "food_name": "spaghetti_bolognese",
                    "display_name": "Spaghetti Bolognese",
                    "classification_confidence": 0.55,
                    "detection_confidence": 0.80,
                    "top3_predictions": [
                        {"label": "spaghetti_bolognese",
                         "confidence": 0.55},
                        {"label": "spaghetti_carbonara",
                         "confidence": 0.20},
                        {"label": "ramen",
                         "confidence": 0.10}
                    ],
                    "estimated_grams": 250,
                    "portion_method": "plate_reference",
                    "bbox": [100, 50, 400, 350],
                    "nutrition_per_100g": {
                        "calories_kcal": 132,
                        "protein_g": 5.8,
                        "fat_g": 4.5,
                        "carbs_g": 18.1,
                        "fiber_g": 1.4
                    },
                    "nutrition_total": {
                        "calories_kcal": 330,
                        "protein_g": 14.5,
                        "fat_g": 11.3,
                        "carbs_g": 45.3,
                        "fiber_g": 3.5
                    }
                }
            ],
            "totals": {
                "calories_kcal": 330,
                "protein_g": 14.5,
                "fat_g": 11.3,
                "carbs_g": 45.3,
                "fiber_g": 3.5
            }
        }
    }


def call_ollama(
    model: str,
    image_base64: str,
    user_prompt: str,
    system_prompt: str,
):
    """Call Gemma 4 via Ollama local API."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt,
         "images": [image_base64]}
    ]

    response = http_requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
            },
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def extract_json(text: str) -> dict:
    """Extract JSON from model response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```json" in text:
        s = text.index("```json") + 7
        e = text.index("```", s)
        return json.loads(text[s:e].strip())
    elif "```" in text:
        s = text.index("```") + 3
        e = text.index("```", s)
        return json.loads(text[s:e].strip())
    elif "{" in text:
        s = text.index("{")
        e = text.rindex("}") + 1
        return json.loads(text[s:e])
    raise ValueError(f"No JSON found in response")


def test_local(image_path: str, model: str = "gemma4:e4b"):
    print("=" * 55)
    print(f"VLM Local Test — {model} via Ollama")
    print("=" * 55)

    # Validate input
    print(f"\nLoading image: {image_path}")
    request_data = build_test_request(image_path)
    request = VLMRequest(**request_data)
    print(f"  CV confidence: "
          f"{request.cv_output.average_confidence}")
    print(f"  CV prediction: "
          f"{request.cv_output.items[0].food_name}")

    # Build prompt
    user_prompt = build_user_prompt(
        reason=request.reason.value,
        threshold=request.trigger_threshold,
        avg_confidence=request.cv_output.average_confidence,
        cv_output_json=request.cv_output.model_dump_json(),
    )

    # Call Ollama
    print(f"\nCalling {model} locally...")
    print("  (may take 5-60 seconds depending on model)")

    start = time.time()
    raw = call_ollama(
        model=model,
        image_base64=request.image_base64,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
    )
    elapsed = int((time.time() - start) * 1000)

    # Show raw response
    print(f"\n{'=' * 55}")
    print(f"Gemma 4 Response ({elapsed}ms)")
    print(f"{'=' * 55}")
    print(raw)

    # Parse JSON
    print(f"\n{'=' * 55}")
    print("Parsed JSON")
    print(f"{'=' * 55}")
    try:
        parsed = extract_json(raw)
        print(json.dumps(parsed, indent=2))
    except Exception as e:
        print(f"Could not parse JSON: {e}")

    print(f"\nDone in {elapsed}ms")


if __name__ == "__main__":
    # Find test image
    data_dir = os.path.join(PROJECT, "data")
    image_path = None
    for name in ["test_food.jpg", "test_food.png",
                  "test.jpg", "food.jpg"]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            image_path = path
            break

    # Command line arg override
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.exists(arg):
            image_path = arg
        elif arg.startswith("gemma"):
            pass  # model name, no image override

    model = "gemma4:e4b"
    for arg in sys.argv[1:]:
        if arg.startswith("gemma"):
            model = arg

    if not image_path:
        print("No test image found!")
        print(f"Place a food image in: {data_dir}/")
    else:
        test_local(image_path, model)
