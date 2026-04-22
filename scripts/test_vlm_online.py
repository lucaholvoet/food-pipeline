"""
Test VLM with REAL Gemma 4 — Online Test

Uses Google AI Studio (FREE) to send a real food image
to Gemma 4 and get back refined nutrition analysis.

Setup:
  1. Get free API key: https://aistudio.google.com/apikey
  2. Paste key in .env file (GOOGLE_API_KEY=...)
  3. Place a food image in the data/ folder
  4. Run: python scripts/test_vlm_online.py

Requires:
  pip install pydantic google-generativeai Pillow
"""

import sys
import os
import json
import base64
import time

# Fix Windows encoding
sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
PROJECT = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))
sys.path.insert(0, PROJECT)

# Load .env
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT, ".env"))

from vlm.schemas import VLMRequest
from vlm.prompts import SYSTEM_PROMPT, build_user_prompt


def encode_image(image_path: str) -> str:
    """Read image and return base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_test_request(image_path: str) -> dict:
    """Build a mock CV pipeline request for testing."""
    return {
        "image_base64": encode_image(image_path),
        "reason": "low_confidence",
        "trigger_threshold": 0.70,
        "cv_output": {
            "image_id": "test-001",
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


def test_with_gemini(image_path: str):
    """Send image + CV data to Gemma 4 via Google AI Studio."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("ERROR: Set your API key in .env file")
        print("  1. Go to: https://aistudio.google.com/apikey")
        print("  2. Click 'Create API Key'")
        print("  3. Paste it in .env as GOOGLE_API_KEY=...")
        return

    import google.generativeai as genai
    genai.configure(api_key=api_key)

    print("=" * 55)
    print("VLM Online Test — Gemma 4 via Google AI Studio")
    print("=" * 55)

    # Build request
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

    # Prepare image for Gemini
    image_bytes = base64.b64decode(request.image_base64)

    # Call Gemma 4
    print("\nCalling Gemma 4 (gemma-4-27b-it)...")
    print("  (this may take 10-30 seconds)")

    model = genai.GenerativeModel(
        model_name="gemma-4-27b-it",
        system_instruction=SYSTEM_PROMPT,
    )

    start = time.time()
    response = model.generate_content([
        user_prompt,
        {
            "mime_type": "image/jpeg",
            "data": image_bytes,
        },
    ])
    elapsed = int((time.time() - start) * 1000)

    # Print raw response
    print(f"\n{'=' * 55}")
    print(f"Gemma 4 Response ({elapsed}ms)")
    print(f"{'=' * 55}")

    raw_text = response.text
    print(raw_text)

    # Try to parse as JSON
    print(f"\n{'=' * 55}")
    print("Parsed JSON")
    print(f"{'=' * 55}")
    try:
        if "```json" in raw_text:
            start_i = raw_text.index("```json") + 7
            end_i = raw_text.index("```", start_i)
            json_str = raw_text[start_i:end_i].strip()
        elif "```" in raw_text:
            start_i = raw_text.index("```") + 3
            end_i = raw_text.index("```", start_i)
            json_str = raw_text[start_i:end_i].strip()
        elif "{" in raw_text:
            start_i = raw_text.index("{")
            end_i = raw_text.rindex("}") + 1
            json_str = raw_text[start_i:end_i]
        else:
            json_str = raw_text

        parsed = json.loads(json_str)
        print(json.dumps(parsed, indent=2))
    except Exception as e:
        print(f"Could not parse JSON: {e}")

    print(f"\nDone in {elapsed}ms")


if __name__ == "__main__":
    # Look for a test image
    data_dir = os.path.join(PROJECT, "data")

    # Check for any image file
    image_path = None
    for name in ["test_food.jpg", "test_food.png",
                  "test.jpg", "test.png",
                  "food.jpg", "food.png"]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            image_path = path
            break

    if not image_path:
        print("=" * 55)
        print("No test image found!")
        print("=" * 55)
        print(f"\nPlace any food image in: {data_dir}/")
        print("Name it: test_food.jpg")
        print("\nOr pass image path as argument:")
        print(
            "  python scripts/test_vlm_online.py "
            "C:/path/to/food.jpg"
        )

        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            if os.path.exists(image_path):
                test_with_gemini(image_path)
            else:
                print(f"\nFile not found: {image_path}")
    else:
        test_with_gemini(image_path)
