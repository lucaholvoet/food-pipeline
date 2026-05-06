"""
Quick test for the VLM module.

Tests schema validation, prompt building, and client setup.
Does NOT require Gemma 4 running — tests are offline.

Run:
    python scripts/test_vlm.py
"""

import sys
import os

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

import json
from vlm.schemas import VLMRequest, VLMResponse
from vlm.prompts import build_user_prompt, SYSTEM_PROMPT


def test_schemas():
    """Test that example JSON validates against schemas."""
    print("📋 Testing schemas...")

    with open("vlm/example_input.json") as f:
        input_data = json.load(f)

    # Validate input
    request = VLMRequest(**input_data)
    print(f"  ✅ Input schema valid — image_id: "
          f"{request.cv_output.image_id}")
    print(f"     Items: {len(request.cv_output.items)}, "
          f"Avg confidence: "
          f"{request.cv_output.average_confidence}")

    with open("vlm/example_output.json") as f:
        output_data = json.load(f)

    # Validate output
    response = VLMResponse(
        **output_data["vlm_refinement"]
    )
    print(f"  ✅ Output schema valid — "
          f"{len(response.items)} items")
    print(f"     Action: {response.items[0].action.value}")
    print(f"     Refined: "
          f"{response.items[0].refined.food_name}")


def test_prompt():
    """Test prompt building."""
    print("\n📝 Testing prompt builder...")

    with open("vlm/example_input.json") as f:
        input_data = json.load(f)

    request = VLMRequest(**input_data)

    prompt = build_user_prompt(
        reason=request.reason.value,
        threshold=request.trigger_threshold,
        avg_confidence=request.cv_output.average_confidence,
        cv_output_json=request.cv_output.model_dump_json(),
    )

    print(f"  ✅ System prompt: {len(SYSTEM_PROMPT)} chars")
    print(f"  ✅ User prompt: {len(prompt)} chars")


def test_client_setup():
    """Test client can be initialized (no API call)."""
    print("\n🔌 Testing client setup...")

    from vlm.client import Gemma4Client

    # Ollama
    client = Gemma4Client(backend="ollama", model="gemma4:26b")
    print(f"  ✅ Ollama client — model: {client.model}")

    # Google
    try:
        client = Gemma4Client(
            backend="google",
            model="gemma-4-27b-it",
            api_key="test-key",
        )
        print(f"  ✅ Google client — model: {client.model}")
    except ImportError:
        print("  ⚠️  Google SDK not installed "
              "(pip install google-generativeai)")


if __name__ == "__main__":
    print("=" * 50)
    print("VLM Module — Offline Tests")
    print("=" * 50)

    test_schemas()
    test_prompt()
    test_client_setup()

    print("\n" + "=" * 50)
    print("All tests passed! ✅")
    print("=" * 50)
