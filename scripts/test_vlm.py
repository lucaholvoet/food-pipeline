"""Offline smoke tests for the VLM module."""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from vlm.prompts import SYSTEM_PROMPT, build_user_prompt
from vlm.schemas import VLMRequest, VLMResponse

INPUT_JSON = os.path.join(PROJECT, "vlm", "example_input.json")
OUTPUT_JSON = os.path.join(PROJECT, "vlm", "example_output.json")


def test_schemas() -> None:
    print("Testing schemas...")

    with open(INPUT_JSON, "r", encoding="utf-8") as handle:
        input_data = json.load(handle)
    request = VLMRequest(**input_data)
    print(f"  Input OK: {request.cv_output.image_id}")

    with open(OUTPUT_JSON, "r", encoding="utf-8") as handle:
        output_data = json.load(handle)
    response = VLMResponse(**output_data["vlm_refinement"])
    print(f"  Output OK: {len(response.items)} items")


def test_prompt() -> None:
    print("Testing prompts...")

    with open(INPUT_JSON, "r", encoding="utf-8") as handle:
        input_data = json.load(handle)
    request = VLMRequest(**input_data)

    prompt = build_user_prompt(
        reason=request.reason.value,
        threshold=request.trigger_threshold,
        avg_confidence=request.cv_output.average_confidence,
        cv_output_json=request.cv_output.model_dump_json(),
    )
    print(f"  System prompt chars: {len(SYSTEM_PROMPT)}")
    print(f"  User prompt chars:   {len(prompt)}")


def test_client_setup() -> None:
    print("Testing client setup...")
    from vlm.client import Gemma4Client

    client = Gemma4Client(backend="ollama", model="gemma4:e4b")
    print(f"  Ollama client OK: {client.model}")


if __name__ == "__main__":
    print("=" * 50)
    print("VLM Module Offline Tests")
    print("=" * 50)
    test_schemas()
    test_prompt()
    test_client_setup()
    print("\nAll offline tests passed.")
