"""
Playground - understand the VLM module step by step

Run in VS Code terminal:
    venv\\Scripts\\activate
    python -u playground.py
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8")

# ── STEP 1: Load the example input ──
print("=" * 50)
print("STEP 1: What the VLM receives from CV pipeline")
print("=" * 50)
with open("vlm/example_input.json") as f:
    data = json.load(f)
print(json.dumps(data, indent=2))

# ── STEP 2: Validate it with Pydantic ──
print("\n" + "=" * 50)
print("STEP 2: Validate with Pydantic schemas")
print("=" * 50)
from vlm.schemas import VLMRequest
request = VLMRequest(**data)
print(f"Valid! Image ID: {request.cv_output.image_id}")
print(f"Items: {len(request.cv_output.items)}")
print(f"Avg confidence: {request.cv_output.average_confidence}")
print(f"Food: {request.cv_output.items[0].food_name}")

# ── STEP 3: See what prompt gets built ──
print("\n" + "=" * 50)
print("STEP 3: The prompt sent to Gemma 4")
print("=" * 50)
from vlm.prompts import SYSTEM_PROMPT, build_user_prompt
print(f"System prompt ({len(SYSTEM_PROMPT)} chars):")
print(SYSTEM_PROMPT[:200] + "...\n")

prompt = build_user_prompt(
    reason=request.reason.value,
    threshold=request.trigger_threshold,
    avg_confidence=request.cv_output.average_confidence,
    cv_output_json=request.cv_output.model_dump_json(),
)
print(f"User prompt ({len(prompt)} chars):")
print(prompt[:300] + "...")

# ── STEP 4: Validate example output ──
print("\n" + "=" * 50)
print("STEP 4: What the VLM returns")
print("=" * 50)
with open("vlm/example_output.json") as f:
    out_data = json.load(f)
from vlm.schemas import VLMResponse
response = VLMResponse(**out_data["vlm_refinement"])
print(f"Status: {response.refinement_status}")
print(f"Model: {response.vlm_model}")
print(f"Action: {response.items[0].action.value}")
print(f"Original: {response.items[0].original.food_name}")
print(f"Corrected to: {response.items[0].refined.food_name}")
print(f"VLM confidence: {response.items[0].refined.vlm_confidence}")
print(f"Portion: {response.items[0].portion.estimated_grams}g")

# ── STEP 5: See full output JSON ──
print("\n" + "=" * 50)
print("STEP 5: Full output JSON")
print("=" * 50)
print(response.model_dump_json(indent=2))
