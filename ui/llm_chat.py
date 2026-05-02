import os
import json
from google import genai
from google.genai import types

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
MAX_TURNS = 6  # max back-and-forth messages

def _get_client():
    return genai.Client(api_key=GOOGLE_API_KEY)

def _format_pipeline_result(pipeline_data: dict) -> str:
    """Convert CV/VLM pipeline JSON to readable summary for LLM context."""
    items = pipeline_data.get("items", [])
    totals = pipeline_data.get("totals", {})
    is_vlm = "refinement_status" in pipeline_data

    lines = ["The food analysis pipeline detected the following:"]
    for item in items:
        if is_vlm:
            refined = item.get("refined", {})
            portion = item.get("portion", {})
            name = refined.get("display_name", "Unknown")
            grams = portion.get("estimated_grams", 0)
            cal = item.get("nutrition_total", {}).get("calories_kcal", 0)
        else:
            name = item.get("display_name", "Unknown")
            grams = item.get("estimated_grams", 0)
            cal = item.get("nutrition_total", {}).get("calories_kcal", 0)
        lines.append(f"- {name}: {grams:.0f}g, {cal:.0f} kcal")

    lines.append(f"\nTotal: {totals.get('calories_kcal', 0):.0f} kcal, "
                 f"{totals.get('protein_g', 0):.0f}g protein, "
                 f"{totals.get('fat_g', 0):.0f}g fat, "
                 f"{totals.get('carbs_g', 0):.0f}g carbs")
    return "\n".join(lines)

CORRECTION_SYSTEM_PROMPT = """You are a nutrition assistant helping a user correct a food analysis result.

The user was shown an automatic food detection result and disagrees with it. Your job is to:
1. Understand what the user actually ate
2. Ask clarifying questions if needed (e.g. portion size, cooking method)
3. Estimate the nutrition based on their description
4. When you have enough information, output a JSON block with the corrected meal

Rules:
- Be conversational and friendly
- Maximum 6 exchanges total
- When ready to log, output EXACTLY this JSON block (no other text around it):
```json
{
  "ready_to_log": true,
  "meal_name": "Corrected meal name",
  "items_description": "Brief description of what was eaten",
  "totals": {
    "calories_kcal": 0,
    "protein_g": 0,
    "fat_g": 0,
    "carbs_g": 0,
    "fiber_g": 0
  }
}
```
- Use realistic nutrition values based on standard food databases
- If the user confirms they are happy, output the JSON
- Never output the JSON until the user has confirmed what they ate"""

MANUAL_SYSTEM_PROMPT = """You are a nutrition assistant helping a user log a meal manually.

The user will describe what they ate. Your job is to:
1. Understand what they ate and how much
2. Ask clarifying questions about portions if needed
3. Estimate nutrition based on their description
4. When ready, output a JSON block to log the meal

Rules:
- Be conversational and friendly  
- Maximum 6 exchanges total
- When ready to log, output EXACTLY this JSON block:
```json
{
  "ready_to_log": true,
  "meal_name": "Meal name",
  "items_description": "Brief description",
  "log_date": "YYYY-MM-DD",
  "totals": {
    "calories_kcal": 0,
    "protein_g": 0,
    "fat_g": 0,
    "carbs_g": 0,
    "fiber_g": 0
  }
}
```
- Use realistic nutrition values
- Ask for the date if not provided (default to today)
- Always confirm with the user before outputting JSON"""

def chat_correction(history: list, user_message: str, pipeline_data: dict) -> tuple[list, dict | None]:
    """
    Send a message in the correction chat.
    Returns (updated_history, log_data_if_ready)
    history format: list of {"role": "user"|"assistant", "content": str}
    """
    client = _get_client()

    # Build context on first message
    if not history:
        context = _format_pipeline_result(pipeline_data)
        system = CORRECTION_SYSTEM_PROMPT + f"\n\nOriginal detection result:\n{context}"
    else:
        system = CORRECTION_SYSTEM_PROMPT

    # Build messages for API
    messages = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        messages.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    messages.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(system_instruction=system),
        contents=messages,
    )

    assistant_text = response.text.strip()

    # Update history
    new_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_text},
    ]

    # Check if ready to log
    log_data = _extract_log_data(assistant_text)

    return new_history, log_data

def chat_manual(history: list, user_message: str, default_date: str) -> tuple[list, dict | None]:
    """
    Send a message in the manual logging chat.
    Returns (updated_history, log_data_if_ready)
    """
    client = _get_client()

    system = MANUAL_SYSTEM_PROMPT + f"\n\nDefault date if not specified: {default_date}"

    messages = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        messages.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    messages.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(system_instruction=system),
        contents=messages,
    )

    assistant_text = response.text.strip()

    new_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_text},
    ]

    log_data = _extract_log_data(assistant_text)
    return new_history, log_data

def _extract_log_data(text: str) -> dict | None:
    """Extract JSON log data from assistant response if present."""
    try:
        start = text.find("```json")
        end = text.find("```", start + 6)
        if start == -1 or end == -1:
            return None
        json_str = text[start + 7:end].strip()
        data = json.loads(json_str)
        if data.get("ready_to_log"):
            return data
    except Exception:
        pass
    return None