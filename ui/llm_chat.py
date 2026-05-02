import os
import json
from google import genai
from google.genai import types
from datetime import date, timedelta

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

The user disagrees with the automatic detection. Be decisive and efficient.

Rules:
- After the user's FIRST message, immediately estimate the nutrition and output the JSON
- Do NOT ask multiple clarifying questions — make reasonable assumptions
- If portion size is unclear, assume a standard serving
- Maximum 2 exchanges before you must output the JSON
- Always output the JSON after your response so the user can confirm

When ready to log, output this JSON block at the end of your response:
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
- Use realistic nutrition values
- Output the JSON on the FIRST or SECOND response, not later
- The user will confirm by clicking a button — you don't need to ask "is this okay?" in text"""

MANUAL_SYSTEM_PROMPT = """You are a nutrition assistant helping a user log a meal manually.

Be decisive and efficient. Make reasonable assumptions about portions.

Rules:
- After the user describes their meal, immediately estimate nutrition and output the JSON
- Do NOT ask multiple clarifying questions — assume standard portions if unclear
- Maximum 2 exchanges before you must output the JSON
- Always output the JSON after your response so the user can confirm

When ready to log, output this JSON block at the end of your response:
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
- Default log_date to today unless user specifies otherwise
- Output JSON on the FIRST or SECOND response
- The user confirms by clicking a button"""

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

    today = date.today()
    # Calculate recent weekdays for context
    days_context = "\n\nDate context:"
    days_context += f"\n- Today is {today.strftime('%A %d %B %Y')} ({today.isoformat()})"
    for i in range(1, 8):
        past = today - timedelta(days=i)
        days_context += f"\n- {past.strftime('%A')} was {past.strftime('%d %B %Y')} ({past.isoformat()})"

    system = MANUAL_SYSTEM_PROMPT + days_context

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