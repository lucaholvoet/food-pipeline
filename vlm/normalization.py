"""Helpers for mapping noisy VLM labels back to valid Food-101 labels."""

from vlm.prompts import FOOD101_CLASSES

FOOD101_SET = set(FOOD101_CLASSES)

_ALIAS_MAP = {
    "burger": "hamburger",
    "cheeseburger": "hamburger",
    "beef_burger": "hamburger",
    "hamburger": "hamburger",
    "icecream": "ice_cream",
    "ice_cream": "ice_cream",
    "ice cream": "ice_cream",
    "ice-cream": "ice_cream",
    "sushi_roll": "sushi",
    "sushi_rolls": "sushi",
    "sushi platter": "sushi",
    "cheese_pancakes": "pancakes",
    "cheesy_pancakes": "pancakes",
    "pancake_stack": "pancakes",
    "pasta": "spaghetti_bolognese",
    "spaghetti": "spaghetti_bolognese",
    "meat_pasta": "spaghetti_bolognese",
    "salad": "greek_salad",
    "salad_bowl": "greek_salad",
    "green_salad": "greek_salad",
    "noodle_soup": "ramen",
    "shrimp_noodle_soup": "ramen",
    "noodles": "ramen",
    "fries": "french_fries",
    "salmon": "grilled_salmon",
    "mac_and_cheese": "macaroni_and_cheese",
}


def _canonicalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("-", "_")
    text = text.replace("/", "_")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.replace(" ", "_")


def normalize_food101_label(raw_label: str, fallback: str | None = None) -> str:
    """Return a valid Food-101 label, falling back if needed."""
    normalized = _canonicalize(raw_label)

    if normalized in FOOD101_SET:
        return normalized

    if raw_label in _ALIAS_MAP:
        return _ALIAS_MAP[raw_label]

    if normalized in _ALIAS_MAP:
        return _ALIAS_MAP[normalized]

    for key, value in _ALIAS_MAP.items():
        if key in normalized:
            return value

    for label in FOOD101_CLASSES:
        if normalized == label.replace("_", ""):
            return label
        if normalized in label or label in normalized:
            return label

    if fallback and _canonicalize(fallback) in FOOD101_SET:
        return _canonicalize(fallback)

    return fallback or raw_label


def display_name_from_label(label: str) -> str:
    return label.replace("_", " ").title()
