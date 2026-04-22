"""
VLM Client — Gemma 4 API wrapper

Supports two backends:
  1. Ollama (local)    — gemma4:26b, gemma4:31b, gemma4:e4b, etc.
  2. Google AI Studio  — cloud API with API key

Usage:
    from vlm.client import Gemma4Client

    client = Gemma4Client(backend="ollama", model="gemma4:26b")
    # or
    client = Gemma4Client(backend="google", model="gemma-4-27b-it")

    response = client.analyze_image(image_base64, prompt)
"""

import json
import base64
import time
import requests
from typing import Optional


class Gemma4Client:
    """Unified client for Gemma 4 via Ollama or Google AI Studio."""

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "gemma4:26b",
        api_key: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        timeout: int = 120,
    ):
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.timeout = timeout

        if backend == "google":
            self._setup_google()

    def _setup_google(self):
        """Initialize Google AI Studio SDK."""
        try:
            import google.generativeai as genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
            self._genai = genai
        except ImportError:
            raise ImportError(
                "Google AI SDK not installed. Run:\n"
                "  pip install google-generativeai"
            )

    # ──────────────────────────────────────────
    #  Main method
    # ──────────────────────────────────────────

    def analyze_image(
        self,
        image_base64: str,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Send image + prompt to Gemma 4 and return parsed JSON.

        Args:
            image_base64: Base64-encoded image string
            prompt: User prompt (from prompts.py)
            system_prompt: Optional system prompt

        Returns:
            dict: Parsed JSON response from the model
        """
        start = time.time()

        if self.backend == "ollama":
            raw = self._call_ollama(
                image_base64, prompt, system_prompt
            )
        elif self.backend == "google":
            raw = self._call_google(
                image_base64, prompt, system_prompt
            )
        else:
            raise ValueError(
                f"Unknown backend: {self.backend}. "
                f"Use 'ollama' or 'google'."
            )

        elapsed_ms = int((time.time() - start) * 1000)

        # Parse JSON from model response
        parsed = self._extract_json(raw)
        parsed["_processing_time_ms"] = elapsed_ms

        return parsed

    # ──────────────────────────────────────────
    #  Ollama backend
    # ──────────────────────────────────────────

    def _call_ollama(
        self,
        image_base64: str,
        prompt: str,
        system_prompt: Optional[str],
    ) -> str:
        """Call Gemma 4 via Ollama local API."""
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt,
            "images": [image_base64]
        })

        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    # ──────────────────────────────────────────
    #  Google AI Studio backend
    # ──────────────────────────────────────────

    def _call_google(
        self,
        image_base64: str,
        prompt: str,
        system_prompt: Optional[str],
    ) -> str:
        """Call Gemma 4 via Google AI Studio API."""
        import google.generativeai as genai

        model = self._genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt,
        )

        # Decode base64 for Google SDK
        image_bytes = base64.b64decode(image_base64)

        response = model.generate_content([
            prompt,
            {
                "mime_type": "image/jpeg",
                "data": image_bytes,
            },
        ])

        return response.text

    # ──────────────────────────────────────────
    #  JSON extraction helper
    # ──────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> dict:
        """
        Extract JSON from model response.
        Handles cases where model wraps JSON in markdown.
        """
        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            return json.loads(text[start:end].strip())
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return json.loads(text[start:end].strip())

        # Try finding JSON object boundaries
        if "{" in text and "}" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])

        raise ValueError(
            f"Could not extract JSON from model response:\n"
            f"{text[:500]}"
        )
