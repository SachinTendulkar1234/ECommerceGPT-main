"""The only LLM client used by this experiment: Ollama."""

import os
from typing import Optional

import requests


class OllamaUnavailable(RuntimeError):
    """Raised when the local Ollama service cannot be reached."""


class OllamaClient:
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if not response.ok:
                return False
            models = response.json().get("models", [])
            return any(
                item.get("name") == self.model or item.get("model") == self.model
                for item in models
            )
        except (requests.RequestException, ValueError):
            return False

    def status_message(self) -> str:
        if self.is_available():
            return f"Ollama is available. Configured model: {self.model}"
        return (
            "Ollama is unavailable. Start Ollama and run "
            f"'ollama pull {self.model}' to enable generated responses."
        )

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Generate text without silently switching to another provider."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False,
                      "options": {"temperature": temperature}},
                timeout=90,
            )
            response.raise_for_status()
            result = response.json().get("response", "").strip()
        except (requests.RequestException, ValueError) as error:
            raise OllamaUnavailable(
                f"Could not generate with Ollama ({self.model}): {error}"
            ) from error
        if not result:
            raise OllamaUnavailable("Ollama returned an empty response.")
        return result