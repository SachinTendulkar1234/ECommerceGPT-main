"""Ollama-based intent detection with an explicit local preview fallback."""

import re
from typing import Dict, Optional, Set

from ollama_client import OllamaClient, OllamaUnavailable


INTENTS = ("FAQ", "RECOMMENDATION", "PRODUCT", "REVIEW")


def _local_preview_intent(query: str, excluded_intents: Set[str]) -> str:
    text = query.lower()
    candidates = [
        ("REVIEW", r"review|rating|opinion|feedback"),
        ("RECOMMENDATION", r"recommend|suggest|best|looking for|buy"),
        ("FAQ", r"track|parcel|delivery|return|cancel|refund|warranty|payment|order"),
        ("PRODUCT", r".*"),
    ]
    for intent, pattern in candidates:
        if intent not in excluded_intents and re.search(pattern, text):
            return intent
    return next(intent for intent in INTENTS if intent not in excluded_intents)


def detect_intent(
    query: str,
    ollama: Optional[OllamaClient] = None,
    excluded_intents: Optional[Set[str]] = None,
) -> Dict[str, str]:
    client = ollama or OllamaClient()
    excluded = excluded_intents or set()
    allowed = tuple(intent for intent in INTENTS if intent not in excluded)
    allowed_labels = ", ".join(allowed)
    prompt = (
        f"Classify the query into exactly one label: {allowed_labels}. "
        f"Do not return any excluded label. Return only the label.\nQuery: {query}"
    )
    try:
        output = client.generate(prompt, temperature=0)
        match = re.search(r"\b(FAQ|RECOMMENDATION|PRODUCT|REVIEW)\b", output.upper())
        if match and match.group(1) in allowed:
            return {"intent": match.group(1), "source": "Ollama", "warning": ""}
    except OllamaUnavailable as error:
        warning = str(error)
    else:
        warning = "Ollama returned no valid intent label."
    return {
        "intent": _local_preview_intent(query, excluded),
        "source": "local preview fallback",
        "warning": warning,
    }