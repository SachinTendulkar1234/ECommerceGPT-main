"""Review sentiment analysis and evidence-based review intelligence."""

from collections import Counter
from typing import Any, Dict, Optional

from data_utils import load_reviews
from ollama_client import OllamaClient, OllamaUnavailable


class ReviewModelUnavailable(RuntimeError):
    """Raised when the requested pre-trained sentiment model is not usable."""


class ReviewModule:
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self, ollama: Optional[OllamaClient] = None):
        self.ollama = ollama or OllamaClient()
        self.reviews = load_reviews()
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline("sentiment-analysis", model=self.MODEL_NAME)
            except Exception as error:
                raise ReviewModelUnavailable(
                    "The pre-trained sentiment model could not be loaded. "
                    f"Install requirements and allow the first model download: {error}"
                ) from error
        return self._pipeline

    def analyze_product(self, product_name: str) -> Dict[str, Any]:
        selected = self.reviews[
            self.reviews["ProductName"].str.contains(product_name, case=False, na=False)
        ]
        if selected.empty:
            return {"product_name": product_name, "found": False, "message": "No matching reviews found."}
        sentiment_model = self._get_pipeline()
        details = []
        for _, row in selected.iterrows():
            prediction = sentiment_model(str(row["Review"]))[0]
            label = "Positive" if prediction["label"].upper() == "POSITIVE" else "Negative"
            if float(row["Rating"]) == 3:
                label = "Neutral"
            details.append({
                "review": str(row["Review"]),
                "sentiment": label,
                "score": float(prediction["score"]),
                "rating": int(row["Rating"]),
            })
        counts = Counter(item["sentiment"] for item in details)
        strengths = [item["review"] for item in details if item["sentiment"] == "Positive"][:3]
        weaknesses = [item["review"] for item in details if item["sentiment"] == "Negative"][:3]
        evidence = "\n".join(
            f"- {item['review']} ({item['sentiment']}, score {item['score']:.3f})"
            for item in details
        )
        prompt = (
            "Summarize only the customer review evidence below. Return sections "
            "Overall Opinion, Strengths, Weaknesses, and Buying Suggestion. "
            "Do not invent themes.\n"
            f"Product: {selected.iloc[0]['ProductName']}\n{evidence}"
        )
        try:
            generated = self.ollama.generate(prompt)
        except OllamaUnavailable as error:
            generated = None
            warning = str(error)
        else:
            warning = None
        return {
            "product_name": str(selected.iloc[0]["ProductName"]),
            "found": True,
            "counts": {
                "Positive": counts.get("Positive", 0),
                "Neutral": counts.get("Neutral", 0),
                "Negative": counts.get("Negative", 0),
            },
            "average_rating": float(selected["Rating"].mean()),
            "details": details,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "ollama_summary": generated,
            "ollama_warning": warning,
        }