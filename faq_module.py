"""FAQ semantic search using stored embeddings, threshold, and Ollama."""

import json
from typing import Any, Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from create_embeddings import EMBEDDING_FILE, create_faq_embeddings
from data_utils import load_faqs
from ollama_client import OllamaClient, OllamaUnavailable


class FAQModule:
    def __init__(self, threshold: float = 0.20, ollama: Optional[OllamaClient] = None):
        self.threshold = threshold
        self.ollama = ollama or OllamaClient()
        self.embeddings_file = EMBEDDING_FILE
        self.faqs = load_faqs()
        self._load_stored_embeddings()

    def _load_stored_embeddings(self) -> None:
        """Load the persisted FAQ matrix and rebuild only the query vectorizer."""
        if not self.embeddings_file.exists():
            create_faq_embeddings(self.embeddings_file)

        try:
            payload = json.loads(self.embeddings_file.read_text(encoding="utf-8"))
            questions = payload["questions"]
            answers = payload["answers"]
            vectors = np.asarray(payload["vectors"], dtype=float)
            vocabulary = {
                str(term): int(index)
                for term, index in payload["vocabulary"].items()
            }
            idf = np.asarray(payload["idf"], dtype=float)
            metadata = payload["vectorizer"]
            if (
                questions != self.faqs["Question"].fillna("").tolist()
                or answers != self.faqs["Answer"].fillna("").tolist()
                or vectors.ndim != 2
                or vectors.shape[0] != len(questions)
                or vectors.shape[1] != len(vocabulary)
                or len(idf) != len(vocabulary)
            ):
                raise ValueError("Stored FAQ embeddings do not match the current FAQ dataset.")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            create_faq_embeddings(self.embeddings_file)
            payload = json.loads(self.embeddings_file.read_text(encoding="utf-8"))
            questions = payload["questions"]
            answers = payload["answers"]
            vectors = np.asarray(payload["vectors"], dtype=float)
            vocabulary = {
                str(term): int(index)
                for term, index in payload["vocabulary"].items()
            }
            idf = np.asarray(payload["idf"], dtype=float)
            metadata = payload["vectorizer"]

        self.questions = [str(question) for question in questions]
        self.answers = [str(answer) for answer in answers]
        self.matrix = vectors
        self.idf = idf
        self.vectorizer = CountVectorizer(
            vocabulary=vocabulary,
            lowercase=bool(metadata["lowercase"]),
            stop_words=metadata["stop_words"],
            token_pattern=metadata["token_pattern"],
        )

    def _query_embedding(self, question: str) -> np.ndarray:
        counts = self.vectorizer.transform([question]).toarray()[0].astype(float)
        weighted = counts * self.idf
        norm = np.linalg.norm(weighted)
        return weighted / norm if norm else weighted

    def search(self, question: str) -> Optional[Dict[str, Any]]:
        query_vector = self._query_embedding(question)
        scores = cosine_similarity(query_vector.reshape(1, -1), self.matrix)[0]
        best_index = int(np.argmax(scores))
        score = float(scores[best_index])
        return {
            "question": self.questions[best_index],
            "answer": self.answers[best_index],
            "score": score,
            "matched": score >= self.threshold,
        }

    def answer(self, question: str) -> Dict[str, Any]:
        match = self.search(question)
        if not match or not match["matched"]:
            return {
                "matched": False,
                "message": "No FAQ passed the similarity threshold; try another chatbot module.",
                "match": match,
                "ollama_used": False,
            }
        prompt = (
            "Answer the user's question using only the retrieved FAQ answer. "
            "Do not add policies or facts. Be concise.\n"
            f"User question: {question}\n"
            f"Retrieved FAQ question: {match['question']}\n"
            f"Retrieved FAQ answer: {match['answer']}"
        )
        try:
            response = self.ollama.generate(prompt)
            return {**match, "response": response, "ollama_used": True}
        except OllamaUnavailable as error:
            return {
                **match,
                "response": None,
                "ollama_used": False,
                "warning": str(error),
            }