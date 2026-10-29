"""Product-specific RAG retrieval with context-constrained question answering."""

from typing import Any, Dict, Optional

from recommendation_module import RecommendationModule
from ollama_client import OllamaClient, OllamaUnavailable


class ProductModule:
    def __init__(self, ollama: Optional[OllamaClient] = None):
        self.recommender = RecommendationModule(ollama)
        self.ollama = self.recommender.ollama
        self.last_product: Optional[Dict[str, Any]] = None

    @staticmethod
    def structured_context(product: Dict[str, Any]) -> str:
        return (
            f"Product name: {product['ProductName']}\n"
            f"Category: {product['Category']}\n"
            f"Brand: {product['Brand']}\n"
            f"Price: ₹{product['Price']}\n"
            f"Catalog rating: {product['Rating']}/5\n"
            f"Description: {product['Description']}"
        )

    def answer(self, question: str) -> Dict[str, Any]:
        retrieved = self.recommender.retrieve(question, top_k=1)
        product = retrieved[0] if retrieved and retrieved[0]["score"] > 0 else self.last_product
        if product is None:
            return {
                "product": None,
                "response": "No product was found in the uploaded catalog.",
                "ollama_used": False,
            }
        self.last_product = product
        context = self.structured_context(product)
        prompt = (
            "Answer only from the PRODUCT CONTEXT. If the answer is not present, "
            "say that the context does not contain the information. Never invent "
            "specifications, features, or prices.\n"
            f"Question: {question}\nPRODUCT CONTEXT:\n{context}"
        )
        try:
            response = self.ollama.generate(prompt)
            return {
                "product": product,
                "similarity": product["score"],
                "context": context,
                "response": response,
                "ollama_used": True,
            }
        except OllamaUnavailable as error:
            return {
                "product": product,
                "similarity": product["score"],
                "context": context,
                "response": None,
                "ollama_used": False,
                "warning": str(error),
            }