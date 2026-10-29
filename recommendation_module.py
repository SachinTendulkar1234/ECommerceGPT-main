"""TF-IDF product retrieval and Ollama recommendation with justification."""

from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_utils import load_products
from ollama_client import OllamaClient, OllamaUnavailable


class RecommendationModule:
    def __init__(self, ollama: Optional[OllamaClient] = None):
        self.ollama = ollama or OllamaClient()
        self.products = load_products()
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self.documents = (
            self.products["ProductName"].fillna("")
            + " "
            + self.products["Category"].fillna("")
            + " "
            + self.products["Brand"].fillna("")
            + " "
            + self.products["Description"].fillna("")
        )
        self.matrix = self.vectorizer.fit_transform(self.documents)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # Prefer an explicit product name in a product question. This keeps
        # follow-up RAG answers grounded in the product the user named.
        normalized_query = query.lower()
        exact = self.products[
            self.products["ProductName"].str.lower().apply(
                lambda name: name in normalized_query
            )
        ]
        if not exact.empty:
            results = []
            for _, product in exact.head(top_k).iterrows():
                item = product.to_dict()
                item["score"] = 1.0
                results.append(item)
            return results
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        ranked = scores.argsort()[::-1][:top_k]
        results = []
        for index in ranked:
            product = self.products.iloc[int(index)].to_dict()
            product["score"] = float(scores[index])
            results.append(product)
        return results

    def recommend(self, query: str) -> Dict[str, Any]:
        products = self.retrieve(query)
        context = "\n".join(
            f"{item['ProductName']} | Brand: {item['Brand']} | "
            f"Category: {item['Category']} | Price: ₹{item['Price']} | "
            f"Rating: {item['Rating']} | {item['Description']}"
            for item in products
        )
        prompt = (
            "Choose exactly one best product from the retrieved products. "
            "Give a short reason using only this context.\n"
            f"User request: {query}\nRetrieved products:\n{context}"
        )
        try:
            response = self.ollama.generate(prompt)
            return {"products": products, "response": response, "ollama_used": True}
        except OllamaUnavailable as error:
            return {
                "products": products,
                "response": None,
                "ollama_used": False,
                "warning": str(error),
            }