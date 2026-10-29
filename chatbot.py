"""Continuous Q13 chatbot that routes every query to the correct module."""

from data_utils import show_dataset_report
from faq_module import FAQModule
from intent_module import detect_intent
from ollama_client import OllamaClient
from product_module import ProductModule
from recommendation_module import RecommendationModule
from review_module import ReviewModelUnavailable, ReviewModule


def print_ollama_status(client: OllamaClient) -> None:
    print(f"\nLLM status: {client.status_message()}")
    if not client.is_available():
        print("The retrieval logic can still be inspected, but generated Ollama text is disabled.")


def run() -> None:
    print("ECommerceGPT — Experiment 1")
    print("AI-Powered E-Commerce Chatbot using Ollama")
    show_dataset_report()

    ollama = OllamaClient()
    faq = FAQModule(ollama=ollama)
    recommender = RecommendationModule(ollama=ollama)
    product = ProductModule(ollama=ollama)
    reviews = ReviewModule(ollama=ollama)
    print_ollama_status(ollama)
    print("\nType a question, or type 'exit' to stop.")

    while True:
        query = input("\nYou: ").strip()
        if query.lower() == "exit":
            print("Assistant: Goodbye.")
            break
        if not query:
            print("Assistant: Please enter a question.")
            continue

        intent_result = detect_intent(query, ollama)
        faq_result = None
        if intent_result["intent"] == "FAQ":
            faq_result = faq.answer(query)
            if not faq_result.get("matched", False):
                print(
                    "FAQ similarity is below the threshold; "
                    "rerouting this query to the next appropriate module."
                )
                intent_result = detect_intent(
                    query,
                    ollama,
                    excluded_intents={"FAQ"},
                )

        print(f"\nDetected intent: {intent_result['intent']} ({intent_result['source']})")
        if intent_result["warning"]:
            print(f"Note: {intent_result['warning']}")

        if intent_result["intent"] == "FAQ":
            result = faq_result or faq.answer(query)
            if result.get("match"):
                print(f"Best Matching FAQ: {result['match']['question']}")
                print(f"Similarity Score: {result['match']['score']:.3f}")
            print("Assistant:", result.get("response") or result.get("message"))
            if result.get("warning"):
                print("Ollama note:", result["warning"])
        elif intent_result["intent"] == "RECOMMENDATION":
            result = recommender.recommend(query)
            print("\nTop-3 retrieved products:")
            for item in result["products"]:
                print(f"- {item['ProductName']} | ₹{item['Price']} | score {item['score']:.3f}")
            print("Assistant:", result.get("response") or "Ollama is unavailable; inspect the retrieved products above.")
            if result.get("warning"):
                print("Ollama note:", result["warning"])
        elif intent_result["intent"] == "PRODUCT":
            result = product.answer(query)
            if result["product"]:
                print(f"Retrieved Product: {result['product']['ProductName']}")
                print(f"Similarity: {result.get('similarity', 0):.3f}")
                print("Structured context:\n" + result["context"])
            print("Assistant:", result.get("response") or "Ollama is unavailable; the structured context above is the grounded evidence.")
            if result.get("warning"):
                print("Ollama note:", result["warning"])
        else:
            try:
                result = reviews.analyze_product(query)
                if not result["found"]:
                    print("Assistant:", result["message"])
                else:
                    print("Sentiment counts:", result["counts"])
                    print(f"Average rating: {result['average_rating']:.2f}/5")
                    print("GPT summary:", result["ollama_summary"] or "Ollama is unavailable; sentiment evidence is shown above.")
                    if result.get("ollama_warning"):
                        print("Ollama note:", result["ollama_warning"])
            except ReviewModelUnavailable as error:
                print("Review module limitation:", error)


if __name__ == "__main__":
    run()