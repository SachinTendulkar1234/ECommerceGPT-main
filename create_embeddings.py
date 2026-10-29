"""Create and store lightweight FAQ embeddings for cosine similarity search."""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from data_utils import PROJECT_DIR, load_faqs

EMBEDDING_FILE = PROJECT_DIR / "models" / "faq_embeddings.json"


def create_faq_embeddings(output_file: Path = EMBEDDING_FILE) -> Path:
    faqs = load_faqs()
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    matrix = vectorizer.fit_transform(faqs["Question"].fillna("").tolist())
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "questions": faqs["Question"].tolist(),
        "answers": faqs["Answer"].tolist(),
        "vectors": matrix.toarray().tolist(),
        "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist(),
        "vectorizer": {
            "lowercase": True,
            "stop_words": "english",
            "token_pattern": r"(?u)\b\w\w+\b",
            "norm": "l2",
            "use_idf": True,
            "smooth_idf": True,
            "sublinear_tf": False,
        },
    }
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Stored {len(payload['questions'])} FAQ embeddings in {output_file}")
    return output_file


if __name__ == "__main__":
    create_faq_embeddings()