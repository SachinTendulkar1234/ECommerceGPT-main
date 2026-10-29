# ECommerceGPT

## Experiment 1 — Building an AI-Powered E-Commerce Chatbot using Ollama

This is a modular Retrieval-Augmented Generation (RAG) laboratory project for
24CS2506 – Building Applications using GPT-4. It uses the three supplied CSV
files to answer store FAQs, recommend products, answer product questions,
analyze reviews, and route queries through an Ollama large language model.

The project never replaces the supplied datasets with sample data. The files
copied into `data/` are the original uploaded datasets:

- `data/products.csv`
- `data/faqs.csv`
- `data/reviews.csv`

## Problem statement

An online store receives many questions about orders, products, and customer
opinions. A single chatbot should retrieve the relevant evidence, send only
that evidence to Ollama, and return an understandable response.

## Objective

Build a simple, modular chatbot that demonstrates:

1. Dataset preparation
2. Stored FAQ embeddings and cosine similarity
3. Similarity threshold filtering
4. TF-IDF product retrieval and Top-3 recommendations
5. Product-specific RAG question answering
6. Pre-trained sentiment analysis for customer reviews
7. Ollama-based intent detection and response generation
8. One continuous chatbot loop until `exit`

## Dataset report

The CSV column names were inspected directly; no assumed column names were
used.

| Dataset | Rows | Columns | Meaning |
| --- | ---: | --- | --- |
| Products | 100 | `ProductID`, `ProductName`, `Category`, `Brand`, `Price`, `Description`, `Rating` | The product catalog used for product retrieval, recommendations, and grounded product answers. |
| FAQs | 20 | `FAQID`, `Question`, `Answer` | Store questions and their official answers used by the FAQ semantic-search module. |
| Reviews | 1,000 | `ReviewID`, `ProductID`, `ProductName`, `Review`, `Rating` | Customer review text and rating used for sentiment and review intelligence. |

The products contain 25 Laptops, 25 Mobiles, 25 Headphones, and 25
Smartwatches. There are 10 reviews for each catalog product.

### Dataset-to-module mapping

- **Products:** Q1, Q2, Q6, Q7, Q8, Q9, and Q13.
- **FAQs:** Q1, Q2, Q3, Q4, Q5, and Q13.
- **Reviews:** Q1, Q2, Q10, Q11, and Q13.

## Workflow

```text
User query
    ↓
Ollama intent detection
    ↓
FAQ | Recommendation | Product RAG | Review intelligence
    ↓
Embeddings / cosine similarity / TF-IDF / sentiment model
    ↓
Retrieved evidence
    ↓
Ollama natural-language response
```

## Project structure

```text
ECommerceGPT/
├── chatbot.py                 # Q13 continuous chatbot and router
├── create_embeddings.py       # Q3 stored FAQ embedding generator
├── data_utils.py              # Actual CSV loading and Q1 report
├── faq_module.py              # Q3–Q5 FAQ search and threshold
├── recommendation_module.py   # Q6–Q7 TF-IDF and Top-3 recommendation
├── product_module.py          # Q8–Q9 product RAG and constrained prompts
├── review_module.py           # Q10–Q11 sentiment and review intelligence
├── intent_module.py           # Q12 Ollama intent detector
├── ollama_client.py           # Ollama-only LLM client
├── requirements.txt
├── data/
│   ├── products.csv
│   ├── faqs.csv
│   └── reviews.csv
├── models/                    # faq_embeddings.json is created here
└── screenshots/
    └── README.md
```

## Installation for a beginner

### 1. Open a terminal in the `ECommerceGPT` folder

Windows users can right-click the folder and choose **Open in Terminal**.

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python requirements

```bash
pip install -r requirements.txt
```

The first review-model run may download the pre-trained
`distilbert-base-uncased-finetuned-sst-2-english` model.

## Ollama setup

Ollama is the required LLM provider for this experiment. It is not silently
replaced by OpenAI, Gemini, Claude, or a made-up response.

1. Install Ollama from [ollama.com](https://ollama.com).
2. Open a new terminal.
3. Download a local model:

```powershell
ollama pull llama3.2:3b
```

4. Confirm Ollama is running:

```powershell
ollama list
```

5. Run the project with the same terminal environment.

To use another Ollama model, set `OLLAMA_MODEL`:

```powershell
$env:OLLAMA_MODEL = "llama3.2:3b"
```

Optional advanced setting:

```powershell
$env:OLLAMA_BASE_URL = "http://localhost:11434"
```

If Ollama is not available, the Python modules do not pretend that an
LLM-generated response succeeded. They print the retrieved evidence and a
clear limitation message. Intent detection has an explicitly labeled local
preview fallback so the rest of the routing can still be demonstrated.

## How to run

From inside `ECommerceGPT/`:

```bash
python create_embeddings.py
python chatbot.py
```

The first command writes `models/faq_embeddings.json`. The second command
prints the Q1 dataset report and opens a continuous prompt:

```text
You: Where is my parcel?
You: Recommend a gaming laptop
You: Tell me about Lenovo Laptop 3
You: What is its price?
You: Show me reviews for Lenovo Laptop 3
You: exit
```

The product module remembers the last retrieved product, so follow-up
questions such as “What is its price?” and “Which brand is it?” are grounded in
the previous product context.

## Module explanation

### `data_utils.py`

Loads the actual CSV paths relative to the project directory. The
`show_dataset_report()` function displays first five rows, row counts, column
names, and pandas data types for Q1.

### `create_embeddings.py`

Uses `TfidfVectorizer` on every FAQ question. Each resulting vector is a
numeric FAQ embedding and is stored in `models/faq_embeddings.json` with the
questions, answers, vocabulary, IDF values, vectors, and all vectorizer
settings needed to reproduce the query vector.

### `faq_module.py`

Loads the stored FAQ embeddings and vectorizer metadata instead of fitting a
new FAQ matrix on every startup. It transforms only the user's question into
the stored vector space, calculates cosine similarity, retrieves the best FAQ,
and checks the `0.20` threshold. Only a match that passes the threshold is sent
to Ollama for a natural response.

### `recommendation_module.py`

Builds a TF-IDF document for each product from its name, category, brand, and
description. It vectorizes the user request, calculates cosine similarity,
returns the Top-3 products, and sends those three records to Ollama for a
recommendation with justification.

### `product_module.py`

Retrieves the most relevant product and creates a structured context containing
only product name, category, brand, price, catalog rating, and description.
Its prompt explicitly tells Ollama not to invent information. The previous
product is retained for follow-up questions.

### `review_module.py`

Loads the pre-trained
`distilbert-base-uncased-finetuned-sst-2-english` sentiment model through the
Hugging Face `transformers` pipeline. It classifies each selected review as
Positive, Neutral, or Negative and shows a confidence score. Rating 3 is
reported as Neutral for the lab's three-class display. The actual review
evidence is then sent to Ollama for an overall opinion, strengths, weaknesses,
and buying suggestion.

### `intent_module.py`

Asks Ollama to return exactly one of `FAQ`, `RECOMMENDATION`, `PRODUCT`, or
`REVIEW`. It validates the returned label and can exclude FAQ when a matched
FAQ does not pass the threshold. If Ollama is unavailable, it uses a small
explicitly labeled local preview classifier and reports that Ollama was not
used.

### `chatbot.py`

Loads all modules once, repeatedly reads user input, detects intent, routes the
query, prints retrieval evidence and the response, reroutes below-threshold FAQ
queries to the next detected module, and exits cleanly when the user types
`exit`.

## Q1–Q13 traceability

| Question | Requirement | Implementation |
| --- | --- | --- |
| Q1 | Load datasets and display first five records, rows, columns, attributes | `data_utils.py` → `show_dataset_report()` |
| Q2 | Explain dataset purpose and module mapping | This README, Dataset report and Dataset-to-module mapping |
| Q3 | Generate and store FAQ embeddings | `create_embeddings.py` → `models/faq_embeddings.json`; `faq_module.py` loads the stored vectors and metadata |
| Q4 | Retrieve best FAQ with cosine similarity | `faq_module.py` → `FAQModule.search()` |
| Q5 | Apply threshold, use Ollama for matched FAQ, and reroute below-threshold queries | `faq_module.py` → `FAQModule.answer()`; `chatbot.py` reroutes with an excluded FAQ intent |
| Q6 | Create TF-IDF product vectors | `recommendation_module.py` → `RecommendationModule` |
| Q7 | Retrieve Top-3 and recommend with Ollama justification | `recommendation_module.py` → `recommend()` |
| Q8 | Retrieve relevant product and structured context | `product_module.py` → `structured_context()` |
| Q9 | Constrained product RAG answer | `product_module.py` → `answer()` prompt |
| Q10 | Load pre-trained sentiment model | `review_module.py` → `ReviewModule._get_pipeline()`; no keyword sentiment is used by the Python project |
| Q11 | Analyze reviews and generate summary and buying suggestion | `review_module.py` → `analyze_product()` |
| Q12 | Detect FAQ, Recommendation, Product, Review intent with Ollama | `intent_module.py` → `detect_intent()`; fallback is explicitly labeled |
| Q13 | Integrate modules into continuous chatbot until Exit | `chatbot.py` → `run()`; intent → module routing → response → repeated input |

## Testing checklist

Run these tests with the actual CSV files. Record the result honestly; do not
claim an Ollama or transformer result when that dependency is unavailable.

- [ ] Dataset loading: confirm 100 products, 20 FAQs, and 1,000 reviews.
- [ ] FAQ search: ask `Where is my parcel?`; inspect the best FAQ and score.
- [ ] FAQ threshold: ask an unrelated question and confirm it can fail the threshold.
- [ ] Product recommendation: ask `Recommend a gaming laptop`; confirm three retrieved products.
- [ ] Product RAG: ask `Tell me about Lenovo Laptop 3`; inspect structured context.
- [ ] Follow-up product question: ask `What is its price?` and `Which brand is it?`.
- [ ] Review analysis: ask `Show me reviews for Lenovo Laptop 3`; inspect sentiment details.
- [ ] Intent detection: try one FAQ, recommendation, product, and review query.
- [ ] Continuous chatbot: submit multiple queries without restarting.
- [ ] Exit command: type `exit` and confirm a clean goodbye.
- [ ] Ollama check: with Ollama stopped, confirm the limitation message; with Ollama running, confirm the generated response.

### Windows Ollama verification

Use Python 3.10 or 3.11 for the smoothest `torch` installation on Windows.
Open the Ollama desktop application before running the chatbot, then verify
that the configured model is installed:

```powershell
ollama list
ollama pull llama3.2:3b
Invoke-WebRequest http://localhost:11434/api/tags
```

The Python client checks both the Ollama service and the configured model name.
If the service is running but the model is missing, it reports Ollama as
unavailable instead of claiming that generation succeeded.

### Replit browser demo limitation

The browser demo intentionally reports Ollama availability honestly. Its API
server checks its own local Ollama address, not a developer's Windows
localhost. The browser demo can still show dataset loading, FAQ similarity,
product retrieval, structured product context, and review evidence. Run
`chatbot.py` locally on Windows for the complete Ollama-backed experiment.

## Submission screenshots

Use the checklist in `screenshots/README.md`. Take screenshots of the real
terminal or browser output only:

1. Dataset report
2. FAQ search
3. Product recommendation
4. Product RAG plus follow-up
5. Review sentiment and summary
6. Intent detection
7. Final continuous chatbot and exit

Do not create fake screenshots.

## GitHub upload steps

1. Sign in to GitHub and select **New repository**.
2. Name the repository exactly `ECommerceGPT`.
3. Keep the repository public if your college requires a public link.
4. Upload the complete `ECommerceGPT` folder.
5. Confirm `chatbot.py`, all module files, `requirements.txt`, and `README.md` are present.
6. Open `data/` and confirm all three actual CSV files are present.
7. Confirm `models/` is present after running `create_embeddings.py`.
8. Upload only real submission screenshots into `screenshots/`.
9. Open the repository's **Code** menu and copy the HTTPS repository link.
10. Submit that link to your college.

Never upload passwords, API keys, or private credentials.

## Short viva answers

**What is RAG?** Retrieval-Augmented Generation first retrieves trusted
evidence, then gives that evidence to an LLM to generate an answer.

**What are embeddings?** Embeddings are numeric vectors that represent the
meaning or features of text.

**What is cosine similarity?** It measures the angle between two vectors. A
larger value means the texts point in a more similar direction.

**What is TF-IDF?** TF-IDF gives higher weight to words that are important in
one document but uncommon across the collection.

**Why use TF-IDF?** It is simple, fast, explainable, and works well for
retrieving products from a small catalog.

**What is sentiment analysis?** It classifies text by opinion, such as
Positive, Neutral, or Negative, with a confidence score.

**What is Ollama?** Ollama runs open-source language models locally.

**What is an LLM?** A Large Language Model generates and understands text
using patterns learned from large datasets.

**What is intent detection?** It identifies what the user wants, such as an FAQ
answer, recommendation, product answer, or review summary.

**What is a similarity threshold?** It is the minimum score required before a
retrieved result is accepted as relevant.

**Why separate modules?** Separate modules keep each experiment step simple,
testable, and easy to explain in a viva.

**What does `chatbot.py` do?** It keeps the conversation running, detects
intent, routes the query, prints the result, and handles `exit`.

**What does `create_embeddings.py` do?** It vectorizes FAQ questions and stores
the vectors for later semantic search.

**How does the chatbot route queries?** Ollama returns one validated intent
label, and `chatbot.py` calls the matching module.

**How does product retrieval work?** Product text becomes TF-IDF vectors; the
query vector is compared with each product using cosine similarity, and the
highest-scoring products are returned.

## Conclusion

This project demonstrates an explainable e-commerce chatbot pipeline using
actual catalog, FAQ, and review data. Retrieval results are kept visible so a
student can explain what evidence was used before Ollama generates a response.