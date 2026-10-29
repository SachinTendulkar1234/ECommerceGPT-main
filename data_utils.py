"""Small helpers for loading the three supplied CSV files."""

from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"


def load_products() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "products.csv")


def load_faqs() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "faqs.csv")


def load_reviews() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "reviews.csv")


def show_dataset_report() -> None:
    """Print Q1's first five records, shape, columns, and data types."""
    datasets = {
        "Products": load_products(),
        "FAQs": load_faqs(),
        "Reviews": load_reviews(),
    }
    for name, frame in datasets.items():
        print(f"\n{name}")
        print("-" * len(name))
        print(f"Rows: {len(frame)}")
        print(f"Columns: {list(frame.columns)}")
        print("First five records:")
        print(frame.head().to_string(index=False))
        print("Attributes / data types:")
        print(frame.dtypes.to_string())