import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    """
    Basic text cleaning:
    - lowercase
    - remove punctuation & numbers
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


def load_and_preprocess_tickets(path):
    df = pd.read_csv(path)

    # Drop rows with missing values
    df = df.dropna(subset=["ticket_text", "category"])

    # Clean text
    df["ticket_text"] = df["ticket_text"].apply(clean_text)

    X_text = df["ticket_text"]
    y = df["category"].values

    return X_text, y
