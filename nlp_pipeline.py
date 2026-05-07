import re
import string

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


try:
    import nltk
    from nltk.stem import WordNetLemmatizer

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    _LEMMATIZER = WordNetLemmatizer()
except Exception:
    _LEMMATIZER = None


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
TOKEN_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z']*\b")


def simple_lemma(token):
    """Small fallback if WordNet data is unavailable."""
    for suffix in ("ing", "edly", "edly", "ed", "ly", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]
    return token


def lemmatize_token(token):
    if _LEMMATIZER is None:
        return simple_lemma(token)
    return _LEMMATIZER.lemmatize(token)


def clean_text(text):
    text = str(text).lower()
    text = URL_PATTERN.sub(" ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = TOKEN_PATTERN.findall(cleaned)
    tokens = [
        lemmatize_token(token)
        for token in tokens
        if token not in ENGLISH_STOP_WORDS and len(token) > 2
    ]
    return " ".join(tokens)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([preprocess_text(text) for text in X], dtype=object)


class TextStats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            value = str(text)
            cleaned = clean_text(value)
            words = TOKEN_PATTERN.findall(cleaned)
            rows.append(
                [
                    len(value),
                    len(words),
                    value.count("!"),
                    value.count("?"),
                    sum(word in cleaned for word in ["sad", "empty", "hopeless", "tired"]),
                    sum(word in cleaned for word in ["panic", "worry", "nervous", "scared"]),
                    sum(word in cleaned for word in ["deadline", "pressure", "overwhelmed", "workload"]),
                ]
            )
        return np.array(rows, dtype=float)
