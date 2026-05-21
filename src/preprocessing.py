"""Text cleaning utilities for Amazon review sentiment analysis."""
from __future__ import annotations

import re
import html

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG = re.compile(r"<[^>]+>")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs, keep only letters and spaces.

    Kept intentionally simple — TF-IDF + n-grams handles most of the
    nuance. Heavy preprocessing (lemmatization, stopword removal) often
    hurts as much as it helps on review data.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = text.lower()
    text = _URL.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def clean_series(series):
    return series.fillna("").map(clean_text)
