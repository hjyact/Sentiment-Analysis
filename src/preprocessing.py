import html
import re

URL = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG = re.compile(r"<[^>]+>")
NON_ALPHA = re.compile(r"[^a-z\s]")
WHITESPACE = re.compile(r"\s+")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text).lower()
    text = URL.sub(" ", text)
    text = HTML_TAG.sub(" ", text)
    text = NON_ALPHA.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def clean_series(series):
    return series.fillna("").map(clean_text)
