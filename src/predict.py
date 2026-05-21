"""Command-line inference: predict sentiment for one or more reviews.

Usage:
    python -m src.predict "This product is amazing!" "Terrible quality, broke in a week."
"""
from __future__ import annotations

import sys
import joblib

from . import config


def main(texts: list[str]):
    if not config.MODEL_PATH.exists():
        print(f"Model not found at {config.MODEL_PATH}")
        print("Run `python -m src.download_data` then `python -m src.train` first.")
        sys.exit(1)

    bundle = joblib.load(config.MODEL_PATH)
    probs = bundle.predict_proba(texts)
    for text, p in zip(texts, probs):
        label = "Positive" if p >= 0.5 else "Negative"
        print(f"[{label} {p:.3f}]  {text}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = [
            "This product is amazing — exceeded my expectations!",
            "Terrible quality, broke after one week of use.",
        ]
    main(args)
