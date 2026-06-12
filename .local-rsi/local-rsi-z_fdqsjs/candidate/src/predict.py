import sys

import joblib

from . import config


def main(texts):
    if not config.MODEL_PATH.exists():
        sys.exit("model not found; run download_data and train first")
    bundle = joblib.load(config.MODEL_PATH)
    for text, p in zip(texts, bundle.predict_proba(texts)):
        tag = "+" if p >= 0.5 else "-"
        print(f"[{tag} {p:.3f}] {text}")


if __name__ == "__main__":
    args = sys.argv[1:] or [
        "This product is amazing, exceeded my expectations.",
        "Terrible quality, broke after one week.",
    ]
    main(args)
