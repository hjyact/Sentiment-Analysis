"""Show the top positive / negative weighted words the model learned.

The hand-built logistic regression has one weight per TF-IDF feature
(word or bigram). A large positive weight means "this word pushes the
prediction toward Positive"; a large negative weight does the opposite.
Inspecting these is a quick sanity check — the model should associate
'amazing', 'love', 'perfect' with positive, and 'terrible', 'waste',
'broken' with negative.

Usage:
    python -m src.inspect_weights
"""
from __future__ import annotations

import joblib

from . import config


def main(top_k: int = 20):
    if not config.MODEL_PATH.exists():
        print(f"Model not found at {config.MODEL_PATH}. Run `python -m src.train` first.")
        return

    bundle = joblib.load(config.MODEL_PATH)
    w = bundle.classifier.w
    vocab = bundle.vectorizer.get_feature_names_out()
    assert len(w) == len(vocab), "weight / vocab size mismatch"

    order = w.argsort()
    most_negative = [(vocab[i], w[i]) for i in order[:top_k]]
    most_positive = [(vocab[i], w[i]) for i in order[::-1][:top_k]]

    print(f"\n=== Top {top_k} POSITIVE words (push toward 'positive review') ===")
    for word, weight in most_positive:
        print(f"  {weight:+.3f}   {word}")

    print(f"\n=== Top {top_k} NEGATIVE words (push toward 'negative review') ===")
    for word, weight in most_negative:
        print(f"  {weight:+.3f}   {word}")

    print(f"\nbias (b): {bundle.classifier.b:+.3f}")
    print(f"total vocab size: {len(vocab):,}")


if __name__ == "__main__":
    main()
