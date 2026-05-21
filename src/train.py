"""Train the sentiment classifier and save the bundled model.

Usage:
    python -m src.train
"""
from __future__ import annotations

import json
import time
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from . import config
from .preprocessing import clean_series
from .features import build_vectorizer
from .model import LogisticRegressionScratch, SentimentModel


def _load_split(path):
    df = pd.read_csv(path)
    df["text"] = clean_series(df["text"])
    return df


def main():
    print("Loading processed splits ...")
    train_df = _load_split(config.TRAIN_CSV)
    dev_df = _load_split(config.DEV_CSV)
    test_df = _load_split(config.TEST_CSV)
    print(f"  train={len(train_df):,}  dev={len(dev_df):,}  test={len(test_df):,}")

    print("Fitting TF-IDF vectorizer ...")
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_df["text"])
    X_dev = vectorizer.transform(dev_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    print(f"  feature matrix: {X_train.shape}  nnz/row≈{X_train.nnz / X_train.shape[0]:.1f}")

    y_train = train_df["label"].to_numpy()
    y_dev = dev_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    print("\nTraining hand-built logistic regression (Andrew Ng style) ...")
    clf = LogisticRegressionScratch(
        learning_rate=config.LEARNING_RATE,
        n_iters=config.N_ITERS,
        lambda_reg=config.LAMBDA_REG,
        momentum=config.MOMENTUM,
        print_every=config.PRINT_EVERY,
    )
    t0 = time.time()
    clf.fit(X_train, y_train, X_dev=X_dev, y_dev=y_dev)
    elapsed = time.time() - t0
    print(f"  training time: {elapsed:.1f}s")

    # Evaluate on all splits
    metrics = {}
    for name, X, y in [("train", X_train, y_train), ("dev", X_dev, y_dev), ("test", X_test, y_test)]:
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        p, r, f1, _ = precision_recall_fscore_support(y, preds, average="binary")
        metrics[name] = {"accuracy": float(acc), "precision": float(p),
                         "recall": float(r), "f1": float(f1)}
        print(f"  {name:5s}  acc={acc:.4f}  p={p:.4f}  r={r:.4f}  f1={f1:.4f}")

    metrics["history"] = clf.history
    metrics["config"] = {
        "learning_rate": config.LEARNING_RATE,
        "n_iters": config.N_ITERS,
        "lambda_reg": config.LAMBDA_REG,
        "momentum": config.MOMENTUM,
        "max_features": config.MAX_FEATURES,
        "ngram_range": list(config.NGRAM_RANGE),
        "sample_size_per_class": config.SAMPLE_SIZE_PER_CLASS,
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = SentimentModel(vectorizer=vectorizer, classifier=clf)
    joblib.dump(bundle, config.MODEL_PATH, compress=3)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model    -> {config.MODEL_PATH}")
    print(f"Saved metrics  -> {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
