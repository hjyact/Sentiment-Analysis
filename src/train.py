import json
import time

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from . import config
from .features import build_vectorizer
from .model import LogisticRegressionScratch, SklearnLR, SentimentModel
from .preprocessing import clean_series


def load_split(path):
    df = pd.read_csv(path)
    df["text"] = clean_series(df["text"])
    return df


def main():
    train_df = load_split(config.TRAIN_CSV)
    dev_df = load_split(config.DEV_CSV)
    test_df = load_split(config.TEST_CSV)

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_df["text"])
    X_dev = vectorizer.transform(dev_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    y_train = train_df["label"].to_numpy()
    y_dev = dev_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()
    print(f"features: {X_train.shape}")

    if config.USE_SKLEARN:
        print("using sklearn LogisticRegression (L-BFGS)")
        clf = SklearnLR(C=config.SKLEARN_C, max_iter=config.SKLEARN_MAX_ITER)
    else:
        print("using hand-built LogisticRegression (GD + momentum)")
        clf = LogisticRegressionScratch(
            learning_rate=config.LEARNING_RATE,
            n_iters=config.N_ITERS,
            lambda_reg=config.LAMBDA_REG,
            momentum=config.MOMENTUM,
            print_every=config.PRINT_EVERY,
        )
    t0 = time.time()
    clf.fit(X_train, y_train, X_dev=X_dev, y_dev=y_dev)
    print(f"trained in {time.time() - t0:.1f}s")

    metrics = {}
    for name, X, y in [("train", X_train, y_train), ("dev", X_dev, y_dev), ("test", X_test, y_test)]:
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        p, r, f1, _ = precision_recall_fscore_support(y, preds, average="binary")
        metrics[name] = {
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        }
        print(f"{name:5s} acc={acc:.4f} p={p:.4f} r={r:.4f} f1={f1:.4f}")

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
    joblib.dump(SentimentModel(vectorizer, clf), config.MODEL_PATH, compress=3)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
