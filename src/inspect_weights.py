import joblib

from . import config


def main(top_k=20):
    if not config.MODEL_PATH.exists():
        print("model not found")
        return

    bundle = joblib.load(config.MODEL_PATH)
    w = bundle.classifier.w
    vocab = bundle.vectorizer.get_feature_names_out()
    order = w.argsort()

    print(f"\nTop {top_k} positive")
    for i in order[::-1][:top_k]:
        print(f"  {w[i]:+.3f}  {vocab[i]}")

    print(f"\nTop {top_k} negative")
    for i in order[:top_k]:
        print(f"  {w[i]:+.3f}  {vocab[i]}")

    print(f"\nbias {bundle.classifier.b:+.3f}   vocab {len(vocab):,}")


if __name__ == "__main__":
    main()
