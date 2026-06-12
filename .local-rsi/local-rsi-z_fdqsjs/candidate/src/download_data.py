import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets")

    print(f"loading {config.DATASET_NAME}")
    ds = load_dataset(config.DATASET_NAME)
    df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()], ignore_index=True)

    df["text"] = (df["title"].fillna("") + ". " + df["content"].fillna("")).str.strip()
    df = df[["text", "label"]]

    n = config.SAMPLE_SIZE_PER_CLASS
    pos = df[df.label == 1].sample(n=n, random_state=config.RANDOM_SEED)
    neg = df[df.label == 0].sample(n=n, random_state=config.RANDOM_SEED)
    sample = (
        pd.concat([pos, neg])
        .sample(frac=1.0, random_state=config.RANDOM_SEED)
        .reset_index(drop=True)
    )

    train_df, test_df = train_test_split(
        sample,
        test_size=config.TEST_SIZE,
        stratify=sample.label,
        random_state=config.RANDOM_SEED,
    )
    train_df, dev_df = train_test_split(
        train_df,
        test_size=config.DEV_SIZE / (1 - config.TEST_SIZE),
        stratify=train_df.label,
        random_state=config.RANDOM_SEED,
    )

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.TRAIN_CSV, index=False)
    dev_df.to_csv(config.DEV_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)
    print(f"train={len(train_df):,}  dev={len(dev_df):,}  test={len(test_df):,}")


if __name__ == "__main__":
    main()
