"""Download and sample the Amazon Polarity dataset.

Source: HuggingFace `amazon_polarity` (3.6M Amazon reviews labeled
positive/negative). We sample SAMPLE_SIZE_PER_CLASS reviews from each
class for a balanced, fast-to-train subset.

Usage:
    python -m src.download_data
"""
from __future__ import annotations

import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Missing dependency: pip install datasets")
        sys.exit(1)

    print(f"Loading {config.DATASET_NAME} from HuggingFace ...")
    ds = load_dataset(config.DATASET_NAME)
    df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()], ignore_index=True)
    print(f"  total rows: {len(df):,}")

    # `amazon_polarity` columns: label (0=neg, 1=pos), title, content
    df["text"] = (df["title"].fillna("") + ". " + df["content"].fillna("")).str.strip()
    df = df[["text", "label"]]

    # Balanced sample
    per_class = config.SAMPLE_SIZE_PER_CLASS
    pos = df[df.label == 1].sample(n=per_class, random_state=config.RANDOM_SEED)
    neg = df[df.label == 0].sample(n=per_class, random_state=config.RANDOM_SEED)
    sample = pd.concat([pos, neg], ignore_index=True).sample(
        frac=1.0, random_state=config.RANDOM_SEED
    ).reset_index(drop=True)
    print(f"  sampled: {len(sample):,} (pos={per_class:,}, neg={per_class:,})")

    # train / dev / test
    train_df, test_df = train_test_split(
        sample, test_size=config.TEST_SIZE,
        stratify=sample.label, random_state=config.RANDOM_SEED,
    )
    train_df, dev_df = train_test_split(
        train_df,
        test_size=config.DEV_SIZE / (1 - config.TEST_SIZE),
        stratify=train_df.label, random_state=config.RANDOM_SEED,
    )

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.TRAIN_CSV, index=False)
    dev_df.to_csv(config.DEV_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)
    print(f"  wrote {config.TRAIN_CSV.name} ({len(train_df):,})")
    print(f"  wrote {config.DEV_CSV.name}   ({len(dev_df):,})")
    print(f"  wrote {config.TEST_CSV.name}  ({len(test_df):,})")


if __name__ == "__main__":
    main()
