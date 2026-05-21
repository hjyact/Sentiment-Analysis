"""Central configuration: paths, hyperparameters, dataset choice."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# Dataset
DATASET_NAME = "fancyzhx/amazon_polarity"   # HuggingFace dataset id (canonical alias of amazon_polarity)
SAMPLE_SIZE_PER_CLASS = 25_000        # 25K pos + 25K neg = 50K total (balanced)
RANDOM_SEED = 42

# Train / dev / test split
TEST_SIZE = 0.10
DEV_SIZE = 0.10

# TF-IDF features
MAX_FEATURES = 30_000
NGRAM_RANGE = (1, 2)    # unigram + bigram (trigrams hurt more than help here)
MIN_DF = 2
MAX_DF = 0.95

# Logistic regression (hand-built, Andrew Ng style) with momentum
# Notes on tuning:
#  - TF-IDF values are in [0, 1] so the gradient signal is small;
#    a larger learning rate is needed than for raw-count features.
#  - Momentum (Andrew Ng DL course 2) smooths updates → faster convergence.
#    At steady state EMA velocity equals the gradient, so keep LR the same
#    as without momentum (5.0) — momentum just removes zig-zag, not magnitude.
LEARNING_RATE = 5.0
N_ITERS = 2000
LAMBDA_REG = 0.01
MOMENTUM = 0.9
PRINT_EVERY = 100

# Artifacts
MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
TRAIN_CSV = DATA_PROCESSED / "train.csv"
DEV_CSV = DATA_PROCESSED / "dev.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"
