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
MAX_FEATURES = 20_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95

# Logistic regression (hand-built, Andrew Ng style)
# Notes on tuning:
#  - TF-IDF values are in [0, 1] so the gradient signal is small;
#    a larger learning rate is needed than for raw-count features.
#  - On 50K balanced samples with 20K sparse features, weak L2 works best.
LEARNING_RATE = 5.0
N_ITERS = 1500
LAMBDA_REG = 0.01     # L2 regularization strength
PRINT_EVERY = 100

# Artifacts
MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
TRAIN_CSV = DATA_PROCESSED / "train.csv"
DEV_CSV = DATA_PROCESSED / "dev.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"
