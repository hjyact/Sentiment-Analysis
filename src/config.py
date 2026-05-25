from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

DATASET_NAME = "fancyzhx/amazon_polarity"
SAMPLE_SIZE_PER_CLASS = 25_000
RANDOM_SEED = 42

TEST_SIZE = 0.10
DEV_SIZE = 0.10

MAX_FEATURES = 30_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95

LEARNING_RATE = 5.0
N_ITERS = 2000
LAMBDA_REG = 0.01
MOMENTUM = 0.9
PRINT_EVERY = 100

MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
TRAIN_CSV = DATA_PROCESSED / "train.csv"
DEV_CSV = DATA_PROCESSED / "dev.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"
