from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "NFLX", "JPM",
    "CRM", "ADBE", "INTC", "QCOM", "PYPL",
]

TARGET_COL = "target_drawdown_5d_7pct"

TRAIN_END = "2022-08-31"
VALID_END = "2023-08-31"
EMBARGO_DAYS = 7

RANDOM_STATE = 42
