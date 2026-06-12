from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

DEFAULT_DATA_PATH = DATA_DIR / "train.csv"
DEMO_DATA_PATH = DATA_DIR / "demo_train.csv"

TARGET_COLUMN = "sales"
DATE_COLUMN = "date"
GROUP_COLUMNS = ["store", "item"]
