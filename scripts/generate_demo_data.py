from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import DATA_DIR, DEMO_DATA_PATH
from src.data import generate_demo_data


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_demo_data()
    df.to_csv(DEMO_DATA_PATH, index=False)
    print(f"Demo dataset saved to {Path(DEMO_DATA_PATH).resolve()}")
    print(f"Rows: {len(df):,}")


if __name__ == "__main__":
    main()
