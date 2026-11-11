"""Inspect the current daily page; never activate the game or move the mouse."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.daily import observe_daily


if __name__ == "__main__":
    report, path = observe_daily(ROOT)
    print(json.dumps({key: value for key, value in report.items() if key != "ocr"},
                     ensure_ascii=False, indent=2))
    print(f"Evidence: {path}")
    sys.exit(1 if report.get("error") else 0)
