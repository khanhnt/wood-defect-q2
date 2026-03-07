"""Input/output helpers."""

from pathlib import Path
import json
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data, path: str | Path) -> None:
    """Save JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save DataFrame to CSV."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
