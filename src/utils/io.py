"""Input/output helpers."""

from __future__ import annotations

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


def save_jsonl(records, path: str | Path) -> None:
    """Save iterable records as JSONL."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save DataFrame to CSV."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def compute_dir_size_bytes(path: str | Path) -> int:
    """Compute the total size of files under a directory."""
    root = Path(path)
    if not root.exists():
        return 0
    return sum(file_path.stat().st_size for file_path in root.rglob("*") if file_path.is_file())


def format_num_bytes(num_bytes: int) -> str:
    """Format a byte count into a compact human-readable string."""
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"
