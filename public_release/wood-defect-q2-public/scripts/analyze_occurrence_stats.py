#!/usr/bin/env python3
"""Analyze source-level occurrence and co-occurrence patterns from a manifest JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.occurrence_stats import (
    build_cooccurrence_statistics,
    build_occurrence_statistics,
    load_manifest_jsonl,
)
from src.utils.config import expand_path
from src.utils.io import ensure_dir, save_csv, save_json
from src.utils.logger import setup_logger

logger = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="data/processed/large_scale_wood_surface_defects_manifest.jsonl",
        help="Manifest JSONL to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/tables",
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help="Optional output stem. Defaults to manifest stem + '_occurrence_stats'.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional class filter. Only these classes are retained before aggregation.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Optional split filter for processed manifests: all/train/val/test.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top pair/triple rows to log to stdout. CSV files always contain full tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = expand_path(args.manifest_path)
    output_dir = expand_path(args.output_dir)
    if manifest_path is None or output_dir is None:
        raise ValueError("manifest-path and output-dir must resolve to non-empty paths.")

    records = load_manifest_jsonl(manifest_path)
    per_class_df, occurrence_summary = build_occurrence_statistics(
        records=records,
        class_names=args.classes,
        split=args.split,
    )
    pair_df, triple_df, cooccurrence_summary = build_cooccurrence_statistics(
        records=records,
        class_names=args.classes,
        split=args.split,
        top_k=None,
    )

    output_dir = ensure_dir(output_dir)
    output_stem = args.output_stem or f"{manifest_path.stem}_occurrence_stats"
    occurrence_csv_path = output_dir / f"{output_stem}.csv"
    pair_csv_path = output_dir / f"{output_stem}_pair_cooccurrence.csv"
    triple_csv_path = output_dir / f"{output_stem}_triple_cooccurrence.csv"
    summary_path = output_dir / f"{output_stem}_summary.json"

    save_csv(per_class_df, occurrence_csv_path)
    save_csv(pair_df, pair_csv_path)
    save_csv(triple_df, triple_csv_path)
    save_json(
        {
            "occurrence": occurrence_summary,
            "cooccurrence": cooccurrence_summary,
        },
        summary_path,
    )

    logger.info("Occurrence stats CSV: %s", occurrence_csv_path)
    logger.info("Pair co-occurrence CSV: %s", pair_csv_path)
    logger.info("Triple co-occurrence CSV: %s", triple_csv_path)
    logger.info("Occurrence/co-occurrence summary: %s", summary_path)
    logger.info("Occurrence summary: %s", occurrence_summary)

    top_k = max(int(args.top_k), 0)
    if top_k > 0:
        logger.info("Top %d pair co-occurrences:\n%s", top_k, pair_df.head(top_k).to_string(index=False))
        logger.info("Top %d triple co-occurrences:\n%s", top_k, triple_df.head(top_k).to_string(index=False))
    logger.info("Per-class occurrence stats:\n%s", per_class_df.to_string(index=False))


if __name__ == "__main__":
    main()
