#!/usr/bin/env python3
"""Compare source-level occurrence/co-occurrence statistics across benchmark manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.occurrence_stats import compare_manifest_occurrence_statistics
from src.utils.config import expand_path
from src.utils.io import ensure_dir, save_csv, save_json
from src.utils.logger import setup_logger

logger = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-dirs",
        nargs="+",
        required=True,
        help="Benchmark root directories containing manifest.jsonl and optional metadata.json.",
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
        default="benchmark_occurrence_compare",
        help="Stem for generated output files.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional class filter. Only these classes are retained before aggregation.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["all"],
        help="Splits to analyze. Example: all train val test",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Keep top-k pair/triple rows per benchmark and split in the comparison CSVs.",
    )
    return parser.parse_args()


def _load_benchmark_spec(benchmark_dir: Path) -> dict[str, object]:
    manifest_path = benchmark_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")

    metadata_path = benchmark_dir / "metadata.json"
    metadata: dict[str, object] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    selection = metadata.get("selection", {}) if isinstance(metadata, dict) else {}
    if not isinstance(selection, dict):
        selection = {}

    benchmark_name = str(metadata.get("dataset_name") or benchmark_dir.name)
    selection_mode = str(selection.get("selection_mode") or benchmark_dir.name)
    return {
        "benchmark_name": benchmark_name,
        "selection_mode": selection_mode,
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    args = parse_args()
    output_dir = expand_path(args.output_dir)
    if output_dir is None:
        raise ValueError("output-dir must resolve to a non-empty path.")

    benchmark_specs: list[dict[str, object]] = []
    for benchmark_dir_arg in args.benchmark_dirs:
        benchmark_dir = expand_path(benchmark_dir_arg)
        if benchmark_dir is None:
            raise ValueError(f"Failed to resolve benchmark dir: {benchmark_dir_arg}")
        benchmark_specs.append(_load_benchmark_spec(Path(benchmark_dir)))

    overview_df, per_class_df, pair_df, triple_df, summary = compare_manifest_occurrence_statistics(
        manifest_specs=benchmark_specs,
        class_names=args.classes,
        splits=args.splits,
        top_k=args.top_k,
    )

    output_dir = ensure_dir(output_dir)
    output_stem = str(args.output_stem).strip() or "benchmark_occurrence_compare"
    overview_path = output_dir / f"{output_stem}_overview.csv"
    per_class_path = output_dir / f"{output_stem}_per_class.csv"
    pair_path = output_dir / f"{output_stem}_pair_cooccurrence.csv"
    triple_path = output_dir / f"{output_stem}_triple_cooccurrence.csv"
    summary_path = output_dir / f"{output_stem}_summary.json"

    save_csv(overview_df, overview_path)
    save_csv(per_class_df, per_class_path)
    save_csv(pair_df, pair_path)
    save_csv(triple_df, triple_path)
    save_json(summary, summary_path)

    logger.info("Overview CSV: %s", overview_path)
    logger.info("Per-class CSV: %s", per_class_path)
    logger.info("Pair co-occurrence CSV: %s", pair_path)
    logger.info("Triple co-occurrence CSV: %s", triple_path)
    logger.info("Comparison summary JSON: %s", summary_path)
    if not overview_df.empty:
        logger.info("Overview:\n%s", overview_df.to_string(index=False))
    if not pair_df.empty:
        logger.info("Top pair rows:\n%s", pair_df.head(max(int(args.top_k), 1)).to_string(index=False))
    if not triple_df.empty:
        logger.info("Top triple rows:\n%s", triple_df.head(max(int(args.top_k), 1)).to_string(index=False))


if __name__ == "__main__":
    main()
