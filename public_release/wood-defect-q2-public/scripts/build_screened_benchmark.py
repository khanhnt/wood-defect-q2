#!/usr/bin/env python3
"""Build a reproducible screened 7-class/3600-image benchmark from processed tiles."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.screened_benchmark import (
    DEFAULT_SELECTION_MODE,
    DEFAULT_VSB7_CLASSES,
    SUPPORTED_SELECTION_MODES,
    build_screened_benchmark_from_processed_manifest,
)
from src.utils.config import expand_path
from src.utils.logger import setup_logger

logger = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        type=str,
        default="${WOOD_MAIN_PROCESSED_ROOT}/manifest.jsonl",
        help="Processed tile manifest to screen.",
    )
    parser.add_argument(
        "--output-root-dir",
        type=str,
        default="${WOOD_MAIN_PROCESSED_ROOT}/benchmarks/vsb7_3600",
        help="Output directory for the screened benchmark manifest and metadata.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="large_scale_wood_surface_defects_vsb7_3600",
        help="Dataset name stored in the screened manifest.",
    )
    parser.add_argument(
        "--target-source-images",
        type=int,
        default=3600,
        help="Number of source images to retain after screening.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic subset selection.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_VSB7_CLASSES),
        help="Foreground classes to keep in the screened benchmark.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=list(SUPPORTED_SELECTION_MODES),
        default=DEFAULT_SELECTION_MODE,
        help="Selection strategy for choosing source images inside each split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_manifest = expand_path(args.input_manifest)
    output_root_dir = expand_path(args.output_root_dir)
    if input_manifest is None or output_root_dir is None:
        raise ValueError("input-manifest and output-root-dir must resolve to non-empty paths.")

    result = build_screened_benchmark_from_processed_manifest(
        input_manifest_path=input_manifest,
        output_root_dir=output_root_dir,
        dataset_name=args.dataset_name,
        kept_classes=args.classes,
        target_source_images=args.target_source_images,
        seed=args.seed,
        selection_mode=args.selection_mode,
    )

    logger.info("Screened benchmark manifest: %s", result["manifest_path"])
    logger.info("Selected source ids: %s", result["selected_source_ids_path"])
    logger.info("Metadata summary: %s", result["metadata_path"])
    logger.info("Selection summary: %s", result["summary"]["selection"])


if __name__ == "__main__":
    main()
