#!/usr/bin/env python3
"""Build a full-data class-filtered manifest from the processed tile manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.class_filtered_manifest import build_class_filtered_manifest
from src.datasets.screened_benchmark import DEFAULT_VSB7_CLASSES
from src.utils.config import expand_path
from src.utils.logger import setup_logger

logger = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        type=str,
        default="${WOOD_MAIN_PROCESSED_ROOT}/manifest.jsonl",
        help="Processed tile manifest to filter.",
    )
    parser.add_argument(
        "--output-root-dir",
        type=str,
        default="${WOOD_MAIN_PROCESSED_ROOT}/benchmarks/full_7class",
        help="Output directory for the class-filtered manifest and metadata.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="large_scale_wood_surface_defects_full_7class",
        help="Dataset name stored in the filtered manifest.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_VSB7_CLASSES),
        help="Foreground classes to keep in the filtered dataset.",
    )
    parser.add_argument(
        "--keep-source-images-without-kept-classes",
        action="store_true",
        help="Keep source images that become empty after removing dropped classes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_manifest = expand_path(args.input_manifest)
    output_root_dir = expand_path(args.output_root_dir)
    if input_manifest is None or output_root_dir is None:
        raise ValueError("input-manifest and output-root-dir must resolve to non-empty paths.")

    result = build_class_filtered_manifest(
        input_manifest_path=input_manifest,
        output_root_dir=output_root_dir,
        dataset_name=args.dataset_name,
        kept_classes=args.classes,
        drop_source_images_without_kept_classes=not bool(args.keep_source_images_without_kept_classes),
    )

    logger.info("Class-filtered manifest: %s", result["manifest_path"])
    logger.info("Selected source ids: %s", result["selected_source_ids_path"])
    logger.info("Metadata summary: %s", result["metadata_path"])
    logger.info("Filtering summary: %s", result["summary"])


if __name__ == "__main__":
    main()
