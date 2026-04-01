#!/usr/bin/env python3
"""Export a manifest-backed dataset into YOLOv8/Ultralytics format."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.screened_benchmark import DEFAULT_VSB7_CLASSES
from src.datasets.yolo_export import export_manifest_to_yolo
from src.utils.config import expand_path
from src.utils.logger import setup_logger

logger = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        type=str,
        default="${WOOD_MAIN_PROCESSED_ROOT}/benchmarks/full_7class/manifest.jsonl",
        help="Manifest to export in YOLO format.",
    )
    parser.add_argument(
        "--image-root-dir",
        type=str,
        default="${WOOD_MAIN_PROCESSED_ROOT}",
        help="Root dir used to resolve relative image_path values.",
    )
    parser.add_argument(
        "--output-root-dir",
        type=str,
        default="${WOOD_MAIN_PROCESSED_ROOT}/benchmarks/full_7class_yolo",
        help="Output directory for YOLO dataset files.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="large_scale_wood_surface_defects_full_7class_yolo",
        help="Dataset name written into metadata.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_VSB7_CLASSES),
        help="Foreground classes to export.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symlinks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_manifest = expand_path(args.input_manifest)
    image_root_dir = expand_path(args.image_root_dir)
    output_root_dir = expand_path(args.output_root_dir)
    if input_manifest is None or output_root_dir is None:
        raise ValueError("input-manifest and output-root-dir must resolve to non-empty paths.")

    result = export_manifest_to_yolo(
        input_manifest_path=input_manifest,
        image_root_dir=image_root_dir,
        output_root_dir=output_root_dir,
        dataset_name=args.dataset_name,
        classes=args.classes,
        prefer_symlink=not bool(args.copy_images),
    )

    logger.info("YOLO dataset yaml: %s", result["dataset_yaml_path"])
    logger.info("YOLO metadata: %s", result["metadata_path"])
    logger.info("YOLO export summary: %s", result["summary"])


if __name__ == "__main__":
    main()
