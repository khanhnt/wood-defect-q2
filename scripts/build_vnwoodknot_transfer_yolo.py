#!/usr/bin/env python3
"""Build a 2-class VNWoodKnot transfer dataset and export it to YOLO format."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.vnwoodknot_transfer import (
    DEFAULT_VNWOODKNOT_TRANSFER_CLASSES,
    build_vnwoodknot_transfer_yolo_dataset,
)
from src.utils.config import expand_path
from src.utils.logger import setup_logger

logger = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        type=str,
        default="${WOOD_VN_PROCESSED_ROOT}/manifest.jsonl",
        help="Processed VNWoodKnot manifest to convert.",
    )
    parser.add_argument(
        "--image-root-dir",
        type=str,
        default="${WOOD_VN_PROCESSED_ROOT}",
        help="Root dir used to resolve relative VNWoodKnot image paths.",
    )
    parser.add_argument(
        "--manifest-output-root-dir",
        type=str,
        default="${WOOD_VN_PROCESSED_ROOT}/benchmarks/vnwoodknot_live_dead_2class",
        help="Output directory for the filtered 2-class manifest.",
    )
    parser.add_argument(
        "--yolo-output-root-dir",
        type=str,
        default="${WOOD_VN_PROCESSED_ROOT}/benchmarks/vnwoodknot_live_dead_2class_yolo",
        help="Output directory for the YOLO-exported VNWoodKnot dataset.",
    )
    parser.add_argument(
        "--filtered-dataset-name",
        type=str,
        default="vnwoodknot_live_dead_2class",
        help="Dataset name stored in the filtered manifest.",
    )
    parser.add_argument(
        "--yolo-dataset-name",
        type=str,
        default="vnwoodknot_live_dead_2class_yolo",
        help="Dataset name stored in the YOLO metadata.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_VNWOODKNOT_TRANSFER_CLASSES),
        help="Foreground classes kept for transfer learning.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symlinks in the YOLO export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_manifest = expand_path(args.input_manifest)
    image_root_dir = expand_path(args.image_root_dir)
    manifest_output_root_dir = expand_path(args.manifest_output_root_dir)
    yolo_output_root_dir = expand_path(args.yolo_output_root_dir)
    if None in {input_manifest, image_root_dir, manifest_output_root_dir, yolo_output_root_dir}:
        raise ValueError("All VNWoodKnot transfer paths must resolve to non-empty values.")

    result = build_vnwoodknot_transfer_yolo_dataset(
        input_manifest_path=input_manifest,
        image_root_dir=image_root_dir,
        manifest_output_root_dir=manifest_output_root_dir,
        yolo_output_root_dir=yolo_output_root_dir,
        filtered_dataset_name=args.filtered_dataset_name,
        yolo_dataset_name=args.yolo_dataset_name,
        classes=args.classes,
        prefer_symlink=not bool(args.copy_images),
    )

    logger.info("VNWoodKnot filtered manifest: %s", result["filtered_manifest"]["manifest_path"])
    logger.info("VNWoodKnot filtered metadata: %s", result["filtered_manifest"]["metadata_path"])
    logger.info("VNWoodKnot YOLO dataset yaml: %s", result["yolo_export"]["dataset_yaml_path"])
    logger.info("VNWoodKnot YOLO metadata: %s", result["yolo_export"]["metadata_path"])
    logger.info("VNWoodKnot transfer build summary: %s", result)


if __name__ == "__main__":
    main()
