#!/usr/bin/env python3
"""Prepare and audit the VNWoodKnot dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.base_dataset import export_prepared_dataset
from src.datasets.vnwoodknot_dataset import parse_vnwoodknot_dataset
from src.utils.config import load_yaml
from src.utils.logger import setup_logger

logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=None,
        help="Optional dataset root passed positionally",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_transfer.yaml",
        help="Path to dataset config",
    )
    parser.add_argument(
        "--root-dir",
        "--dataset-root",
        dest="root_dir",
        type=str,
        default=None,
        help="Optional override for the extracted dataset root",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory to store JSONL manifests",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/tables",
        help="Directory to store compact audit tables",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default="outputs/figures",
        help="Directory to store compact audit figures",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="docs",
        help="Directory to store markdown audit notes",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    effective_root = args.root_dir or args.dataset_root or config.get("root_dir")
    if effective_root:
        config["root_dir"] = effective_root
    if args.max_images is not None:
        config["max_images"] = args.max_images

    logger.info("Preparing VNWoodKnot dataset from %s", config.get("root_dir"))
    records, report = parse_vnwoodknot_dataset(config)
    artifacts = export_prepared_dataset(
        records=records,
        dataset_name=report["dataset_name"],
        class_to_idx=report["class_to_idx"],
        validation_counts=report["validation_counts"],
        small_defect_config=config.get("small_defect"),
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        docs_dir=args.docs_dir,
    )

    logger.info("Prepared %d images with %d classes", len(records), len(report["class_to_idx"]))
    logger.info("Manifest saved to %s", artifacts["manifest_path"])
    logger.info("Audit summary saved to %s", artifacts["summary_path"])


if __name__ == "__main__":
    main()
