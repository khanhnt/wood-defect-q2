#!/usr/bin/env python3
"""Preprocess VNWoodKnot into a compact server-upload format."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.vnwoodknot_dataset import preprocess_vnwoodknot_for_server
from src.utils.config import load_yaml
from src.utils.logger import setup_logger

logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=None,
        help="Optional raw dataset root passed positionally",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocess_vnwoodknot.yaml",
        help="Path to preprocessing config",
    )
    parser.add_argument(
        "--root-dir",
        "--dataset-root",
        dest="root_dir",
        type=str,
        default=None,
        help="Optional override for the external raw dataset root",
    )
    parser.add_argument(
        "--processed-root-dir",
        type=str,
        default=None,
        help="Optional override for the external processed dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/tables",
        help="Directory inside the repo for compact preprocessing summaries",
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
    if args.processed_root_dir is not None:
        config["processed_root_dir"] = args.processed_root_dir
    if args.max_images is not None:
        config["max_images"] = args.max_images
    config["repo_output_dir"] = args.output_dir

    logger.info("Preprocessing VNWoodKnot from %s", config.get("root_dir"))
    logger.info("Processed output root: %s", config.get("processed_root_dir"))
    result = preprocess_vnwoodknot_for_server(config)

    logger.info("Source images: %d", result["summary"]["num_source_images"])
    logger.info("Processed images: %d", result["summary"]["num_processed_images"])
    logger.info("Disk usage: %s", result["summary"]["disk_usage_human"])
    logger.info("Processed manifest: %s", result["artifacts"]["manifest_path"])
    logger.info("Repo summary: %s", result["artifacts"]["repo_summary_path"])


if __name__ == "__main__":
    main()
