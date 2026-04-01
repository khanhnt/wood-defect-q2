#!/usr/bin/env python3
"""Retile an existing processed VNWoodKnot dataset into a matched-tiling external protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.vnwoodknot_dataset import build_tiled_vnwoodknot_from_processed_manifest
from src.utils.config import expand_path
from src.utils.logger import setup_logger

logger = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        type=str,
        default="${WOOD_VN_PROCESSED_ROOT}/manifest.jsonl",
        help="Processed VNWoodKnot manifest to retile.",
    )
    parser.add_argument(
        "--image-root-dir",
        type=str,
        default="${WOOD_VN_PROCESSED_ROOT}",
        help="Root used to resolve relative image_path values in the processed manifest.",
    )
    parser.add_argument(
        "--output-root-dir",
        type=str,
        default="${WOOD_VN_TILED_PROCESSED_ROOT}",
        help="Output directory for the tiled processed dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="vnwoodknot_tiled",
        help="Dataset name written into metadata.",
    )
    parser.add_argument(
        "--repo-output-dir",
        type=str,
        default="outputs/tables",
        help="Directory inside the repo for compact preprocessing summaries.",
    )
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size in pixels.")
    parser.add_argument("--tile-overlap", type=int, default=128, help="Tile overlap in pixels.")
    parser.add_argument(
        "--min-box-visibility",
        type=float,
        default=0.5,
        help="Minimum retained area ratio required to keep a box inside a tile.",
    )
    parser.add_argument(
        "--drop-negative-tiles",
        action="store_true",
        help="Drop most empty tiles instead of keeping them all.",
    )
    parser.add_argument(
        "--negative-ratio-to-positive",
        type=float,
        default=0.25,
        help="Used only when --drop-negative-tiles is set.",
    )
    parser.add_argument(
        "--negative-max-per-source",
        type=int,
        default=2,
        help="Used only when --drop-negative-tiles is set.",
    )
    parser.add_argument(
        "--negative-empty-source-keep",
        type=int,
        default=0,
        help="Used only when --drop-negative-tiles is set.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=97, help="JPEG quality for tiled images.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_manifest = expand_path(args.input_manifest)
    image_root_dir = expand_path(args.image_root_dir)
    output_root_dir = expand_path(args.output_root_dir)

    if input_manifest is None or output_root_dir is None:
        raise ValueError("input-manifest and output-root-dir must resolve to non-empty paths.")

    result = build_tiled_vnwoodknot_from_processed_manifest(
        input_manifest_path=input_manifest,
        image_root_dir=image_root_dir,
        output_root_dir=output_root_dir,
        dataset_name=args.dataset_name,
        repo_output_dir=args.repo_output_dir,
        tile_cfg={
            "size": args.tile_size,
            "overlap": args.tile_overlap,
            "min_box_visibility": args.min_box_visibility,
            "keep_all_negative_tiles": not bool(args.drop_negative_tiles),
        },
        negative_cfg={
            "enabled": bool(args.drop_negative_tiles),
            "ratio_to_positive": args.negative_ratio_to_positive,
            "max_per_source_image": args.negative_max_per_source,
            "empty_source_keep": args.negative_empty_source_keep,
        },
        jpeg_quality=args.jpeg_quality,
        max_images=args.max_images,
    )

    logger.info("Tiled VNWoodKnot manifest: %s", result["artifacts"]["manifest_path"])
    logger.info("Metadata summary: %s", result["artifacts"]["metadata_path"])
    logger.info("Repo preprocess summary: %s", result["artifacts"]["repo_summary_path"])
    logger.info(
        "Tiled summary: %s",
        {
            "num_source_images": result["summary"]["num_source_images"],
            "num_processed_images": result["summary"]["num_processed_images"],
            "num_processed_annotations": result["summary"]["num_processed_annotations"],
            "split_distribution": result["summary"]["split_distribution"],
        },
    )


if __name__ == "__main__":
    main()
