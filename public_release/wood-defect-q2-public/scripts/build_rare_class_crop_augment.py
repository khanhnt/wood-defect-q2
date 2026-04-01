#!/usr/bin/env python3
"""Build an offline rare-class crop-augmented processed manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.rare_class_crop_augment import (
    DEFAULT_RARE_CLASS_TARGETS,
    build_rare_class_crop_augmented_dataset,
)
from src.utils.config import expand_path
from src.utils.logger import setup_logger

logger = setup_logger()


def _parse_class_max_crops(values: list[str] | None) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values or []:
        class_name, separator, limit_text = str(value).partition("=")
        if not separator:
            raise ValueError(
                f"Invalid --class-max-crops entry: {value!r}. Expected format <class_name>=<limit>."
            )
        parsed[class_name.strip()] = int(limit_text)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        type=str,
        required=True,
        help="Processed tile manifest to augment.",
    )
    parser.add_argument(
        "--image-root-dir",
        type=str,
        required=True,
        help="Root directory used to resolve relative image_path values in the input manifest.",
    )
    parser.add_argument(
        "--output-root-dir",
        type=str,
        required=True,
        help="Directory where augmented crop JPEGs and the new manifest will be written.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Dataset name stored in the augmented manifest.",
    )
    parser.add_argument(
        "--target-classes",
        nargs="+",
        default=list(DEFAULT_RARE_CLASS_TARGETS),
        help="Rare classes to target with crop augmentation.",
    )
    parser.add_argument(
        "--class-max-crops",
        nargs="*",
        default=None,
        help="Optional per-class global caps in the form class_name=limit.",
    )
    parser.add_argument(
        "--candidate-selection-mode",
        type=str,
        choices=("manifest", "balanced"),
        default="balanced",
        help="How to choose crop candidates globally after proposal generation.",
    )
    parser.add_argument(
        "--head-classes",
        nargs="+",
        default=["live_knot", "dead_knot"],
        help="Head classes treated as dominant context when ranking or filtering crop candidates.",
    )
    parser.add_argument(
        "--max-window-head-annotation-count",
        type=int,
        default=None,
        help="Optional hard cap on how many head-class annotations may appear inside one kept crop.",
    )
    parser.add_argument(
        "--max-crops-per-record",
        type=int,
        default=2,
        help="Maximum augmented crops to generate from one train tile record.",
    )
    parser.add_argument(
        "--edge-margin-px",
        type=float,
        default=24.0,
        help="Reject target boxes that lie closer than this margin to a tile edge.",
    )
    parser.add_argument(
        "--merge-iou-threshold",
        type=float,
        default=0.1,
        help="Cluster same-class boxes when IoU exceeds this threshold.",
    )
    parser.add_argument(
        "--merge-center-distance-px",
        type=float,
        default=96.0,
        help="Cluster same-class boxes when center distance is below this threshold.",
    )
    parser.add_argument(
        "--min-retained-ratio-target",
        type=float,
        default=0.9,
        help="Minimum retained area ratio for target-class boxes after cropping.",
    )
    parser.add_argument(
        "--min-retained-ratio-context",
        type=float,
        default=0.75,
        help="Minimum retained area ratio for non-target context boxes after cropping.",
    )
    parser.add_argument(
        "--min-box-size-px",
        type=float,
        default=8.0,
        help="Drop boxes smaller than this size after cropping.",
    )
    parser.add_argument(
        "--target-border-margin-px",
        type=float,
        default=12.0,
        help="Reject crops whose retained target boxes touch the crop border within this margin.",
    )
    parser.add_argument(
        "--max-window-iou",
        type=float,
        default=0.6,
        help="Reject crop windows that overlap an already kept window more than this IoU.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=97,
        help="JPEG quality for generated crop images.",
    )
    parser.add_argument(
        "--small-defect-combine",
        type=str,
        default="any",
        choices=("any", "all"),
        help="Small-defect combine rule for recomputing augmented crop tags.",
    )
    parser.add_argument(
        "--small-defect-min-area-ratio",
        type=float,
        default=0.01,
        help="Small-defect area-ratio threshold for recomputing augmented crop tags.",
    )
    parser.add_argument(
        "--small-defect-min-width-px",
        type=float,
        default=16.0,
        help="Small-defect width threshold for recomputing augmented crop tags.",
    )
    parser.add_argument(
        "--small-defect-min-height-px",
        type=float,
        default=16.0,
        help="Small-defect height threshold for recomputing augmented crop tags.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_manifest = expand_path(args.input_manifest)
    image_root_dir = expand_path(args.image_root_dir)
    output_root_dir = expand_path(args.output_root_dir)
    if input_manifest is None or image_root_dir is None or output_root_dir is None:
        raise ValueError("input-manifest, image-root-dir, and output-root-dir must resolve to non-empty paths.")

    result = build_rare_class_crop_augmented_dataset(
        input_manifest_path=input_manifest,
        image_root_dir=image_root_dir,
        output_root_dir=output_root_dir,
        dataset_name=args.dataset_name,
        target_classes=args.target_classes,
        class_max_crops=_parse_class_max_crops(args.class_max_crops),
        candidate_selection_mode=args.candidate_selection_mode,
        head_classes=args.head_classes,
        max_window_head_annotation_count=args.max_window_head_annotation_count,
        max_crops_per_record=args.max_crops_per_record,
        edge_margin_px=args.edge_margin_px,
        merge_iou_threshold=args.merge_iou_threshold,
        merge_center_distance_px=args.merge_center_distance_px,
        min_retained_ratio_target=args.min_retained_ratio_target,
        min_retained_ratio_context=args.min_retained_ratio_context,
        min_box_size_px=args.min_box_size_px,
        target_border_margin_px=args.target_border_margin_px,
        max_window_iou=args.max_window_iou,
        jpeg_quality=args.jpeg_quality,
        small_defect_rule={
            "enabled": True,
            "combine": args.small_defect_combine,
            "min_area_ratio": args.small_defect_min_area_ratio,
            "min_width_px": args.small_defect_min_width_px,
            "min_height_px": args.small_defect_min_height_px,
        },
    )

    logger.info("Augmented manifest: %s", result["manifest_path"])
    logger.info("Metadata summary: %s", result["metadata_path"])
    logger.info("Repo preprocess summary: %s", result["repo_summary_path"])
    logger.info("Augmentation summary: %s", result["summary"]["augmentation"])


if __name__ == "__main__":
    main()
