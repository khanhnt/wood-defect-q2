"""Helpers for VNWoodKnot transfer-learning experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.datasets.class_filtered_manifest import build_class_filtered_manifest
from src.datasets.yolo_export import export_manifest_to_yolo


DEFAULT_VNWOODKNOT_TRANSFER_CLASSES = ("live_knot", "dead_knot")


def build_vnwoodknot_transfer_yolo_dataset(
    *,
    input_manifest_path: str | Path,
    image_root_dir: str | Path,
    manifest_output_root_dir: str | Path,
    yolo_output_root_dir: str | Path,
    filtered_dataset_name: str = "vnwoodknot_live_dead_2class",
    yolo_dataset_name: str = "vnwoodknot_live_dead_2class_yolo",
    classes: Sequence[str] = DEFAULT_VNWOODKNOT_TRANSFER_CLASSES,
    prefer_symlink: bool = True,
) -> dict[str, Any]:
    """Build a 2-class VNWoodKnot manifest and export it to YOLO format.

    The filtered manifest keeps source images that become empty after removing
    ``knot_free`` so that negative images remain available during target-domain
    training.
    """

    filtered_result = build_class_filtered_manifest(
        input_manifest_path=input_manifest_path,
        output_root_dir=manifest_output_root_dir,
        dataset_name=filtered_dataset_name,
        kept_classes=classes,
        drop_source_images_without_kept_classes=False,
    )

    yolo_result = export_manifest_to_yolo(
        input_manifest_path=filtered_result["manifest_path"],
        image_root_dir=image_root_dir,
        output_root_dir=yolo_output_root_dir,
        dataset_name=yolo_dataset_name,
        classes=classes,
        prefer_symlink=prefer_symlink,
    )

    return {
        "filtered_manifest": filtered_result,
        "yolo_export": yolo_result,
    }
