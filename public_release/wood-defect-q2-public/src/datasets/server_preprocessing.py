"""Helpers for building compact processed detection datasets for server upload."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from math import ceil
from pathlib import Path
from random import Random
from typing import Any, Dict, Mapping, Sequence

import pandas as pd
from PIL import Image

from src.datasets.base_dataset import assign_class_ids, build_annotation, normalize_class_name
from src.utils.io import compute_dir_size_bytes, ensure_dir, format_num_bytes, save_csv, save_json, save_jsonl


def normalize_split_name(split: str | None) -> str | None:
    """Normalize split names for portable processed manifests."""
    if split is None:
        return None
    split = str(split).lower()
    return "val" if split == "validation" else split


def assign_splits_by_source_image(
    records: Sequence[Dict[str, Any]],
    split_config: Mapping[str, Any] | None = None,
    preserve_existing: bool = False,
) -> list[Dict[str, Any]]:
    """Assign deterministic train/val/test splits at the source-image level."""
    split_config = dict(split_config or {})
    seed = int(split_config.get("seed", 42))
    train_ratio = float(split_config.get("train_ratio", 0.8))
    val_ratio = float(split_config.get("val_ratio", 0.1))
    test_ratio = float(split_config.get("test_ratio", 0.1))

    if train_ratio < 0.0 or val_ratio < 0.0 or test_ratio < 0.0:
        raise ValueError("Split ratios must be non-negative.")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0.0:
        raise ValueError("At least one split ratio must be greater than zero.")

    normalized_train = train_ratio / ratio_sum
    normalized_val = val_ratio / ratio_sum

    copied_records = [deepcopy(record) for record in records]
    if preserve_existing and all(record.get("split") for record in copied_records):
        for record in copied_records:
            record["split"] = normalize_split_name(record.get("split"))
        return copied_records

    source_ids = sorted({record["image_id"] for record in copied_records})
    rng = Random(seed)
    rng.shuffle(source_ids)

    num_sources = len(source_ids)
    train_count = int(num_sources * normalized_train)
    val_count = int(num_sources * normalized_val)
    val_count = min(val_count, max(num_sources - train_count, 0))

    split_map: dict[str, str] = {}
    for source_id in source_ids[:train_count]:
        split_map[source_id] = "train"
    for source_id in source_ids[train_count:train_count + val_count]:
        split_map[source_id] = "val"
    for source_id in source_ids[train_count + val_count:]:
        split_map[source_id] = "test"

    for record in copied_records:
        record["split"] = split_map.get(record["image_id"], "train")

    return copied_records


def generate_tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Generate tile start positions that cover an axis fully."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size.")
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def generate_tile_windows(width: int, height: int, tile_size: int, overlap: int) -> list[Dict[str, int]]:
    """Generate XYXY tile windows for an image."""
    x_starts = generate_tile_starts(width, tile_size, overlap)
    y_starts = generate_tile_starts(height, tile_size, overlap)
    windows: list[Dict[str, int]] = []
    for top in y_starts:
        for left in x_starts:
            right = min(left + tile_size, width)
            bottom = min(top + tile_size, height)
            windows.append(
                {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "width": right - left,
                    "height": bottom - top,
                }
            )
    return windows


def _annotation_to_abs_xyxy(annotation: Mapping[str, Any], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = annotation["bbox_xyxy_norm"]
    return (
        float(x1) * image_width,
        float(y1) * image_height,
        float(x2) * image_width,
        float(y2) * image_height,
    )


def remap_annotations_to_tile(
    annotations: Sequence[Mapping[str, Any]],
    image_width: int,
    image_height: int,
    tile_window: Mapping[str, int],
    min_visibility: float = 0.5,
) -> list[Dict[str, Any]]:
    """Clip source annotations to a tile and remap them into tile-normalized coordinates."""
    remapped: list[Dict[str, Any]] = []
    tile_left = float(tile_window["left"])
    tile_top = float(tile_window["top"])
    tile_right = float(tile_window["right"])
    tile_bottom = float(tile_window["bottom"])
    tile_width = float(tile_window["width"])
    tile_height = float(tile_window["height"])

    for annotation in annotations:
        box_left, box_top, box_right, box_bottom = _annotation_to_abs_xyxy(annotation, image_width, image_height)
        original_width = max(box_right - box_left, 0.0)
        original_height = max(box_bottom - box_top, 0.0)
        original_area = original_width * original_height
        if original_area <= 0.0:
            continue

        clipped_left = max(box_left, tile_left)
        clipped_top = max(box_top, tile_top)
        clipped_right = min(box_right, tile_right)
        clipped_bottom = min(box_bottom, tile_bottom)

        clipped_width = max(clipped_right - clipped_left, 0.0)
        clipped_height = max(clipped_bottom - clipped_top, 0.0)
        clipped_area = clipped_width * clipped_height
        if clipped_area <= 0.0:
            continue

        visible_ratio = clipped_area / original_area
        if visible_ratio < float(min_visibility):
            continue

        remapped.append(
            build_annotation(
                class_name=annotation["class_name"],
                bbox_xyxy_norm=[
                    (clipped_left - tile_left) / tile_width,
                    (clipped_top - tile_top) / tile_height,
                    (clipped_right - tile_left) / tile_width,
                    (clipped_bottom - tile_top) / tile_height,
                ],
                source_label=annotation.get("source_label"),
            )
        )

    return remapped


def choose_negative_tiles(
    negative_tiles: Sequence[Dict[str, Any]],
    num_positive_tiles: int,
    source_image_id: str,
    negative_config: Mapping[str, Any] | None = None,
) -> list[Dict[str, Any]]:
    """Select a controlled subset of negative tiles for one source image."""
    negative_config = dict(negative_config or {})
    if not negative_config.get("enabled", True):
        return []

    ratio_to_positive = float(negative_config.get("ratio_to_positive", 0.25))
    max_per_source = negative_config.get("max_per_source_image", 2)
    empty_source_keep = int(negative_config.get("empty_source_keep", 0))
    seed = int(negative_config.get("seed", 42))

    if num_positive_tiles > 0:
        keep_count = int(ceil(num_positive_tiles * ratio_to_positive))
    else:
        keep_count = empty_source_keep

    if max_per_source is not None:
        keep_count = min(keep_count, int(max_per_source))

    keep_count = min(keep_count, len(negative_tiles))
    if keep_count <= 0:
        return []
    if keep_count >= len(negative_tiles):
        return list(negative_tiles)

    rng = Random(f"{seed}:{source_image_id}")
    chosen_indices = sorted(rng.sample(range(len(negative_tiles)), keep_count))
    return [negative_tiles[index] for index in chosen_indices]


def save_image_as_jpeg(image: Image.Image, output_path: str | Path, quality: int = 97) -> None:
    """Save a PIL image as JPEG with stable settings."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    image.convert("RGB").save(output_path, format="JPEG", quality=int(quality), subsampling=0)


def build_processed_summary(
    dataset_name: str,
    source_records: Sequence[Dict[str, Any]],
    processed_records: Sequence[Dict[str, Any]],
    processed_root_dir: str | Path,
    preprocess_config: Mapping[str, Any] | None = None,
) -> tuple[Dict[str, Any], pd.DataFrame]:
    """Build compact preprocessing summary outputs."""
    preprocess_config = dict(preprocess_config or {})
    class_to_idx = assign_class_ids(processed_records, preferred_class_names=preprocess_config.get("classes"))

    source_class_counter: Counter[str] = Counter()
    processed_class_counter: Counter[str] = Counter()
    processed_class_images: dict[str, set[str]] = {class_name: set() for class_name in class_to_idx}
    split_counter: Counter[str] = Counter()

    for record in source_records:
        for annotation in record.get("annotations", []):
            source_class_counter[annotation["class_name"]] += 1

    positive_processed_images = 0
    negative_processed_images = 0
    processed_annotations = 0
    for record in processed_records:
        split_counter[str(record.get("split", "unspecified"))] += 1
        if record.get("annotations"):
            positive_processed_images += 1
        else:
            negative_processed_images += 1

        for annotation in record.get("annotations", []):
            processed_annotations += 1
            class_name = annotation["class_name"]
            processed_class_counter[class_name] += 1
            processed_class_images.setdefault(class_name, set()).add(record["image_id"])

    class_distribution = pd.DataFrame(
        [
            {
                "class_name": class_name,
                "class_id": class_to_idx[class_name],
                "source_box_count": int(source_class_counter.get(class_name, 0)),
                "processed_box_count": int(processed_class_counter.get(class_name, 0)),
                "processed_image_count": len(processed_class_images.get(class_name, set())),
            }
            for class_name in sorted(class_to_idx, key=lambda name: class_to_idx[name])
        ]
    )

    disk_usage_bytes = compute_dir_size_bytes(processed_root_dir)
    summary = {
        "dataset_name": dataset_name,
        "num_source_images": len(source_records),
        "num_source_annotations": int(sum(len(record.get("annotations", [])) for record in source_records)),
        "num_processed_images": len(processed_records),
        "num_positive_processed_images": positive_processed_images,
        "num_negative_processed_images": negative_processed_images,
        "num_processed_annotations": processed_annotations,
        "split_distribution": dict(sorted(split_counter.items())),
        "class_names": list(class_to_idx.keys()),
        "disk_usage_bytes": int(disk_usage_bytes),
        "disk_usage_human": format_num_bytes(disk_usage_bytes),
        "preprocess": dict(preprocess_config),
    }
    return summary, class_distribution


def export_processed_dataset(
    dataset_name: str,
    processed_root_dir: str | Path,
    processed_records: Sequence[Dict[str, Any]],
    summary: Mapping[str, Any],
    class_distribution: pd.DataFrame,
    repo_output_dir: str | Path = "outputs/tables",
) -> Dict[str, Path]:
    """Export the processed dataset manifest outside the repo and compact summaries inside the repo."""
    processed_root_dir = Path(processed_root_dir)
    repo_output_dir = Path(repo_output_dir)
    ensure_dir(processed_root_dir)
    ensure_dir(repo_output_dir)

    dataset_slug = normalize_class_name(dataset_name)
    manifest_path = processed_root_dir / "manifest.jsonl"
    metadata_path = processed_root_dir / "metadata.json"
    repo_summary_path = repo_output_dir / f"{dataset_slug}_preprocess_summary.json"
    repo_class_path = repo_output_dir / f"{dataset_slug}_preprocess_class_distribution.csv"

    save_jsonl(processed_records, manifest_path)
    save_json(dict(summary), metadata_path)
    save_json(dict(summary), repo_summary_path)
    save_csv(class_distribution, repo_class_path)

    return {
        "manifest_path": manifest_path,
        "metadata_path": metadata_path,
        "repo_summary_path": repo_summary_path,
        "repo_class_distribution_path": repo_class_path,
    }
