"""Minimal manifest-backed detection dataset and dataloader helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.datasets.base_dataset import normalize_class_name, resolve_small_defect_rule
from src.utils.config import expand_path, load_yaml


def _load_dataset_config(config_or_path: str | Path | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(config_or_path, (str, Path)):
        return load_yaml(config_or_path)
    return dict(config_or_path)


def _resolve_manifest_path(dataset_config: Dict[str, Any]) -> Path:
    manifest_value = dataset_config.get("manifest_path")
    if manifest_value:
        manifest_path = expand_path(manifest_value)
        if manifest_path is None:
            raise ValueError("manifest_path is empty.")
        return manifest_path

    dataset_name = dataset_config.get("dataset_name")
    if not dataset_name:
        raise ValueError("dataset_name or manifest_path is required in dataset config.")

    root_dir = expand_path(dataset_config.get("root_dir"))
    if root_dir is not None and (root_dir / "manifest.jsonl").exists():
        return root_dir / "manifest.jsonl"

    return Path("data/processed") / f"{dataset_name}_manifest.jsonl"


def _read_manifest_records(manifest_path: Path) -> list[Dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

    records: list[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return records


def _normalize_split_name(split: str | None) -> str | None:
    if split is None:
        return None
    split = str(split).lower()
    return "val" if split == "validation" else split


def _assign_synthetic_splits(
    records: Sequence[Dict[str, Any]],
    split_seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
) -> list[Dict[str, Any]]:
    copied_records = [dict(record) for record in records]
    ordered_indices = list(range(len(copied_records)))
    ordered_indices.sort(key=lambda index: copied_records[index]["image_id"])

    rng = random.Random(split_seed)
    rng.shuffle(ordered_indices)

    train_count = int(len(ordered_indices) * train_ratio)
    val_count = int(len(ordered_indices) * val_ratio)
    val_count = min(val_count, len(ordered_indices) - train_count)

    split_assignment: dict[int, str] = {}
    for index in ordered_indices[:train_count]:
        split_assignment[index] = "train"
    for index in ordered_indices[train_count:train_count + val_count]:
        split_assignment[index] = "val"
    for index in ordered_indices[train_count + val_count:]:
        split_assignment[index] = "test"

    for index, record in enumerate(copied_records):
        record["resolved_split"] = split_assignment[index]
    return copied_records


def load_manifest_records(
    dataset_config_or_path: str | Path | Dict[str, Any],
    split: str | None = None,
    split_seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    max_samples: int | None = None,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Load manifest records and resolve train/val/test filtering."""
    dataset_config = _load_dataset_config(dataset_config_or_path)
    manifest_path = _resolve_manifest_path(dataset_config)
    records = _read_manifest_records(manifest_path)

    has_explicit_splits = any(record.get("split") is not None for record in records)
    normalized_split = _normalize_split_name(split)

    if has_explicit_splits:
        filtered_records = []
        for record in records:
            resolved = _normalize_split_name(record.get("split"))
            record_copy = dict(record)
            record_copy["resolved_split"] = resolved
            if normalized_split in {None, "all"} or resolved == normalized_split:
                filtered_records.append(record_copy)
    else:
        synthetic_records = _assign_synthetic_splits(
            records=records,
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        filtered_records = [
            record
            for record in synthetic_records
            if normalized_split in {None, "all"} or record.get("resolved_split") == normalized_split
        ]

    if max_samples is not None:
        filtered_records = filtered_records[: int(max_samples)]

    class_names = resolve_class_names(records, dataset_config)
    image_root_dir = expand_path(dataset_config.get("root_dir")) or manifest_path.parent
    metadata = {
        "dataset_config": dataset_config,
        "manifest_path": str(manifest_path),
        "image_root_dir": str(image_root_dir),
        "class_names": class_names,
        "num_classes": len(class_names),
        "split": normalized_split or "all",
    }
    return filtered_records, metadata


def resolve_class_names(records: Sequence[Dict[str, Any]], dataset_config: Dict[str, Any]) -> list[str]:
    """Resolve foreground class names using config first, then manifest annotations."""
    configured_names = [
        normalize_class_name(class_name)
        for class_name in dataset_config.get("classes", []) or []
    ]
    if configured_names:
        return configured_names

    class_pairs = {
        (int(annotation.get("class_id", -1)), annotation["class_name"])
        for record in records
        for annotation in record.get("annotations", [])
        if int(annotation.get("class_id", -1)) >= 0
    }
    if class_pairs:
        return [name for _, name in sorted(class_pairs, key=lambda item: item[0])]

    return sorted(
        {
            annotation["class_name"]
            for record in records
            for annotation in record.get("annotations", [])
        }
    )


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a float tensor in CHW format without torchvision."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def build_small_defect_sampler(
    records: Sequence[Dict[str, Any]],
    sampler_config: Dict[str, Any] | None,
    small_defect_config: Dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[WeightedRandomSampler | None, Dict[str, Any] | None]:
    """Build a simple weighted sampler that favors tiles containing small defects."""
    if not sampler_config or not bool(sampler_config.get("enabled", False)):
        return None, None

    small_weight = float(sampler_config.get("small_weight", 3.0))
    positive_weight = float(sampler_config.get("positive_weight", 1.5))
    negative_weight = float(sampler_config.get("negative_weight", 0.5))

    weights: list[float] = []
    bucket_counts = {
        "small_defect_records": 0,
        "positive_records": 0,
        "negative_records": 0,
    }
    small_defect_rule = resolve_small_defect_rule(small_defect_config)

    def infer_small_annotations(record: Dict[str, Any]) -> int:
        explicit_count = record.get("num_small_annotations")
        if explicit_count is not None and int(explicit_count) > 0:
            return int(explicit_count)

        annotations = record.get("annotations", [])
        if not annotations:
            return 0

        record_width = float(record.get("width", 0) or 0)
        record_height = float(record.get("height", 0) or 0)
        inferred_count = 0

        for annotation in annotations:
            if bool(annotation.get("is_small_defect", False)):
                inferred_count += 1
                continue

            checks = []
            if small_defect_rule.get("min_area_ratio") is not None:
                checks.append(float(annotation.get("bbox_area_norm", 0.0)) <= float(small_defect_rule["min_area_ratio"]))

            if small_defect_rule.get("min_width_px") is not None and record_width > 0:
                width_px = float(annotation.get("bbox_width_norm", 0.0)) * record_width
                checks.append(width_px <= float(small_defect_rule["min_width_px"]))

            if small_defect_rule.get("min_height_px") is not None and record_height > 0:
                height_px = float(annotation.get("bbox_height_norm", 0.0)) * record_height
                checks.append(height_px <= float(small_defect_rule["min_height_px"]))

            if not small_defect_rule["enabled"] or not checks:
                is_small = False
            elif small_defect_rule["combine"] == "all":
                is_small = all(checks)
            else:
                is_small = any(checks)

            if is_small:
                inferred_count += 1

        return inferred_count

    for record in records:
        num_small_annotations = infer_small_annotations(record)
        num_annotations = len(record.get("annotations", []))
        if num_small_annotations > 0:
            weights.append(small_weight)
            bucket_counts["small_defect_records"] += 1
        elif num_annotations > 0:
            weights.append(positive_weight)
            bucket_counts["positive_records"] += 1
        else:
            weights.append(negative_weight)
            bucket_counts["negative_records"] += 1

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    summary = {
        "enabled": True,
        "small_weight": small_weight,
        "positive_weight": positive_weight,
        "negative_weight": negative_weight,
        "small_defect_rule": small_defect_rule,
        **bucket_counts,
    }
    return sampler, summary


class ManifestDetectionDataset(Dataset):
    """A minimal PyTorch detection dataset backed by prepared JSONL manifests."""

    def __init__(self, records: Sequence[Dict[str, Any]], image_root_dir: str | Path | None = None) -> None:
        self.records = list(records)
        self.image_root_dir = expand_path(image_root_dir)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = Path(record["image_path"])
        if not image_path.is_absolute():
            if self.image_root_dir is not None:
                image_path = self.image_root_dir / image_path
            else:
                image_path = image_path.resolve()

        image = Image.open(image_path).convert("RGB")
        image_tensor = _pil_to_tensor(image)

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        for annotation in record.get("annotations", []):
            x1, y1, x2, y2 = annotation["bbox_xyxy_norm"]
            width = float(record["width"])
            height = float(record["height"])
            box = [
                float(x1) * width,
                float(y1) * height,
                float(x2) * width,
                float(y2) * height,
            ]
            boxes.append(box)
            labels.append(int(annotation["class_id"]) + 1)
            areas.append(max(0.0, (box[2] - box[0]) * (box[3] - box[1])))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        if len(boxes) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)

        metadata = {
            "image_id": record["image_id"],
            "dataset_name": record["dataset_name"],
            "resolved_split": record.get("resolved_split"),
            "source_image_id": record.get("source_image_id"),
            "tile_origin_xy": record.get("tile_origin_xy"),
            "record_width": int(record["width"]),
            "record_height": int(record["height"]),
        }
        return image_tensor, target, metadata


def collate_detection_batch(batch):
    """Collate function for torchvision detection models."""
    images, targets, metadata = zip(*batch)
    return list(images), list(targets), list(metadata)


def build_detection_dataloader(
    dataset_config_or_path: str | Path | Dict[str, Any],
    split: str | None,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    split_seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    max_samples: int | None = None,
    sampler_config: Dict[str, Any] | None = None,
) -> tuple[DataLoader, Dict[str, Any]]:
    """Build a minimal detection dataloader from a manifest-backed dataset."""
    records, metadata = load_manifest_records(
        dataset_config_or_path=dataset_config_or_path,
        split=split,
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_samples=max_samples,
    )
    dataset = ManifestDetectionDataset(records, image_root_dir=metadata["image_root_dir"])
    sampler, sampler_summary = build_small_defect_sampler(
        records=records,
        sampler_config=sampler_config,
        small_defect_config=metadata["dataset_config"].get("small_defect"),
        seed=split_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle and sampler is None),
        num_workers=num_workers,
        collate_fn=collate_detection_batch,
        sampler=sampler,
    )
    metadata["num_images"] = len(dataset)
    if sampler_summary is not None:
        metadata["sampler"] = sampler_summary
    return loader, metadata
