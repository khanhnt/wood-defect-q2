"""Export manifest-backed datasets into Ultralytics YOLO detection format."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.datasets.base_dataset import normalize_class_name
from src.datasets.screened_benchmark import load_jsonl_records
from src.utils.io import ensure_dir, save_json


def _resolve_image_path(record: Mapping[str, Any], image_root_dir: str | Path | None) -> Path:
    image_path = Path(str(record.get("image_path") or ""))
    if image_path.is_absolute():
        return image_path
    if image_root_dir is None:
        raise ValueError(f"Relative image_path requires image_root_dir: {image_path}")
    return Path(image_root_dir) / image_path


def _sanitize_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "__", value.strip())
    stem = re.sub(r"__+", "__", stem).strip("._")
    return stem or "record"


def _xyxy_to_yolo_xywh(box_xyxy: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


def _link_or_copy(src: Path, dst: Path, prefer_symlink: bool) -> str:
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if prefer_symlink:
        try:
            os.symlink(src, dst)
            return "symlink"
        except OSError:
            pass

    shutil.copy2(src, dst)
    return "copy"


def export_manifest_to_yolo(
    input_manifest_path: str | Path,
    image_root_dir: str | Path | None,
    output_root_dir: str | Path,
    dataset_name: str,
    classes: Sequence[str],
    prefer_symlink: bool = True,
) -> dict[str, Any]:
    """Export a manifest-backed dataset into YOLO images/labels layout plus dataset YAML."""
    normalized_classes = [normalize_class_name(name) for name in classes]
    class_to_id = {class_name: index for index, class_name in enumerate(normalized_classes)}
    records = load_jsonl_records(input_manifest_path)

    output_root = ensure_dir(output_root_dir)
    images_root = ensure_dir(output_root / "images")
    labels_root = ensure_dir(output_root / "labels")
    dataset_yaml_path = output_root / "dataset.yaml"
    metadata_path = output_root / "metadata.json"

    export_counts_by_split: dict[str, int] = {}
    positive_counts_by_split: dict[str, int] = {}
    negative_counts_by_split: dict[str, int] = {}
    annotation_count_by_class = {class_name: 0 for class_name in normalized_classes}
    link_mode_counter = {"symlink": 0, "copy": 0}

    for record in records:
        split_name = str(record.get("split") or "train").lower()
        if split_name == "validation":
            split_name = "val"
        if split_name not in {"train", "val", "test"}:
            continue

        src_image_path = _resolve_image_path(record, image_root_dir=image_root_dir)
        if not src_image_path.exists():
            raise FileNotFoundError(f"Image does not exist for YOLO export: {src_image_path}")

        export_stem = _sanitize_stem(str(record.get("image_id") or src_image_path.stem))
        image_suffix = src_image_path.suffix or ".jpg"
        split_image_dir = ensure_dir(images_root / split_name)
        split_label_dir = ensure_dir(labels_root / split_name)
        dst_image_path = split_image_dir / f"{export_stem}{image_suffix}"
        dst_label_path = split_label_dir / f"{export_stem}.txt"

        link_mode = _link_or_copy(src=src_image_path, dst=dst_image_path, prefer_symlink=prefer_symlink)
        link_mode_counter[link_mode] += 1

        yolo_lines: list[str] = []
        for annotation in record.get("annotations", []):
            class_name = normalize_class_name(annotation["class_name"])
            if class_name not in class_to_id:
                continue
            cx, cy, width, height = _xyxy_to_yolo_xywh(annotation["bbox_xyxy_norm"])
            yolo_lines.append(
                f"{class_to_id[class_name]} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"
            )
            annotation_count_by_class[class_name] += 1

        dst_label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        export_counts_by_split[split_name] = export_counts_by_split.get(split_name, 0) + 1
        if yolo_lines:
            positive_counts_by_split[split_name] = positive_counts_by_split.get(split_name, 0) + 1
        else:
            negative_counts_by_split[split_name] = negative_counts_by_split.get(split_name, 0) + 1

    dataset_yaml_lines = [
        f"path: {output_root}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for class_id, class_name in enumerate(normalized_classes):
        dataset_yaml_lines.append(f"  {class_id}: {class_name}")
    dataset_yaml_path.write_text("\n".join(dataset_yaml_lines) + "\n", encoding="utf-8")

    metadata = {
        "dataset_name": dataset_name,
        "input_manifest_path": str(Path(input_manifest_path)),
        "dataset_yaml_path": str(dataset_yaml_path),
        "output_root_dir": str(output_root),
        "classes": normalized_classes,
        "prefer_symlink": bool(prefer_symlink),
        "link_mode_counts": {key: int(value) for key, value in sorted(link_mode_counter.items())},
        "num_records_by_split": {split: int(count) for split, count in sorted(export_counts_by_split.items())},
        "positive_records_by_split": {split: int(count) for split, count in sorted(positive_counts_by_split.items())},
        "negative_records_by_split": {split: int(count) for split, count in sorted(negative_counts_by_split.items())},
        "annotation_count_by_class": {class_name: int(annotation_count_by_class[class_name]) for class_name in normalized_classes},
    }
    save_json(metadata, metadata_path)

    return {
        "dataset_yaml_path": dataset_yaml_path,
        "metadata_path": metadata_path,
        "summary": metadata,
    }
