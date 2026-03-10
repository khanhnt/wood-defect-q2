"""Parser and lightweight dataset wrapper for VNWoodKnot."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Sequence

from PIL import Image

from src.datasets.base_dataset import (
    BaseWoodDefectDataset,
    assign_class_ids,
    build_annotation,
    clip_and_validate_bbox_xyxy,
    read_image_size,
    resolve_optional_path,
    xywh_to_xyxy_norm,
)
from src.datasets.server_preprocessing import (
    assign_splits_by_source_image,
    build_processed_summary,
    export_processed_dataset,
    normalize_split_name,
    save_image_as_jpeg,
)
from src.utils.config import expand_path


DEFAULT_DATASET_NAME = "vnwoodknot"
DEFAULT_SOURCE_CATEGORY_MAP = {"0": "knot_free", "1": "live_knot", "2": "dead_knot"}
DEFAULT_YOLO_CLASS_MAP = {"0": "live_knot", "1": "dead_knot"}
DEFAULT_SPLITS = ("train", "validation", "test")


class VNWoodKnotDataset(BaseWoodDefectDataset):
    """Record-backed dataset for VNWoodKnot."""

    def load_records(self) -> Sequence[Dict[str, Any]]:
        return load_vnwoodknot_annotations(self.config)


def _resolve_root_dir(config: Dict[str, Any]) -> Path:
    root_value = config.get("root_dir")
    if not root_value:
        raise ValueError(
            "VNWoodKnot root_dir is required. Pass it via config or "
            "'python scripts/prepare_vnwoodknot.py --dataset-root /path/to/dataset'."
        )

    root_dir = expand_path(root_value)
    if root_dir is None:
        raise ValueError("VNWoodKnot root_dir is empty.")
    if not root_dir.exists():
        raise FileNotFoundError(
            f"VNWoodKnot root does not exist: {root_dir}. "
            "Pass the extracted dataset root via '--dataset-root' or '--root-dir'."
        )

    if not (root_dir / "train").exists() and (root_dir / "VNWoodKnot").exists():
        root_dir = root_dir / "VNWoodKnot"

    return root_dir


def _parse_label_file(
    label_path: Path,
    yolo_class_map: Dict[str, str],
    expected_category: str,
) -> tuple[list[Dict[str, Any]], int, int, list[str], str | None]:
    annotations: list[Dict[str, Any]] = []
    invalid_boxes = 0
    clipped_boxes = 0
    issues: list[str] = []

    raw_text = label_path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return annotations, invalid_boxes, clipped_boxes, issues, "empty_annotation_file"

    for line in lines:
        parts = line.replace(",", ".").split()
        if len(parts) != 5:
            invalid_boxes += 1
            continue

        class_id_token = parts[0]
        class_name = yolo_class_map.get(class_id_token)
        if class_name is None:
            invalid_boxes += 1
            issues.append("unknown_yolo_class_id")
            continue

        if expected_category in {"live_knot", "dead_knot"} and class_name != expected_category:
            issues.append("folder_label_mismatch")

        try:
            cx, cy, width, height = [float(value) for value in parts[1:]]
        except ValueError:
            invalid_boxes += 1
            continue

        bbox_xyxy_norm = xywh_to_xyxy_norm(cx, cy, width, height)
        bbox_xyxy_norm, bbox_issues = clip_and_validate_bbox_xyxy(bbox_xyxy_norm)
        if "clipped_box" in bbox_issues:
            clipped_boxes += 1
        if bbox_xyxy_norm is None:
            invalid_boxes += 1
            continue

        annotations.append(
            build_annotation(
                class_name=class_name,
                bbox_xyxy_norm=bbox_xyxy_norm,
                source_label=class_id_token,
            )
        )

    empty_reason = None if annotations else "invalid_annotations_only"
    return annotations, invalid_boxes, clipped_boxes, sorted(set(issues)), empty_reason


def parse_vnwoodknot_dataset(config: Dict[str, Any]) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Parse VNWoodKnot into the unified internal format."""
    dataset_name = config.get("dataset_name", DEFAULT_DATASET_NAME)
    root_dir = _resolve_root_dir(config)
    source_category_map = {
        str(key): value
        for key, value in config.get("source_category_map", DEFAULT_SOURCE_CATEGORY_MAP).items()
    }
    yolo_class_map = {
        str(key): value
        for key, value in config.get("yolo_class_map", DEFAULT_YOLO_CLASS_MAP).items()
    }
    requested_image_dir = resolve_optional_path(root_dir, config.get("image_dir"))
    data_root = requested_image_dir if requested_image_dir is not None else root_dir

    validation_counts: Counter[str] = Counter()
    records: list[Dict[str, Any]] = []

    splits = tuple(config.get("splits", DEFAULT_SPLITS))
    max_images = config.get("max_images")

    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            validation_counts["missing_split_directories"] += 1
            continue

        class_dirs = sorted([path for path in split_dir.iterdir() if path.is_dir()], key=lambda path: path.name)
        for class_dir in class_dirs:
            category_name = source_category_map.get(class_dir.name, f"unknown_{class_dir.name}")
            image_paths = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")))
            if max_images:
                remaining = int(max_images) - len(records)
                if remaining <= 0:
                    break
                image_paths = image_paths[:remaining]

            image_stems = {path.stem for path in image_paths}
            label_paths = sorted(class_dir.glob("*.txt"))
            label_stems = {path.stem for path in label_paths}

            if max_images is None:
                validation_counts["orphan_annotation_files"] += len(label_stems - image_stems)

            for image_path in image_paths:
                width, height = read_image_size(image_path)
                label_path = image_path.with_suffix(".txt")

                issues: list[str] = []
                annotations: list[Dict[str, Any]] = []
                invalid_boxes = 0
                clipped_boxes = 0
                empty_reason: str | None = None

                if label_path.exists():
                    parsed = _parse_label_file(
                        label_path=label_path,
                        yolo_class_map=yolo_class_map,
                        expected_category=category_name,
                    )
                    annotations, invalid_boxes, clipped_boxes, issues, empty_reason = parsed
                    if class_dir.name == "0" and annotations:
                        issues.append("background_folder_with_labels")
                else:
                    if class_dir.name == "0":
                        empty_reason = "background_sample"
                    else:
                        empty_reason = "missing_annotation_file"

                record = {
                    "dataset_name": dataset_name,
                    "image_id": str(image_path.relative_to(root_dir).with_suffix("")).replace("\\", "/"),
                    "image_path": str(image_path),
                    "split": split,
                    "source_category": category_name,
                    "width": width,
                    "height": height,
                    "annotations": annotations,
                    "is_empty": empty_reason is not None,
                    "empty_reason": empty_reason,
                    "issues": sorted(set(issues)),
                    "num_invalid_boxes": invalid_boxes,
                    "num_clipped_boxes": clipped_boxes,
                    "semantic_map_path": None,
                    "annotation_path": str(label_path) if label_path.exists() else None,
                }
                records.append(record)

            if max_images and len(records) >= int(max_images):
                break

        if max_images and len(records) >= int(max_images):
            break

    class_to_idx = assign_class_ids(records, preferred_class_names=config.get("classes"))
    report = {
        "dataset_name": dataset_name,
        "root_dir": str(root_dir),
        "class_to_idx": class_to_idx,
        "validation_counts": dict(validation_counts),
    }
    return records, report


def load_vnwoodknot_annotations(config: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Load raw annotations and convert them to the unified internal format."""
    records, _ = parse_vnwoodknot_dataset(config)
    return records


def preprocess_vnwoodknot_for_server(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a compact processed VNWoodKnot dataset for server-side training."""
    source_records, report = parse_vnwoodknot_dataset(config)
    source_records = assign_splits_by_source_image(
        source_records,
        split_config=config.get("split"),
        preserve_existing=True,
    )

    processed_root_dir = expand_path(config.get("processed_root_dir"))
    if processed_root_dir is None:
        raise ValueError("processed_root_dir is required for preprocessing.")

    jpeg_quality = int(config.get("jpeg_quality", 97))
    repo_output_dir = config.get("repo_output_dir", "outputs/tables")
    max_images = config.get("max_images")
    processed_records: list[Dict[str, Any]] = []

    for record_index, source_record in enumerate(source_records):
        if max_images is not None and record_index >= int(max_images):
            break

        split_name = normalize_split_name(source_record.get("split")) or "train"
        source_category = source_record.get("source_category") or "unspecified"
        relative_stem = Path(source_record["image_id"])
        relative_image_path = Path("images") / split_name / source_category / f"{relative_stem.name}.jpg"

        with Image.open(source_record["image_path"]) as image:
            save_image_as_jpeg(
                image=image,
                output_path=processed_root_dir / relative_image_path,
                quality=jpeg_quality,
            )

        processed_record = {
            "dataset_name": report["dataset_name"],
            "image_id": str(relative_image_path.with_suffix("")).replace("\\", "/"),
            "image_path": str(relative_image_path).replace("\\", "/"),
            "split": split_name,
            "source_category": source_category,
            "source_image_id": source_record["image_id"],
            "width": int(source_record["width"]),
            "height": int(source_record["height"]),
            "annotations": deepcopy(source_record.get("annotations", [])),
            "is_empty": bool(source_record.get("is_empty", False)),
            "empty_reason": source_record.get("empty_reason"),
            "issues": list(source_record.get("issues", [])),
            "num_invalid_boxes": int(source_record.get("num_invalid_boxes", 0)),
            "num_clipped_boxes": int(source_record.get("num_clipped_boxes", 0)),
            "annotation_path": None,
            "semantic_map_path": None,
        }
        processed_records.append(processed_record)

    summary, class_distribution = build_processed_summary(
        dataset_name=report["dataset_name"],
        source_records=source_records[: len(source_records) if max_images is None else int(max_images)],
        processed_records=processed_records,
        processed_root_dir=processed_root_dir,
        preprocess_config=config,
    )
    artifacts = export_processed_dataset(
        dataset_name=report["dataset_name"],
        processed_root_dir=processed_root_dir,
        processed_records=processed_records,
        summary=summary,
        class_distribution=class_distribution,
        repo_output_dir=repo_output_dir,
    )

    return {
        "source_records": source_records,
        "processed_records": processed_records,
        "summary": summary,
        "class_distribution": class_distribution,
        "artifacts": artifacts,
    }
