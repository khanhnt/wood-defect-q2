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
    choose_negative_tiles,
    export_processed_dataset,
    generate_tile_windows,
    normalize_split_name,
    remap_annotations_to_tile,
    save_image_as_jpeg,
)
from src.datasets.screened_benchmark import load_jsonl_records
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


def _copy_source_record_to_processed(
    *,
    dataset_name: str,
    processed_root_dir: Path,
    source_record: Dict[str, Any],
    jpeg_quality: int,
) -> Dict[str, Any]:
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

    return {
        "dataset_name": dataset_name,
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


def _tile_source_record_to_processed(
    *,
    dataset_name: str,
    processed_root_dir: Path,
    source_record: Dict[str, Any],
    jpeg_quality: int,
    tile_cfg: Dict[str, Any],
    negative_cfg: Dict[str, Any],
) -> list[Dict[str, Any]]:
    source_width = int(source_record["width"])
    source_height = int(source_record["height"])
    split_name = normalize_split_name(source_record.get("split")) or "train"
    source_category = source_record.get("source_category") or "unspecified"
    source_image_id = str(source_record["image_id"])
    source_stem = Path(source_image_id).name
    tile_size = int(tile_cfg.get("size", 1024))
    tile_overlap = int(tile_cfg.get("overlap", 128))
    min_box_visibility = float(tile_cfg.get("min_box_visibility", 0.5))
    keep_all_negative_tiles = bool(tile_cfg.get("keep_all_negative_tiles", True))

    tile_entries: list[Dict[str, Any]] = []
    negative_tile_entries: list[Dict[str, Any]] = []
    positive_tile_count = 0

    windows = generate_tile_windows(
        width=source_width,
        height=source_height,
        tile_size=tile_size,
        overlap=tile_overlap,
    )

    for tile_index, window in enumerate(windows):
        remapped_annotations = remap_annotations_to_tile(
            annotations=source_record.get("annotations", []),
            image_width=source_width,
            image_height=source_height,
            tile_window=window,
            min_visibility=min_box_visibility,
        )

        tile_name = (
            f"{source_stem}__x{window['left']:04d}_y{window['top']:04d}"
            f"_w{window['width']:04d}_h{window['height']:04d}"
        )
        relative_image_path = Path("images") / split_name / source_category / f"{tile_name}.jpg"
        tile_record = {
            "dataset_name": dataset_name,
            "image_id": str(relative_image_path.with_suffix("")).replace("\\", "/"),
            "image_path": str(relative_image_path).replace("\\", "/"),
            "split": split_name,
            "source_category": source_category,
            "source_image_id": source_image_id,
            "width": int(window["width"]),
            "height": int(window["height"]),
            "annotations": remapped_annotations,
            "is_empty": len(remapped_annotations) == 0,
            "empty_reason": None if remapped_annotations else "negative_tile",
            "issues": list(source_record.get("issues", [])),
            "num_invalid_boxes": 0,
            "num_clipped_boxes": 0,
            "annotation_path": None,
            "semantic_map_path": None,
            "tile_origin_xy": [int(window["left"]), int(window["top"])],
            "tile_index": tile_index,
        }
        tile_entry = {
            "record": tile_record,
            "window": window,
            "relative_image_path": relative_image_path,
        }
        if remapped_annotations:
            positive_tile_count += 1
            tile_entries.append(tile_entry)
        elif keep_all_negative_tiles:
            tile_entries.append(tile_entry)
        else:
            negative_tile_entries.append(tile_entry)

    if not keep_all_negative_tiles:
        tile_entries.extend(
            choose_negative_tiles(
                negative_tiles=negative_tile_entries,
                num_positive_tiles=positive_tile_count,
                source_image_id=source_image_id,
                negative_config=negative_cfg,
            )
        )

    processed_records: list[Dict[str, Any]] = []
    if not tile_entries:
        return processed_records

    with Image.open(source_record["image_path"]) as source_image:
        for tile_entry in tile_entries:
            window = tile_entry["window"]
            tile_image = source_image.crop(
                (window["left"], window["top"], window["right"], window["bottom"])
            )
            output_path = processed_root_dir / tile_entry["relative_image_path"]
            save_image_as_jpeg(tile_image, output_path, quality=jpeg_quality)
            processed_records.append(deepcopy(tile_entry["record"]))

    return processed_records


def _resolve_processed_record_image_path(
    record: Mapping[str, Any],
    *,
    image_root_dir: Path | None,
    input_manifest_path: Path,
) -> Path:
    image_path_value = record.get("image_path")
    if not image_path_value:
        raise ValueError(f"Processed record {record.get('image_id')!r} is missing image_path.")

    image_path = Path(str(image_path_value))
    if image_path.is_absolute():
        if not image_path.exists():
            raise FileNotFoundError(f"Processed image does not exist: {image_path}")
        return image_path

    candidates: list[Path] = []
    if image_root_dir is not None:
        candidates.append(image_root_dir / image_path)
    candidates.append(input_manifest_path.parent / image_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Processed image {image_path!s} could not be resolved from "
        f"image_root_dir={image_root_dir!s} or manifest dir {input_manifest_path.parent!s}."
    )


def build_tiled_vnwoodknot_from_processed_manifest(
    *,
    input_manifest_path: str | Path,
    output_root_dir: str | Path,
    image_root_dir: str | Path | None = None,
    dataset_name: str = "vnwoodknot_tiled",
    repo_output_dir: str | Path = "outputs/tables",
    tile_cfg: Mapping[str, Any] | None = None,
    negative_cfg: Mapping[str, Any] | None = None,
    jpeg_quality: int = 97,
    max_images: int | None = None,
) -> Dict[str, Any]:
    """Retile an existing processed VNWoodKnot manifest into a matched-tiling external protocol."""
    manifest_path = Path(input_manifest_path)
    processed_root_dir = Path(output_root_dir)
    resolved_image_root = Path(image_root_dir) if image_root_dir is not None else None

    source_records = load_jsonl_records(manifest_path)
    tile_cfg_resolved = dict(tile_cfg or {})
    tile_cfg_resolved.setdefault("size", 1024)
    tile_cfg_resolved.setdefault("overlap", 128)
    tile_cfg_resolved.setdefault("min_box_visibility", 0.5)
    tile_cfg_resolved.setdefault("keep_all_negative_tiles", True)
    negative_cfg_resolved = dict(negative_cfg or {})

    processed_records: list[Dict[str, Any]] = []
    limited_source_records = source_records[: len(source_records) if max_images is None else int(max_images)]

    for source_record in limited_source_records:
        source_record_copy = deepcopy(source_record)
        source_record_copy["image_path"] = str(
            _resolve_processed_record_image_path(
                source_record_copy,
                image_root_dir=resolved_image_root,
                input_manifest_path=manifest_path,
            )
        )
        source_record_copy["source_image_id"] = str(
            source_record_copy.get("source_image_id") or source_record_copy["image_id"]
        )
        processed_records.extend(
            _tile_source_record_to_processed(
                dataset_name=dataset_name,
                processed_root_dir=processed_root_dir,
                source_record=source_record_copy,
                jpeg_quality=int(jpeg_quality),
                tile_cfg=tile_cfg_resolved,
                negative_cfg=negative_cfg_resolved,
            )
        )

    summary, class_distribution = build_processed_summary(
        dataset_name=dataset_name,
        source_records=limited_source_records,
        processed_records=processed_records,
        processed_root_dir=processed_root_dir,
        preprocess_config={
            "dataset_name": dataset_name,
            "input_manifest_path": str(manifest_path),
            "image_root_dir": str(resolved_image_root) if resolved_image_root is not None else None,
            "processed_root_dir": str(processed_root_dir),
            "repo_output_dir": str(repo_output_dir),
            "jpeg_quality": int(jpeg_quality),
            "tile": tile_cfg_resolved,
            "negative_sampling": negative_cfg_resolved,
            "max_images": max_images,
        },
    )
    artifacts = export_processed_dataset(
        dataset_name=dataset_name,
        processed_root_dir=processed_root_dir,
        processed_records=processed_records,
        summary=summary,
        class_distribution=class_distribution,
        repo_output_dir=repo_output_dir,
    )

    return {
        "source_records": limited_source_records,
        "processed_records": processed_records,
        "summary": summary,
        "class_distribution": class_distribution,
        "artifacts": artifacts,
    }


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
    tile_cfg = dict(config.get("tile", {}))
    tile_enabled = bool(tile_cfg.get("enabled", False))
    negative_cfg = dict(config.get("negative_sampling", {}))
    repo_output_dir = config.get("repo_output_dir", "outputs/tables")
    max_images = config.get("max_images")
    processed_records: list[Dict[str, Any]] = []

    for record_index, source_record in enumerate(source_records):
        if max_images is not None and record_index >= int(max_images):
            break

        if tile_enabled:
            processed_records.extend(
                _tile_source_record_to_processed(
                    dataset_name=report["dataset_name"],
                    processed_root_dir=processed_root_dir,
                    source_record=source_record,
                    jpeg_quality=jpeg_quality,
                    tile_cfg=tile_cfg,
                    negative_cfg=negative_cfg,
                )
            )
        else:
            processed_records.append(
                _copy_source_record_to_processed(
                    dataset_name=report["dataset_name"],
                    processed_root_dir=processed_root_dir,
                    source_record=source_record,
                    jpeg_quality=jpeg_quality,
                )
            )

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
