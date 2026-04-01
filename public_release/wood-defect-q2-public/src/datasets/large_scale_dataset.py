"""Parser and lightweight dataset wrapper for the large-scale wood defect dataset."""

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
    natural_sort_key,
    read_image_size,
    resolve_optional_path,
)
from src.datasets.server_preprocessing import (
    assign_splits_by_source_image,
    build_processed_summary,
    choose_negative_tiles,
    export_processed_dataset,
    generate_tile_windows,
    remap_annotations_to_tile,
    save_image_as_jpeg,
)
from src.utils.config import expand_path


DEFAULT_DATASET_NAME = "large_scale_wood_surface_defects"
DEFAULT_ANNOTATION_DIR = "Bouding Boxes"
DEFAULT_SEMANTIC_DIR = "Semantic Maps"


class LargeScaleWoodDefectDataset(BaseWoodDefectDataset):
    """Record-backed dataset for the large-scale wood surface defects dataset."""

    def load_records(self) -> Sequence[Dict[str, Any]]:
        return load_large_scale_annotations(self.config)


def _resolve_root_dir(config: Dict[str, Any]) -> Path:
    root_value = config.get("root_dir")
    if not root_value:
        raise ValueError(
            "Large-scale dataset root_dir is required. Pass it via config or "
            "'python scripts/prepare_large_scale.py --dataset-root /path/to/dataset'."
        )

    root_dir = expand_path(root_value)
    if root_dir is None:
        raise ValueError("Large-scale dataset root_dir is empty.")
    if not root_dir.exists():
        raise FileNotFoundError(
            f"Large-scale dataset root does not exist: {root_dir}. "
            "Pass the extracted dataset root via '--dataset-root' or '--root-dir'."
        )

    extracted_dir = root_dir / "extracted"
    has_images = any(path.is_dir() and path.name.startswith("Images") for path in root_dir.iterdir())
    if not has_images and extracted_dir.exists():
        root_dir = extracted_dir

    return root_dir


def _resolve_image_dirs(root_dir: Path, config: Dict[str, Any]) -> list[Path]:
    image_dir = resolve_optional_path(root_dir, config.get("image_dir"))
    search_root = image_dir if image_dir is not None else root_dir

    if search_root.is_dir() and search_root.name.startswith("Images"):
        return [search_root]

    image_dirs = [
        path
        for path in search_root.iterdir()
        if path.is_dir() and path.name.startswith("Images")
    ]
    image_dirs = sorted(image_dirs, key=lambda path: natural_sort_key(path.name))
    if not image_dirs:
        raise FileNotFoundError(
            f"No extracted Images* directories found under: {search_root}. "
            "Pass the extracted dataset root that contains Images1..Images10."
        )
    return image_dirs


def _parse_annotation_file(annotation_path: Path) -> tuple[list[Dict[str, Any]], int, int, str | None]:
    annotations: list[Dict[str, Any]] = []
    invalid_boxes = 0
    clipped_boxes = 0

    raw_text = annotation_path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return annotations, invalid_boxes, clipped_boxes, "empty_annotation_file"

    for line in lines:
        parts = line.replace(",", ".").split()
        if len(parts) != 5:
            invalid_boxes += 1
            continue

        class_name = parts[0]
        try:
            bbox_values = [float(value) for value in parts[1:]]
        except ValueError:
            invalid_boxes += 1
            continue

        bbox_xyxy_norm, issues = clip_and_validate_bbox_xyxy(bbox_values)
        if "clipped_box" in issues:
            clipped_boxes += 1
        if bbox_xyxy_norm is None:
            invalid_boxes += 1
            continue

        annotations.append(
            build_annotation(
                class_name=class_name,
                bbox_xyxy_norm=bbox_xyxy_norm,
                source_label=class_name,
            )
        )

    empty_reason = None if annotations else "invalid_annotations_only"
    return annotations, invalid_boxes, clipped_boxes, empty_reason


def parse_large_scale_dataset(config: Dict[str, Any]) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Parse the dataset into a unified internal manifest."""
    dataset_name = config.get("dataset_name", DEFAULT_DATASET_NAME)
    root_dir = _resolve_root_dir(config)
    annotation_dir = resolve_optional_path(root_dir, config.get("annotation_path")) or (root_dir / DEFAULT_ANNOTATION_DIR)
    use_semantic_maps = bool(config.get("use_semantic_maps", True))
    semantic_dir = None
    if use_semantic_maps:
        semantic_dir = resolve_optional_path(root_dir, config.get("semantic_map_path")) or (root_dir / DEFAULT_SEMANTIC_DIR)
    image_dirs = _resolve_image_dirs(root_dir, config)

    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory does not exist: {annotation_dir}")

    image_map: dict[str, Path] = {}
    for image_dir in image_dirs:
        for image_path in sorted(image_dir.glob("*.bmp")):
            image_map[image_path.stem] = image_path

    annotation_map = {
        path.name.replace("_anno.txt", ""): path
        for path in sorted(annotation_dir.glob("*_anno.txt"))
    }

    semantic_map = {}
    if semantic_dir is not None and semantic_dir.exists():
        semantic_map = {
            path.name.replace("_segm.bmp", ""): path
            for path in sorted(semantic_dir.glob("*_segm.bmp"))
        }

    image_ids = sorted(image_map.keys(), key=natural_sort_key)
    max_images = config.get("max_images")
    if max_images:
        image_ids = image_ids[: int(max_images)]

    validation_counts: Counter[str] = Counter()
    if max_images is None:
        selected_image_ids = set(image_ids)
        validation_counts["orphan_annotation_files"] = sum(
            image_id not in selected_image_ids for image_id in annotation_map
        )
        if semantic_dir is not None and semantic_dir.exists():
            validation_counts["orphan_semantic_map_files"] = sum(
                image_id not in selected_image_ids for image_id in semantic_map
            )

    records: list[Dict[str, Any]] = []
    for image_id in image_ids:
        image_path = image_map[image_id]
        width, height = read_image_size(image_path)
        annotation_path = annotation_map.get(image_id)
        semantic_path = semantic_map.get(image_id)

        issues: list[str] = []
        annotations: list[Dict[str, Any]] = []
        invalid_boxes = 0
        clipped_boxes = 0
        empty_reason: str | None = None

        if annotation_path is None:
            empty_reason = "missing_annotation_file"
        else:
            annotations, invalid_boxes, clipped_boxes, empty_reason = _parse_annotation_file(annotation_path)

        if semantic_dir is not None and semantic_dir.exists() and semantic_path is None:
            issues.append("missing_semantic_map_file")

        record = {
            "dataset_name": dataset_name,
            "image_id": str(image_path.relative_to(root_dir).with_suffix("")).replace("\\", "/"),
            "image_path": str(image_path),
            "split": None,
            "source_category": None,
            "width": width,
            "height": height,
            "annotations": annotations,
            "is_empty": empty_reason is not None,
            "empty_reason": empty_reason,
            "issues": issues,
            "num_invalid_boxes": invalid_boxes,
            "num_clipped_boxes": clipped_boxes,
            "semantic_map_path": str(semantic_path) if semantic_path is not None else None,
            "annotation_path": str(annotation_path) if annotation_path is not None else None,
        }
        records.append(record)

    class_to_idx = assign_class_ids(records, preferred_class_names=config.get("classes"))
    report = {
        "dataset_name": dataset_name,
        "root_dir": str(root_dir),
        "annotation_dir": str(annotation_dir),
        "semantic_dir": str(semantic_dir) if semantic_dir is not None and semantic_dir.exists() else None,
        "class_to_idx": class_to_idx,
        "validation_counts": dict(validation_counts),
    }
    return records, report


def load_large_scale_annotations(config: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Load raw annotations and convert them to the unified internal format."""
    records, _ = parse_large_scale_dataset(config)
    return records


def preprocess_large_scale_for_server(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a compact tiled JPEG dataset for server-side training."""
    config = dict(config)
    config["use_semantic_maps"] = False
    source_records, report = parse_large_scale_dataset(config)
    source_records = assign_splits_by_source_image(
        source_records,
        split_config=config.get("split"),
        preserve_existing=False,
    )

    processed_root_dir = expand_path(config.get("processed_root_dir"))
    if processed_root_dir is None:
        raise ValueError("processed_root_dir is required for preprocessing.")

    jpeg_quality = int(config.get("jpeg_quality", 97))
    tile_cfg = dict(config.get("tile", {}))
    tile_size = int(tile_cfg.get("size", 1024))
    tile_overlap = int(tile_cfg.get("overlap", 128))
    min_box_visibility = float(tile_cfg.get("min_box_visibility", 0.5))
    negative_cfg = dict(config.get("negative_sampling", {}))
    repo_output_dir = config.get("repo_output_dir", "outputs/tables")
    max_images = config.get("max_images")

    processed_records: list[Dict[str, Any]] = []

    for record_index, source_record in enumerate(source_records):
        if max_images is not None and record_index >= int(max_images):
            break

        source_width = int(source_record["width"])
        source_height = int(source_record["height"])
        split_name = str(source_record.get("split") or "train")
        source_image_id = source_record["image_id"]
        source_stem = Path(source_image_id).name
        source_folder = Path(source_image_id).parent.as_posix()

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
            relative_image_path = Path("images") / split_name
            if source_folder and source_folder != ".":
                relative_image_path = relative_image_path / source_folder
            relative_image_path = relative_image_path / f"{tile_name}.jpg"

            tile_record = {
                "dataset_name": report["dataset_name"],
                "image_id": str(relative_image_path.with_suffix("")).replace("\\", "/"),
                "image_path": str(relative_image_path).replace("\\", "/"),
                "split": split_name,
                "source_category": source_record.get("source_category"),
                "source_image_id": source_image_id,
                "width": int(window["width"]),
                "height": int(window["height"]),
                "annotations": remapped_annotations,
                "is_empty": len(remapped_annotations) == 0,
                "empty_reason": None if remapped_annotations else "negative_tile",
                "issues": [],
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
            else:
                negative_tile_entries.append(tile_entry)

        tile_entries.extend(
            choose_negative_tiles(
                negative_tiles=negative_tile_entries,
                num_positive_tiles=positive_tile_count,
                source_image_id=source_image_id,
                negative_config=negative_cfg,
            )
        )

        if not tile_entries:
            continue

        with Image.open(source_record["image_path"]) as source_image:
            for tile_entry in tile_entries:
                window = tile_entry["window"]
                tile_image = source_image.crop(
                    (window["left"], window["top"], window["right"], window["bottom"])
                )
                output_path = processed_root_dir / tile_entry["relative_image_path"]
                save_image_as_jpeg(tile_image, output_path, quality=jpeg_quality)
                processed_records.append(deepcopy(tile_entry["record"]))

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
