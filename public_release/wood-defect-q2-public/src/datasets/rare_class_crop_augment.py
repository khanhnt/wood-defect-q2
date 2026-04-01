"""Offline rare-class crop augmentation for processed detection tile manifests."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from src.datasets.base_dataset import build_annotation, normalize_class_name, tag_small_defects
from src.datasets.screened_benchmark import load_jsonl_records
from src.datasets.server_preprocessing import (
    build_processed_summary,
    export_processed_dataset,
    save_image_as_jpeg,
)
from src.utils.config import expand_path


DEFAULT_RARE_CLASS_TARGETS = [
    "resin",
    "knot_with_crack",
    "crack",
    "marrow",
    "quartzity",
    "knot_missing",
]

DEFAULT_CLASS_CROP_PROFILES: dict[str, dict[str, float]] = {
    "resin": {"scale": 2.2, "min_side": 320.0, "max_side": 640.0},
    "knot_with_crack": {"scale": 3.0, "min_side": 512.0, "max_side": 768.0},
    "crack": {"scale": 3.2, "min_side": 512.0, "max_side": 768.0},
    "marrow": {"scale": 2.0, "min_side": 384.0, "max_side": 640.0},
    "quartzity": {"scale": 2.2, "min_side": 384.0, "max_side": 640.0},
    "knot_missing": {"scale": 2.8, "min_side": 448.0, "max_side": 704.0},
}

DEFAULT_SMALL_DEFECT_RULE = {
    "enabled": True,
    "combine": "any",
    "min_area_ratio": 0.01,
    "min_width_px": 16.0,
    "min_height_px": 16.0,
}


@dataclass(frozen=True)
class CropCandidate:
    """A candidate crop proposal for one clustered target-class region."""

    class_name: str
    cluster_index: int
    member_indices: tuple[int, ...]
    window_xyxy: tuple[int, int, int, int]
    primary_box_count: int
    parent_annotation_count: int
    parent_present_class_count: int
    parent_head_annotation_count: int
    window_annotation_count: int
    window_present_class_count: int
    window_head_annotation_count: int


@dataclass(frozen=True)
class CropPlan:
    """A crop candidate plus the parent record index it comes from."""

    record_index: int
    candidate: CropCandidate


def _resolve_image_path(record: Mapping[str, Any], image_root_dir: str | Path | None) -> Path:
    image_path = Path(str(record["image_path"]))
    if image_path.is_absolute():
        return image_path
    resolved_root = expand_path(image_root_dir)
    if resolved_root is None:
        raise ValueError("image_root_dir must be provided when manifest image_path values are relative.")
    return resolved_root / image_path


def _resolve_class_names(records: Sequence[Mapping[str, Any]]) -> list[str]:
    class_pairs = {
        (int(annotation.get("class_id", -1)), normalize_class_name(annotation["class_name"]))
        for record in records
        for annotation in record.get("annotations", [])
        if int(annotation.get("class_id", -1)) >= 0
    }
    if class_pairs:
        return [class_name for _, class_name in sorted(class_pairs, key=lambda item: item[0])]

    discovered = sorted(
        {
            normalize_class_name(annotation["class_name"])
            for record in records
            for annotation in record.get("annotations", [])
        }
    )
    return discovered


def _bbox_norm_to_px(annotation: Mapping[str, Any], width: float, height: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = annotation["bbox_xyxy_norm"]
    return (
        float(x1) * float(width),
        float(y1) * float(height),
        float(x2) * float(width),
        float(y2) * float(height),
    )


def _bbox_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    inter_left = max(float(box_a[0]), float(box_b[0]))
    inter_top = max(float(box_a[1]), float(box_b[1]))
    inter_right = min(float(box_a[2]), float(box_b[2]))
    inter_bottom = min(float(box_a[3]), float(box_b[3]))
    inter_area = _bbox_area((inter_left, inter_top, inter_right, inter_bottom))
    if inter_area <= 0.0:
        return 0.0
    union_area = _bbox_area(box_a) + _bbox_area(box_b) - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _center_distance(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax = (float(box_a[0]) + float(box_a[2])) * 0.5
    ay = (float(box_a[1]) + float(box_a[3])) * 0.5
    bx = (float(box_b[0]) + float(box_b[2])) * 0.5
    by = (float(box_b[1]) + float(box_b[3])) * 0.5
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _target_is_near_edge(
    box_xyxy: Sequence[float],
    width: float,
    height: float,
    edge_margin_px: float,
) -> bool:
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    return (
        x1 < edge_margin_px
        or y1 < edge_margin_px
        or x2 > float(width) - edge_margin_px
        or y2 > float(height) - edge_margin_px
    )


def _cluster_indices(
    boxes_xyxy: Sequence[Sequence[float]],
    merge_iou_threshold: float,
    merge_center_distance_px: float,
) -> list[list[int]]:
    clusters: list[list[int]] = []
    for box_index, box_xyxy in enumerate(boxes_xyxy):
        matching_clusters: list[int] = []
        for cluster_index, cluster in enumerate(clusters):
            if any(
                _bbox_iou(box_xyxy, boxes_xyxy[member_index]) > merge_iou_threshold
                or _center_distance(box_xyxy, boxes_xyxy[member_index]) < merge_center_distance_px
                for member_index in cluster
            ):
                matching_clusters.append(cluster_index)

        if not matching_clusters:
            clusters.append([box_index])
            continue

        first_cluster_index = matching_clusters[0]
        clusters[first_cluster_index].append(box_index)
        for merged_cluster_index in reversed(matching_clusters[1:]):
            clusters[first_cluster_index].extend(clusters[merged_cluster_index])
            del clusters[merged_cluster_index]

    return clusters


def _union_box(boxes_xyxy: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    return (
        min(float(box[0]) for box in boxes_xyxy),
        min(float(box[1]) for box in boxes_xyxy),
        max(float(box[2]) for box in boxes_xyxy),
        max(float(box[3]) for box in boxes_xyxy),
    )


def _build_square_window(
    center_x: float,
    center_y: float,
    requested_side: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    side = int(round(min(float(requested_side), float(image_width), float(image_height))))
    side = max(side, 1)

    left = int(round(center_x - (side / 2.0)))
    top = int(round(center_y - (side / 2.0)))
    max_left = max(int(image_width) - side, 0)
    max_top = max(int(image_height) - side, 0)
    left = min(max(left, 0), max_left)
    top = min(max(top, 0), max_top)
    right = left + side
    bottom = top + side
    return left, top, right, bottom


def _build_crop_candidates(
    record: Mapping[str, Any],
    target_classes: Sequence[str],
    class_crop_profiles: Mapping[str, Mapping[str, float]],
    head_classes: Sequence[str],
    edge_margin_px: float,
    merge_iou_threshold: float,
    merge_center_distance_px: float,
    max_crops_per_record: int,
    max_window_iou: float,
    rejection_counter: Counter[str],
) -> list[CropCandidate]:
    width = int(record["width"])
    height = int(record["height"])
    normalized_targets = [normalize_class_name(name) for name in target_classes]
    normalized_head_classes = {normalize_class_name(name) for name in head_classes}
    annotations = list(record.get("annotations", []))
    candidates: list[CropCandidate] = []
    parent_present_classes = {
        normalize_class_name(annotation["class_name"])
        for annotation in annotations
    }
    parent_head_annotation_count = sum(
        1 for annotation in annotations if normalize_class_name(annotation["class_name"]) in normalized_head_classes
    )

    for class_name in normalized_targets:
        matching_indices: list[int] = []
        matching_boxes: list[tuple[float, float, float, float]] = []
        for annotation_index, annotation in enumerate(annotations):
            annotation_class = normalize_class_name(annotation["class_name"])
            if annotation_class != class_name:
                continue
            box_xyxy = _bbox_norm_to_px(annotation, width=width, height=height)
            if _target_is_near_edge(
                box_xyxy=box_xyxy,
                width=width,
                height=height,
                edge_margin_px=edge_margin_px,
            ):
                rejection_counter["target_near_tile_edge"] += 1
                continue
            matching_indices.append(annotation_index)
            matching_boxes.append(box_xyxy)

        if not matching_boxes:
            continue

        cluster_groups = _cluster_indices(
            boxes_xyxy=matching_boxes,
            merge_iou_threshold=merge_iou_threshold,
            merge_center_distance_px=merge_center_distance_px,
        )
        cluster_groups = sorted(
            cluster_groups,
            key=lambda cluster: (
                -len(cluster),
                -_bbox_area(_union_box([matching_boxes[index] for index in cluster])),
            ),
        )

        profile = dict(class_crop_profiles[class_name])
        for cluster_index, cluster in enumerate(cluster_groups):
            union_xyxy = _union_box([matching_boxes[index] for index in cluster])
            union_width = float(union_xyxy[2]) - float(union_xyxy[0])
            union_height = float(union_xyxy[3]) - float(union_xyxy[1])
            base_side = max(union_width, union_height)
            requested_side = max(
                float(profile["min_side"]),
                min(float(profile["max_side"]), base_side * float(profile["scale"])),
            )
            window_xyxy = _build_square_window(
                center_x=(float(union_xyxy[0]) + float(union_xyxy[2])) * 0.5,
                center_y=(float(union_xyxy[1]) + float(union_xyxy[3])) * 0.5,
                requested_side=requested_side,
                image_width=width,
                image_height=height,
            )

            window_present_classes = set()
            window_annotation_count = 0
            window_head_annotation_count = 0
            for annotation in annotations:
                clipped_box, retained_ratio = _clip_annotation_to_window(
                    annotation=annotation,
                    record_width=width,
                    record_height=height,
                    window_xyxy=window_xyxy,
                )
                if clipped_box is None or retained_ratio <= 0.0:
                    continue
                annotation_class = normalize_class_name(annotation["class_name"])
                window_present_classes.add(annotation_class)
                window_annotation_count += 1
                if annotation_class in normalized_head_classes:
                    window_head_annotation_count += 1

            if min(
                float(window_xyxy[2] - window_xyxy[0]),
                float(window_xyxy[3] - window_xyxy[1]),
            ) < min(float(profile["min_side"]), float(width), float(height)):
                rejection_counter["window_below_min_side"] += 1
                continue

            if any(
                _bbox_iou(window_xyxy, existing.window_xyxy) > max_window_iou
                for existing in candidates
            ):
                rejection_counter["window_overlap_rejected"] += 1
                continue

            candidates.append(
                CropCandidate(
                    class_name=class_name,
                    cluster_index=cluster_index,
                    member_indices=tuple(matching_indices[index] for index in cluster),
                    window_xyxy=window_xyxy,
                    primary_box_count=len(cluster),
                    parent_annotation_count=len(annotations),
                    parent_present_class_count=len(parent_present_classes),
                    parent_head_annotation_count=parent_head_annotation_count,
                    window_annotation_count=window_annotation_count,
                    window_present_class_count=len(window_present_classes),
                    window_head_annotation_count=window_head_annotation_count,
                )
            )
            if len(candidates) >= int(max_crops_per_record):
                rejection_counter["max_crops_per_record_reached"] += 1
                return candidates

    return candidates


def _crop_plan_quality_key(plan: CropPlan) -> tuple[float, ...]:
    candidate = plan.candidate
    return (
        float(candidate.window_present_class_count),
        float(candidate.window_head_annotation_count),
        float(candidate.parent_present_class_count),
        float(candidate.parent_annotation_count),
        -float(candidate.primary_box_count),
        float(candidate.window_annotation_count),
        float(plan.record_index),
        float(candidate.cluster_index),
    )


def _order_crop_plans(
    crop_plans: Sequence[CropPlan],
    target_classes: Sequence[str],
    candidate_selection_mode: str,
) -> list[CropPlan]:
    normalized_mode = str(candidate_selection_mode).lower()
    if normalized_mode == "manifest":
        return list(crop_plans)

    if normalized_mode != "balanced":
        raise ValueError(
            f"Unsupported candidate_selection_mode={candidate_selection_mode!r}. Expected 'manifest' or 'balanced'."
        )

    normalized_targets = [normalize_class_name(name) for name in target_classes]
    plans_by_class: dict[str, list[CropPlan]] = {class_name: [] for class_name in normalized_targets}
    for crop_plan in crop_plans:
        plans_by_class.setdefault(crop_plan.candidate.class_name, []).append(crop_plan)

    for class_name in plans_by_class:
        plans_by_class[class_name].sort(key=_crop_plan_quality_key)

    ordered: list[CropPlan] = []
    remaining = True
    while remaining:
        remaining = False
        for class_name in normalized_targets:
            queue = plans_by_class.get(class_name, [])
            if queue:
                ordered.append(queue.pop(0))
                remaining = True
    return ordered


def _clip_annotation_to_window(
    annotation: Mapping[str, Any],
    record_width: int,
    record_height: int,
    window_xyxy: Sequence[int],
) -> tuple[tuple[float, float, float, float] | None, float]:
    original_box = _bbox_norm_to_px(annotation, width=record_width, height=record_height)
    crop_left, crop_top, crop_right, crop_bottom = [float(value) for value in window_xyxy]
    inter_left = max(float(original_box[0]), crop_left)
    inter_top = max(float(original_box[1]), crop_top)
    inter_right = min(float(original_box[2]), crop_right)
    inter_bottom = min(float(original_box[3]), crop_bottom)
    if inter_right <= inter_left or inter_bottom <= inter_top:
        return None, 0.0

    original_area = _bbox_area(original_box)
    clipped_box = (inter_left, inter_top, inter_right, inter_bottom)
    retained_ratio = 0.0 if original_area <= 0.0 else (_bbox_area(clipped_box) / original_area)
    return clipped_box, retained_ratio


def _extract_relative_output_parts(record: Mapping[str, Any], split_name: str) -> list[str]:
    image_path = Path(str(record["image_path"]))
    parts = list(image_path.parts)
    if "images" in parts:
        image_root_index = parts.index("images")
        tail = parts[image_root_index + 2 : -1]
        return [part for part in tail if part not in {"", "."}]

    image_id = Path(str(record["image_id"]))
    tail_parts = [part for part in image_id.parts[-2:-1] if part not in {"", "."}]
    if tail_parts:
        return tail_parts
    return [str(split_name)]


def _build_augmented_record(
    record: Mapping[str, Any],
    image_root_dir: str | Path,
    output_root_dir: str | Path,
    dataset_name: str,
    candidate: CropCandidate,
    record_crop_index: int,
    min_retained_ratio_target: float,
    min_retained_ratio_context: float,
    min_box_size_px: float,
    target_border_margin_px: float,
    jpeg_quality: int,
    small_defect_rule: Mapping[str, Any],
    rejection_counter: Counter[str],
) -> dict[str, Any] | None:
    source_image_path = _resolve_image_path(record, image_root_dir=image_root_dir)
    crop_left, crop_top, crop_right, crop_bottom = candidate.window_xyxy
    crop_width = int(crop_right - crop_left)
    crop_height = int(crop_bottom - crop_top)
    if crop_width <= 0 or crop_height <= 0:
        rejection_counter["invalid_window_geometry"] += 1
        return None

    rebuilt_annotations: list[dict[str, Any]] = []
    kept_primary_member_indices: set[int] = set()
    record_width = int(record["width"])
    record_height = int(record["height"])

    for annotation_index, annotation in enumerate(record.get("annotations", [])):
        clipped_box, retained_ratio = _clip_annotation_to_window(
            annotation=annotation,
            record_width=record_width,
            record_height=record_height,
            window_xyxy=candidate.window_xyxy,
        )
        if clipped_box is None:
            continue

        annotation_class = normalize_class_name(annotation["class_name"])
        required_ratio = (
            float(min_retained_ratio_target)
            if annotation_class == candidate.class_name
            else float(min_retained_ratio_context)
        )
        if retained_ratio < required_ratio:
            if annotation_index in candidate.member_indices:
                rejection_counter["primary_bbox_low_retention"] += 1
            continue

        new_x1 = float(clipped_box[0]) - float(crop_left)
        new_y1 = float(clipped_box[1]) - float(crop_top)
        new_x2 = float(clipped_box[2]) - float(crop_left)
        new_y2 = float(clipped_box[3]) - float(crop_top)
        new_width = new_x2 - new_x1
        new_height = new_y2 - new_y1
        if new_width < float(min_box_size_px) or new_height < float(min_box_size_px):
            if annotation_index in candidate.member_indices:
                rejection_counter["primary_bbox_below_min_size"] += 1
            continue

        rebuilt_annotation = build_annotation(
            class_name=annotation_class,
            bbox_xyxy_norm=[
                new_x1 / float(crop_width),
                new_y1 / float(crop_height),
                new_x2 / float(crop_width),
                new_y2 / float(crop_height),
            ],
            source_label=annotation.get("source_label"),
        )
        rebuilt_annotation["class_id"] = int(annotation["class_id"])
        rebuilt_annotations.append(rebuilt_annotation)

        if annotation_index in candidate.member_indices:
            touches_border = (
                new_x1 < float(target_border_margin_px)
                or new_y1 < float(target_border_margin_px)
                or new_x2 > float(crop_width) - float(target_border_margin_px)
                or new_y2 > float(crop_height) - float(target_border_margin_px)
            )
            if touches_border:
                rejection_counter["primary_bbox_touches_crop_border"] += 1
                return None
            kept_primary_member_indices.add(annotation_index)

    if len(kept_primary_member_indices) != len(candidate.member_indices):
        rejection_counter["primary_cluster_not_fully_retained"] += 1
        return None
    if not rebuilt_annotations:
        rejection_counter["empty_crop_after_filtering"] += 1
        return None

    source_image = Image.open(source_image_path).convert("RGB")
    try:
        crop_image = source_image.crop((crop_left, crop_top, crop_right, crop_bottom))
    finally:
        source_image.close()

    split_name = str(record.get("split") or "train").lower()
    relative_tail = _extract_relative_output_parts(record, split_name=split_name)
    parent_stem = Path(str(record["image_id"])).name
    crop_stem = (
        f"{parent_stem}__aug_{candidate.class_name}_c{record_crop_index:02d}"
        f"_x{crop_left:04d}_y{crop_top:04d}_s{crop_width:04d}"
    )
    relative_output_path = Path("images") / f"{split_name}_aug"
    for part in relative_tail:
        relative_output_path /= part
    relative_output_path /= f"{crop_stem}.jpg"

    output_root = Path(output_root_dir)
    absolute_output_path = output_root / relative_output_path
    save_image_as_jpeg(crop_image, absolute_output_path, quality=jpeg_quality)

    parent_tile_origin = record.get("tile_origin_xy") or [0, 0]
    if not isinstance(parent_tile_origin, Sequence) or len(parent_tile_origin) != 2:
        parent_tile_origin = [0, 0]
    crop_tile_origin = [
        int(parent_tile_origin[0]) + int(crop_left),
        int(parent_tile_origin[1]) + int(crop_top),
    ]

    augmented_record = {
        "dataset_name": dataset_name,
        "image_id": str(relative_output_path.with_suffix("")).replace("\\", "/"),
        "image_path": str(absolute_output_path),
        "split": split_name,
        "source_category": record.get("source_category"),
        "source_image_id": record.get("source_image_id"),
        "width": crop_width,
        "height": crop_height,
        "annotations": rebuilt_annotations,
        "is_empty": False,
        "empty_reason": None,
        "issues": [],
        "num_invalid_boxes": 0,
        "num_clipped_boxes": 0,
        "annotation_path": record.get("annotation_path"),
        "semantic_map_path": record.get("semantic_map_path"),
        "tile_origin_xy": crop_tile_origin,
        "tile_index": None,
        "parent_image_id": record.get("image_id"),
        "augmentation_type": "rare_class_crop",
        "augmentation_primary_class": candidate.class_name,
        "crop_xyxy_parent": [int(crop_left), int(crop_top), int(crop_right), int(crop_bottom)],
        "augmentation_parent_annotation_count": int(candidate.parent_annotation_count),
        "augmentation_parent_present_class_count": int(candidate.parent_present_class_count),
        "augmentation_parent_head_annotation_count": int(candidate.parent_head_annotation_count),
        "augmentation_window_annotation_count": int(candidate.window_annotation_count),
        "augmentation_window_present_class_count": int(candidate.window_present_class_count),
        "augmentation_window_head_annotation_count": int(candidate.window_head_annotation_count),
    }
    tag_small_defects([augmented_record], rule_config=small_defect_rule)
    return augmented_record


def build_rare_class_crop_augmented_dataset(
    input_manifest_path: str | Path,
    image_root_dir: str | Path,
    output_root_dir: str | Path,
    dataset_name: str,
    target_classes: Sequence[str] | None = None,
    class_crop_profiles: Mapping[str, Mapping[str, float]] | None = None,
    max_crops_per_record: int = 2,
    edge_margin_px: float = 24.0,
    merge_iou_threshold: float = 0.1,
    merge_center_distance_px: float = 96.0,
    min_retained_ratio_target: float = 0.9,
    min_retained_ratio_context: float = 0.75,
    min_box_size_px: float = 8.0,
    target_border_margin_px: float = 12.0,
    max_window_iou: float = 0.6,
    jpeg_quality: int = 97,
    small_defect_rule: Mapping[str, Any] | None = None,
    class_max_crops: Mapping[str, int] | None = None,
    candidate_selection_mode: str = "manifest",
    head_classes: Sequence[str] | None = None,
    max_window_head_annotation_count: int | None = None,
    repo_output_dir: str | Path = "outputs/tables",
) -> dict[str, Any]:
    """Append rare-class crop records to a processed manifest and export a new manifest."""
    input_manifest = Path(input_manifest_path)
    output_root = Path(output_root_dir)
    target_classes = [normalize_class_name(name) for name in (target_classes or DEFAULT_RARE_CLASS_TARGETS)]
    head_classes = [normalize_class_name(name) for name in (head_classes or ["live_knot", "dead_knot"])]
    class_crop_profiles = {
        normalize_class_name(class_name): {
            "scale": float(profile["scale"]),
            "min_side": float(profile["min_side"]),
            "max_side": float(profile["max_side"]),
        }
        for class_name, profile in dict(class_crop_profiles or DEFAULT_CLASS_CROP_PROFILES).items()
    }
    class_max_crops = {
        normalize_class_name(class_name): int(limit)
        for class_name, limit in dict(class_max_crops or {}).items()
    }
    for class_name in target_classes:
        if class_name not in class_crop_profiles:
            raise ValueError(f"Missing crop profile for target class: {class_name}")

    records = load_jsonl_records(input_manifest)
    normalized_original_records: list[dict[str, Any]] = []
    augmented_records: list[dict[str, Any]] = []
    generated_by_class: Counter[str] = Counter()
    rejection_counter: Counter[str] = Counter()
    accepted_parent_present_class_count: Counter[int] = Counter()
    accepted_window_present_class_count: Counter[int] = Counter()
    accepted_window_head_annotation_count: Counter[int] = Counter()
    small_defect_rule = dict(DEFAULT_SMALL_DEFECT_RULE | dict(small_defect_rule or {}))
    crop_plans: list[CropPlan] = []

    for record_index, record in enumerate(records):
        normalized_record = deepcopy(dict(record))
        normalized_record["dataset_name"] = dataset_name
        normalized_record["image_path"] = str(_resolve_image_path(record, image_root_dir=image_root_dir))
        normalized_original_records.append(normalized_record)

        split_name = str(record.get("split") or "").lower()
        if split_name != "train":
            continue
        if record.get("augmentation_type"):
            rejection_counter["skip_already_augmented"] += 1
            continue
        if not record.get("annotations"):
            rejection_counter["skip_negative_record"] += 1
            continue

        candidates = _build_crop_candidates(
            record=record,
            target_classes=target_classes,
            class_crop_profiles=class_crop_profiles,
            head_classes=head_classes,
            edge_margin_px=edge_margin_px,
            merge_iou_threshold=merge_iou_threshold,
            merge_center_distance_px=merge_center_distance_px,
            max_crops_per_record=max_crops_per_record,
            max_window_iou=max_window_iou,
            rejection_counter=rejection_counter,
        )
        for candidate in candidates:
            crop_plans.append(
                CropPlan(
                    record_index=record_index,
                    candidate=candidate,
                )
            )

    ordered_crop_plans = _order_crop_plans(
        crop_plans=crop_plans,
        target_classes=target_classes,
        candidate_selection_mode=candidate_selection_mode,
    )
    generated_per_record: Counter[int] = Counter()

    for crop_plan in ordered_crop_plans:
        record = records[crop_plan.record_index]
        candidate = crop_plan.candidate
        if generated_per_record[crop_plan.record_index] >= int(max_crops_per_record):
            rejection_counter["record_cap_reached_post_ordering"] += 1
            continue
        if (
            max_window_head_annotation_count is not None
            and int(candidate.window_head_annotation_count) > int(max_window_head_annotation_count)
        ):
            rejection_counter["window_head_context_rejected"] += 1
            continue

        class_cap = class_max_crops.get(candidate.class_name)
        if class_cap is not None and generated_by_class[candidate.class_name] >= class_cap:
            rejection_counter[f"class_cap_reached::{candidate.class_name}"] += 1
            continue

        augmented_record = _build_augmented_record(
            record=record,
            image_root_dir=image_root_dir,
            output_root_dir=output_root,
            dataset_name=dataset_name,
            candidate=candidate,
            record_crop_index=generated_per_record[crop_plan.record_index],
            min_retained_ratio_target=min_retained_ratio_target,
            min_retained_ratio_context=min_retained_ratio_context,
            min_box_size_px=min_box_size_px,
            target_border_margin_px=target_border_margin_px,
            jpeg_quality=jpeg_quality,
            small_defect_rule=small_defect_rule,
            rejection_counter=rejection_counter,
        )
        if augmented_record is None:
            continue

        augmented_records.append(augmented_record)
        generated_by_class[candidate.class_name] += 1
        generated_per_record[crop_plan.record_index] += 1
        accepted_parent_present_class_count[int(candidate.parent_present_class_count)] += 1
        accepted_window_present_class_count[int(candidate.window_present_class_count)] += 1
        accepted_window_head_annotation_count[int(candidate.window_head_annotation_count)] += 1

    combined_records = normalized_original_records + augmented_records
    class_names = _resolve_class_names(combined_records)
    summary, class_distribution = build_processed_summary(
        dataset_name=dataset_name,
        source_records=normalized_original_records,
        processed_records=combined_records,
        processed_root_dir=output_root,
        preprocess_config={
            "classes": class_names,
            "augmentation_type": "rare_class_crop",
            "target_classes": target_classes,
        },
    )
    summary.update(
        {
            "input_manifest_path": str(input_manifest),
            "image_root_dir": str(expand_path(image_root_dir)),
            "output_root_dir": str(output_root),
            "num_original_records": len(normalized_original_records),
            "num_augmented_records": len(augmented_records),
            "augmentation": {
                "target_classes": target_classes,
                "generated_by_class": dict(sorted(generated_by_class.items())),
                "rejection_counts": dict(sorted(rejection_counter.items())),
                "class_crop_profiles": class_crop_profiles,
                "class_max_crops": class_max_crops,
                "candidate_selection_mode": str(candidate_selection_mode),
                "head_classes": list(head_classes),
                "max_window_head_annotation_count": None
                if max_window_head_annotation_count is None
                else int(max_window_head_annotation_count),
                "max_crops_per_record": int(max_crops_per_record),
                "edge_margin_px": float(edge_margin_px),
                "merge_iou_threshold": float(merge_iou_threshold),
                "merge_center_distance_px": float(merge_center_distance_px),
                "min_retained_ratio_target": float(min_retained_ratio_target),
                "min_retained_ratio_context": float(min_retained_ratio_context),
                "min_box_size_px": float(min_box_size_px),
                "target_border_margin_px": float(target_border_margin_px),
                "max_window_iou": float(max_window_iou),
                "jpeg_quality": int(jpeg_quality),
                "small_defect_rule": dict(small_defect_rule),
                "accepted_parent_present_class_count": {
                    str(key): int(value) for key, value in sorted(accepted_parent_present_class_count.items())
                },
                "accepted_window_present_class_count": {
                    str(key): int(value) for key, value in sorted(accepted_window_present_class_count.items())
                },
                "accepted_window_head_annotation_count": {
                    str(key): int(value) for key, value in sorted(accepted_window_head_annotation_count.items())
                },
            },
        }
    )
    artifacts = export_processed_dataset(
        dataset_name=dataset_name,
        processed_root_dir=output_root,
        processed_records=combined_records,
        summary=summary,
        class_distribution=class_distribution,
        repo_output_dir=repo_output_dir,
    )
    return {
        "manifest_path": artifacts["manifest_path"],
        "metadata_path": artifacts["metadata_path"],
        "repo_summary_path": artifacts["repo_summary_path"],
        "repo_class_distribution_path": artifacts["repo_class_distribution_path"],
        "summary": summary,
    }
