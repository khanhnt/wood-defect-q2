"""Evaluate external detector predictions against manifest-backed targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from src.datasets.base_dataset import normalize_class_name, resolve_small_defect_rule
from src.metrics.detection_metrics import compute_detection_metrics
from src.utils.io import ensure_dir, save_csv, save_json


def build_targets_from_manifest_records(
    records: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
) -> list[Dict[str, Any]]:
    """Convert manifest records into pixel-space target payloads."""
    class_to_id = {normalize_class_name(name): class_id for class_id, name in enumerate(class_names)}
    targets: list[Dict[str, Any]] = []

    for record in records:
        record_width = int(record.get("width", 0) or 0)
        record_height = int(record.get("height", 0) or 0)
        boxes = []
        labels = []
        for annotation in record.get("annotations", []):
            class_name = normalize_class_name(annotation["class_name"])
            if class_name not in class_to_id:
                continue
            x1, y1, x2, y2 = annotation["bbox_xyxy_norm"]
            boxes.append(
                [
                    float(x1) * record_width,
                    float(y1) * record_height,
                    float(x2) * record_width,
                    float(y2) * record_height,
                ]
            )
            labels.append(int(class_to_id[class_name]))

        targets.append(
            {
                "image_id": str(record["image_id"]),
                "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
            }
        )
    return targets


def _annotation_is_small(
    annotation: Dict[str, Any],
    record_width: int,
    record_height: int,
    small_defect_rule: Dict[str, Any],
) -> bool:
    if bool(annotation.get("is_small_defect", False)):
        return True

    checks: list[bool] = []
    if small_defect_rule.get("min_area_ratio") is not None:
        checks.append(float(annotation.get("bbox_area_norm", 0.0)) <= float(small_defect_rule["min_area_ratio"]))
    if small_defect_rule.get("min_width_px") is not None and record_width > 0:
        width_px = float(annotation.get("bbox_width_norm", 0.0)) * float(record_width)
        checks.append(width_px <= float(small_defect_rule["min_width_px"]))
    if small_defect_rule.get("min_height_px") is not None and record_height > 0:
        height_px = float(annotation.get("bbox_height_norm", 0.0)) * float(record_height)
        checks.append(height_px <= float(small_defect_rule["min_height_px"]))

    if not small_defect_rule["enabled"] or not checks:
        return False
    if small_defect_rule["combine"] == "all":
        return all(checks)
    return any(checks)


def build_small_defect_eval_payloads_from_records(
    records: Sequence[Dict[str, Any]],
    predictions: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    small_defect_config: Dict[str, Any] | None,
    score_threshold: float,
) -> Dict[str, Any] | None:
    """Build small-defect evaluation payloads using manifest records plus external predictions."""
    if not small_defect_config:
        return None

    small_defect_rule = resolve_small_defect_rule(small_defect_config)
    class_to_id = {normalize_class_name(name): class_id for class_id, name in enumerate(class_names)}
    prediction_by_image = {prediction["image_id"]: prediction for prediction in predictions}

    small_target_targets: list[Dict[str, Any]] = []
    small_target_predictions: list[Dict[str, Any]] = []
    small_image_targets: list[Dict[str, Any]] = []
    small_image_predictions: list[Dict[str, Any]] = []

    for record in records:
        image_id = str(record["image_id"])
        prediction = prediction_by_image.get(image_id)
        if prediction is None:
            continue

        record_width = int(record.get("width", 0) or 0)
        record_height = int(record.get("height", 0) or 0)
        all_boxes = []
        all_labels = []
        small_boxes = []
        small_labels = []

        for annotation in record.get("annotations", []):
            class_name = normalize_class_name(annotation["class_name"])
            if class_name not in class_to_id:
                continue
            x1, y1, x2, y2 = annotation["bbox_xyxy_norm"]
            box = [
                float(x1) * record_width,
                float(y1) * record_height,
                float(x2) * record_width,
                float(y2) * record_height,
            ]
            class_id = int(class_to_id[class_name])
            all_boxes.append(box)
            all_labels.append(class_id)

            if _annotation_is_small(
                annotation=annotation,
                record_width=record_width,
                record_height=record_height,
                small_defect_rule=small_defect_rule,
            ):
                small_boxes.append(box)
                small_labels.append(class_id)

        if not all_boxes:
            continue

        if small_boxes:
            small_target_targets.append(
                {
                    "image_id": image_id,
                    "boxes": torch.as_tensor(small_boxes, dtype=torch.float32).reshape(-1, 4),
                    "labels": torch.as_tensor(small_labels, dtype=torch.int64),
                }
            )
            small_target_predictions.append(prediction)

            small_image_targets.append(
                {
                    "image_id": image_id,
                    "boxes": torch.as_tensor(all_boxes, dtype=torch.float32).reshape(-1, 4),
                    "labels": torch.as_tensor(all_labels, dtype=torch.int64),
                }
            )
            small_image_predictions.append(prediction)

    if not small_target_targets:
        return None

    small_target_metrics = compute_detection_metrics(
        predictions=small_target_predictions,
        targets=small_target_targets,
        class_names=class_names,
        score_threshold=score_threshold,
    )
    small_image_metrics = compute_detection_metrics(
        predictions=small_image_predictions,
        targets=small_image_targets,
        class_names=class_names,
        score_threshold=score_threshold,
    )
    return {
        "rule": small_defect_rule,
        "small_target": small_target_metrics,
        "small_image": small_image_metrics,
    }


def save_prediction_evaluation_outputs(
    output_dir: str | Path,
    experiment_name: str,
    split_name: str,
    summary: Dict[str, Any],
    per_class: Any,
    small_defect_eval_payload: Dict[str, Any] | None = None,
) -> None:
    """Save prediction evaluation outputs matching the repo's usual tables layout."""
    tables_dir = ensure_dir(Path(output_dir) / "tables")
    summary_path = tables_dir / f"{experiment_name}_{split_name}_summary.json"
    per_class_path = tables_dir / f"{experiment_name}_{split_name}_per_class.csv"
    save_json(summary, summary_path)
    save_csv(per_class, per_class_path)

    if small_defect_eval_payload is not None:
        small_target_summary_path = tables_dir / f"{experiment_name}_{split_name}_small_target_summary.json"
        small_target_per_class_path = tables_dir / f"{experiment_name}_{split_name}_small_target_per_class.csv"
        small_image_summary_path = tables_dir / f"{experiment_name}_{split_name}_small_image_subset_summary.json"
        small_image_per_class_path = tables_dir / f"{experiment_name}_{split_name}_small_image_subset_per_class.csv"
        save_json(small_defect_eval_payload["small_target"]["summary"], small_target_summary_path)
        save_csv(small_defect_eval_payload["small_target"]["per_class"], small_target_per_class_path)
        save_json(small_defect_eval_payload["small_image"]["summary"], small_image_summary_path)
        save_csv(small_defect_eval_payload["small_image"]["per_class"], small_image_per_class_path)
