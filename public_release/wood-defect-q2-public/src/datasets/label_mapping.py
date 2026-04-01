"""Explicit label mapping helpers for cross-dataset evaluation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from src.datasets.base_dataset import normalize_class_name


def _normalize_optional_class_name(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "ignore", "ignored"}:
        return None
    return normalize_class_name(text)


def resolve_cross_dataset_label_mapping(
    source_class_names: Sequence[str],
    target_class_names: Sequence[str],
    label_mapping: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Resolve an explicit target-to-source class mapping for overlap-only evaluation."""
    normalized_source = [normalize_class_name(name) for name in source_class_names]
    normalized_target = [normalize_class_name(name) for name in target_class_names]
    normalized_mapping = {
        normalize_class_name(key): _normalize_optional_class_name(value)
        for key, value in dict(label_mapping or {}).items()
    }

    source_index = {name: index for index, name in enumerate(normalized_source)}
    mapping_rows: list[Dict[str, Any]] = []
    target_to_mapped: dict[int, int] = {}
    source_to_mapped: dict[int, int] = {}
    mapped_target_names: list[str] = []
    used_source_names: set[str] = set()

    for target_id, target_name in enumerate(normalized_target):
        if target_name not in normalized_mapping:
            mapping_rows.append(
                {
                    "target_class_id": target_id,
                    "target_class_name": target_name,
                    "source_class_id": None,
                    "source_class_name": None,
                    "mapped_class_id": None,
                    "mapped_class_name": None,
                    "status": "unmatched_target_class",
                }
            )
            continue

        mapped_source_name = normalized_mapping[target_name]
        if mapped_source_name is None:
            mapping_rows.append(
                {
                    "target_class_id": target_id,
                    "target_class_name": target_name,
                    "source_class_id": None,
                    "source_class_name": None,
                    "mapped_class_id": None,
                    "mapped_class_name": None,
                    "status": "ignored_target_class",
                }
            )
            continue

        source_id = source_index.get(mapped_source_name)
        if source_id is None:
            mapping_rows.append(
                {
                    "target_class_id": target_id,
                    "target_class_name": target_name,
                    "source_class_id": None,
                    "source_class_name": mapped_source_name,
                    "mapped_class_id": None,
                    "mapped_class_name": None,
                    "status": "unmatched_source_class",
                }
            )
            continue

        if source_id in source_to_mapped:
            raise ValueError(
                f"Duplicate source class mapping is not supported: {mapped_source_name}"
            )

        mapped_class_id = len(mapped_target_names)
        mapped_target_names.append(target_name)
        target_to_mapped[target_id] = mapped_class_id
        source_to_mapped[source_id] = mapped_class_id
        used_source_names.add(mapped_source_name)
        mapping_rows.append(
            {
                "target_class_id": target_id,
                "target_class_name": target_name,
                "source_class_id": source_id,
                "source_class_name": mapped_source_name,
                "mapped_class_id": mapped_class_id,
                "mapped_class_name": target_name,
                "status": "mapped",
            }
        )

    unmatched_source_names = [
        name for name in normalized_source if name not in used_source_names
    ]
    for source_name in unmatched_source_names:
        mapping_rows.append(
            {
                "target_class_id": None,
                "target_class_name": None,
                "source_class_id": source_index[source_name],
                "source_class_name": source_name,
                "mapped_class_id": None,
                "mapped_class_name": None,
                "status": "unmatched_source_class",
            }
        )

    mapping_df = pd.DataFrame(mapping_rows)
    return {
        "mapping_table": mapping_df,
        "mapped_class_names": mapped_target_names,
        "target_to_mapped": target_to_mapped,
        "source_to_mapped": source_to_mapped,
        "mapped_target_classes": mapping_df.loc[mapping_df["status"] == "mapped", "target_class_name"].tolist(),
        "ignored_target_classes": mapping_df.loc[mapping_df["status"] == "ignored_target_class", "target_class_name"].tolist(),
        "unmatched_target_classes": mapping_df.loc[mapping_df["status"] == "unmatched_target_class", "target_class_name"].tolist(),
        "unmatched_source_classes": unmatched_source_names,
    }


def _remap_single_detection_entry(
    entry: Dict[str, Any],
    label_map: Mapping[int, int],
    keep_scores: bool,
) -> tuple[Dict[str, Any], int]:
    labels = np.asarray(entry["labels"], dtype=np.int64)
    keep_mask = np.asarray([label in label_map for label in labels], dtype=bool)
    filtered = {
        "image_id": entry["image_id"],
        "boxes": np.asarray(entry["boxes"])[keep_mask],
        "labels": np.asarray([label_map[label] for label in labels[keep_mask]], dtype=np.int64),
    }
    if keep_scores:
        filtered["scores"] = np.asarray(entry["scores"])[keep_mask]
    return filtered, int((~keep_mask).sum())


def remap_predictions_and_targets_for_cross_dataset(
    predictions: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    mapping_report: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, int]]:
    """Filter and remap predictions/targets to the overlapping label space only."""
    remapped_predictions: list[Dict[str, Any]] = []
    remapped_targets: list[Dict[str, Any]] = []
    ignored_prediction_count = 0
    ignored_target_count = 0

    source_to_mapped = mapping_report["source_to_mapped"]
    target_to_mapped = mapping_report["target_to_mapped"]

    for prediction in predictions:
        filtered_prediction, ignored_count = _remap_single_detection_entry(
            entry=prediction,
            label_map=source_to_mapped,
            keep_scores=True,
        )
        remapped_predictions.append(filtered_prediction)
        ignored_prediction_count += ignored_count

    for target in targets:
        filtered_target, ignored_count = _remap_single_detection_entry(
            entry=target,
            label_map=target_to_mapped,
            keep_scores=False,
        )
        remapped_targets.append(filtered_target)
        ignored_target_count += ignored_count

    remap_counts = {
        "ignored_prediction_count": ignored_prediction_count,
        "ignored_target_annotation_count": ignored_target_count,
        "mapped_prediction_count": int(sum(len(entry["labels"]) for entry in remapped_predictions)),
        "mapped_target_annotation_count": int(sum(len(entry["labels"]) for entry in remapped_targets)),
    }
    return remapped_predictions, remapped_targets, remap_counts
