"""Build class-filtered processed manifests from an existing tile manifest."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.datasets.base_dataset import natural_sort_key, normalize_class_name
from src.datasets.screened_benchmark import (
    DEFAULT_VSB7_CLASSES,
    _build_source_index,
    load_jsonl_records,
)
from src.utils.io import ensure_dir, save_json, save_jsonl


def build_class_filtered_manifest(
    input_manifest_path: str | Path,
    output_root_dir: str | Path,
    dataset_name: str,
    kept_classes: Sequence[str] | None = None,
    drop_source_images_without_kept_classes: bool = True,
) -> dict[str, Any]:
    """Filter a processed manifest to a reduced class set while preserving source-level splits."""
    normalized_classes = [normalize_class_name(name) for name in (kept_classes or DEFAULT_VSB7_CLASSES)]
    class_to_new_id = {class_name: index for index, class_name in enumerate(normalized_classes)}

    processed_records = load_jsonl_records(input_manifest_path)
    source_index = _build_source_index(processed_records=processed_records, kept_classes=normalized_classes)

    all_source_ids = set(source_index.keys())
    selected_source_ids = {
        source_image_id
        for source_image_id, payload in source_index.items()
        if payload["class_counter"]
    }
    if not drop_source_images_without_kept_classes:
        selected_source_ids = set(all_source_ids)

    filtered_records: list[dict[str, Any]] = []
    selected_annotation_counts: Counter[str] = Counter()
    selected_source_presence_counts: dict[str, Counter[str]] = {}
    selected_source_counts_by_split: Counter[str] = Counter()

    for source_image_id in sorted(selected_source_ids, key=natural_sort_key):
        payload = source_index[source_image_id]
        split_name = str(payload["split"])
        selected_source_counts_by_split[split_name] += 1
        selected_source_presence_counts.setdefault(split_name, Counter()).update(set(payload["class_counter"].keys()))

    for record in processed_records:
        source_image_id = str(record.get("source_image_id") or "")
        if source_image_id not in selected_source_ids:
            continue

        kept_annotations = []
        for annotation in record.get("annotations", []):
            class_name = normalize_class_name(annotation["class_name"])
            if class_name not in class_to_new_id:
                continue
            annotation_copy = dict(annotation)
            annotation_copy["class_name"] = class_name
            annotation_copy["class_id"] = class_to_new_id[class_name]
            kept_annotations.append(annotation_copy)
            selected_annotation_counts[class_name] += 1

        record_copy = dict(record)
        record_copy["dataset_name"] = dataset_name
        record_copy["annotations"] = kept_annotations
        record_copy["is_empty"] = len(kept_annotations) == 0
        record_copy["empty_reason"] = None if kept_annotations else "negative_tile_class_filtered"
        record_copy["num_small_annotations"] = sum(
            1 for annotation in kept_annotations if bool(annotation.get("is_small_defect", False))
        )
        filtered_records.append(record_copy)

    if not filtered_records:
        raise ValueError("Class-filtered manifest produced no records.")

    output_root = ensure_dir(output_root_dir)
    manifest_path = output_root / "manifest.jsonl"
    metadata_path = output_root / "metadata.json"
    selected_ids_path = output_root / "selected_source_ids.txt"

    save_jsonl(filtered_records, manifest_path)
    selected_ids_path.write_text("\n".join(sorted(selected_source_ids, key=natural_sort_key)) + "\n", encoding="utf-8")

    summary = {
        "dataset_name": dataset_name,
        "input_manifest_path": str(Path(input_manifest_path)),
        "manifest_path": str(manifest_path),
        "selected_source_ids_path": str(selected_ids_path),
        "kept_classes": normalized_classes,
        "drop_source_images_without_kept_classes": bool(drop_source_images_without_kept_classes),
        "num_processed_records": len(filtered_records),
        "num_all_source_images": len(all_source_ids),
        "num_selected_source_images": len(selected_source_ids),
        "num_dropped_source_images_without_kept_classes": len(all_source_ids - selected_source_ids),
        "selected_source_images_by_split": {
            split_name: int(count)
            for split_name, count in sorted(selected_source_counts_by_split.items())
        },
        "selected_source_presence_by_split": {
            split_name: {
                class_name: int(split_counter.get(class_name, 0))
                for class_name in normalized_classes
            }
            for split_name, split_counter in sorted(selected_source_presence_counts.items())
        },
        "selected_annotation_count_by_class": {
            class_name: int(selected_annotation_counts.get(class_name, 0))
            for class_name in normalized_classes
        },
    }
    save_json(summary, metadata_path)

    return {
        "manifest_path": manifest_path,
        "metadata_path": metadata_path,
        "selected_source_ids_path": selected_ids_path,
        "summary": summary,
    }
