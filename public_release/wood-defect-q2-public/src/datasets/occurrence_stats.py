"""Source-level occurrence and co-occurrence statistics for manifest-backed datasets."""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.datasets.base_dataset import natural_sort_key, normalize_class_name


def load_manifest_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a manifest JSONL file into memory."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return records


def _resolve_class_names(
    records: Sequence[Mapping[str, Any]],
    explicit_class_names: Sequence[str] | None = None,
) -> list[str]:
    if explicit_class_names:
        return [normalize_class_name(name) for name in explicit_class_names]

    discovered = sorted(
        {
            normalize_class_name(annotation["class_name"])
            for record in records
            for annotation in record.get("annotations", [])
        }
    )
    return discovered


def aggregate_source_level_records(
    records: Sequence[Mapping[str, Any]],
    class_names: Sequence[str] | None = None,
    split: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Aggregate raw/source or processed/tile manifests into source-level records."""
    normalized_class_names = _resolve_class_names(records=records, explicit_class_names=class_names)
    allowed = set(normalized_class_names)
    requested_split = None if split in {None, "", "all"} else str(split).lower()

    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        record_split = record.get("split") or record.get("resolved_split")
        normalized_split = None if record_split in {None, ""} else str(record_split).lower()
        if requested_split is not None and normalized_split != requested_split:
            continue

        source_image_id = str(record.get("source_image_id") or record.get("image_id") or "").strip()
        if not source_image_id:
            raise ValueError("Each record must provide source_image_id or image_id.")

        entry = grouped.setdefault(
            source_image_id,
            {
                "source_image_id": source_image_id,
                "split": normalized_split,
                "annotations": [],
                "tile_image_ids": [],
            },
        )
        if entry["split"] is None:
            entry["split"] = normalized_split

        image_id = record.get("image_id")
        if image_id is not None:
            entry["tile_image_ids"].append(str(image_id))

        for annotation in record.get("annotations", []):
            class_name = normalize_class_name(annotation["class_name"])
            if class_name not in allowed:
                continue
            annotation_copy = dict(annotation)
            annotation_copy["class_name"] = class_name
            entry["annotations"].append(annotation_copy)

    aggregated_records = [
        {
            "source_image_id": source_image_id,
            "split": payload["split"],
            "tile_count": len(payload["tile_image_ids"]),
            "annotations": payload["annotations"],
        }
        for source_image_id, payload in sorted(grouped.items(), key=lambda item: natural_sort_key(item[0]))
    ]
    return aggregated_records, normalized_class_names


def build_occurrence_statistics(
    records: Sequence[Mapping[str, Any]],
    class_names: Sequence[str] | None = None,
    split: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build per-class source-level occurrence statistics."""
    source_records, normalized_class_names = aggregate_source_level_records(
        records=records,
        class_names=class_names,
        split=split,
    )

    image_count_by_class: Counter[str] = Counter()
    annotation_count_by_class: Counter[str] = Counter()
    only_this_class_image_count: Counter[str] = Counter()
    exactly_one_annotation_image_count: Counter[str] = Counter()
    only_this_class_multi_annotation_count: Counter[str] = Counter()
    multi_class_image_with_class_count: Counter[str] = Counter()

    num_images = 0
    num_empty_images = 0
    num_positive_images = 0
    num_single_annotation_images = 0
    num_single_class_images = 0
    num_multi_class_images = 0

    for record in source_records:
        num_images += 1
        annotations = record.get("annotations", [])
        total_annotations = len(annotations)

        if total_annotations == 0:
            num_empty_images += 1
            continue

        num_positive_images += 1
        if total_annotations == 1:
            num_single_annotation_images += 1

        class_counter = Counter(annotation["class_name"] for annotation in annotations)
        present_classes = sorted(class_counter.keys())

        for class_name, count in class_counter.items():
            image_count_by_class[class_name] += 1
            annotation_count_by_class[class_name] += int(count)

        if len(present_classes) == 1:
            class_name = present_classes[0]
            num_single_class_images += 1
            only_this_class_image_count[class_name] += 1
            if total_annotations == 1:
                exactly_one_annotation_image_count[class_name] += 1
            else:
                only_this_class_multi_annotation_count[class_name] += 1
        else:
            num_multi_class_images += 1
            for class_name in present_classes:
                multi_class_image_with_class_count[class_name] += 1

    rows: list[dict[str, Any]] = []
    for class_name in normalized_class_names:
        images_with_class = int(image_count_by_class.get(class_name, 0))
        only_this_class_images = int(only_this_class_image_count.get(class_name, 0))
        exactly_one_annotation_images = int(exactly_one_annotation_image_count.get(class_name, 0))
        rows.append(
            {
                "class_name": class_name,
                "annotation_count": int(annotation_count_by_class.get(class_name, 0)),
                "images_with_class": images_with_class,
                "images_only_this_class": only_this_class_images,
                "images_only_this_class_multi_annotation": int(
                    only_this_class_multi_annotation_count.get(class_name, 0)
                ),
                "images_exactly_one_annotation_total": exactly_one_annotation_images,
                "images_multi_class_with_class": int(multi_class_image_with_class_count.get(class_name, 0)),
                "images_with_class_but_not_singleton": max(images_with_class - exactly_one_annotation_images, 0),
                "singleton_rate_within_class_images": round(
                    exactly_one_annotation_images / max(images_with_class, 1),
                    6,
                ),
                "only_class_rate_within_class_images": round(
                    only_this_class_images / max(images_with_class, 1),
                    6,
                ),
            }
        )

    per_class_df = pd.DataFrame(rows)
    summary = {
        "num_images": int(num_images),
        "num_empty_images": int(num_empty_images),
        "num_positive_images": int(num_positive_images),
        "num_single_annotation_images": int(num_single_annotation_images),
        "num_single_class_images": int(num_single_class_images),
        "num_multi_class_images": int(num_multi_class_images),
        "class_names": normalized_class_names,
        "split": None if split in {None, "", "all"} else str(split).lower(),
    }
    return per_class_df, summary


def build_cooccurrence_statistics(
    records: Sequence[Mapping[str, Any]],
    class_names: Sequence[str] | None = None,
    split: str | None = None,
    top_k: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build pair and triple co-occurrence tables at the source-image level."""
    source_records, normalized_class_names = aggregate_source_level_records(
        records=records,
        class_names=class_names,
        split=split,
    )

    pair_counter: Counter[tuple[str, str]] = Counter()
    triple_counter: Counter[tuple[str, str, str]] = Counter()
    positive_source_images = 0

    for record in source_records:
        present_classes = sorted({annotation["class_name"] for annotation in record.get("annotations", [])})
        if not present_classes:
            continue

        positive_source_images += 1
        for combo in combinations(present_classes, 2):
            pair_counter[combo] += 1
        for combo in combinations(present_classes, 3):
            triple_counter[combo] += 1

    pair_rows = [
        {
            "class_a": combo[0],
            "class_b": combo[1],
            "combo_key": " + ".join(combo),
            "source_image_count": int(count),
            "positive_image_ratio": round(count / max(positive_source_images, 1), 6),
        }
        for combo, count in pair_counter.items()
    ]
    triple_rows = [
        {
            "class_a": combo[0],
            "class_b": combo[1],
            "class_c": combo[2],
            "combo_key": " + ".join(combo),
            "source_image_count": int(count),
            "positive_image_ratio": round(count / max(positive_source_images, 1), 6),
        }
        for combo, count in triple_counter.items()
    ]

    pair_df = pd.DataFrame(pair_rows).sort_values(
        by=["source_image_count", "combo_key"],
        ascending=[False, True],
    ) if pair_rows else pd.DataFrame(
        columns=["class_a", "class_b", "combo_key", "source_image_count", "positive_image_ratio"]
    )
    triple_df = pd.DataFrame(triple_rows).sort_values(
        by=["source_image_count", "combo_key"],
        ascending=[False, True],
    ) if triple_rows else pd.DataFrame(
        columns=["class_a", "class_b", "class_c", "combo_key", "source_image_count", "positive_image_ratio"]
    )

    if top_k is not None and top_k > 0:
        pair_df = pair_df.head(int(top_k)).reset_index(drop=True)
        triple_df = triple_df.head(int(top_k)).reset_index(drop=True)
    else:
        pair_df = pair_df.reset_index(drop=True)
        triple_df = triple_df.reset_index(drop=True)

    summary = {
        "num_source_images": int(len(source_records)),
        "num_positive_source_images": int(positive_source_images),
        "num_pair_combos": int(len(pair_counter)),
        "num_triple_combos": int(len(triple_counter)),
        "class_names": normalized_class_names,
        "split": None if split in {None, "", "all"} else str(split).lower(),
    }
    return pair_df, triple_df, summary


def compare_manifest_occurrence_statistics(
    manifest_specs: Sequence[Mapping[str, Any]],
    class_names: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    top_k: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare per-manifest occurrence/co-occurrence statistics across one or more splits."""
    requested_splits = list(splits or ["all"])

    overview_rows: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    triple_frames: list[pd.DataFrame] = []
    benchmark_summaries: list[dict[str, Any]] = []

    for spec in manifest_specs:
        benchmark_name = str(spec.get("benchmark_name") or spec.get("name") or "").strip()
        manifest_path = spec.get("manifest_path")
        if not benchmark_name:
            raise ValueError("Each manifest spec must provide benchmark_name or name.")
        if not manifest_path:
            raise ValueError("Each manifest spec must provide manifest_path.")

        selection_mode = str(spec.get("selection_mode") or "").strip() or None
        records = load_manifest_jsonl(manifest_path)

        split_summaries: list[dict[str, Any]] = []
        for requested_split in requested_splits:
            per_class_df, occurrence_summary = build_occurrence_statistics(
                records=records,
                class_names=class_names,
                split=requested_split,
            )
            pair_df, triple_df, cooccurrence_summary = build_cooccurrence_statistics(
                records=records,
                class_names=class_names,
                split=requested_split,
                top_k=None,
            )

            split_label = occurrence_summary["split"] or "all"

            if not per_class_df.empty:
                per_class_df = per_class_df.copy()
                per_class_df.insert(0, "split", split_label)
                per_class_df.insert(0, "selection_mode", selection_mode)
                per_class_df.insert(0, "benchmark_name", benchmark_name)
                per_class_frames.append(per_class_df)

            pair_top_df = pair_df if top_k is None or top_k <= 0 else pair_df.head(int(top_k))
            if not pair_top_df.empty:
                pair_top_df = pair_top_df.copy()
                pair_top_df.insert(0, "rank", range(1, len(pair_top_df) + 1))
                pair_top_df.insert(0, "split", split_label)
                pair_top_df.insert(0, "selection_mode", selection_mode)
                pair_top_df.insert(0, "benchmark_name", benchmark_name)
                pair_frames.append(pair_top_df)

            triple_top_df = triple_df if top_k is None or top_k <= 0 else triple_df.head(int(top_k))
            if not triple_top_df.empty:
                triple_top_df = triple_top_df.copy()
                triple_top_df.insert(0, "rank", range(1, len(triple_top_df) + 1))
                triple_top_df.insert(0, "split", split_label)
                triple_top_df.insert(0, "selection_mode", selection_mode)
                triple_top_df.insert(0, "benchmark_name", benchmark_name)
                triple_frames.append(triple_top_df)

            top_pair_combo = None if pair_df.empty else str(pair_df.iloc[0]["combo_key"])
            top_pair_count = 0 if pair_df.empty else int(pair_df.iloc[0]["source_image_count"])
            top_triple_combo = None if triple_df.empty else str(triple_df.iloc[0]["combo_key"])
            top_triple_count = 0 if triple_df.empty else int(triple_df.iloc[0]["source_image_count"])

            overview_rows.append(
                {
                    "benchmark_name": benchmark_name,
                    "selection_mode": selection_mode,
                    "split": split_label,
                    "num_images": int(occurrence_summary["num_images"]),
                    "num_empty_images": int(occurrence_summary["num_empty_images"]),
                    "num_positive_images": int(occurrence_summary["num_positive_images"]),
                    "num_single_annotation_images": int(occurrence_summary["num_single_annotation_images"]),
                    "num_single_class_images": int(occurrence_summary["num_single_class_images"]),
                    "num_multi_class_images": int(occurrence_summary["num_multi_class_images"]),
                    "num_pair_combos": int(cooccurrence_summary["num_pair_combos"]),
                    "num_triple_combos": int(cooccurrence_summary["num_triple_combos"]),
                    "top_pair_combo": top_pair_combo,
                    "top_pair_count": top_pair_count,
                    "top_triple_combo": top_triple_combo,
                    "top_triple_count": top_triple_count,
                }
            )
            split_summaries.append(
                {
                    "split": split_label,
                    "occurrence": occurrence_summary,
                    "cooccurrence": cooccurrence_summary,
                    "top_pair_combo": top_pair_combo,
                    "top_pair_count": top_pair_count,
                    "top_triple_combo": top_triple_combo,
                    "top_triple_count": top_triple_count,
                }
            )

        benchmark_summaries.append(
            {
                "benchmark_name": benchmark_name,
                "selection_mode": selection_mode,
                "manifest_path": str(manifest_path),
                "splits": split_summaries,
            }
        )

    overview_df = pd.DataFrame(overview_rows)
    per_class_df = pd.concat(per_class_frames, ignore_index=True) if per_class_frames else pd.DataFrame()
    pair_df = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    triple_df = pd.concat(triple_frames, ignore_index=True) if triple_frames else pd.DataFrame()
    summary = {
        "benchmarks": benchmark_summaries,
        "class_names": _resolve_class_names([], explicit_class_names=class_names)
        if class_names
        else None,
        "splits": requested_splits,
        "top_k": None if top_k is None else int(top_k),
    }
    return overview_df, per_class_df, pair_df, triple_df, summary
