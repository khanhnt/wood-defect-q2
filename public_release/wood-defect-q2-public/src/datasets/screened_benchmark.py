"""Helpers to build reproducible screened benchmark subsets from processed manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence

from src.datasets.base_dataset import natural_sort_key, normalize_class_name
from src.utils.io import ensure_dir, save_json, save_jsonl


DEFAULT_VSB7_CLASSES = [
    "live_knot",
    "dead_knot",
    "resin",
    "knot_with_crack",
    "crack",
    "marrow",
    "knot_missing",
]

DEFAULT_SPLIT_RATIOS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}

DEFAULT_SELECTION_MODE = "random"
SUPPORTED_SELECTION_MODES = ("random", "stratified", "rare_first")


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL records into memory."""
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


def _build_source_index(
    processed_records: Sequence[Mapping[str, Any]],
    kept_classes: Sequence[str],
) -> dict[str, dict[str, Any]]:
    allowed = {normalize_class_name(name) for name in kept_classes}
    source_index: dict[str, dict[str, Any]] = {}

    for record in processed_records:
        source_image_id = str(record.get("source_image_id") or "").strip()
        if not source_image_id:
            raise ValueError("Processed benchmark selection requires source_image_id on every record.")

        split_name = str(record.get("split") or "train").lower()
        entry = source_index.setdefault(
            source_image_id,
            {
                "split": split_name,
                "tile_count": 0,
                "positive_tile_count": 0,
                "class_counter": Counter(),
            },
        )
        if entry["split"] != split_name:
            raise ValueError(
                f"Mixed split assignments detected for source image {source_image_id!r}: "
                f"{entry['split']!r} vs {split_name!r}"
            )

        entry["tile_count"] += 1
        has_positive_tile = False
        for annotation in record.get("annotations", []):
            class_name = normalize_class_name(annotation["class_name"])
            if class_name not in allowed:
                continue
            entry["class_counter"][class_name] += 1
            has_positive_tile = True
        if has_positive_tile:
            entry["positive_tile_count"] += 1

    return source_index


def _allocate_split_targets(
    split_candidates: Mapping[str, Sequence[str]],
    target_source_images: int,
    split_ratios: Mapping[str, float] | None = None,
) -> dict[str, int]:
    ratios = dict(DEFAULT_SPLIT_RATIOS)
    ratios.update({str(key).lower(): float(value) for key, value in dict(split_ratios or {}).items()})

    available_counts = {
        split_name: len(source_ids)
        for split_name, source_ids in split_candidates.items()
        if len(source_ids) > 0
    }
    total_available = sum(available_counts.values())
    if target_source_images > total_available:
        raise ValueError(
            f"Requested {target_source_images} source images, but only {total_available} are available "
            "after screening the processed manifest."
        )

    normalized_splits = list(available_counts.keys())
    ratio_sum = sum(ratios.get(split_name, 0.0) for split_name in normalized_splits)
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    targets = {split_name: 0 for split_name in normalized_splits}
    fractional_parts: list[tuple[float, str]] = []

    for split_name in normalized_splits:
        raw_target = target_source_images * (ratios.get(split_name, 0.0) / ratio_sum)
        base_target = min(int(raw_target), available_counts[split_name])
        targets[split_name] = base_target
        fractional_parts.append((raw_target - int(raw_target), split_name))

    allocated = sum(targets.values())
    remainder = target_source_images - allocated

    for _, split_name in sorted(fractional_parts, key=lambda item: (-item[0], item[1])):
        if remainder <= 0:
            break
        if targets[split_name] < available_counts[split_name]:
            targets[split_name] += 1
            remainder -= 1

    while remainder > 0:
        spare_splits = [
            split_name
            for split_name in normalized_splits
            if targets[split_name] < available_counts[split_name]
        ]
        if not spare_splits:
            raise ValueError("Unable to satisfy target source image count with the requested split ratios.")

        spare_splits.sort(
            key=lambda split_name: (
                available_counts[split_name] - targets[split_name],
                split_name == "train",
                split_name,
            ),
            reverse=True,
        )
        chosen_split = spare_splits[0]
        targets[chosen_split] += 1
        remainder -= 1

    return targets


def _sorted_present_classes(class_counter: Mapping[str, int], kept_classes: Sequence[str]) -> list[str]:
    """Return normalized present classes in a stable order."""
    normalized_kept = [normalize_class_name(name) for name in kept_classes]
    return [
        class_name
        for class_name in normalized_kept
        if int(class_counter.get(class_name, 0)) > 0
    ]


def _select_random_ids(
    candidate_ids: Sequence[str],
    target_count: int,
    seed: int,
    split_name: str,
) -> list[str]:
    candidate_ids = list(candidate_ids)
    split_rng = Random(f"{seed}:{split_name}:random")
    split_rng.shuffle(candidate_ids)
    return sorted(candidate_ids[:target_count], key=natural_sort_key)


def _select_stratified_ids(
    source_index: Mapping[str, Mapping[str, Any]],
    candidate_ids: Sequence[str],
    kept_classes: Sequence[str],
    target_count: int,
    seed: int,
    split_name: str,
) -> tuple[list[str], Counter[str]]:
    normalized_kept = [normalize_class_name(name) for name in kept_classes]
    tie_break_order = list(candidate_ids)
    split_rng = Random(f"{seed}:{split_name}:stratified")
    split_rng.shuffle(tie_break_order)
    tie_break_rank = {source_image_id: index for index, source_image_id in enumerate(tie_break_order)}

    remaining_ids = set(candidate_ids)
    selected_ids: list[str] = []
    selected_presence_counts: Counter[str] = Counter()

    while remaining_ids and len(selected_ids) < target_count:
        best_source_id = None
        best_score = None
        for source_image_id in remaining_ids:
            present_classes = _sorted_present_classes(
                class_counter=source_index[source_image_id]["class_counter"],
                kept_classes=normalized_kept,
            )
            if not present_classes:
                continue

            score = (
                sum(1.0 / (1.0 + selected_presence_counts[class_name]) for class_name in present_classes),
                len(present_classes),
                -tie_break_rank[source_image_id],
            )
            if best_score is None or score > best_score:
                best_score = score
                best_source_id = source_image_id

        if best_source_id is None:
            break

        selected_ids.append(best_source_id)
        remaining_ids.remove(best_source_id)
        for class_name in _sorted_present_classes(
            class_counter=source_index[best_source_id]["class_counter"],
            kept_classes=normalized_kept,
        ):
            selected_presence_counts[class_name] += 1

    return sorted(selected_ids, key=natural_sort_key), selected_presence_counts


def _select_rare_first_ids(
    source_index: Mapping[str, Mapping[str, Any]],
    candidate_ids: Sequence[str],
    kept_classes: Sequence[str],
    target_count: int,
    seed: int,
    split_name: str,
) -> tuple[list[str], Counter[str]]:
    normalized_kept = [normalize_class_name(name) for name in kept_classes]
    source_presence_counts: Counter[str] = Counter()
    for source_image_id in candidate_ids:
        present_classes = _sorted_present_classes(
            class_counter=source_index[source_image_id]["class_counter"],
            kept_classes=normalized_kept,
        )
        source_presence_counts.update(set(present_classes))

    tie_break_order = list(candidate_ids)
    split_rng = Random(f"{seed}:{split_name}:rare_first")
    split_rng.shuffle(tie_break_order)
    tie_break_rank = {source_image_id: index for index, source_image_id in enumerate(tie_break_order)}

    scored_candidates = []
    for source_image_id in candidate_ids:
        present_classes = _sorted_present_classes(
            class_counter=source_index[source_image_id]["class_counter"],
            kept_classes=normalized_kept,
        )
        rarity_score = sum(1.0 / max(1, source_presence_counts[class_name]) for class_name in present_classes)
        scored_candidates.append(
            (
                rarity_score,
                len(present_classes),
                -tie_break_rank[source_image_id],
                source_image_id,
            )
        )

    scored_candidates.sort(reverse=True)
    chosen_ids = [source_image_id for _, _, _, source_image_id in scored_candidates[:target_count]]
    selected_presence_counts: Counter[str] = Counter()
    for source_image_id in chosen_ids:
        selected_presence_counts.update(
            set(
                _sorted_present_classes(
                    class_counter=source_index[source_image_id]["class_counter"],
                    kept_classes=normalized_kept,
                )
            )
        )
    return sorted(chosen_ids, key=natural_sort_key), selected_presence_counts


def select_screened_source_ids(
    processed_records: Sequence[Mapping[str, Any]],
    kept_classes: Sequence[str],
    target_source_images: int,
    seed: int = 42,
    split_ratios: Mapping[str, float] | None = None,
    selection_mode: str = DEFAULT_SELECTION_MODE,
) -> tuple[set[str], dict[str, Any]]:
    """Select a reproducible screened source-image subset from processed tiles."""
    selection_mode = str(selection_mode).lower()
    if selection_mode not in SUPPORTED_SELECTION_MODES:
        raise ValueError(
            f"Unsupported selection_mode={selection_mode!r}. "
            f"Expected one of: {', '.join(SUPPORTED_SELECTION_MODES)}."
        )

    source_index = _build_source_index(processed_records=processed_records, kept_classes=kept_classes)

    split_candidates: dict[str, list[str]] = {}
    split_source_presence_counts: dict[str, Counter[str]] = {}
    for source_image_id, payload in source_index.items():
        if not payload["class_counter"]:
            continue
        split_name = payload["split"]
        split_candidates.setdefault(split_name, []).append(source_image_id)
        split_source_presence_counts.setdefault(split_name, Counter()).update(
            set(_sorted_present_classes(payload["class_counter"], kept_classes))
        )

    split_targets = _allocate_split_targets(
        split_candidates=split_candidates,
        target_source_images=int(target_source_images),
        split_ratios=split_ratios,
    )

    selected_source_ids: set[str] = set()
    selected_class_counts: Counter[str] = Counter()
    selected_split_counts: dict[str, int] = {}
    selected_source_presence_counts_by_split: dict[str, dict[str, int]] = {}

    for split_name, target_count in split_targets.items():
        candidate_ids = sorted(split_candidates.get(split_name, []), key=natural_sort_key)
        if selection_mode == "random":
            chosen_ids = _select_random_ids(
                candidate_ids=candidate_ids,
                target_count=target_count,
                seed=seed,
                split_name=split_name,
            )
            split_presence_counts = Counter()
            for source_image_id in chosen_ids:
                split_presence_counts.update(
                    set(_sorted_present_classes(source_index[source_image_id]["class_counter"], kept_classes))
                )
        elif selection_mode == "stratified":
            chosen_ids, split_presence_counts = _select_stratified_ids(
                source_index=source_index,
                candidate_ids=candidate_ids,
                kept_classes=kept_classes,
                target_count=target_count,
                seed=seed,
                split_name=split_name,
            )
        else:
            chosen_ids, split_presence_counts = _select_rare_first_ids(
                source_index=source_index,
                candidate_ids=candidate_ids,
                kept_classes=kept_classes,
                target_count=target_count,
                seed=seed,
                split_name=split_name,
            )

        selected_split_counts[split_name] = len(chosen_ids)
        selected_source_presence_counts_by_split[split_name] = {
            class_name: int(split_presence_counts.get(class_name, 0))
            for class_name in [normalize_class_name(name) for name in kept_classes]
        }
        selected_source_ids.update(chosen_ids)
        for source_image_id in chosen_ids:
            selected_class_counts.update(source_index[source_image_id]["class_counter"])

    summary = {
        "target_source_images": int(target_source_images),
        "selected_source_images": len(selected_source_ids),
        "seed": int(seed),
        "selection_mode": selection_mode,
        "kept_classes": [normalize_class_name(name) for name in kept_classes],
        "available_source_images_by_split": {
            split_name: len(source_ids)
            for split_name, source_ids in sorted(split_candidates.items())
        },
        "available_source_presence_by_split": {
            split_name: {
                class_name: int(split_source_presence_counts.get(split_name, Counter()).get(class_name, 0))
                for class_name in [normalize_class_name(name) for name in kept_classes]
            }
            for split_name in sorted(split_candidates.keys())
        },
        "selected_source_images_by_split": dict(sorted(selected_split_counts.items())),
        "selected_source_presence_by_split": dict(sorted(selected_source_presence_counts_by_split.items())),
        "selected_annotation_count_by_class": {
            class_name: int(selected_class_counts.get(class_name, 0))
            for class_name in [normalize_class_name(name) for name in kept_classes]
        },
    }
    return selected_source_ids, summary


def build_screened_processed_records(
    processed_records: Sequence[Mapping[str, Any]],
    selected_source_ids: set[str],
    kept_classes: Sequence[str],
    dataset_name: str,
) -> list[dict[str, Any]]:
    """Filter/remap processed tile records for a screened benchmark subset."""
    normalized_classes = [normalize_class_name(name) for name in kept_classes]
    class_to_new_id = {class_name: index for index, class_name in enumerate(normalized_classes)}
    selected_records: list[dict[str, Any]] = []

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

        record_copy = dict(record)
        record_copy["dataset_name"] = dataset_name
        record_copy["annotations"] = kept_annotations
        record_copy["is_empty"] = len(kept_annotations) == 0
        record_copy["empty_reason"] = None if kept_annotations else "negative_tile_screened_benchmark"
        record_copy["num_small_annotations"] = sum(
            1 for annotation in kept_annotations if bool(annotation.get("is_small_defect", False))
        )
        selected_records.append(record_copy)

    if not selected_records:
        raise ValueError("Screened benchmark selection produced no processed records.")
    return selected_records


def build_screened_benchmark_from_processed_manifest(
    input_manifest_path: str | Path,
    output_root_dir: str | Path,
    dataset_name: str,
    kept_classes: Sequence[str] | None = None,
    target_source_images: int = 3600,
    seed: int = 42,
    split_ratios: Mapping[str, float] | None = None,
    selection_mode: str = DEFAULT_SELECTION_MODE,
) -> dict[str, Any]:
    """Create a screened processed-manifest benchmark without re-tiling source images."""
    normalized_classes = [normalize_class_name(name) for name in (kept_classes or DEFAULT_VSB7_CLASSES)]
    processed_records = load_jsonl_records(input_manifest_path)
    selected_source_ids, selection_summary = select_screened_source_ids(
        processed_records=processed_records,
        kept_classes=normalized_classes,
        target_source_images=int(target_source_images),
        seed=int(seed),
        split_ratios=split_ratios,
        selection_mode=selection_mode,
    )
    screened_records = build_screened_processed_records(
        processed_records=processed_records,
        selected_source_ids=selected_source_ids,
        kept_classes=normalized_classes,
        dataset_name=dataset_name,
    )

    output_root = ensure_dir(output_root_dir)
    manifest_path = output_root / "manifest.jsonl"
    metadata_path = output_root / "metadata.json"
    selected_ids_path = output_root / "selected_source_ids.txt"

    save_jsonl(screened_records, manifest_path)
    selected_ids_path.write_text("\n".join(sorted(selected_source_ids, key=natural_sort_key)) + "\n", encoding="utf-8")

    summary = {
        "dataset_name": dataset_name,
        "input_manifest_path": str(Path(input_manifest_path)),
        "manifest_path": str(manifest_path),
        "selected_source_ids_path": str(selected_ids_path),
        "num_processed_records": len(screened_records),
        "num_selected_source_images": len(selected_source_ids),
        "kept_classes": normalized_classes,
        "selection": selection_summary,
    }
    save_json(summary, metadata_path)

    return {
        "manifest_path": manifest_path,
        "metadata_path": metadata_path,
        "selected_source_ids_path": selected_ids_path,
        "summary": summary,
    }
