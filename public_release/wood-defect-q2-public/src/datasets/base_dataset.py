"""Shared utilities for wood defect dataset parsing and audit export."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
import math
import os
import re
import tempfile

cache_dir = Path(tempfile.gettempdir()) / "wood-defect-q2-mpl-cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.utils.io import ensure_dir, save_csv, save_json, save_jsonl


class BaseWoodDefectDataset:
    """Small record-backed dataset wrapper used by preparation scripts and training code."""

    def __init__(self, config: Dict[str, Any], records: Sequence[Dict[str, Any]] | None = None) -> None:
        self.config = config
        self.records = list(records) if records is not None else list(self.load_records())

    def load_records(self) -> Sequence[Dict[str, Any]]:
        """Load records for the dataset instance."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


DEFAULT_SMALL_DEFECT_RULE = {
    "enabled": True,
    "combine": "any",
    "min_area_ratio": 0.01,
    "min_width_px": 16,
    "min_height_px": 16,
}


def normalize_class_name(name: str) -> str:
    """Normalize class names across datasets."""
    normalized = name.strip().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.lower()


def natural_sort_key(value: str) -> list[Any]:
    """Sort strings containing digits in human order."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def resolve_optional_path(root_dir: Path, value: str | None) -> Path | None:
    """Resolve an optional path relative to a dataset root."""
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root_dir / path


def read_image_size(image_path: Path) -> tuple[int, int]:
    """Read image size without decoding the full image into memory."""
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Failed to read image metadata: {image_path}") from exc
    return int(width), int(height)


def clip_and_validate_bbox_xyxy(
    bbox_xyxy: Sequence[float],
    decimals: int = 6,
) -> tuple[list[float] | None, list[str]]:
    """Clip normalized XYXY boxes into [0, 1] and validate them."""
    values = [float(value) for value in bbox_xyxy]
    issues: list[str] = []

    if any(not math.isfinite(value) for value in values):
        return None, ["non_finite_box"]

    if any(value < 0.0 or value > 1.0 for value in values):
        issues.append("clipped_box")

    x1, y1, x2, y2 = [min(1.0, max(0.0, value)) for value in values]
    if x2 <= x1 or y2 <= y1:
        issues.append("invalid_box")
        return None, issues

    rounded = [round(x1, decimals), round(y1, decimals), round(x2, decimals), round(y2, decimals)]
    return rounded, issues


def xywh_to_xyxy_norm(cx: float, cy: float, width: float, height: float) -> list[float]:
    """Convert normalized YOLO XYWH boxes into normalized XYXY boxes."""
    return [
        cx - (width / 2.0),
        cy - (height / 2.0),
        cx + (width / 2.0),
        cy + (height / 2.0),
    ]


def build_annotation(
    class_name: str,
    bbox_xyxy_norm: Sequence[float],
    source_label: str | None = None,
) -> Dict[str, Any]:
    """Create the unified annotation dictionary."""
    x1, y1, x2, y2 = bbox_xyxy_norm
    bbox_width = round(x2 - x1, 6)
    bbox_height = round(y2 - y1, 6)
    bbox_area = round(bbox_width * bbox_height, 6)
    normalized_name = normalize_class_name(class_name)
    return {
        "class_name": normalized_name,
        "class_id": -1,
        "bbox_xyxy_norm": [round(float(value), 6) for value in bbox_xyxy_norm],
        "bbox_width_norm": bbox_width,
        "bbox_height_norm": bbox_height,
        "bbox_area_norm": bbox_area,
        "is_small_defect": False,
        "source_label": source_label or class_name,
    }


def resolve_small_defect_rule(rule_config: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Resolve a transparent small-defect rule from config with stable defaults."""
    rule = dict(DEFAULT_SMALL_DEFECT_RULE)
    if rule_config:
        rule.update(dict(rule_config))

    combine = str(rule.get("combine", "any")).lower()
    if combine not in {"any", "all"}:
        raise ValueError("small_defect.combine must be either 'any' or 'all'.")

    resolved = {
        "enabled": bool(rule.get("enabled", True)),
        "combine": combine,
        "min_area_ratio": rule.get("min_area_ratio"),
        "min_width_px": rule.get("min_width_px"),
        "min_height_px": rule.get("min_height_px"),
    }

    for key in ("min_area_ratio", "min_width_px", "min_height_px"):
        value = resolved[key]
        resolved[key] = None if value is None else float(value)

    return resolved


def tag_small_defects(
    records: Sequence[Dict[str, Any]],
    rule_config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Tag annotations using a compact small-defect rule."""
    rule = resolve_small_defect_rule(rule_config)
    active_checks = [
        key
        for key in ("min_area_ratio", "min_width_px", "min_height_px")
        if rule.get(key) is not None
    ]

    for record in records:
        width = int(record["width"])
        height = int(record["height"])
        small_count = 0

        for annotation in record.get("annotations", []):
            width_px = float(annotation["bbox_width_norm"]) * width
            height_px = float(annotation["bbox_height_norm"]) * height
            area_ratio = float(annotation["bbox_area_norm"])

            checks = []
            if rule["min_area_ratio"] is not None:
                checks.append(area_ratio <= float(rule["min_area_ratio"]))
            if rule["min_width_px"] is not None:
                checks.append(width_px <= float(rule["min_width_px"]))
            if rule["min_height_px"] is not None:
                checks.append(height_px <= float(rule["min_height_px"]))

            if not rule["enabled"] or not active_checks:
                is_small = False
            elif rule["combine"] == "all":
                is_small = all(checks)
            else:
                is_small = any(checks)

            annotation["is_small_defect"] = bool(is_small)
            if is_small:
                small_count += 1

        record["num_small_annotations"] = small_count

    return rule


def assign_class_ids(
    records: Sequence[Dict[str, Any]],
    preferred_class_names: Sequence[str] | None = None,
) -> Dict[str, int]:
    """Assign stable class ids to a list of unified records."""
    ordered_names: list[str] = []
    seen_names: set[str] = set()

    for class_name in preferred_class_names or []:
        normalized = normalize_class_name(class_name)
        if normalized not in seen_names:
            ordered_names.append(normalized)
            seen_names.add(normalized)

    discovered_names = sorted(
        {
            annotation["class_name"]
            for record in records
            for annotation in record.get("annotations", [])
        }
    )
    for class_name in discovered_names:
        if class_name not in seen_names:
            ordered_names.append(class_name)
            seen_names.add(class_name)

    class_to_idx = {class_name: index for index, class_name in enumerate(ordered_names)}
    for record in records:
        for annotation in record.get("annotations", []):
            annotation["class_id"] = class_to_idx[annotation["class_name"]]
    return class_to_idx


def _describe_metric(metric_name: str, values: Sequence[float]) -> Dict[str, Any]:
    """Build compact distribution statistics for one metric."""
    if not values:
        return {
            "metric": metric_name,
            "count": 0,
            "min": None,
            "mean": None,
            "std": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }

    array = np.asarray(values, dtype=float)
    return {
        "metric": metric_name,
        "count": int(array.size),
        "min": round(float(np.min(array)), 6),
        "mean": round(float(np.mean(array)), 6),
        "std": round(float(np.std(array)), 6),
        "p25": round(float(np.quantile(array, 0.25)), 6),
        "p50": round(float(np.quantile(array, 0.50)), 6),
        "p75": round(float(np.quantile(array, 0.75)), 6),
        "p95": round(float(np.quantile(array, 0.95)), 6),
        "max": round(float(np.max(array)), 6),
    }


def build_audit_tables(
    records: Sequence[Dict[str, Any]],
    dataset_name: str,
    class_to_idx: Mapping[str, int],
    validation_counts: Mapping[str, int] | None = None,
    small_defect_rule: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build compact audit summaries from unified records."""
    class_counter: Counter[str] = Counter()
    class_image_ids: dict[str, set[str]] = defaultdict(set)
    small_class_counter: Counter[str] = Counter()
    small_class_image_ids: dict[str, set[str]] = defaultdict(set)
    image_size_counter: Counter[tuple[int, int]] = Counter()
    split_counter: Counter[str] = Counter()
    validation_counter: Counter[str] = Counter(validation_counts or {})
    bbox_width_norm: list[float] = []
    bbox_height_norm: list[float] = []
    bbox_area_norm: list[float] = []
    bbox_width_px: list[float] = []
    bbox_height_px: list[float] = []
    bbox_area_px: list[float] = []

    empty_images = 0
    total_annotations = 0
    total_small_annotations = 0
    images_with_small_annotations = 0

    for record in records:
        width = int(record["width"])
        height = int(record["height"])
        image_size_counter[(width, height)] += 1

        split = record.get("split")
        if split:
            split_counter[split] += 1

        if record.get("empty_reason"):
            empty_images += 1
            validation_counter[record["empty_reason"]] += 1

        if record.get("num_invalid_boxes", 0):
            validation_counter["invalid_boxes"] += int(record["num_invalid_boxes"])

        if record.get("num_clipped_boxes", 0):
            validation_counter["clipped_boxes"] += int(record["num_clipped_boxes"])

        for issue in record.get("issues", []):
            validation_counter[issue] += 1

        if record.get("num_small_annotations", 0):
            images_with_small_annotations += 1

        for annotation in record.get("annotations", []):
            total_annotations += 1
            class_name = annotation["class_name"]
            class_counter[class_name] += 1
            class_image_ids[class_name].add(record["image_id"])

            if annotation.get("is_small_defect", False):
                total_small_annotations += 1
                small_class_counter[class_name] += 1
                small_class_image_ids[class_name].add(record["image_id"])

            width_norm = float(annotation["bbox_width_norm"])
            height_norm = float(annotation["bbox_height_norm"])
            area_norm = float(annotation["bbox_area_norm"])
            bbox_width_norm.append(width_norm)
            bbox_height_norm.append(height_norm)
            bbox_area_norm.append(area_norm)
            bbox_width_px.append(width_norm * width)
            bbox_height_px.append(height_norm * height)
            bbox_area_px.append(area_norm * width * height)

    class_distribution = pd.DataFrame(
        [
            {
                "class_name": class_name,
                "class_id": class_to_idx.get(class_name, -1),
                "box_count": int(class_counter.get(class_name, 0)),
                "image_count": len(class_image_ids.get(class_name, set())),
                "small_box_count": int(small_class_counter.get(class_name, 0)),
                "small_box_ratio": round(
                    float(small_class_counter.get(class_name, 0)) / float(class_counter.get(class_name, 1)),
                    6,
                ) if class_counter.get(class_name, 0) else 0.0,
                "small_image_count": len(small_class_image_ids.get(class_name, set())),
            }
            for class_name in sorted(
                class_to_idx.keys(),
                key=lambda item: (-class_counter.get(item, 0), item),
            )
        ]
    )

    image_size_distribution = pd.DataFrame(
        [
            {"width": width, "height": height, "image_count": count}
            for (width, height), count in sorted(
                image_size_counter.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ]
    )

    bbox_distribution = pd.DataFrame(
        [
            _describe_metric("bbox_width_norm", bbox_width_norm),
            _describe_metric("bbox_height_norm", bbox_height_norm),
            _describe_metric("bbox_area_norm", bbox_area_norm),
            _describe_metric("bbox_width_px", bbox_width_px),
            _describe_metric("bbox_height_px", bbox_height_px),
            _describe_metric("bbox_area_px", bbox_area_px),
        ]
    )

    validation_summary = pd.DataFrame(
        [
            {"issue": issue, "count": count}
            for issue, count in sorted(
                validation_counter.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
    )

    summary = {
        "dataset_name": dataset_name,
        "num_images": len(records),
        "num_annotations": total_annotations,
        "num_classes": len(class_to_idx),
        "num_empty_images": empty_images,
        "num_small_annotations": total_small_annotations,
        "small_annotation_ratio": round(
            float(total_small_annotations) / float(total_annotations),
            6,
        ) if total_annotations else 0.0,
        "num_images_with_small_annotations": images_with_small_annotations,
        "images_with_small_annotation_ratio": round(
            float(images_with_small_annotations) / float(len(records)),
            6,
        ) if records else 0.0,
        "split_distribution": dict(sorted(split_counter.items())),
        "class_names": list(class_to_idx.keys()),
        "validation_counts": {
            str(issue): int(count)
            for issue, count in validation_counter.items()
        },
        "small_defect_rule": dict(small_defect_rule or resolve_small_defect_rule()),
    }

    return {
        "summary": summary,
        "class_distribution": class_distribution,
        "image_size_distribution": image_size_distribution,
        "bbox_distribution": bbox_distribution,
        "validation_summary": validation_summary,
    }


def _format_table(df: pd.DataFrame) -> str:
    """Format a small dataframe for markdown docs without extra dependencies."""
    if df.empty:
        return "_No rows_"
    return "```\n" + df.to_string(index=False) + "\n```"


def export_object_size_distribution_figure(
    records: Sequence[Dict[str, Any]],
    dataset_name: str,
    figure_path: str | Path,
    small_defect_rule: Mapping[str, Any],
) -> Path:
    """Export a compact bbox-area distribution figure with the small-defect threshold."""
    figure_path = Path(figure_path)
    ensure_dir(figure_path.parent)

    area_values = np.asarray(
        [
            float(annotation["bbox_area_norm"])
            for record in records
            for annotation in record.get("annotations", [])
        ],
        dtype=float,
    )
    small_area_values = np.asarray(
        [
            float(annotation["bbox_area_norm"])
            for record in records
            for annotation in record.get("annotations", [])
            if annotation.get("is_small_defect", False)
        ],
        dtype=float,
    )

    display_values = np.clip(area_values, 1e-6, None)
    display_small_values = np.clip(small_area_values, 1e-6, None)
    bins = np.logspace(-6, 0, 40)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    if display_values.size:
        ax.hist(display_values, bins=bins, color="#7f8c8d", alpha=0.75, label="All boxes")
    else:
        ax.text(0.5, 0.5, "No annotations", ha="center", va="center", transform=ax.transAxes)

    if display_small_values.size:
        ax.hist(display_small_values, bins=bins, color="#d95f02", alpha=0.85, label="Tagged small boxes")

    ax.set_xscale("log")
    ax.set_xlabel("BBox area ratio")
    ax.set_ylabel("Count")
    ax.set_title(f"{dataset_name} object-size distribution")
    ax.grid(alpha=0.2, axis="y")

    threshold = small_defect_rule.get("min_area_ratio")
    if threshold is not None:
        threshold_value = max(float(threshold), 1e-6)
        ax.axvline(threshold_value, color="#1b9e77", linestyle="--", linewidth=1.2, label=f"Area threshold={threshold:g}")

    if display_values.size:
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def export_prepared_dataset(
    records: Sequence[Dict[str, Any]],
    dataset_name: str,
    class_to_idx: Mapping[str, int],
    validation_counts: Mapping[str, int] | None = None,
    small_defect_config: Mapping[str, Any] | None = None,
    processed_dir: str | Path = "data/processed",
    output_dir: str | Path = "outputs/tables",
    figure_dir: str | Path = "outputs/figures",
    docs_dir: str | Path = "docs",
) -> Dict[str, Path]:
    """Save unified manifest plus compact audit tables and markdown summary."""
    artifact_stem = normalize_class_name(dataset_name)
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    docs_dir = Path(docs_dir)

    ensure_dir(processed_dir)
    ensure_dir(output_dir)
    ensure_dir(figure_dir)
    ensure_dir(docs_dir)

    small_defect_rule = tag_small_defects(records, small_defect_config)

    manifest_path = processed_dir / f"{artifact_stem}_manifest.jsonl"
    summary_path = output_dir / f"{artifact_stem}_summary.json"
    class_path = output_dir / f"{artifact_stem}_class_distribution.csv"
    image_size_path = output_dir / f"{artifact_stem}_image_size_distribution.csv"
    bbox_path = output_dir / f"{artifact_stem}_bbox_distribution.csv"
    validation_path = output_dir / f"{artifact_stem}_validation_summary.csv"
    figure_path = figure_dir / f"{artifact_stem}_object_size_distribution.png"
    doc_path = docs_dir / f"{artifact_stem}_audit.md"

    audit_tables = build_audit_tables(
        records=records,
        dataset_name=dataset_name,
        class_to_idx=class_to_idx,
        validation_counts=validation_counts,
        small_defect_rule=small_defect_rule,
    )
    export_object_size_distribution_figure(
        records=records,
        dataset_name=dataset_name,
        figure_path=figure_path,
        small_defect_rule=small_defect_rule,
    )

    save_jsonl(records, manifest_path)
    save_json(audit_tables["summary"], summary_path)
    save_csv(audit_tables["class_distribution"], class_path)
    save_csv(audit_tables["image_size_distribution"], image_size_path)
    save_csv(audit_tables["bbox_distribution"], bbox_path)
    save_csv(audit_tables["validation_summary"], validation_path)

    markdown_lines = [
        f"# {dataset_name} Audit",
        "",
        f"- Images: {audit_tables['summary']['num_images']}",
        f"- Annotations: {audit_tables['summary']['num_annotations']}",
        f"- Classes: {audit_tables['summary']['num_classes']}",
        f"- Empty images: {audit_tables['summary']['num_empty_images']}",
        f"- Small annotations: {audit_tables['summary']['num_small_annotations']}",
        f"- Small annotation ratio: {audit_tables['summary']['small_annotation_ratio']:.4f}",
    ]

    split_distribution = audit_tables["summary"].get("split_distribution", {})
    if split_distribution:
        markdown_lines.append(f"- Splits: {split_distribution}")

    markdown_lines.extend(
        [
            "",
            "## Small-Defect Rule",
            f"- Enabled: {small_defect_rule['enabled']}",
            f"- Combine mode: `{small_defect_rule['combine']}`",
            f"- Min area ratio: {small_defect_rule['min_area_ratio']}",
            f"- Min width px: {small_defect_rule['min_width_px']}",
            f"- Min height px: {small_defect_rule['min_height_px']}",
            "",
            "## Class Distribution",
            _format_table(audit_tables["class_distribution"]),
            "",
            "## Image Size Distribution",
            _format_table(audit_tables["image_size_distribution"]),
            "",
            "## Bounding Box Distribution",
            _format_table(audit_tables["bbox_distribution"]),
            "",
            "## Validation Summary",
            _format_table(audit_tables["validation_summary"]),
            "",
            f"- Manifest: `{manifest_path}`",
            f"- Summary JSON: `{summary_path}`",
            f"- Size Figure: `{figure_path}`",
        ]
    )

    doc_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    return {
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "class_distribution_path": class_path,
        "image_size_distribution_path": image_size_path,
        "bbox_distribution_path": bbox_path,
        "validation_summary_path": validation_path,
        "figure_path": figure_path,
        "doc_path": doc_path,
    }
