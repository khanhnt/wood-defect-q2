#!/usr/bin/env python3
"""Build a publication-ready in-domain qualitative comparison figure.

This script is designed for the curated-benchmark / in-domain setting of the
wood-defect paper. It expects:

1. A manifest JSONL describing the benchmark split and image paths.
2. Per-image prediction JSONL files for the compared models. The format matches
   the optional `save_predictions` output from the repo evaluator.

The script selects representative cases using reproducible rules, renders a
compact four-column figure, and exports a manifest plus short note describing
the chosen examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.datasets.screened_benchmark import load_jsonl_records
from src.utils.io import ensure_dir, save_json


DEFAULT_CLASS_NAMES = [
    "live_knot",
    "dead_knot",
    "resin",
    "knot_with_crack",
    "crack",
    "marrow",
    "knot_missing",
]

CLASS_COLORS = {
    "live_knot": "#1f77b4",
    "dead_knot": "#ff7f0e",
    "resin": "#2ca02c",
    "knot_with_crack": "#d62728",
    "crack": "#9467bd",
    "marrow": "#8c564b",
    "knot_missing": "#e377c2",
}

KNOT_RELATED = {"live_knot", "dead_knot", "knot_with_crack", "knot_missing"}
CRACK_RELATED = {"crack", "knot_with_crack"}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    header: str
    run_name: str
    predictions_path: Path


@dataclass
class BoxEntry:
    box: list[float]
    label: str
    score: float | None = None


@dataclass
class MatchStats:
    matched_gt_indices: set[int]
    tp: int
    fp: int
    fn: int
    confusion_count: int
    matched_scores: list[float]
    fp_entries: list[BoxEntry]
    confusion_entries: list[BoxEntry]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Curated benchmark manifest JSONL.")
    parser.add_argument(
        "--image-root-dir",
        type=str,
        default=None,
        help="Optional root used when manifest image paths are relative or stale absolute paths need remapping.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for figure artifacts.")
    parser.add_argument("--split", type=str, default="test", help="Preferred split to visualize. Default: test.")
    parser.add_argument("--rows", type=int, default=5, help="Number of qualitative cases to export.")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=list(DEFAULT_CLASS_NAMES),
        help="Class order used by prediction JSONL label IDs.",
    )
    parser.add_argument(
        "--baseline-run-name",
        type=str,
        default="baseline_mobilenet_hr_vsb7_3600_rarefirst_full",
        help="Display/run name for the two-stage reference.",
    )
    parser.add_argument(
        "--baseline-header",
        type=str,
        default="Faster R-CNN",
        help="Short column header for the two-stage reference.",
    )
    parser.add_argument(
        "--baseline-predictions",
        type=str,
        required=True,
        help="Prediction JSONL for the two-stage reference.",
    )
    parser.add_argument(
        "--yolo-run-name",
        type=str,
        default="y0_yolov8s_vsb7_3600_rarefirst_e200",
        help="Display/run name for the YOLO baseline.",
    )
    parser.add_argument(
        "--yolo-header",
        type=str,
        default="YOLOv8s",
        help="Short column header for the YOLO baseline.",
    )
    parser.add_argument(
        "--yolo-predictions",
        type=str,
        required=True,
        help="Prediction JSONL for the YOLO baseline.",
    )
    parser.add_argument(
        "--variant-run-name",
        type=str,
        default="y2_yolov8s_wniou_vsb7_3600_rarefirst_e200",
        help="Display/run name for the YOLO variant.",
    )
    parser.add_argument(
        "--variant-header",
        type=str,
        default="YOLO variant",
        help="Short column header for the YOLO variant.",
    )
    parser.add_argument(
        "--variant-predictions",
        type=str,
        required=True,
        help="Prediction JSONL for the YOLO variant.",
    )
    parser.add_argument("--panel-width", type=int, default=420, help="Panel width in pixels.")
    parser.add_argument("--panel-height", type=int, default=220, help="Panel height in pixels.")
    parser.add_argument("--score-threshold", type=float, default=0.25, help="Minimum score to visualize.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for TP / confusion analysis.")
    return parser.parse_args()


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _normalize_path_string(value: str) -> str:
    normalized = value.replace("\\", "/")
    normalized = normalized.replace("Large-scale image dataset /", "Large-scale image dataset/")
    normalized = normalized.replace("Semantic Maps /", "Semantic Maps/")
    normalized = re.sub(r"/{2,}", "/", normalized)
    return normalized


def resolve_image_path(record: Mapping[str, Any], image_root_dir: Path | None) -> Path:
    raw_image_path = str(record.get("image_path") or "")
    if not raw_image_path:
        raise ValueError(f"Record {record.get('image_id')!r} is missing image_path.")

    candidates: list[Path] = []
    path = Path(raw_image_path)
    if path.is_absolute():
        candidates.append(path)
        candidates.append(Path(_normalize_path_string(raw_image_path)))
    else:
        candidates.append(path)
        if image_root_dir is not None:
            candidates.append(image_root_dir / path)
            candidates.append(Path(_normalize_path_string(str(image_root_dir / path))))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve image path for {record.get('image_id')!r}. "
        f"Tried: {[str(candidate) for candidate in candidates]}"
    )


def _box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


def build_gt_entries(record: Mapping[str, Any]) -> list[BoxEntry]:
    width = float(record["width"])
    height = float(record["height"])
    entries: list[BoxEntry] = []
    for annotation in record.get("annotations", []):
        x1, y1, x2, y2 = annotation["bbox_xyxy_norm"]
        entries.append(
            BoxEntry(
                box=[x1 * width, y1 * height, x2 * width, y2 * height],
                label=str(annotation["class_name"]),
                score=None,
            )
        )
    return entries


def load_predictions(
    predictions_path: Path,
    class_names: Sequence[str],
    score_threshold: float,
) -> dict[str, list[BoxEntry]]:
    prediction_records = load_jsonl_records(predictions_path)
    loaded: dict[str, list[BoxEntry]] = {}
    for record in prediction_records:
        image_id = str(record["image_id"])
        boxes = list(record.get("boxes", []))
        labels = list(record.get("labels", []))
        scores = list(record.get("scores", []))
        entries: list[BoxEntry] = []
        for box, label_id, score in zip(boxes, labels, scores):
            label_index = int(label_id)
            if label_index < 0 or label_index >= len(class_names):
                continue
            score_value = float(score)
            if score_value < score_threshold:
                continue
            entries.append(
                BoxEntry(
                    box=[float(v) for v in box],
                    label=str(class_names[label_index]),
                    score=score_value,
                )
            )
        entries.sort(key=lambda item: item.score if item.score is not None else -1.0, reverse=True)
        loaded[image_id] = entries
    return loaded


def analyze_predictions(
    gt_entries: Sequence[BoxEntry],
    pred_entries: Sequence[BoxEntry],
    iou_threshold: float,
) -> MatchStats:
    gt_matched: set[int] = set()
    tp = 0
    fp = 0
    confusion_count = 0
    matched_scores: list[float] = []
    fp_entries: list[BoxEntry] = []
    confusion_entries: list[BoxEntry] = []

    for prediction in pred_entries:
        best_same_index = None
        best_same_iou = 0.0
        best_diff_iou = 0.0
        best_diff_index = None
        for gt_index, gt_entry in enumerate(gt_entries):
            iou = _box_iou(prediction.box, gt_entry.box)
            if gt_entry.label == prediction.label and gt_index not in gt_matched and iou > best_same_iou:
                best_same_iou = iou
                best_same_index = gt_index
            if gt_entry.label != prediction.label and iou > best_diff_iou:
                best_diff_iou = iou
                best_diff_index = gt_index

        if best_same_index is not None and best_same_iou >= iou_threshold:
            gt_matched.add(best_same_index)
            tp += 1
            if prediction.score is not None:
                matched_scores.append(float(prediction.score))
            continue

        if best_diff_index is not None and best_diff_iou >= iou_threshold:
            confusion_count += 1
            confusion_entries.append(prediction)
        else:
            fp += 1
            fp_entries.append(prediction)

    fn = len(gt_entries) - len(gt_matched)
    return MatchStats(
        matched_gt_indices=gt_matched,
        tp=tp,
        fp=fp,
        fn=fn,
        confusion_count=confusion_count,
        matched_scores=matched_scores,
        fp_entries=fp_entries,
        confusion_entries=confusion_entries,
    )


def _safe_mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_image_analysis(
    record: Mapping[str, Any],
    gt_entries: Sequence[BoxEntry],
    model_predictions: Mapping[str, Sequence[BoxEntry]],
    iou_threshold: float,
) -> dict[str, Any]:
    model_stats = {
        model_key: analyze_predictions(gt_entries=gt_entries, pred_entries=predictions, iou_threshold=iou_threshold)
        for model_key, predictions in model_predictions.items()
    }
    gt_labels = [entry.label for entry in gt_entries]
    has_small_defect = any(bool(annotation.get("is_small_defect", False)) for annotation in record.get("annotations", []))
    has_knot_related = any(label in KNOT_RELATED for label in gt_labels)
    has_crack_related = any(label in CRACK_RELATED for label in gt_labels)
    all_predictions = [entry for predictions in model_predictions.values() for entry in predictions]
    all_fp_count = sum(stats.fp for stats in model_stats.values())
    all_fn_count = sum(stats.fn for stats in model_stats.values())
    all_confusion_count = sum(stats.confusion_count for stats in model_stats.values())

    return {
        "image_id": str(record["image_id"]),
        "split": str(record.get("split") or "unspecified"),
        "width": int(record["width"]),
        "height": int(record["height"]),
        "gt_entries": list(gt_entries),
        "model_stats": model_stats,
        "model_predictions": {key: list(value) for key, value in model_predictions.items()},
        "num_gt": len(gt_entries),
        "gt_labels": gt_labels,
        "has_small_defect": has_small_defect,
        "has_knot_related": has_knot_related,
        "has_crack_related": has_crack_related,
        "is_negative": len(gt_entries) == 0,
        "all_fp_count": all_fp_count,
        "all_fn_count": all_fn_count,
        "all_confusion_count": all_confusion_count,
        "mean_matched_score": _safe_mean(
            [score for stats in model_stats.values() for score in stats.matched_scores]
        ),
        "prediction_count": len(all_predictions),
    }


def _choose_best_candidate(
    analyses: Sequence[dict[str, Any]],
    *,
    used_image_ids: set[str],
    predicate,
    sort_key,
) -> dict[str, Any] | None:
    candidates = [item for item in analyses if item["image_id"] not in used_image_ids and predicate(item)]
    if not candidates:
        return None
    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


def select_cases(
    analyses: Sequence[dict[str, Any]],
    *,
    baseline_key: str,
    yolo_key: str,
    variant_key: str,
    max_rows: int,
) -> list[dict[str, Any]]:
    used_image_ids: set[str] = set()
    rows: list[dict[str, Any]] = []

    def add_row(case_type: str, reason: str, item: dict[str, Any] | None) -> None:
        if item is None or item["image_id"] in used_image_ids or len(rows) >= max_rows:
            return
        rows.append(
            {
                "case_type": case_type,
                "reason": reason,
                "image_id": item["image_id"],
                "split": item["split"],
                "analysis": item,
            }
        )
        used_image_ids.add(item["image_id"])

    add_row(
        "easy_correct",
        "All compared models detect the defect cleanly with no false positives or missed targets.",
        _choose_best_candidate(
            analyses,
            used_image_ids=used_image_ids,
            predicate=lambda item: (
                item["num_gt"] > 0
                and all(
                    stats.fp == 0 and stats.fn == 0 and stats.confusion_count == 0
                    for stats in item["model_stats"].values()
                )
            ),
            sort_key=lambda item: (
                item["mean_matched_score"],
                item["num_gt"],
            ),
        ),
    )

    add_row(
        "small_or_low_contrast",
        "A small-defect case that remains correct overall but is visually harder than the easiest examples.",
        _choose_best_candidate(
            analyses,
            used_image_ids=used_image_ids,
            predicate=lambda item: (
                item["has_small_defect"]
                and item["num_gt"] > 0
                and item["model_stats"][yolo_key].fn == 0
            ),
            sort_key=lambda item: (
                -item["model_stats"][yolo_key].tp,
                -item["model_stats"][variant_key].tp,
                -item["mean_matched_score"],
            ),
        ),
    )

    add_row(
        "false_positive_texture",
        "A negative/background tile that triggers at least one false positive on wood texture or grain.",
        _choose_best_candidate(
            analyses,
            used_image_ids=used_image_ids,
            predicate=lambda item: item["is_negative"] and item["all_fp_count"] > 0,
            sort_key=lambda item: (
                item["all_fp_count"],
                max(
                    (entry.score or 0.0)
                    for stats in item["model_stats"].values()
                    for entry in stats.fp_entries
                ) if item["all_fp_count"] > 0 else 0.0,
            ),
        ),
    )

    add_row(
        "missed_crack",
        "A crack or knot-with-crack case with at least one clear false negative under the shared test protocol.",
        _choose_best_candidate(
            analyses,
            used_image_ids=used_image_ids,
            predicate=lambda item: item["has_crack_related"] and item["all_fn_count"] > 0,
            sort_key=lambda item: (
                item["all_fn_count"],
                item["has_small_defect"],
                -item["mean_matched_score"],
            ),
        ),
    )

    add_row(
        "knot_related_confusion",
        "A semantically difficult knot-related case with class confusion or strong cross-model disagreement.",
        _choose_best_candidate(
            analyses,
            used_image_ids=used_image_ids,
            predicate=lambda item: item["has_knot_related"] and (item["all_confusion_count"] > 0 or item["all_fn_count"] > 0),
            sort_key=lambda item: (
                item["all_confusion_count"],
                item["all_fn_count"],
                item["num_gt"],
            ),
        ),
    )

    add_row(
        "model_disagreement",
        "A held-out test image where the compared models disagree substantially in false positives or missed detections.",
        _choose_best_candidate(
            analyses,
            used_image_ids=used_image_ids,
            predicate=lambda item: (
                item["num_gt"] > 0
                and (
                    len({item["model_stats"][key].fn for key in (baseline_key, yolo_key, variant_key)}) > 1
                    or len({item["model_stats"][key].fp for key in (baseline_key, yolo_key, variant_key)}) > 1
                )
            ),
            sort_key=lambda item: (
                len({item["model_stats"][key].fn for key in (baseline_key, yolo_key, variant_key)})
                + len({item["model_stats"][key].fp for key in (baseline_key, yolo_key, variant_key)}),
                item["all_confusion_count"],
                item["num_gt"],
            ),
        ),
    )

    return rows[:max_rows]


def _union_box(entries: Iterable[BoxEntry]) -> list[float] | None:
    entries = list(entries)
    if not entries:
        return None
    x1 = min(entry.box[0] for entry in entries)
    y1 = min(entry.box[1] for entry in entries)
    x2 = max(entry.box[2] for entry in entries)
    y2 = max(entry.box[3] for entry in entries)
    return [x1, y1, x2, y2]


def compute_crop_box(row: Mapping[str, Any], padding: int = 80) -> tuple[int, int, int, int]:
    analysis = row["analysis"]
    relevant_entries: list[BoxEntry] = list(analysis["gt_entries"])
    if row["case_type"] == "false_positive_texture":
        relevant_entries = [
            entry
            for stats in analysis["model_stats"].values()
            for entry in stats.fp_entries
        ]
    elif row["case_type"] == "knot_related_confusion":
        relevant_entries.extend(
            entry
            for stats in analysis["model_stats"].values()
            for entry in stats.confusion_entries
        )
    else:
        relevant_entries.extend(
            entry
            for predictions in analysis["model_predictions"].values()
            for entry in predictions
        )

    union = _union_box(relevant_entries)
    image_width = int(analysis["width"])
    image_height = int(analysis["height"])
    if union is None:
        return 0, 0, image_width, image_height

    x1, y1, x2, y2 = union
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    crop_w = max(640.0, box_w + 2 * padding)
    crop_h = max(360.0, box_h + 2 * padding)
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5

    left = int(round(center_x - crop_w * 0.5))
    top = int(round(center_y - crop_h * 0.5))
    right = int(round(center_x + crop_w * 0.5))
    bottom = int(round(center_y + crop_h * 0.5))

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_width:
        left -= right - image_width
        right = image_width
    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height
    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)
    return left, top, right, bottom


def _draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fill: str, font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle(
        [bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2],
        fill=fill,
        outline=fill,
    )
    draw.text((x, y), text, fill="white", font=font)


def _draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    color: str,
    width: int = 2,
    dash: int = 8,
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    segments = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for start, end in segments:
        if start[0] == end[0]:
            length = abs(end[1] - start[1])
            step = 1 if end[1] >= start[1] else -1
            for offset in range(0, length, dash * 2):
                y_start = start[1] + step * offset
                y_end = start[1] + step * min(offset + dash, length)
                draw.line([(start[0], y_start), (end[0], y_end)], fill=color, width=width)
        else:
            length = abs(end[0] - start[0])
            step = 1 if end[0] >= start[0] else -1
            for offset in range(0, length, dash * 2):
                x_start = start[0] + step * offset
                x_end = start[0] + step * min(offset + dash, length)
                draw.line([(x_start, start[1]), (x_end, end[1])], fill=color, width=width)


def render_panel(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    *,
    gt_entries: Sequence[BoxEntry],
    pred_entries: Sequence[BoxEntry],
    show_predictions: bool,
    panel_width: int,
    panel_height: int,
    font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
) -> Image.Image:
    left, top, right, bottom = crop_box
    cropped = image.crop(crop_box).convert("RGB")
    canvas = Image.new("RGB", (panel_width, panel_height), "white")
    fitted = ImageOps.contain(cropped, (panel_width, panel_height))
    offset_x = (panel_width - fitted.width) // 2
    offset_y = (panel_height - fitted.height) // 2
    canvas.paste(fitted, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)

    scale_x = fitted.width / max(1, right - left)
    scale_y = fitted.height / max(1, bottom - top)

    def transform_box(box: Sequence[float]) -> list[float]:
        x1, y1, x2, y2 = box
        return [
            offset_x + (x1 - left) * scale_x,
            offset_y + (y1 - top) * scale_y,
            offset_x + (x2 - left) * scale_x,
            offset_y + (y2 - top) * scale_y,
        ]

    for gt_entry in gt_entries:
        gt_box = transform_box(gt_entry.box)
        _draw_dashed_rectangle(draw, gt_box, color="#4d4d4d", width=2, dash=6)
        _draw_label(
            draw,
            int(gt_box[0]) + 2,
            max(2, int(gt_box[1]) - 16),
            gt_entry.label.replace("_", " "),
            fill="#4d4d4d",
            font=label_font,
        )

    if show_predictions:
        for pred_entry in pred_entries:
            pred_box = transform_box(pred_entry.box)
            color = CLASS_COLORS.get(pred_entry.label, "#d62728")
            draw.rectangle(pred_box, outline=color, width=3)
            _draw_label(
                draw,
                int(pred_box[0]) + 2,
                max(2, int(pred_box[1]) - 16),
                pred_entry.label.replace("_", " "),
                fill=color,
                font=label_font,
            )

    return canvas


def render_figure(
    selected_rows: Sequence[dict[str, Any]],
    records_by_image: Mapping[str, Mapping[str, Any]],
    image_root_dir: Path | None,
    model_specs: Sequence[ModelSpec],
    panel_width: int,
    panel_height: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    header_font = _load_font(20)
    label_font = _load_font(14)
    row_label_font = _load_font(13)

    columns = ["GT"] + [spec.header for spec in model_specs]
    header_height = 42
    row_note_width = 140
    gap_x = 10
    gap_y = 10
    width = row_note_width + len(columns) * panel_width + (len(columns) - 1) * gap_x + 20
    height = header_height + len(selected_rows) * panel_height + (len(selected_rows) - 1) * gap_y + 20
    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)

    for column_index, header in enumerate(columns):
        x = row_note_width + column_index * (panel_width + gap_x) + panel_width // 2
        bbox = draw.textbbox((0, 0), header, font=header_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, 8), header, fill="black", font=header_font)

    for row_index, row in enumerate(selected_rows):
        analysis = row["analysis"]
        record = records_by_image[row["image_id"]]
        image_path = resolve_image_path(record, image_root_dir=image_root_dir)
        with Image.open(image_path) as image:
            crop_box = compute_crop_box(row)
            y = header_height + row_index * (panel_height + gap_y)
            row_label = row["case_type"].replace("_", " ")
            draw.text((10, y + 6), row_label, fill="black", font=row_label_font)

            gt_panel = render_panel(
                image=image,
                crop_box=crop_box,
                gt_entries=analysis["gt_entries"],
                pred_entries=[],
                show_predictions=False,
                panel_width=panel_width,
                panel_height=panel_height,
                font=header_font,
                label_font=label_font,
            )
            figure.paste(gt_panel, (row_note_width, y))

            for column_offset, spec in enumerate(model_specs, start=1):
                panel = render_panel(
                    image=image,
                    crop_box=crop_box,
                    gt_entries=analysis["gt_entries"],
                    pred_entries=analysis["model_predictions"][spec.key],
                    show_predictions=True,
                    panel_width=panel_width,
                    panel_height=panel_height,
                    font=header_font,
                    label_font=label_font,
                )
                x = row_note_width + column_offset * (panel_width + gap_x)
                figure.paste(panel, (x, y))

    png_path = output_dir / "in_domain_qualitative_comparison.png"
    pdf_path = output_dir / "in_domain_qualitative_comparison.pdf"
    figure.save(png_path)
    figure.save(pdf_path, resolution=300.0)
    return png_path, pdf_path


def export_selection_manifest(selected_rows: Sequence[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    manifest_rows = []
    for row in selected_rows:
        manifest_rows.append(
            {
                "image_id": row["image_id"],
                "split": row["split"],
                "case_type": row["case_type"],
                "reason": row["reason"],
                "gt_labels": ";".join(row["analysis"]["gt_labels"]),
            }
        )

    json_path = output_dir / "in_domain_qualitative_manifest.json"
    csv_path = output_dir / "in_domain_qualitative_manifest.csv"
    save_json(manifest_rows, json_path)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return json_path, csv_path


def export_note(
    output_dir: Path,
    *,
    split_name: str,
    selected_rows: Sequence[dict[str, Any]],
    model_specs: Sequence[ModelSpec],
) -> Path:
    note_path = output_dir / "in_domain_qualitative_note.txt"
    lines = [
        "In-domain qualitative comparison on the curated benchmark.",
        f"Preferred split: {split_name}.",
        "Compared runs:",
    ]
    for spec in model_specs:
        lines.append(f"- {spec.header}: {spec.run_name}")
    lines.append("Selected cases:")
    for row in selected_rows:
        lines.append(f"- {row['case_type']}: {row['image_id']} ({row['reason']})")
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note_path


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = ensure_dir(args.output_dir)
    image_root_dir = Path(args.image_root_dir) if args.image_root_dir else None
    class_names = [str(name) for name in args.class_names]

    model_specs = [
        ModelSpec(
            key="baseline",
            header=args.baseline_header,
            run_name=args.baseline_run_name,
            predictions_path=Path(args.baseline_predictions),
        ),
        ModelSpec(
            key="yolo",
            header=args.yolo_header,
            run_name=args.yolo_run_name,
            predictions_path=Path(args.yolo_predictions),
        ),
        ModelSpec(
            key="variant",
            header=args.variant_header,
            run_name=args.variant_run_name,
            predictions_path=Path(args.variant_predictions),
        ),
    ]

    missing_paths = [str(path) for path in [manifest_path, *(spec.predictions_path for spec in model_specs)] if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing required input artifacts for the qualitative figure:\n"
            + "\n".join(missing_paths)
        )

    records = load_jsonl_records(manifest_path)
    filtered_records = [record for record in records if str(record.get("split") or "").lower() == args.split.lower()]
    if not filtered_records:
        raise ValueError(f"No records found for split={args.split!r} in {manifest_path}.")
    records_by_image = {str(record["image_id"]): record for record in filtered_records}

    predictions_by_model = {
        spec.key: load_predictions(
            predictions_path=spec.predictions_path,
            class_names=class_names,
            score_threshold=float(args.score_threshold),
        )
        for spec in model_specs
    }

    analyses: list[dict[str, Any]] = []
    for image_id, record in records_by_image.items():
        gt_entries = build_gt_entries(record)
        model_predictions = {
            spec.key: predictions_by_model[spec.key].get(image_id, [])
            for spec in model_specs
        }
        analyses.append(
            build_image_analysis(
                record=record,
                gt_entries=gt_entries,
                model_predictions=model_predictions,
                iou_threshold=float(args.iou_threshold),
            )
        )

    selected_rows = select_cases(
        analyses=analyses,
        baseline_key="baseline",
        yolo_key="yolo",
        variant_key="variant",
        max_rows=int(args.rows),
    )
    if not selected_rows:
        raise ValueError("Case selection returned no examples. Check manifest split or prediction inputs.")

    png_path, pdf_path = render_figure(
        selected_rows=selected_rows,
        records_by_image=records_by_image,
        image_root_dir=image_root_dir,
        model_specs=model_specs,
        panel_width=int(args.panel_width),
        panel_height=int(args.panel_height),
        output_dir=output_dir,
    )
    json_manifest_path, csv_manifest_path = export_selection_manifest(selected_rows, output_dir=output_dir)
    note_path = export_note(
        output_dir=output_dir,
        split_name=args.split,
        selected_rows=selected_rows,
        model_specs=model_specs,
    )

    summary = {
        "figure_png": str(png_path),
        "figure_pdf": str(pdf_path),
        "manifest_json": str(json_manifest_path),
        "manifest_csv": str(csv_manifest_path),
        "note_path": str(note_path),
        "selected_image_ids": [row["image_id"] for row in selected_rows],
        "model_runs": {spec.header: spec.run_name for spec in model_specs},
    }
    save_json(summary, output_dir / "in_domain_qualitative_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
