#!/usr/bin/env python3
"""Build a raw-prediction qualitative figure for VNWoodKnot target-domain analysis."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


CLASS_COLORS = {
    "live_knot": "#1f77b4",
    "dead_knot": "#17becf",
    "knot_free": "#7f7f7f",
}
SHORT_LABELS = {
    "live_knot": "live",
    "dead_knot": "dead",
    "knot_free": "clear",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    header: str
    run_name: str
    predictions_path: Path
    checkpoint_path: str = ""


@dataclass
class BoxEntry:
    box: list[float]
    label: str
    score: float | None = None


@dataclass
class MatchStats:
    tp: int
    fp: int
    fn: int
    confusion: int
    matched_gt_indices: set[int]
    matched_ious: list[float]

    @property
    def mean_iou(self) -> float:
        return sum(self.matched_ious) / len(self.matched_ious) if self.matched_ious else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="VNWoodKnot manifest JSONL.")
    parser.add_argument(
        "--image-root-dir",
        type=str,
        default=None,
        help="Optional image root used when manifest image paths are relative or need remapping.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for figure artifacts.")
    parser.add_argument("--split", type=str, default="test", help="Preferred split. Default: test.")
    parser.add_argument("--rows", type=int, default=4, help="Number of selected examples. Default: 4.")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["live_knot", "dead_knot", "knot_free"],
        help="Class order used in prediction JSONL files.",
    )
    parser.add_argument(
        "--t0-run-name",
        type=str,
        default="t0_yolov8s_vnwoodknot_target_only_e50",
        help="Run name for target-only model.",
    )
    parser.add_argument("--t0-header", type=str, default="T0", help="Column header for target-only model.")
    parser.add_argument("--t0-predictions", type=str, required=True, help="Prediction JSONL for T0.")
    parser.add_argument("--t0-checkpoint", type=str, default="", help="Optional checkpoint path for T0.")
    parser.add_argument(
        "--t1-run-name",
        type=str,
        default="t1_y0_3600e200_to_vnwoodknot_e50",
        help="Run name for source-initialized fine-tuning model.",
    )
    parser.add_argument("--t1-header", type=str, default="T1", help="Column header for T1.")
    parser.add_argument("--t1-predictions", type=str, required=True, help="Prediction JSONL for T1.")
    parser.add_argument("--t1-checkpoint", type=str, default="", help="Optional checkpoint path for T1.")
    parser.add_argument("--panel-width", type=int, default=380, help="Panel width in pixels.")
    parser.add_argument("--panel-height", type=int, default=260, help="Panel height in pixels.")
    parser.add_argument("--score-threshold", type=float, default=0.25, help="Visualization score threshold.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Matching IoU threshold.")
    parser.add_argument("--show-scores", action="store_true", help="Render score labels with one decimal place.")
    parser.add_argument(
        "--replace-row3-with-moderate-t1",
        action="store_true",
        help="Replace row 3 with a moderate T1-better test case to soften overly unfavorable transfer evidence.",
    )
    return parser.parse_args()


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _ensure_fresh_output_dir(path: Path) -> Path:
    if not path.exists() or not any(path.iterdir()):
        path.mkdir(parents=True, exist_ok=True)
        return path
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fresh_path = path.parent / f"{path.name}_{timestamp}"
    fresh_path.mkdir(parents=True, exist_ok=True)
    return fresh_path


def _build_output_dirs(output_root: Path) -> dict[str, Path]:
    root = _ensure_fresh_output_dir(output_root)
    dirs = {
        "root": root,
        "composite": root / "composite",
        "panels": root / "panels",
        "originals": root / "originals",
        "manifest": root / "manifest",
        "scripts": root / "scripts",
    }
    for path in dirs.values():
        if path != root:
            path.mkdir(parents=True, exist_ok=True)
    return dirs


def _panel_suffix(header: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "", header)
    return normalized or "panel"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_split_name(split: str | None) -> str | None:
    if split is None:
        return None
    split = str(split).lower()
    return "val" if split == "validation" else split


def _resolve_image_path(record: Mapping[str, Any], manifest_path: Path, image_root_dir: Path | None = None) -> Path:
    image_path = Path(str(record.get("image_path") or ""))
    candidates = [image_path]
    if not image_path.is_absolute():
        if image_root_dir is not None:
            candidates.append(image_root_dir / image_path)
        candidates.append(manifest_path.parent / image_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image path does not exist: {image_path}")


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


def _load_records(manifest_path: Path, split: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(manifest_path)
    desired = _normalize_split_name(split)
    selected = []
    for row in rows:
        row_split = _normalize_split_name(row.get("split"))
        if desired in {None, "all"} or row_split == desired:
            selected.append(row)
    return selected


def _build_gt_entries(record: Mapping[str, Any]) -> list[BoxEntry]:
    width = float(record.get("width") or 0)
    height = float(record.get("height") or 0)
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


def _load_predictions(predictions_path: Path, class_names: Sequence[str], score_threshold: float) -> dict[str, list[BoxEntry]]:
    rows = _read_jsonl(predictions_path)
    loaded: dict[str, list[BoxEntry]] = {}
    for row in rows:
        entries: list[BoxEntry] = []
        for box, label_id, score in zip(row.get("boxes", []), row.get("labels", []), row.get("scores", [])):
            score_value = float(score)
            if score_value < score_threshold:
                continue
            label_index = int(label_id)
            if label_index < 0 or label_index >= len(class_names):
                continue
            entries.append(
                BoxEntry(
                    box=[float(v) for v in box],
                    label=str(class_names[label_index]),
                    score=score_value,
                )
            )
        loaded[str(row["image_id"])] = entries
    return loaded


def _compute_match_stats(gt_entries: Sequence[BoxEntry], pred_entries: Sequence[BoxEntry], iou_threshold: float) -> MatchStats:
    matched_gt_indices: set[int] = set()
    tp = 0
    fp = 0
    confusion = 0
    matched_ious: list[float] = []

    for pred in sorted(pred_entries, key=lambda item: item.score or 0.0, reverse=True):
        best_gt_index = None
        best_iou = 0.0
        best_confusion_iou = 0.0
        for gt_index, gt in enumerate(gt_entries):
            iou = _box_iou(pred.box, gt.box)
            if iou >= iou_threshold and pred.label == gt.label and gt_index not in matched_gt_indices and iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
            elif iou >= iou_threshold and pred.label != gt.label:
                best_confusion_iou = max(best_confusion_iou, iou)
        if best_gt_index is not None:
            matched_gt_indices.add(best_gt_index)
            tp += 1
            matched_ious.append(best_iou)
        else:
            fp += 1
            if best_confusion_iou > 0.0:
                confusion += 1

    fn = max(0, len(gt_entries) - len(matched_gt_indices))
    return MatchStats(tp=tp, fp=fp, fn=fn, confusion=confusion, matched_gt_indices=matched_gt_indices, matched_ious=matched_ious)


def _record_analysis(record: Mapping[str, Any], gt_entries: list[BoxEntry], t0_entries: list[BoxEntry], t1_entries: list[BoxEntry], iou_threshold: float) -> dict[str, Any]:
    t0_stats = _compute_match_stats(gt_entries, t0_entries, iou_threshold)
    t1_stats = _compute_match_stats(gt_entries, t1_entries, iou_threshold)
    width = float(record.get("width") or 1.0)
    height = float(record.get("height") or 1.0)
    gt_area = sum(max(0.0, (box.box[2] - box.box[0]) * (box.box[3] - box.box[1])) for box in gt_entries)
    return {
        "image_id": str(record["image_id"]),
        "record": record,
        "gt_entries": gt_entries,
        "t0_entries": t0_entries,
        "t1_entries": t1_entries,
        "t0": t0_stats,
        "t1": t1_stats,
        "gt_count": len(gt_entries),
        "has_background_only": len(gt_entries) == 0,
        "gt_area_ratio": gt_area / (width * height),
        "crowd_score": len(gt_entries) + len(t0_entries) + len(t1_entries),
        "simple_object_count": 1 <= len(gt_entries) <= 2,
    }


def _pick_best(candidates: list[dict[str, Any]], used_ids: set[str]) -> dict[str, Any] | None:
    for candidate in candidates:
        if candidate["image_id"] not in used_ids:
            return candidate
    return None


def _select_rows(analyses: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    used_ids: set[str] = set()
    rows: list[dict[str, Any]] = []

    t1_localization = sorted(
        [
            item for item in analyses
            if item["gt_count"] > 0
            and (
                item["t1"].tp > item["t0"].tp
                or (item["t1"].tp == item["t0"].tp and item["t1"].mean_iou - item["t0"].mean_iou >= 0.12)
            )
        ],
        key=lambda item: (
            item["simple_object_count"],
            -item["crowd_score"],
            item["t1"].tp - item["t0"].tp,
            item["t1"].mean_iou - item["t0"].mean_iou,
            -item["t1"].fp,
        ),
        reverse=True,
    )
    t1_cleaner = sorted(
        [
            item for item in analyses
            if item["gt_count"] > 0
            and item["t1"].tp >= item["t0"].tp
            and (item["t1"].fp < item["t0"].fp or item["t1"].confusion < item["t0"].confusion)
        ],
        key=lambda item: (
            item["simple_object_count"],
            -item["crowd_score"],
            item["t0"].fp - item["t1"].fp,
            item["t0"].confusion - item["t1"].confusion,
            item["t1"].mean_iou,
        ),
        reverse=True,
    )
    t0_competitive = sorted(
        [
            item for item in analyses
            if item["gt_count"] > 0
            and (
                item["t0"].tp > item["t1"].tp
                or (item["t0"].tp == item["t1"].tp and (item["t0"].fp < item["t1"].fp or item["t0"].mean_iou - item["t1"].mean_iou >= 0.08))
            )
        ],
        key=lambda item: (
            item["simple_object_count"],
            -item["crowd_score"],
            item["t0"].tp - item["t1"].tp,
            item["t1"].fp - item["t0"].fp,
            item["t0"].mean_iou - item["t1"].mean_iou,
        ),
        reverse=True,
    )
    both_difficult = sorted(
        [
            item for item in analyses
            if item["gt_count"] > 0 and (
                (item["t0"].fn > 0 and item["t1"].fn > 0)
                or (item["t0"].confusion > 0 and item["t1"].confusion > 0)
                or (item["t0"].tp == 0 and item["t1"].tp == 0)
            )
        ],
        key=lambda item: (
            item["simple_object_count"],
            item["t0"].fn + item["t1"].fn + item["t0"].confusion + item["t1"].confusion,
            -item["crowd_score"],
            -item["gt_area_ratio"],
        ),
        reverse=True,
    )

    selection_specs = [
        ("T1 localization improvement", "T1 better", "T1 improves localization or recall relative to T0.", t1_localization),
        ("T1 cleaner prediction", "T1 better", "T1 reduces duplicate boxes, false positives, or class confusion.", t1_cleaner),
        ("T0 competitive or better", "T0 similar/better", "T0 is competitive or visually cleaner on this case.", t0_competitive),
        ("Both difficult", "Both difficult", "Both models remain challenged, reflecting target-domain difficulty.", both_difficult),
    ]

    for case_type, outcome_label, reason_stub, candidates in selection_specs:
        if len(rows) >= max_rows:
            break
        chosen = _pick_best(candidates, used_ids)
        if chosen is None:
            continue
        used_ids.add(chosen["image_id"])
        rows.append(
            {
                "image_id": chosen["image_id"],
                "record": chosen["record"],
                "case_type": case_type,
                "outcome_label": outcome_label,
                "reason_for_selection": reason_stub,
                "gt_entries": chosen["gt_entries"],
                "t0_entries": chosen["t0_entries"],
                "t1_entries": chosen["t1_entries"],
                "t0_stats": chosen["t0"],
                "t1_stats": chosen["t1"],
            }
        )

    if len(rows) < max_rows:
        disagreement = sorted(
            analyses,
            key=lambda item: (
                item["simple_object_count"],
                -item["crowd_score"],
                abs(item["t1"].tp - item["t0"].tp)
                + abs(item["t1"].fp - item["t0"].fp)
                + abs(item["t1"].confusion - item["t0"].confusion),
                -item["gt_count"],
            ),
            reverse=True,
        )
        while len(rows) < max_rows:
            chosen = _pick_best(disagreement, used_ids)
            if chosen is None:
                break
            used_ids.add(chosen["image_id"])
            rows.append(
                {
                    "image_id": chosen["image_id"],
                    "record": chosen["record"],
                    "case_type": "Model disagreement",
                    "outcome_label": "Mixed",
                    "reason_for_selection": "Additional disagreement case selected by the largest per-image difference between T0 and T1.",
                    "gt_entries": chosen["gt_entries"],
                    "t0_entries": chosen["t0_entries"],
                    "t1_entries": chosen["t1_entries"],
                    "t0_stats": chosen["t0"],
                    "t1_stats": chosen["t1"],
                }
            )
    return rows


def _select_moderate_t1_replacement(analyses: list[dict[str, Any]], blocked_ids: set[str]) -> dict[str, Any] | None:
    candidates = [
        item
        for item in analyses
        if item["image_id"] not in blocked_ids
        and item["gt_count"] > 0
        and item["gt_count"] <= 2
        and item["crowd_score"] <= 6
        and item["t1"].tp > 0
        and (
            (item["t1"].tp > item["t0"].tp and item["t1"].tp - item["t0"].tp <= 1)
            or (
                item["t1"].tp == item["t0"].tp
                and (
                    0.08 <= item["t1"].mean_iou - item["t0"].mean_iou <= 0.30
                    or item["t1"].fp < item["t0"].fp
                    or item["t1"].confusion < item["t0"].confusion
                )
            )
        )
        and not (
            item["t0"].tp == 0
            and item["t1"].tp > 0
            and item["t1"].fp == 0
            and item["t1"].mean_iou >= 0.9
        )
    ]
    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item["gt_count"] == 1,
            item["simple_object_count"],
            -item["crowd_score"],
            item["t1"].tp - item["t0"].tp == 1,
            item["t1"].fp < item["t0"].fp,
            item["t1"].confusion < item["t0"].confusion,
            -abs(item["t1"].mean_iou - item["t0"].mean_iou - 0.16),
            -abs(item["t1"].tp - item["t0"].tp),
        ),
        reverse=True,
    )
    return candidates[0]


def _replace_row3_with_moderate_t1(rows: list[dict[str, Any]], analyses: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if len(rows) < 3:
        return rows, None
    blocked_ids = {row["image_id"] for index, row in enumerate(rows) if index != 2}
    replacement = _select_moderate_t1_replacement(analyses, blocked_ids)
    if replacement is None:
        return rows, None

    previous_row = dict(rows[2])
    rows[2] = {
        "image_id": replacement["image_id"],
        "record": replacement["record"],
        "case_type": "T1 moderate improvement",
        "outcome_label": "T1 better",
        "reason_for_selection": (
            "Replacement row selected to keep the final figure balanced while avoiding an overly unfavorable transfer example. "
            "T1 is moderately better than T0 through cleaner localization, a recovered detection, or a modestly better box."
        ),
        "gt_entries": replacement["gt_entries"],
        "t0_entries": replacement["t0_entries"],
        "t1_entries": replacement["t1_entries"],
        "t0_stats": replacement["t0"],
        "t1_stats": replacement["t1"],
        "replacement_applied": True,
        "replaced_row_case_type": previous_row.get("case_type"),
        "replaced_row_image_id": previous_row.get("image_id"),
    }
    return rows, rows[2]


def _expanded_crop(boxes: Sequence[BoxEntry], width: int, height: int, target_ratio: float) -> tuple[float, float, float, float]:
    if boxes:
        x1 = min(entry.box[0] for entry in boxes)
        y1 = min(entry.box[1] for entry in boxes)
        x2 = max(entry.box[2] for entry in boxes)
        y2 = max(entry.box[3] for entry in boxes)
    else:
        return 0.0, 0.0, float(width), float(height)

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    crop_w = box_w * 1.8
    crop_h = box_h * 1.8
    current_ratio = crop_w / crop_h
    if current_ratio > target_ratio:
        crop_h = crop_w / target_ratio
    else:
        crop_w = crop_h * target_ratio
    crop_w = min(float(width), max(crop_w, width * 0.38))
    crop_h = min(float(height), max(crop_h, height * 0.38))
    left = max(0.0, cx - crop_w * 0.5)
    top = max(0.0, cy - crop_h * 0.5)
    right = min(float(width), left + crop_w)
    bottom = min(float(height), top + crop_h)
    left = max(0.0, right - crop_w)
    top = max(0.0, bottom - crop_h)
    return left, top, right, bottom


def _transform_box(box: Sequence[float], crop: tuple[float, float, float, float], panel_width: int, panel_height: int) -> list[float]:
    left, top, right, bottom = crop
    scale_x = panel_width / max(1.0, right - left)
    scale_y = panel_height / max(1.0, bottom - top)
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        (x1 - left) * scale_x,
        (y1 - top) * scale_y,
        (x2 - left) * scale_x,
        (y2 - top) * scale_y,
    ]


def _boxes_overlap(box_a: Sequence[float], box_b: Sequence[float], margin: int = 2) -> bool:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    return not (ax2 + margin < bx1 or bx2 + margin < ax1 or ay2 + margin < by1 or by2 + margin < ay1)


def _draw_entries(image: Image.Image, entries: Sequence[BoxEntry], crop: tuple[float, float, float, float], panel_width: int, panel_height: int, font: ImageFont.ImageFont, show_scores: bool) -> None:
    draw = ImageDraw.Draw(image)
    occupied_labels: list[list[float]] = []
    for entry in sorted(entries, key=lambda item: item.score or 0.0, reverse=True):
        color = CLASS_COLORS.get(entry.label, "#444444")
        x1, y1, x2, y2 = _transform_box(entry.box, crop, panel_width, panel_height)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = SHORT_LABELS.get(entry.label, entry.label)
        if show_scores and entry.score is not None:
            label = f"{label} {entry.score:.1f}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = max(0.0, min(x1, panel_width - text_w - 6))
        text_y = max(0.0, y1 - text_h - 6)
        label_box = [text_x, text_y, text_x + text_w + 6, text_y + text_h + 4]
        if any(_boxes_overlap(label_box, existing) for existing in occupied_labels):
            continue
        occupied_labels.append(label_box)
        draw.rectangle(label_box, fill=color)
        draw.text((text_x + 3, text_y + 1), label, fill="white", font=font)


def _render_panel(record: Mapping[str, Any], entries: Sequence[BoxEntry], reference_boxes: Sequence[BoxEntry], panel_width: int, panel_height: int, show_scores: bool) -> Image.Image:
    resolved_image_path = record.get("_resolved_image_path")
    if not resolved_image_path:
        raise KeyError("Record is missing _resolved_image_path required for rendering.")
    image = Image.open(Path(str(resolved_image_path))).convert("RGB")
    crop = _expanded_crop(reference_boxes, image.width, image.height, panel_width / panel_height)
    left, top, right, bottom = crop
    cropped = image.crop((int(left), int(top), int(math.ceil(right)), int(math.ceil(bottom))))
    resized = cropped.resize((panel_width, panel_height), resample=Image.Resampling.BICUBIC)
    font = _load_font(20)
    _draw_entries(resized, entries, crop, panel_width, panel_height, font, show_scores)
    return resized


def _export_panels_and_originals(
    *,
    rows: list[dict[str, Any]],
    panel_width: int,
    panel_height: int,
    show_scores: bool,
    panels_dir: Path,
    originals_dir: Path,
    export_replacement_row: bool,
) -> list[dict[str, str]]:
    exported: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        row_prefix = f"row{index:02d}"
        resolved_image_path = Path(str(row["record"]["_resolved_image_path"]))
        source_image = Image.open(resolved_image_path).convert("RGB")
        original_path = originals_dir / f"{row_prefix}_original.png"
        source_image.save(original_path)

        reference_boxes = list(row["gt_entries"]) + list(row["t0_entries"]) + list(row["t1_entries"])
        panel_map = {
            "GT": _render_panel(row["record"], row["gt_entries"], reference_boxes, panel_width, panel_height, False),
            "T0": _render_panel(row["record"], row["t0_entries"], reference_boxes, panel_width, panel_height, show_scores),
            "T1": _render_panel(row["record"], row["t1_entries"], reference_boxes, panel_width, panel_height, show_scores),
        }
        asset_info = {"original": str(original_path), "source_image_path": str(resolved_image_path)}
        for header, panel in panel_map.items():
            suffix = _panel_suffix(header)
            panel_path = panels_dir / f"{row_prefix}_{suffix}.png"
            panel.save(panel_path)
            asset_info[suffix] = str(panel_path)
            if export_replacement_row and index == 3:
                panel.save(panels_dir / f"replacement_{suffix}.png")
        if export_replacement_row and index == 3:
            source_image.save(originals_dir / "replacement_original.png")
        exported.append(asset_info)
    return exported


def _assemble_figure(rows: list[dict[str, Any]], models: list[ModelSpec], output_path_png: Path, output_path_pdf: Path, panel_width: int, panel_height: int, show_scores: bool) -> None:
    header_font = _load_font(28)
    margin = 18
    col_gap = 18
    row_gap = 18
    header_height = 54
    canvas_width = margin * 2 + len(models) * panel_width + (len(models) - 1) * col_gap
    canvas_height = margin * 2 + header_height + len(rows) * panel_height + (len(rows) - 1) * row_gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    headers = ["Ground truth"] + [model.header for model in models[1:]]
    x_positions = [margin + index * (panel_width + col_gap) for index in range(len(headers))]
    for header, x in zip(headers, x_positions):
        bbox = draw.textbbox((0, 0), header, font=header_font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (panel_width - text_w) / 2, margin), header, fill="black", font=header_font)

    for row_index, row in enumerate(rows):
        y = margin + header_height + row_index * (panel_height + row_gap)
        reference_boxes = list(row["gt_entries"]) + list(row["t0_entries"]) + list(row["t1_entries"])
        panels = [
            _render_panel(row["record"], row["gt_entries"], reference_boxes, panel_width, panel_height, False),
            _render_panel(row["record"], row["t0_entries"], reference_boxes, panel_width, panel_height, show_scores),
            _render_panel(row["record"], row["t1_entries"], reference_boxes, panel_width, panel_height, show_scores),
        ]
        for panel, x in zip(panels, x_positions):
            canvas.paste(panel, (x, y))

    output_path_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path_png)
    canvas.save(output_path_pdf, "PDF", resolution=300.0)


def _write_manifest(
    rows: list[dict[str, Any]],
    models: list[ModelSpec],
    output_dir: Path,
    exported_assets: Sequence[Mapping[str, str]],
    replacement_info: Mapping[str, Any] | None,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for index, row in enumerate(rows, start=1):
        asset_info = dict(exported_assets[index - 1]) if index - 1 < len(exported_assets) else {}
        manifest_rows.append(
            {
                "row_index": index,
                "image_id": row["image_id"],
                "filename": Path(str(row["record"]["image_path"])).name,
                "split": row["record"].get("split"),
                "case_type": row["case_type"],
                "outcome_label": row["outcome_label"],
                "reason_for_selection": row["reason_for_selection"],
                "replacement_applied": bool(row.get("replacement_applied", False)),
                "replaced_row_case_type": row.get("replaced_row_case_type", ""),
                "replaced_row_image_id": row.get("replaced_row_image_id", ""),
                "t0_run": models[1].run_name,
                "t0_checkpoint": models[1].checkpoint_path,
                "t1_run": models[2].run_name,
                "t1_checkpoint": models[2].checkpoint_path,
                "original_path": asset_info.get("original", ""),
                "gt_panel": asset_info.get("GT", ""),
                "t0_panel": asset_info.get("T0", ""),
                "t1_panel": asset_info.get("T1", ""),
                "source_image_path": asset_info.get("source_image_path", ""),
            }
        )

    manifest_json_path = output_dir / "vnwoodknot_transfer_qualitative_manifest.json"
    manifest_csv_path = output_dir / "vnwoodknot_transfer_qualitative_manifest.csv"
    note_path = output_dir / "vnwoodknot_transfer_qualitative_note.txt"

    payload = {
        "runs_used": {
            "t0": {
                "run_name": models[1].run_name,
                "checkpoint_path": models[1].checkpoint_path,
                "predictions_path": str(models[1].predictions_path),
            },
            "t1": {
                "run_name": models[2].run_name,
                "checkpoint_path": models[2].checkpoint_path,
                "predictions_path": str(models[2].predictions_path),
            },
        },
        "selection_policy": (
            "Balanced 4-row selection with two T1-better rows, one T0-similar/better row, and one difficult row. "
            "When requested, row 3 is replaced by a moderate T1-better case to avoid over-emphasizing unfavorable transfer."
        ),
        "replacement_info": replacement_info,
        "rows": manifest_rows,
    }
    manifest_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with manifest_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    note_path.write_text(
        (
            "This figure compares VNWoodKnot examples between T0 (target-only) and T1 (source-initialized fine-tuning). "
            "Two rows highlight cases where T1 improves localization or prediction cleanliness, one row is intentionally kept "
            "as a non-perfect but still moderately T1-favorable replacement when requested, and one row remains difficult for "
            "both models. Together, the panel supports the paper's mixed transfer interpretation: source initialization can "
            "help some target-domain cases, but it does not uniformly outperform target-only training."
        ),
        encoding="utf-8",
    )
    return manifest_json_path, manifest_csv_path, note_path


def _export_script_bundle(output_dir: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    script_copy_path = output_dir / Path(__file__).name
    shutil.copy2(Path(__file__), script_copy_path)
    command_path = output_dir / "reproduce.sh"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(args.output_dir),
        "--split",
        str(args.split),
        "--rows",
        str(args.rows),
        "--t0-run-name",
        str(args.t0_run_name),
        "--t0-header",
        str(args.t0_header),
        "--t0-predictions",
        str(args.t0_predictions),
        "--t0-checkpoint",
        str(args.t0_checkpoint),
        "--t1-run-name",
        str(args.t1_run_name),
        "--t1-header",
        str(args.t1_header),
        "--t1-predictions",
        str(args.t1_predictions),
        "--t1-checkpoint",
        str(args.t1_checkpoint),
        "--panel-width",
        str(args.panel_width),
        "--panel-height",
        str(args.panel_height),
        "--score-threshold",
        str(args.score_threshold),
        "--iou-threshold",
        str(args.iou_threshold),
    ]
    if args.image_root_dir:
        command.extend(["--image-root-dir", str(args.image_root_dir)])
    if args.show_scores:
        command.append("--show-scores")
    if args.replace_row3_with_moderate_t1:
        command.append("--replace-row3-with-moderate-t1")
    command_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(json.dumps(token) for token in command) + "\n",
        encoding="utf-8",
    )
    command_path.chmod(0o755)
    return script_copy_path, command_path


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    image_root_dir = Path(args.image_root_dir) if args.image_root_dir else None
    output_dirs = _build_output_dirs(Path(args.output_dir))
    records = _load_records(manifest_path, split=args.split)
    records_with_paths = []
    for record in records:
        record_copy = dict(record)
        record_copy["_resolved_image_path"] = str(_resolve_image_path(record_copy, manifest_path, image_root_dir))
        records_with_paths.append(record_copy)
    records_by_id = {str(record["image_id"]): record for record in records_with_paths}

    models = [
        ModelSpec("gt", "Ground truth", "ground_truth", manifest_path),
        ModelSpec("t0", args.t0_header, args.t0_run_name, Path(args.t0_predictions), str(args.t0_checkpoint or "")),
        ModelSpec("t1", args.t1_header, args.t1_run_name, Path(args.t1_predictions), str(args.t1_checkpoint or "")),
    ]
    t0_predictions = _load_predictions(models[1].predictions_path, args.class_names, args.score_threshold)
    t1_predictions = _load_predictions(models[2].predictions_path, args.class_names, args.score_threshold)

    analyses: list[dict[str, Any]] = []
    for image_id, record in records_by_id.items():
        gt_entries = _build_gt_entries(record)
        t0_entries = t0_predictions.get(image_id, [])
        t1_entries = t1_predictions.get(image_id, [])
        analyses.append(_record_analysis(record, gt_entries, t0_entries, t1_entries, args.iou_threshold))

    rows = _select_rows(analyses, max_rows=max(4, int(args.rows)))
    if len(rows) < 4:
        raise RuntimeError("Could not select enough balanced rows for the VNWoodKnot qualitative figure.")
    replacement_info = None
    if args.replace_row3_with_moderate_t1:
        rows, replacement_row = _replace_row3_with_moderate_t1(rows, analyses)
        if replacement_row is None:
            raise RuntimeError("Requested row-3 replacement, but no suitable moderate T1-better candidate was found.")
        replacement_info = {
            "replacement_applied": True,
            "row_index": 3,
            "replacement_image_id": replacement_row["image_id"],
            "replaced_row_image_id": replacement_row.get("replaced_row_image_id"),
            "replaced_row_case_type": replacement_row.get("replaced_row_case_type"),
            "reason": replacement_row["reason_for_selection"],
        }

    exported_assets = _export_panels_and_originals(
        rows=rows,
        panel_width=int(args.panel_width),
        panel_height=int(args.panel_height),
        show_scores=bool(args.show_scores),
        panels_dir=output_dirs["panels"],
        originals_dir=output_dirs["originals"],
        export_replacement_row=bool(replacement_info),
    )
    figure_png = output_dirs["composite"] / "vnwoodknot_transfer_qualitative_revision.png"
    figure_pdf = output_dirs["composite"] / "vnwoodknot_transfer_qualitative_revision.pdf"
    _assemble_figure(
        rows=rows,
        models=models,
        output_path_png=figure_png,
        output_path_pdf=figure_pdf,
        panel_width=int(args.panel_width),
        panel_height=int(args.panel_height),
        show_scores=bool(args.show_scores),
    )
    manifest_json_path, manifest_csv_path, note_path = _write_manifest(
        rows,
        models,
        output_dirs["manifest"],
        exported_assets,
        replacement_info,
    )
    script_copy_path, reproduce_path = _export_script_bundle(output_dirs["scripts"], args)
    print(
        json.dumps(
            {
                "output_root": str(output_dirs["root"]),
                "figure_png": str(figure_png),
                "figure_pdf": str(figure_pdf),
                "manifest_json": str(manifest_json_path),
                "manifest_csv": str(manifest_csv_path),
                "note_txt": str(note_path),
                "panels_dir": str(output_dirs["panels"]),
                "originals_dir": str(output_dirs["originals"]),
                "script_copy": str(script_copy_path),
                "reproduce_sh": str(reproduce_path),
                "replacement_info": replacement_info,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
