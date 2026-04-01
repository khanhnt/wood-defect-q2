#!/usr/bin/env python3
"""Evaluate a YOLOv8 checkpoint with the repo's metric protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.label_mapping import (
    remap_predictions_and_targets_for_cross_dataset,
    resolve_cross_dataset_label_mapping,
)
from src.datasets.manifest_detection_dataset import load_manifest_records
from src.engine.prediction_eval import (
    build_small_defect_eval_payloads_from_records,
    build_targets_from_manifest_records,
    save_prediction_evaluation_outputs,
)
from src.metrics.detection_metrics import compute_detection_metrics
from src.utils.config import load_yaml
from src.utils.io import save_csv
from src.utils.logger import setup_logger

logger = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=str, required=True, help="Dataset config YAML for the evaluation split.")
    parser.add_argument("--checkpoint", type=str, required=True, help="YOLOv8 checkpoint path.")
    parser.add_argument("--experiment-name", type=str, required=True, help="Experiment name for saved tables.")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--small-defect-eval", action="store_true")
    parser.add_argument("--cross-dataset", action="store_true", help="Apply overlap-only label remapping for external evaluation.")
    parser.add_argument(
        "--in-domain-summary-path",
        type=str,
        default=None,
        help="Optional in-domain summary JSON used to export a comparison CSV for cross-dataset runs.",
    )
    parser.add_argument(
        "--in-domain-dataset-label",
        type=str,
        default="main_validation",
        help="Label shown for the in-domain row in cross-dataset comparison exports.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def _resolve_image_path(record: dict, image_root_dir: str | Path) -> Path:
    image_path = Path(str(record.get("image_path") or ""))
    if image_path.is_absolute():
        return image_path
    return Path(image_root_dir) / image_path


def _extract_model_class_names(model: object) -> list[str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def _build_cross_dataset_summary(
    *,
    mapping_report: dict,
    remap_counts: dict,
    source_class_names: list[str],
    target_class_names: list[str],
) -> dict:
    return {
        "source_class_names": list(source_class_names),
        "target_class_names": list(target_class_names),
        "evaluated_class_names": list(mapping_report["mapped_class_names"]),
        "mapped_target_classes": list(mapping_report["mapped_target_classes"]),
        "ignored_target_classes": list(mapping_report["ignored_target_classes"]),
        "unmatched_target_classes": list(mapping_report["unmatched_target_classes"]),
        "unmatched_source_classes": list(mapping_report["unmatched_source_classes"]),
        "ignored_prediction_count": int(remap_counts["ignored_prediction_count"]),
        "ignored_target_annotation_count": int(remap_counts["ignored_target_annotation_count"]),
        "mapped_prediction_count": int(remap_counts["mapped_prediction_count"]),
        "mapped_target_annotation_count": int(remap_counts["mapped_target_annotation_count"]),
        "assumption": "Only mapped overlapping classes are evaluated. Unmapped source predictions are ignored for cross-dataset scoring.",
    }


def _export_cross_dataset_reports(
    *,
    output_dir: str | Path,
    experiment_name: str,
    split_name: str,
    current_summary: dict,
    mapping_report: dict,
    in_domain_summary_path: str | None,
    in_domain_dataset_label: str,
) -> None:
    tables_dir = Path(output_dir) / "tables"
    mapping_path = tables_dir / f"{experiment_name}_{split_name}_label_mapping.csv"
    save_csv(mapping_report["mapping_table"], mapping_path)

    comparison_rows = []
    if in_domain_summary_path:
        with open(in_domain_summary_path, "r", encoding="utf-8") as handle:
            in_domain_summary = json.load(handle)
        comparison_rows.append(
            {
                "evaluation_scope": "in_domain",
                "dataset_label": in_domain_dataset_label,
                "split": in_domain_summary.get("split", "val"),
                "mAP50": in_domain_summary.get("mAP50"),
                "mAP50_95": in_domain_summary.get("mAP50_95"),
                "precision50": in_domain_summary.get("precision50"),
                "recall50": in_domain_summary.get("recall50"),
                "num_images": in_domain_summary.get("num_images"),
                "num_targets": in_domain_summary.get("num_targets"),
                "evaluated_classes": ";".join(in_domain_summary.get("class_names", [])),
                "mapped_classes": "",
                "ignored_classes": "",
                "unmatched_classes": "",
            }
        )

    cross_unmatched = sorted(
        set(current_summary.get("unmatched_target_classes", []))
        | set(current_summary.get("unmatched_source_classes", []))
    )
    comparison_rows.append(
        {
            "evaluation_scope": "cross_dataset",
            "dataset_label": current_summary.get("dataset_name", "cross_dataset"),
            "split": split_name,
            "mAP50": current_summary.get("mAP50"),
            "mAP50_95": current_summary.get("mAP50_95"),
            "precision50": current_summary.get("precision50"),
            "recall50": current_summary.get("recall50"),
            "num_images": current_summary.get("num_images"),
            "num_targets": current_summary.get("num_targets"),
            "evaluated_classes": ";".join(current_summary.get("evaluated_class_names", [])),
            "mapped_classes": ";".join(current_summary.get("mapped_target_classes", [])),
            "ignored_classes": ";".join(current_summary.get("ignored_target_classes", [])),
            "unmatched_classes": ";".join(cross_unmatched),
        }
    )
    comparison_path = tables_dir / f"{experiment_name}_{split_name}_comparison.csv"
    save_csv(pd.DataFrame(comparison_rows), comparison_path)


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLOv8 evaluation. Install it with `python -m pip install ultralytics`."
        ) from exc

    args = parse_args()
    if args.cross_dataset and args.small_defect_eval:
        raise ValueError("Small-defect eval is not supported together with cross-dataset remapping.")

    dataset_cfg = load_yaml(args.dataset_config)
    records, metadata = load_manifest_records(
        dataset_config_or_path=dataset_cfg,
        split=args.split,
        split_seed=42,
        train_ratio=0.8,
        val_ratio=0.1,
    )
    target_class_names = list(metadata["class_names"])
    image_root_dir = metadata["image_root_dir"]

    model = YOLO(args.checkpoint)
    source_class_names = _extract_model_class_names(model) or list(target_class_names)
    image_paths = [str(_resolve_image_path(record, image_root_dir)) for record in records]
    predictions = []
    batch_size = max(int(args.batch), 1)
    for start in range(0, len(records), batch_size):
        record_batch = records[start:start + batch_size]
        path_batch = image_paths[start:start + batch_size]
        results = model.predict(
            source=path_batch,
            stream=True,
            imgsz=int(args.imgsz),
            conf=float(args.score_threshold),
            iou=float(args.iou_threshold),
            max_det=int(args.max_detections),
            device=str(args.device),
            batch=len(path_batch),
            verbose=False,
        )

        for record, result in zip(record_batch, results):
            boxes = result.boxes
            if boxes is None:
                xyxy = torch.zeros((0, 4), dtype=torch.float32)
                scores = torch.zeros((0,), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                xyxy = boxes.xyxy.detach().cpu().to(dtype=torch.float32)
                scores = boxes.conf.detach().cpu().to(dtype=torch.float32)
                labels = boxes.cls.detach().cpu().to(dtype=torch.int64)

            predictions.append(
                {
                    "image_id": str(record["image_id"]),
                    "boxes": xyxy,
                    "scores": scores,
                    "labels": labels,
                }
            )

    targets = build_targets_from_manifest_records(records=records, class_names=target_class_names)
    metric_predictions = predictions
    metric_targets = targets
    metric_class_names = list(target_class_names)
    mapping_report = None
    cross_dataset_summary = None

    if bool(args.cross_dataset):
        mapping_report = resolve_cross_dataset_label_mapping(
            source_class_names=source_class_names,
            target_class_names=target_class_names,
            label_mapping=dataset_cfg.get("label_mapping"),
        )
        metric_predictions, metric_targets, remap_counts = remap_predictions_and_targets_for_cross_dataset(
            predictions=predictions,
            targets=targets,
            mapping_report=mapping_report,
        )
        metric_class_names = list(mapping_report["mapped_class_names"])
        if not metric_class_names:
            raise ValueError("Cross-dataset evaluation produced no mapped classes to score.")
        cross_dataset_summary = _build_cross_dataset_summary(
            mapping_report=mapping_report,
            remap_counts=remap_counts,
            source_class_names=source_class_names,
            target_class_names=target_class_names,
        )

    metric_payload = compute_detection_metrics(
        predictions=metric_predictions,
        targets=metric_targets,
        class_names=metric_class_names,
        score_threshold=float(args.score_threshold),
    )

    summary = dict(metric_payload["summary"])
    summary.update(
        {
            "experiment_name": args.experiment_name,
            "dataset_name": dataset_cfg.get("dataset_name", "unknown_dataset"),
            "split": str(args.split),
            "checkpoint_path": str(Path(args.checkpoint)),
            "class_names": list(metric_class_names),
            "evaluation_mode": "cross_dataset" if mapping_report is not None else "in_domain",
            "tile_merge": False,
            "detector_family": "yolov8",
        }
    )
    if cross_dataset_summary is not None:
        summary.update(cross_dataset_summary)

    small_payload = None
    if bool(args.small_defect_eval):
        small_payload = build_small_defect_eval_payloads_from_records(
            records=records,
            predictions=predictions,
            class_names=target_class_names,
            small_defect_config=dataset_cfg.get("small_defect"),
            score_threshold=float(args.score_threshold),
        )
        if small_payload is not None:
            summary["small_defect_rule"] = small_payload["rule"]
            summary["small_target_mAP50"] = small_payload["small_target"]["summary"]["mAP50"]
            summary["small_target_mAP50_95"] = small_payload["small_target"]["summary"]["mAP50_95"]
            summary["small_target_num_images"] = small_payload["small_target"]["summary"]["num_images"]
            summary["small_target_num_targets"] = small_payload["small_target"]["summary"]["num_targets"]
            summary["small_image_subset_mAP50"] = small_payload["small_image"]["summary"]["mAP50"]
            summary["small_image_subset_mAP50_95"] = small_payload["small_image"]["summary"]["mAP50_95"]
            summary["small_image_subset_num_images"] = small_payload["small_image"]["summary"]["num_images"]
            summary["small_image_subset_num_targets"] = small_payload["small_image"]["summary"]["num_targets"]

    save_prediction_evaluation_outputs(
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        split_name=str(args.split),
        summary=summary,
        per_class=metric_payload["per_class"],
        small_defect_eval_payload=small_payload,
    )
    if mapping_report is not None:
        _export_cross_dataset_reports(
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            split_name=str(args.split),
            current_summary=summary,
            mapping_report=mapping_report,
            in_domain_summary_path=args.in_domain_summary_path,
            in_domain_dataset_label=str(args.in_domain_dataset_label),
        )
    logger.info("Saved YOLOv8 evaluation summary to %s/tables", args.output_dir)
    logger.info("Evaluation results: %s", summary)


if __name__ == "__main__":
    main()
