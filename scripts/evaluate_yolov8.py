#!/usr/bin/env python3
"""Evaluate a YOLOv8 checkpoint with the repo's metric protocol."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.manifest_detection_dataset import load_manifest_records
from src.engine.prediction_eval import (
    build_small_defect_eval_payloads_from_records,
    build_targets_from_manifest_records,
    save_prediction_evaluation_outputs,
)
from src.metrics.detection_metrics import compute_detection_metrics
from src.utils.config import load_yaml
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
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def _resolve_image_path(record: dict, image_root_dir: str | Path) -> Path:
    image_path = Path(str(record.get("image_path") or ""))
    if image_path.is_absolute():
        return image_path
    return Path(image_root_dir) / image_path


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLOv8 evaluation. Install it with `python -m pip install ultralytics`."
        ) from exc

    args = parse_args()
    dataset_cfg = load_yaml(args.dataset_config)
    records, metadata = load_manifest_records(
        dataset_config_or_path=dataset_cfg,
        split=args.split,
        split_seed=42,
        train_ratio=0.8,
        val_ratio=0.1,
    )
    class_names = list(metadata["class_names"])
    image_root_dir = metadata["image_root_dir"]

    model = YOLO(args.checkpoint)
    image_paths = [str(_resolve_image_path(record, image_root_dir)) for record in records]
    results = model.predict(
        source=image_paths,
        stream=True,
        imgsz=int(args.imgsz),
        conf=float(args.score_threshold),
        iou=float(args.iou_threshold),
        max_det=int(args.max_detections),
        device=str(args.device),
        batch=int(args.batch),
        verbose=False,
    )

    predictions = []
    for record, result in zip(records, results):
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

    targets = build_targets_from_manifest_records(records=records, class_names=class_names)
    metric_payload = compute_detection_metrics(
        predictions=predictions,
        targets=targets,
        class_names=class_names,
        score_threshold=float(args.score_threshold),
    )

    summary = dict(metric_payload["summary"])
    summary.update(
        {
            "experiment_name": args.experiment_name,
            "dataset_name": dataset_cfg.get("dataset_name", "unknown_dataset"),
            "split": str(args.split),
            "checkpoint_path": str(Path(args.checkpoint)),
            "class_names": class_names,
            "evaluation_mode": "in_domain",
            "tile_merge": False,
            "detector_family": "yolov8",
        }
    )

    small_payload = None
    if bool(args.small_defect_eval):
        small_payload = build_small_defect_eval_payloads_from_records(
            records=records,
            predictions=predictions,
            class_names=class_names,
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
    logger.info("Saved YOLOv8 evaluation summary to %s/tables", args.output_dir)
    logger.info("Evaluation results: %s", summary)


if __name__ == "__main__":
    main()
