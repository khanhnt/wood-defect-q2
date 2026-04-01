#!/usr/bin/env python3
"""Train a YOLOv8 baseline on a YOLO-exported dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, save_json
from src.utils.logger import setup_logger

logger = setup_logger()


def resolve_model_source(model: str | None, weights: str | None) -> str:
    """Resolve the Ultralytics model source from a named model alias or direct checkpoint path."""
    if weights:
        return str(weights)

    model_name = str(model or "").strip()
    if not model_name:
        raise ValueError("Either --model or --weights must be provided.")
    if model_name.endswith((".pt", ".yaml")):
        return model_name
    return f"{model_name}.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the exported YOLO dataset.yaml.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s",
        help="Ultralytics model alias such as yolov8s, yolov8m, or yolo11s. Ignored when --weights is set.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional explicit Ultralytics checkpoint or model yaml. Overrides --model when provided.",
    )
    parser.add_argument("--experiment-name", type=str, required=True, help="Experiment name under outputs/yolo.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--project-dir", type=str, default="outputs/yolo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLOv8 training. Install it with `python -m pip install ultralytics`."
        ) from exc

    project_dir = ensure_dir(args.project_dir)
    model_source = resolve_model_source(args.model, args.weights)
    model = YOLO(model_source)
    results = model.train(
        data=str(Path(args.data).resolve()),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        workers=int(args.workers),
        seed=int(args.seed),
        patience=int(args.patience),
        project=str(project_dir),
        name=args.experiment_name,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    save_json(
        {
            "experiment_name": args.experiment_name,
            "model": args.model,
            "weights": args.weights,
            "model_source": model_source,
            "epochs": int(args.epochs),
            "imgsz": int(args.imgsz),
            "batch": int(args.batch),
            "device": str(args.device),
            "project_dir": str(project_dir),
            "result": str(results),
            "best_checkpoint_path": str(project_dir / args.experiment_name / "weights" / "best.pt"),
        },
        Path("outputs/tables") / f"{args.experiment_name}_train_summary.json",
    )
    logger.info(
        "YOLOv8 training finished for %s. Best checkpoint: %s",
        args.experiment_name,
        project_dir / args.experiment_name / "weights" / "best.pt",
    )


if __name__ == "__main__":
    main()
