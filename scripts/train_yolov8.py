#!/usr/bin/env python3
"""Train a YOLOv8 model on a YOLO-exported dataset."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the exported YOLO dataset.yaml.")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="Ultralytics model weights to start from.")
    parser.add_argument("--experiment-name", type=str, required=True, help="Experiment name under outputs/yolo.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--project-dir", type=str, default="outputs/yolo")
    parser.add_argument(
        "--box-loss",
        type=str,
        default="default",
        choices=["default", "wniou"],
        help="Optional box regression loss override for Ultralytics training.",
    )
    parser.add_argument(
        "--wniou-lambda-nwd",
        type=float,
        default=0.30,
        help="Blend weight for the NWD term inside the WNIoU hybrid loss.",
    )
    parser.add_argument(
        "--wniou-focus-alpha",
        type=float,
        default=0.50,
        help="Difficulty reweighting strength for the WNIoU hybrid loss.",
    )
    parser.add_argument(
        "--wniou-focus-gamma",
        type=float,
        default=1.00,
        help="Exponent used by the WNIoU difficulty reweighting term.",
    )
    parser.add_argument(
        "--wniou-distance-scale",
        type=float,
        default=0.05,
        help="Distance scale for the normalized Wasserstein similarity on xyxy-normalized boxes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for YOLOv8 training. Install it with `python -m pip install ultralytics`."
        ) from exc
    if args.box_loss == "wniou":
        from src.losses.yolo_wniou import apply_wniou_patch

        wniou_config = apply_wniou_patch(
            lambda_nwd=float(args.wniou_lambda_nwd),
            focus_alpha=float(args.wniou_focus_alpha),
            focus_gamma=float(args.wniou_focus_gamma),
            distance_scale=float(args.wniou_distance_scale),
        )
        logger.info("Enabled WNIoU box loss patch: %s", wniou_config)
    else:
        wniou_config = None

    project_dir = ensure_dir(args.project_dir)
    model = YOLO(args.weights)
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
            "weights": args.weights,
            "epochs": int(args.epochs),
            "imgsz": int(args.imgsz),
            "batch": int(args.batch),
            "device": str(args.device),
            "project_dir": str(project_dir),
            "box_loss": str(args.box_loss),
            "wniou_config": wniou_config,
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
