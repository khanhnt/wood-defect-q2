#!/usr/bin/env python3
"""Training entry point for the simple detector pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.trainer import Trainer
from src.models.builder import build_model
from src.utils.config import load_yaml
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger()


BACKBONE_SCALE_OVERRIDES = {
    "default": {
        "stage_channels": [64, 128, 192, 256],
        "neck_out_channels": 128,
    },
    "medium": {
        "stage_channels": [80, 160, 224, 320],
        "neck_out_channels": 160,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training config")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["cnn", "cnn_transformer", "cnn_p2", "cnn_transformer_p2"],
        help="Optional hybrid ablation override",
    )
    parser.add_argument(
        "--backbone-scale",
        type=str,
        choices=["default", "medium"],
        default=None,
        help="Optional hybrid backbone scale override",
    )
    parser.add_argument("--experiment-name", type=str, default=None, help="Optional experiment name override")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory override")
    parser.add_argument("--device", type=str, default=None, help="Optional device override")
    parser.add_argument("--epochs", type=int, default=None, help="Optional training epoch override")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional training batch size override")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional dataloader worker override")
    parser.add_argument("--learning-rate", type=float, default=None, help="Optional learning rate override")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional train subset size override")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional val subset size override")
    parser.add_argument("--score-threshold", type=float, default=None, help="Optional model score threshold override")
    parser.add_argument("--pre-nms-topk", type=int, default=None, help="Optional pre-NMS top-k override")
    parser.add_argument("--max-detections", type=int, default=None, help="Optional max detections per image override")
    return parser.parse_args()


def _apply_variant_override(model_cfg: Dict[str, Any], variant: str | None) -> Dict[str, Any]:
    if variant is None:
        return model_cfg

    overrides = {
        "cnn": {
            "use_transformer": False,
            "num_transformer_blocks": 0,
            "use_p2_branch": False,
        },
        "cnn_transformer": {
            "use_transformer": True,
            "num_transformer_blocks": 1,
            "use_p2_branch": False,
        },
        "cnn_p2": {
            "use_transformer": False,
            "num_transformer_blocks": 0,
            "use_p2_branch": True,
        },
        "cnn_transformer_p2": {
            "use_transformer": True,
            "num_transformer_blocks": 1,
            "use_p2_branch": True,
        },
    }
    merged = dict(model_cfg)
    merged.update(overrides[variant])
    return merged


def _apply_backbone_scale_override(model_cfg: Dict[str, Any], backbone_scale: str | None) -> Dict[str, Any]:
    if backbone_scale is None or model_cfg.get("name") != "hybrid_detector":
        return model_cfg
    merged = dict(model_cfg)
    merged.update(BACKBONE_SCALE_OVERRIDES[backbone_scale])
    return merged


def _apply_train_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(config)
    train_cfg = dict(config.get("train", {}))
    model_cfg = dict(config.get("model", {}))

    model_cfg = _apply_variant_override(model_cfg=model_cfg, variant=args.variant)
    model_cfg = _apply_backbone_scale_override(model_cfg=model_cfg, backbone_scale=args.backbone_scale)

    if args.device is not None:
        config["device"] = args.device
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name

    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        train_cfg["num_workers"] = int(args.num_workers)
    if args.learning_rate is not None:
        train_cfg["learning_rate"] = float(args.learning_rate)
    if args.max_train_samples is not None:
        train_cfg["max_train_samples"] = int(args.max_train_samples)
    if args.max_val_samples is not None:
        train_cfg["max_val_samples"] = int(args.max_val_samples)

    if args.score_threshold is not None:
        model_cfg["score_threshold"] = float(args.score_threshold)
    if args.pre_nms_topk is not None:
        model_cfg["pre_nms_topk"] = int(args.pre_nms_topk)
    if args.max_detections is not None:
        model_cfg["max_detections"] = int(args.max_detections)

    if args.experiment_name is None:
        experiment_parts = []
        if args.variant is not None:
            experiment_parts.append(args.variant)
        if args.backbone_scale is not None and args.backbone_scale != "default":
            experiment_parts.append(args.backbone_scale)
        if experiment_parts:
            config["experiment_name"] = "hybrid_" + "_".join(experiment_parts)

    config["train"] = train_cfg
    config["model"] = model_cfg
    return config


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    config = _apply_train_overrides(config=config, args=args)
    set_seed(config.get("seed", 42))

    model_cfg = config.get("model", {})
    model = build_model(model_config=model_cfg, train_config=config.get("train", {}))
    trainer = Trainer(model=model, config=config)
    logger.info("Starting training for %s", config.get("experiment_name", model_cfg.get("name", "detector")))
    trainer.fit()


if __name__ == "__main__":
    main()
