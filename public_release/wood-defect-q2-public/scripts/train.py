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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training config")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["cnn", "cnn_transformer", "cnn_p2", "cnn_transformer_p2"],
        help="Optional hybrid ablation override",
    )
    parser.add_argument("--experiment-name", type=str, default=None, help="Optional experiment name override")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory override")
    parser.add_argument("--device", type=str, default=None, help="Optional device override")
    parser.add_argument("--epochs", type=int, default=None, help="Optional training epoch override")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional training batch size override")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional dataloader worker override")
    parser.add_argument("--image-size", type=int, default=None, help="Optional image size override")
    parser.add_argument("--learning-rate", type=float, default=None, help="Optional learning rate override")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional train subset size override")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional val subset size override")
    parser.add_argument("--score-threshold", type=float, default=None, help="Optional model score threshold override")
    parser.add_argument("--pre-nms-topk", type=int, default=None, help="Optional pre-NMS top-k override")
    parser.add_argument("--max-detections", type=int, default=None, help="Optional max detections per image override")
    parser.add_argument(
        "--backbone",
        type=str,
        choices=[
            "mobilenet",
            "mobilenet_320",
            "mobilenet_hr",
            "mobilenet_fpn",
            "resnet50",
            "densenet",
            "densenet121",
            "maxvit",
            "maxvit_t",
        ],
        default=None,
        help="Optional baseline backbone override",
    )
    parser.add_argument(
        "--small-defect-profile",
        type=str,
        choices=["none", "small"],
        default=None,
        help="Optional baseline small-defect profile override",
    )
    parser.add_argument(
        "--small-defect-sampler",
        action="store_true",
        help="Enable a weighted sampler that favors tiles containing small defects.",
    )
    parser.add_argument(
        "--small-weight",
        type=float,
        default=None,
        help="Optional sampler weight for records containing small defects.",
    )
    parser.add_argument(
        "--positive-weight",
        type=float,
        default=None,
        help="Optional sampler weight for positive records without small defects.",
    )
    parser.add_argument(
        "--negative-weight",
        type=float,
        default=None,
        help="Optional sampler weight for negative records.",
    )
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


def _apply_train_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(config)
    train_cfg = dict(config.get("train", {}))
    model_cfg = dict(config.get("model", {}))
    sampler_override_requested = bool(
        args.small_defect_sampler
        or any(value is not None for value in (args.small_weight, args.positive_weight, args.negative_weight))
    )

    model_cfg = _apply_variant_override(model_cfg=model_cfg, variant=args.variant)

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
    if args.image_size is not None:
        train_cfg["image_size"] = int(args.image_size)
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
    if args.image_size is not None:
        model_cfg["image_size"] = int(args.image_size)
    if args.backbone is not None:
        model_cfg["backbone"] = args.backbone
    if args.small_defect_profile is not None:
        model_cfg["small_defect_profile"] = args.small_defect_profile
    if sampler_override_requested:
        sampler_cfg = dict(train_cfg.get("small_defect_sampler", {}))
        sampler_cfg["enabled"] = True
        if args.small_weight is not None:
            sampler_cfg["small_weight"] = float(args.small_weight)
        if args.positive_weight is not None:
            sampler_cfg["positive_weight"] = float(args.positive_weight)
        if args.negative_weight is not None:
            sampler_cfg["negative_weight"] = float(args.negative_weight)
        train_cfg["small_defect_sampler"] = sampler_cfg

    if args.experiment_name is None:
        if args.variant is not None:
            config["experiment_name"] = f"hybrid_{args.variant}"
        elif (
            args.backbone is not None
            or args.small_defect_profile not in {None, "none"}
            or sampler_override_requested
        ):
            parts = ["baseline"]
            if args.backbone is not None:
                parts.append(args.backbone)
            if args.small_defect_profile not in {None, "none"}:
                parts.append(args.small_defect_profile)
            if sampler_override_requested:
                parts.append("sdsampler")
            config["experiment_name"] = "_".join(parts)

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
