#!/usr/bin/env python3
"""Evaluation entry point for the simple detector pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.evaluator import Evaluator
from src.models.builder import build_model
from src.utils.config import load_yaml
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint override")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["cnn", "cnn_transformer", "cnn_p2", "cnn_transformer_p2"],
        help="Optional hybrid ablation override",
    )
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
    parser.add_argument("--experiment-name", type=str, default=None, help="Optional experiment name override")
    parser.add_argument("--device", type=str, default=None, help="Optional device override")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional eval batch size override")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional eval num_workers override")
    parser.add_argument("--image-size", type=int, default=None, help="Optional image size override")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional eval subset size override")
    parser.add_argument("--score-threshold", type=float, default=None, help="Optional eval score threshold override")
    parser.add_argument(
        "--small-defect-eval",
        action="store_true",
        help="Export additional metrics on small-target and small-image subsets.",
    )
    parser.add_argument(
        "--tile-merge",
        action="store_true",
        help="Enable tile-aware prediction merge before scoring.",
    )
    parser.add_argument(
        "--tile-merge-iou-threshold",
        type=float,
        default=None,
        help="Optional IoU threshold for tile-merge NMS.",
    )
    parser.add_argument(
        "--source-manifest-path",
        type=str,
        default=None,
        help="Optional source-image manifest used for tile-merge scoring.",
    )
    parser.add_argument(
        "--in-domain-summary-path",
        type=str,
        default=None,
        help="Optional in-domain summary override for cross-dataset comparison exports",
    )
    return parser.parse_args()


def _deep_merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


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


def _apply_eval_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(config)
    eval_cfg = dict(config.get("evaluation", {}))
    model_cfg = _apply_variant_override(dict(config.get("model", {})), args.variant)

    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name
    if args.device is not None:
        config["device"] = args.device
    if args.batch_size is not None:
        eval_cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        eval_cfg["num_workers"] = int(args.num_workers)
    if args.image_size is not None:
        model_cfg["image_size"] = int(args.image_size)
    if args.max_samples is not None:
        eval_cfg["max_samples"] = int(args.max_samples)
    if args.score_threshold is not None:
        eval_cfg["score_threshold"] = float(args.score_threshold)
        model_cfg["score_threshold"] = float(args.score_threshold)
    if args.small_defect_eval:
        eval_cfg["compute_small_defect_eval"] = True
    if args.tile_merge:
        eval_cfg["tile_merge"] = True
    if args.tile_merge_iou_threshold is not None:
        eval_cfg["tile_merge_iou_threshold"] = float(args.tile_merge_iou_threshold)
    if args.source_manifest_path is not None:
        eval_cfg["source_manifest_path"] = args.source_manifest_path
    if args.backbone is not None:
        model_cfg["backbone"] = args.backbone
    if args.small_defect_profile is not None:
        model_cfg["small_defect_profile"] = args.small_defect_profile
    if args.in_domain_summary_path is not None:
        eval_cfg["in_domain_summary_path"] = args.in_domain_summary_path
    if args.experiment_name is None:
        if args.variant is not None:
            suffix = "_tilemerge_eval" if args.tile_merge else "_eval"
            config["experiment_name"] = f"hybrid_{args.variant}{suffix}"
        elif args.backbone is not None or args.small_defect_profile not in {None, "none"}:
            parts = ["baseline"]
            if args.backbone is not None:
                parts.append(args.backbone)
            if args.small_defect_profile not in {None, "none"}:
                parts.append(args.small_defect_profile)
            if args.small_defect_eval:
                parts.append("smalldefect")
            if args.tile_merge:
                parts.append("tilemerge")
            parts.append("eval")
            config["experiment_name"] = "_".join(parts)
        elif args.tile_merge:
            config["experiment_name"] = "baseline_tilemerge_eval"
        elif args.small_defect_eval:
            config["experiment_name"] = "baseline_smalldefect_eval"

    config["model"] = model_cfg
    config["evaluation"] = eval_cfg
    return config


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    config = _apply_eval_overrides(config=config, args=args)
    set_seed(config.get("seed", 42))

    experiment_name = config.get("experiment_name", "baseline_detector")
    output_dir = Path(config.get("output_dir", "outputs"))
    checkpoint_path = Path(
        args.checkpoint
        or config.get("checkpoint_path")
        or (output_dir / "checkpoints" / experiment_name / "best.pt")
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_config = checkpoint.get("config", {})
    config = _deep_merge_dict(train_config, config)
    config["checkpoint_path"] = str(checkpoint_path)

    model_cfg = dict(config.get("model", {}))
    if "num_classes" not in model_cfg:
        model_cfg["num_classes"] = len(checkpoint.get("class_names", []))

    model = build_model(model_config=model_cfg, train_config=config.get("train", {}))
    evaluator = Evaluator(model=model, config=config)
    logger.info("Starting evaluation for %s using %s", experiment_name, checkpoint_path)
    metrics = evaluator.evaluate(
        checkpoint_path=checkpoint_path,
        experiment_name=experiment_name,
        save_outputs=True,
    )
    logger.info("Evaluation results: %s", metrics["summary"])


if __name__ == "__main__":
    main()
