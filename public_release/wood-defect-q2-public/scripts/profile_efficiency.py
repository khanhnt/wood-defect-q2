#!/usr/bin/env python3
"""Profile inference efficiency for internal detectors or YOLO checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.efficiency import (
    build_detection_inputs,
    build_tensor_inputs,
    detection_forward,
    summarize_efficiency,
    tensor_forward,
)
from src.models.builder import build_model
from src.utils.config import load_yaml
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        type=str,
        choices=["internal", "yolo"],
        required=True,
        help="Profile an internal repo detector or an Ultralytics YOLO checkpoint.",
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML for internal repo models.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for YOLO profiling.")
    parser.add_argument("--image-size", type=int, default=1024, help="Square input size for dummy inference.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for latency measurement.")
    parser.add_argument("--device", type=str, default=None, help="Device override: cpu, cuda, or GPU index such as 0.")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations.")
    parser.add_argument("--iterations", type=int, default=30, help="Number of timed iterations.")
    parser.add_argument("--skip-flops", action="store_true", help="Skip THOP MAC/FLOP estimation.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save the JSON summary.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def _load_internal_model(config_path: str) -> tuple[torch.nn.Module, Dict[str, Any]]:
    config = load_yaml(config_path)
    set_seed(config.get("seed", 42))
    model_cfg = dict(config.get("model", {}))
    train_cfg = dict(config.get("train", {}))
    model = build_model(model_config=model_cfg, train_config=train_cfg)
    return model, config


def _load_yolo_model(checkpoint_path: str) -> tuple[torch.nn.Module, Dict[str, Any]]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "ultralytics is required for YOLO profiling. Install it with `python -m pip install ultralytics`."
        ) from exc

    yolo = YOLO(checkpoint_path)
    model = yolo.model
    names = getattr(yolo, "names", None)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "num_classes": len(names) if isinstance(names, (dict, list, tuple)) else None,
    }
    return model, metadata


def _print_summary(summary: Dict[str, Any]) -> None:
    order = [
        "kind",
        "label",
        "device",
        "image_size",
        "batch_size",
        "params_million",
        "trainable_params_million",
        "profile_params_million",
        "macs_giga",
        "flops_giga",
        "latency_ms_mean",
        "latency_ms_std",
        "throughput_images_per_sec",
        "peak_memory_gb",
        "flops_error",
    ]
    logger.info("Efficiency profile")
    for key in order:
        if key in summary:
            logger.info("  %s: %s", key, summary[key])


def main() -> None:
    args = parse_args()

    if args.kind == "internal":
        if args.config is None:
            raise ValueError("--config is required when --kind=internal")
        model, metadata = _load_internal_model(args.config)
        summary = summarize_efficiency(
            model,
            input_builder=build_detection_inputs,
            forward_fn=detection_forward,
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            device=args.device,
            warmup_iterations=int(args.warmup),
            timed_iterations=int(args.iterations),
            include_flops=not args.skip_flops,
        )
        summary["label"] = Path(args.config).stem
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --kind=yolo")
        model, metadata = _load_yolo_model(args.checkpoint)
        summary = summarize_efficiency(
            model,
            input_builder=build_tensor_inputs,
            forward_fn=tensor_forward,
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            device=args.device,
            warmup_iterations=int(args.warmup),
            timed_iterations=int(args.iterations),
            include_flops=not args.skip_flops,
        )
        summary["label"] = Path(args.checkpoint).stem

    summary["kind"] = args.kind
    summary["metadata"] = metadata

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2 if args.pretty else None, sort_keys=True),
            encoding="utf-8",
        )
        logger.info("Saved efficiency summary to %s", output_path)

    if args.pretty:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, sort_keys=True))

    _print_summary(summary)


if __name__ == "__main__":
    main()
