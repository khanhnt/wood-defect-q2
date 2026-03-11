"""Small model builder that keeps baseline and hybrid entrypoints aligned."""

from __future__ import annotations

from typing import Any, Dict

from torch import nn

from src.models.heads.detection_head import build_baseline_detector
from src.models.hybrid_detector import build_hybrid_detector


def build_model(model_config: Dict[str, Any], train_config: Dict[str, Any] | None = None) -> nn.Module:
    """Build a detector model from a compact config dictionary."""
    model_name = str(model_config.get("name", "baseline_detector"))
    if model_name == "baseline_detector":
        return build_baseline_detector(model_config=model_config, train_config=train_config)
    if model_name == "hybrid_detector":
        return build_hybrid_detector(model_config=model_config, train_config=train_config)
    raise NotImplementedError(f"Unsupported model.name={model_name!r}.")
