"""Efficiency evaluation helpers."""

from typing import Dict


def summarize_efficiency(model) -> Dict[str, float]:
    """Return placeholder efficiency stats."""
    # TODO: compute params / FLOPs / latency
    return {
        "params_million": 0.0,
        "flops_giga": 0.0,
        "latency_ms": 0.0,
    }
