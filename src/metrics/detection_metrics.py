"""Detection metric placeholders."""

from typing import Dict


def compute_detection_metrics(predictions, targets) -> Dict[str, float]:
    """Return placeholder detection metrics."""
    # TODO: replace with real mAP / AP implementation
    return {
        "mAP50": 0.0,
        "mAP50_95": 0.0,
    }
