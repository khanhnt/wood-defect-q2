"""Basic model forward test."""

import torch


def test_hybrid_detector_instantiation() -> None:
    from src.models.hybrid_detector import HybridDetector
    model = HybridDetector()
    x = torch.randn(1, 3, 224, 224)
    _ = model(x)
