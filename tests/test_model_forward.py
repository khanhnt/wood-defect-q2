"""Basic hybrid model forward tests."""

import torch


def test_hybrid_detector_variants_forward() -> None:
    from src.models.hybrid_detector import HybridDetector

    variants = [
        HybridDetector(use_transformer=False, use_p2_branch=False),
        HybridDetector(use_transformer=True, num_transformer_blocks=1, use_p2_branch=False),
        HybridDetector(use_transformer=False, use_p2_branch=True),
        HybridDetector(use_transformer=True, num_transformer_blocks=2, use_p2_branch=True),
    ]
    x = torch.randn(1, 3, 224, 224)

    for model in variants:
        outputs = model(x)
        expected_levels = ["p3", "p4", "p5"]
        if model.use_p2_branch:
            expected_levels = ["p2", *expected_levels]

        assert outputs["feature_levels"] == expected_levels
        assert outputs["variant_name"].startswith("cnn")
        assert set(outputs["cls_logits"]) == set(expected_levels)
        assert set(outputs["bbox_regression"]) == set(expected_levels)
        assert set(outputs["objectness"]) == set(expected_levels)
