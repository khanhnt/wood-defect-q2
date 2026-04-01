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
        assert set(outputs["centerness"]) == set(expected_levels)


def test_hybrid_detector_train_and_eval_interfaces() -> None:
    from src.models.hybrid_detector import HybridDetector

    model = HybridDetector(use_transformer=True, num_transformer_blocks=1, use_p2_branch=True)
    train_images = [torch.rand(3, 224, 224), torch.rand(3, 224, 224)]
    targets = [
        {
            "boxes": torch.tensor([[20.0, 30.0, 70.0, 90.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "image_id": torch.tensor([0], dtype=torch.int64),
            "area": torch.tensor([3000.0], dtype=torch.float32),
            "iscrowd": torch.tensor([0], dtype=torch.int64),
        },
        {
            "boxes": torch.tensor([[100.0, 110.0, 170.0, 180.0]], dtype=torch.float32),
            "labels": torch.tensor([2], dtype=torch.int64),
            "image_id": torch.tensor([1], dtype=torch.int64),
            "area": torch.tensor([4900.0], dtype=torch.float32),
            "iscrowd": torch.tensor([0], dtype=torch.int64),
        },
    ]

    loss_dict = model(train_images, targets)
    assert set(loss_dict) == {"loss_cls", "loss_box_reg", "loss_centerness"}

    model.eval()
    with torch.no_grad():
        detections = model([torch.rand(3, 224, 224)])
    assert len(detections) == 1
    assert set(detections[0]) == {"boxes", "scores", "labels"}
