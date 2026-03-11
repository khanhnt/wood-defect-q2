"""Hybrid CNN-Transformer detector with optional P2 branch for ablations."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
from torch import nn

from src.models.backbones.cnn_backbone import CNNBackbone
from src.models.backbones.transformer_block import SimpleTransformerBlock
from src.models.heads.detection_head import DetectionHead
from src.models.necks.light_neck import LightNeck


def _normalize_transformer_levels(levels: Sequence[str] | None, num_blocks: int) -> list[str]:
    default_levels = ["p4", "p5"]
    candidate_levels = list(levels or default_levels)
    candidate_levels = [level for level in candidate_levels if level in {"p4", "p5"}]
    if num_blocks < 0 or num_blocks > 2:
        raise ValueError("HybridDetector supports 0, 1, or 2 transformer blocks.")
    return candidate_levels[:num_blocks]


class HybridDetector(nn.Module):
    """CNN-dominant hybrid detector that keeps ablations explicit and small."""

    def __init__(
        self,
        num_classes: int = 10,
        use_transformer: bool = True,
        num_transformer_blocks: int = 2,
        use_p2_branch: bool = True,
        neck_out_channels: int = 128,
        transformer_heads: int = 4,
        transformer_levels: Sequence[str] | None = None,
        stage_channels: Sequence[int] = (48, 96, 160, 224),
        stage_depths: Sequence[int] = (1, 2, 2, 2),
    ) -> None:
        super().__init__()
        if use_transformer and num_transformer_blocks not in {1, 2}:
            raise ValueError("Set num_transformer_blocks to 1 or 2 when use_transformer is enabled.")
        if not use_transformer:
            num_transformer_blocks = 0

        self.backbone = CNNBackbone(
            stage_channels=stage_channels,
            stage_depths=stage_depths,
        )
        self.use_transformer = bool(use_transformer)
        self.use_p2_branch = bool(use_p2_branch)
        self.transformer_levels = _normalize_transformer_levels(
            levels=transformer_levels,
            num_blocks=num_transformer_blocks,
        )
        self.transformer_blocks = nn.ModuleDict(
            {
                level_name: SimpleTransformerBlock(
                    dim=self.backbone.out_channels[level_name],
                    num_heads=transformer_heads,
                )
                for level_name in self.transformer_levels
            }
        )
        self.neck = LightNeck(
            in_channels=self.backbone.out_channels,
            out_channels=neck_out_channels,
            use_p2_branch=self.use_p2_branch,
        )
        self.head = DetectionHead(
            num_classes=num_classes,
            in_channels=neck_out_channels,
        )

    def get_variant_name(self) -> str:
        """Return a short ablation label for logging and debugging."""
        parts = ["cnn"]
        if self.use_transformer:
            parts.append(f"transformer{len(self.transformer_levels)}")
        if self.use_p2_branch:
            parts.append("p2")
        return "+".join(parts)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Return backbone features, neck features, and dense prediction tensors."""
        backbone_features = self.backbone(x)
        refined_features = dict(backbone_features)

        for level_name in self.transformer_levels:
            refined_features[level_name] = self.transformer_blocks[level_name](refined_features[level_name])

        pyramid_features = self.neck(refined_features)
        predictions = self.head(pyramid_features)
        predictions.update(
            {
                "backbone_features": refined_features,
                "neck_features": pyramid_features,
                "feature_levels": list(pyramid_features.keys()),
                "variant_name": self.get_variant_name(),
            }
        )
        return predictions


def build_hybrid_detector(model_config: Dict[str, Any], train_config: Dict[str, Any] | None = None) -> HybridDetector:
    """Factory wrapper so hybrid ablations can be config-driven later."""
    _ = train_config
    return HybridDetector(
        num_classes=int(model_config.get("num_classes", 10)),
        use_transformer=bool(model_config.get("use_transformer", True)),
        num_transformer_blocks=int(model_config.get("num_transformer_blocks", 2)),
        use_p2_branch=bool(model_config.get("use_p2_branch", True)),
        neck_out_channels=int(model_config.get("neck_out_channels", 128)),
        transformer_heads=int(model_config.get("transformer_heads", 4)),
        transformer_levels=model_config.get("transformer_levels"),
        stage_channels=tuple(model_config.get("stage_channels", (48, 96, 160, 224))),
        stage_depths=tuple(model_config.get("stage_depths", (1, 2, 2, 2))),
    )
