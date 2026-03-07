"""Hybrid CNN-Transformer detector placeholder."""

import torch
from torch import nn

from src.models.backbones.cnn_backbone import CNNBackbone
from src.models.backbones.transformer_block import SimpleTransformerBlock
from src.models.necks.light_neck import LightNeck
from src.models.heads.detection_head import DetectionHead


class HybridDetector(nn.Module):
    """Hybrid detector with CNN backbone, optional Transformer blocks, and a light neck."""

    def __init__(
        self,
        num_classes: int = 10,
        use_transformer: bool = True,
        num_transformer_blocks: int = 2,
        use_p2_branch: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = CNNBackbone()
        self.use_transformer = use_transformer
        self.use_p2_branch = use_p2_branch
        self.transformer_blocks = nn.ModuleList(
            [SimpleTransformerBlock() for _ in range(num_transformer_blocks)]
        )
        self.neck = LightNeck()
        self.head = DetectionHead(num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        features = self.backbone(x)
        if self.use_transformer:
            # TODO: apply transformer blocks to selected feature levels
            pass
        fused = self.neck(features)
        outputs = self.head(fused)
        return outputs
