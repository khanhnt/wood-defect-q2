"""Minimal torchvision classification backbones adapted for Faster R-CNN + FPN."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

from torch import nn


class TorchvisionBackboneWithFPN(nn.Module):
    """Wrap a torchvision backbone with feature extraction and a lightweight FPN."""

    def __init__(
        self,
        body: nn.Module,
        return_nodes: dict[str, str],
        in_channels_list: Sequence[int],
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        from torchvision.models.feature_extraction import create_feature_extractor
        from torchvision.ops import FeaturePyramidNetwork
        from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

        self.body = create_feature_extractor(body, return_nodes=return_nodes)
        self._feature_names = list(return_nodes.values())
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(in_channels_list),
            out_channels=int(out_channels),
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = int(out_channels)

    def forward(self, x):
        features = self.body(x)
        ordered_features = OrderedDict((name, features[name]) for name in self._feature_names)
        return self.fpn(ordered_features)


def _build_densenet121(weights_none_kwargs: dict[str, object]) -> nn.Module:
    from torchvision.models import densenet121

    try:
        return densenet121(**weights_none_kwargs)
    except TypeError:  # pragma: no cover - torchvision version dependent
        return densenet121(pretrained=False)


def _build_maxvit_t(weights_none_kwargs: dict[str, object]) -> nn.Module:
    from torchvision.models import maxvit_t

    try:
        return maxvit_t(**weights_none_kwargs)
    except TypeError:  # pragma: no cover - torchvision version dependent
        return maxvit_t(pretrained=False)


def build_torchvision_fpn_backbone(backbone_name: str, out_channels: int = 256) -> nn.Module:
    """Create a classification backbone adapted for Faster R-CNN via FPN."""

    normalized_name = str(backbone_name).lower()
    weights_none_kwargs = {"weights": None}

    if normalized_name in {"densenet121", "densenet"}:
        model = _build_densenet121(weights_none_kwargs=weights_none_kwargs)
        return TorchvisionBackboneWithFPN(
            body=model,
            return_nodes={
                "features.denseblock1": "0",
                "features.denseblock2": "1",
                "features.denseblock3": "2",
                "features.denseblock4": "3",
            },
            in_channels_list=[256, 512, 1024, 1024],
            out_channels=out_channels,
        )

    if normalized_name in {"maxvit", "maxvit_t"}:
        model = _build_maxvit_t(weights_none_kwargs=weights_none_kwargs)
        return TorchvisionBackboneWithFPN(
            body=model,
            return_nodes={
                "blocks.0": "0",
                "blocks.1": "1",
                "blocks.2": "2",
                "blocks.3": "3",
            },
            in_channels_list=[64, 128, 256, 512],
            out_channels=out_channels,
        )

    raise NotImplementedError(f"Unsupported torchvision FPN backbone: {backbone_name!r}")
