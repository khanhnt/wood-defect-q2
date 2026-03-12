"""Lightweight CNN-dominant backbone for hybrid detector ablations."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn


def _resolve_group_count(channels: int, preferred_groups: int = 8) -> int:
    """Pick a small GroupNorm group count that divides the channel width."""
    for groups in (preferred_groups, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    """Small conv block used throughout the CNN backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_resolve_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    """Lightweight residual block that keeps the backbone CNN-heavy and efficient."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = ConvNormAct(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.use_residual:
            x = x + identity
        return x


class CNNBackbone(nn.Module):
    """A simple backbone that exposes P2-P5 features for detection ablations."""

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 24,
        stage_channels: Sequence[int] = (48, 96, 160, 224),
        stage_depths: Sequence[int] = (1, 2, 2, 2),
    ) -> None:
        super().__init__()
        if len(stage_channels) != 4 or len(stage_depths) != 4:
            raise ValueError("CNNBackbone expects four stages for P2, P3, P4, and P5.")

        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_channels, kernel_size=3, stride=2),
            DepthwiseSeparableBlock(stem_channels, stem_channels, stride=1),
        )

        input_channels = stem_channels
        levels = ("p2", "p3", "p4", "p5")
        self.stages = nn.ModuleDict()
        self.out_channels: Dict[str, int] = {}
        self.out_strides = {"p2": 4, "p3": 8, "p4": 16, "p5": 32}

        for level_name, output_channels, depth in zip(levels, stage_channels, stage_depths):
            blocks = [DepthwiseSeparableBlock(input_channels, output_channels, stride=2)]
            blocks.extend(
                DepthwiseSeparableBlock(output_channels, output_channels, stride=1)
                for _ in range(max(int(depth) - 1, 0))
            )
            self.stages[level_name] = nn.Sequential(*blocks)
            self.out_channels[level_name] = int(output_channels)
            input_channels = output_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return multi-scale CNN feature maps from P2 to P5."""
        features: Dict[str, torch.Tensor] = {}
        x = self.stem(x)
        for level_name in ("p2", "p3", "p4", "p5"):
            x = self.stages[level_name](x)
            features[level_name] = x
        return features
