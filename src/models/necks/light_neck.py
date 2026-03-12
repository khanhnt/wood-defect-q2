"""Lightweight top-down neck with an optional P2 branch."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn.functional as F
from torch import nn


def _resolve_group_count(channels: int, preferred_groups: int = 8) -> int:
    """Pick a small GroupNorm group count that divides the channel width."""
    for groups in (preferred_groups, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_resolve_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class _DepthwiseFusionBlock(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            _ConvNormAct(channels, channels, kernel_size=3, groups=channels),
            _ConvNormAct(channels, channels, kernel_size=1),
        )


class LightNeck(nn.Module):
    """Compact FPN-like neck for CNN, CNN+P2, and hybrid ablations."""

    def __init__(
        self,
        in_channels: Mapping[str, int],
        out_channels: int = 128,
        use_p2_branch: bool = True,
    ) -> None:
        super().__init__()
        self.use_p2_branch = bool(use_p2_branch)
        required_levels = ("p2", "p3", "p4", "p5")
        missing_levels = [level for level in required_levels if level not in in_channels]
        if missing_levels:
            raise ValueError(f"LightNeck missing backbone channels for levels: {missing_levels}")

        self.lateral_convs = nn.ModuleDict(
            {
                level: _ConvNormAct(int(in_channels[level]), out_channels, kernel_size=1)
                for level in required_levels
            }
        )
        self.output_convs = nn.ModuleDict(
            {
                level: _DepthwiseFusionBlock(out_channels)
                for level in required_levels
            }
        )
        self.out_channels = {level: out_channels for level in required_levels}

    def forward(self, features: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Fuse backbone features with a light top-down pathway."""
        p5 = self.lateral_convs["p5"](features["p5"])
        p4 = self.lateral_convs["p4"](features["p4"]) + F.interpolate(
            p5,
            size=features["p4"].shape[-2:],
            mode="nearest",
        )
        p3 = self.lateral_convs["p3"](features["p3"]) + F.interpolate(
            p4,
            size=features["p3"].shape[-2:],
            mode="nearest",
        )

        fused_features = {
            "p3": self.output_convs["p3"](p3),
            "p4": self.output_convs["p4"](p4),
            "p5": self.output_convs["p5"](p5),
        }

        if self.use_p2_branch:
            p2 = self.lateral_convs["p2"](features["p2"]) + F.interpolate(
                p3,
                size=features["p2"].shape[-2:],
                mode="nearest",
            )
            fused_features = {
                "p2": self.output_convs["p2"](p2),
                **fused_features,
            }

        return fused_features
