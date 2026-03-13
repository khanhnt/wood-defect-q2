"""timm feature-extraction backbones adapted for Faster R-CNN + FPN."""

from __future__ import annotations

from collections import OrderedDict

from torch import nn


class TimmBackboneWithFPN(nn.Module):
    """Wrap a timm features_only backbone with a lightweight FPN."""

    def __init__(
        self,
        model_name_candidates: list[str],
        out_channels: int = 256,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - dependency dependent
            raise ImportError(
                "timm is required for MaxViT backbone support. Install it before using backbone=maxvit_t."
            ) from exc

        from torchvision.ops import FeaturePyramidNetwork
        from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

        last_error: Exception | None = None
        body = None
        channels = None
        model_name_used = None

        out_indices_candidates = ((1, 2, 3, 4), (0, 1, 2, 3))
        for model_name in model_name_candidates:
            for out_indices in out_indices_candidates:
                try:
                    candidate = timm.create_model(
                        model_name,
                        pretrained=False,
                        features_only=True,
                        out_indices=out_indices,
                    )
                    candidate_channels = list(candidate.feature_info.channels())
                    if len(candidate_channels) != 4:
                        continue
                    body = candidate
                    channels = candidate_channels
                    model_name_used = model_name
                    break
                except Exception as exc:  # pragma: no cover - dependency/version dependent
                    last_error = exc
            if body is not None:
                break

        if body is None or channels is None:
            raise RuntimeError(
                "Unable to build a timm MaxViT features_only backbone. "
                "Checked model candidates: " + ", ".join(model_name_candidates)
            ) from last_error

        self.body = body
        self.model_name = str(model_name_used)
        self._feature_names = [str(index) for index in range(len(channels))]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=channels,
            out_channels=int(out_channels),
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = int(out_channels)

    def forward(self, x):
        features = self.body(x)
        ordered_features = OrderedDict((name, feature) for name, feature in zip(self._feature_names, features))
        return self.fpn(ordered_features)


def build_timm_fpn_backbone(backbone_name: str, out_channels: int = 256) -> nn.Module:
    """Create a timm backbone adapted for Faster R-CNN via FPN."""

    normalized_name = str(backbone_name).lower()
    if normalized_name in {"maxvit", "maxvit_t"}:
        return TimmBackboneWithFPN(
            model_name_candidates=[
                "maxvit_tiny_tf_224.in1k",
                "maxvit_tiny_tf_224",
                "maxvit_tiny_rw_224.sw_in1k",
                "maxvit_tiny_rw_224",
            ],
            out_channels=out_channels,
        )
    raise NotImplementedError(f"Unsupported timm FPN backbone: {backbone_name!r}")
