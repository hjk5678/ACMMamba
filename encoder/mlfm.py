"""Learnable dual-branch Add/Cat fusion used after every encoder stage."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LayerNorm2d


class FusionConv(nn.Module):
    """Adjust channels and refine one fusion branch."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MLFM(nn.Module):
    """Fuse two modalities with independent Add and Cat branches.

    Both inputs are first projected to ``out_channels``. The two branch results
    are combined exactly as ``F = a * F_add + b * F_cat``, where ``a`` and ``b``
    are independent, unconstrained learnable scalar parameters.
    """

    def __init__(
        self,
        in_channels_a: int,
        in_channels_b: int | None = None,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        in_channels_b = in_channels_a if in_channels_b is None else in_channels_b
        out_channels = in_channels_a if out_channels is None else out_channels

        self.align_a = nn.Conv2d(in_channels_a, out_channels, kernel_size=1, bias=False)
        self.align_b = nn.Conv2d(in_channels_b, out_channels, kernel_size=1, bias=False)
        self.add_branch = FusionConv(out_channels, out_channels)
        self.cat_branch = FusionConv(out_channels * 2, out_channels)

        self.a = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        target_size = feature_a.shape[-2:]
        if feature_b.shape[-2:] != target_size:
            feature_b = F.interpolate(
                feature_b, size=target_size, mode="bilinear", align_corners=False
            )

        aligned_a = self.align_a(feature_a)
        aligned_b = self.align_b(feature_b)
        fused_add = self.add_branch(aligned_a + aligned_b)
        fused_cat = self.cat_branch(torch.cat((aligned_a, aligned_b), dim=1))
        return self.a * fused_add + self.b * fused_cat
