"""Small neural-network layers shared by the dual-modal encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_layers import DropPath


class LayerNorm2d(nn.Module):
    """Apply LayerNorm over channels independently at every spatial position."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


class ConvFFN(nn.Module):
    """Channel MLP with a depthwise convolution for local spatial context."""

    def __init__(self, channels: int, expansion: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion)
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PatchEmbed(nn.Module):
    """Independent two-step patch embedding with an overall stride of four."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden_channels = out_channels // 2
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class PatchMerging(nn.Module):
    """Independent stage downsampling with a spatial stride of two."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)
