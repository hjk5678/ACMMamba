"""NCHW, same-resolution MSCAN refinement for the shallow decoder stages.

Only PyTorch is needed. Upsampling and skip concatenation belong to UpBlock,
not this module. Spatial attention is unbounded modulation, NOT sigmoid gating.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from common_layers import DropPath


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


class _SingletonSafeGroupNorm(nn.GroupNorm):
    """Normal GroupNorm, including its well-defined singleton-group limit."""

    def forward(self, x):
        # PyTorch rejects B=H=W=1 with one channel/group. Its mathematical
        # normalization is zero (mean=x, variance=0); keep autograd connected.
        if x.ndim == 4 and x.numel() // self.num_groups == 1:
            normalized = x - x
            return (normalized * self.weight.view(1, -1, 1, 1)
                    + self.bias.view(1, -1, 1, 1)).to(x.dtype)
        return super().forward(x)


def make_group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    _positive_int(channels, "channels")
    _positive_int(max_groups, "max_groups")
    groups = min(max_groups, channels)
    while channels % groups:
        groups -= 1
    return _SingletonSafeGroupNorm(groups, channels)


class DWConv(nn.Module):
    """Depthwise 3x3: [B,C,H,W] -> [B,C,H,W], no token reshape."""

    def __init__(self, channels: int):
        super().__init__()
        _positive_int(channels, "channels")
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x):
        return self.conv(x)


class MSCANSpatialAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        _positive_int(channels, "channels")
        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, (1, k), padding=(0, k // 2), groups=channels),
                nn.Conv2d(channels, channels, (k, 1), padding=(k // 2, 0), groups=channels),
            ) for k in (7, 11, 21)
        ])
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        original = x                          # [B,C,H,W], modulation operand
        base = self.conv0(x)                  # 5x5 DW, same spatial size
        attn = base
        for branch in self.branches:
            attn = attn + branch(base)        # parallel strip-conv scales
        attn = self.proj(attn)                # 1x1 mixes channels, no sigmoid
        return original * attn               # [B,C,H,W]


class MSCANAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj1 = nn.Conv2d(channels, channels, 1)
        self.activation = nn.GELU()
        self.spatial_attention = MSCANSpatialAttention(channels)
        self.proj2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        shortcut = x
        x = self.activation(self.proj1(x))    # [B,C,H,W]
        x = self.spatial_attention(x)         # [B,C,H,W]
        return self.proj2(x) + shortcut       # requested internal shortcut


class MSCANMLP(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        _positive_int(channels, "channels")
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0 or int(channels * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must be finite and produce at least one hidden channel.")
        if not 0 <= drop < 1:
            raise ValueError("drop must be in [0,1).")
        hidden = int(channels * mlp_ratio)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.dwconv = DWConv(hidden)
        self.activation = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)                       # [B,C,H,W] -> [B,hidden,H,W]
        x = self.dwconv(x)                    # same hidden channels and H,W
        x = self.drop1(self.activation(x))
        return self.drop2(self.fc2(x))        # [B,hidden,H,W] -> [B,C,H,W]


class MSCANBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, drop: float = 0.0,
                 drop_path: float = 0.0, layer_scale_init_value: float = 1e-2):
        super().__init__()
        _positive_int(channels, "channels")
        if not math.isfinite(layer_scale_init_value):
            raise ValueError("layer_scale_init_value must be finite.")
        if not 0 <= drop_path < 1:
            raise ValueError("drop_path must be in [0,1).")
        self.norm1 = make_group_norm(channels)
        self.attention = MSCANAttention(channels)
        self.norm2 = make_group_norm(channels)
        self.mlp = MSCANMLP(channels, mlp_ratio, drop)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(channels))
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(channels))
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        # Both sublayers preserve [B,C,H,W]; channel-wise LayerScale broadcasts.
        x = x + self.drop_path(self.gamma1.view(1, -1, 1, 1) * self.attention(self.norm1(x)))
        x = x + self.drop_path(self.gamma2.view(1, -1, 1, 1) * self.mlp(self.norm2(x)))
        return x


class MSCANStage(nn.Module):
    """[B,Cin,H,W] -> [B,Cout,H,W]; project channels once, then refine."""

    def __init__(self, in_channels: int, out_channels: int, depth: int = 2,
                 mlp_ratio: float = 4.0, drop: float = 0.0, drop_path: float = 0.0,
                 layer_scale_init_value: float = 1e-2):
        super().__init__()
        _positive_int(in_channels, "in_channels")
        _positive_int(out_channels, "out_channels")
        _positive_int(depth, "depth")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            make_group_norm(out_channels),
        ) if in_channels != out_channels else nn.Identity()
        self.blocks = nn.Sequential(*[
            MSCANBlock(out_channels, mlp_ratio, drop, drop_path, layer_scale_init_value)
            for _ in range(depth)
        ])

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels or min(x.shape[-2:]) < 1:
            raise ValueError(f"Expected [B,{self.in_channels},H,W] with positive H,W.")
        expected = (x.shape[0], self.out_channels, *x.shape[-2:])
        x = self.proj(x)                      # channels only, H/W untouched
        x = self.blocks(x)                    # depth independent MSCAN blocks
        if x.shape != expected:
            raise RuntimeError("MSCANStage changed the expected spatial shape.")
        return x
