"""Parallel Residual-Hypergraph Dual-Branch decoder block (RHDB)."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .hypergraph import RegionHypergraphBranch
from .unet_decoder import ResidualStage, _group_count
from .sapa import build_skip_upsampler


class AdaptiveBranchFusion(nn.Module):
    """Fuse increments; in gate mode the local update is never attenuated."""
    def __init__(self, channels: int, fusion_mode: str = "gate"):
        super().__init__()
        if fusion_mode not in {"gate", "add", "concat"}:
            raise ValueError("fusion_mode must be 'gate', 'add' or 'concat'.")
        self.fusion_mode = fusion_mode
        if fusion_mode == "gate":
            self.gate_conv = nn.Conv2d(2 * channels, 1, kernel_size=1, bias=True)
            nn.init.zeros_(self.gate_conv.weight)
            nn.init.constant_(self.gate_conv.bias, -2.0)
        elif fusion_mode == "concat":
            self.projection = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(self, delta_local: torch.Tensor, delta_hg: torch.Tensor) -> torch.Tensor:
        if delta_local.shape != delta_hg.shape or delta_local.ndim != 4:
            raise ValueError("Local and hypergraph increments must have identical [B,C,H,W] shapes.")
        if self.fusion_mode == "add":
            return delta_local + delta_hg
        combined = torch.cat((delta_local, delta_hg), dim=1)
        if self.fusion_mode == "concat":
            return self.projection(combined)
        gate = torch.sigmoid(self.gate_conv(combined))  # [B,1,H,W]
        return delta_local + gate * delta_hg


class RHDBBlock(nn.Module):
    """forward(deep_feat, skip_feat) -> [B,out_channels,H_skip,W_skip]."""
    def __init__(
        self, skip_channels: int, deep_channels: int, out_channels: int,
        use_hypergraph: bool = True, fusion_mode: str = "gate", k: int = 8,
        node_grid: int | Sequence[int] = 16, alpha: float = 0.7,
        hypergraph_hidden_dim: int | None = None, gamma_init: float = 0.0,
        eps: float = 1e-6, num_residual_blocks: int = 1, dropout: float = 0.0,
        upsample_mode: str = "bilinear", sapa_options: dict | None = None,
    ):
        super().__init__()
        if min(skip_channels, deep_channels, out_channels) < 1:
            raise ValueError("All channels must be positive.")
        if fusion_mode not in {"gate", "add", "concat"}:
            raise ValueError("fusion_mode must be 'gate', 'add' or 'concat'.")
        self.skip_channels, self.deep_channels = skip_channels, deep_channels
        self.out_channels, self.use_hypergraph = out_channels, use_hypergraph
        self.fusion_mode = fusion_mode
        self.fuse = nn.Conv2d(skip_channels + deep_channels, out_channels, kernel_size=1)
        self.local_branch = ResidualStage(out_channels, out_channels, num_residual_blocks, dropout)
        if use_hypergraph:
            self.hypergraph_branch = RegionHypergraphBranch(
                out_channels, hypergraph_hidden_dim, node_grid, k, alpha, gamma_init, eps
            )
            self.branch_fusion = AdaptiveBranchFusion(out_channels, fusion_mode)
        # Disabled branch allocates neither HG parameters nor unused gates (DDP).
        self.post = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )
        self.sapa = build_skip_upsampler(skip_channels, deep_channels, upsample_mode, sapa_options)

    def forward(self, deep_feat: torch.Tensor, skip_feat: torch.Tensor) -> torch.Tensor:
        if deep_feat.ndim != 4 or skip_feat.ndim != 4:
            raise ValueError("deep_feat and skip_feat must be BCHW tensors.")
        if deep_feat.shape[0] != skip_feat.shape[0]:
            raise ValueError("deep_feat and skip_feat batch sizes must match.")
        if deep_feat.shape[1] != self.deep_channels or skip_feat.shape[1] != self.skip_channels:
            raise ValueError("Input channels do not match the RHDB configuration.")
        if min(*deep_feat.shape[-2:], *skip_feat.shape[-2:]) < 1:
            raise ValueError("Spatial sizes must be positive.")
        up = (F.interpolate(deep_feat, size=skip_feat.shape[-2:], mode="bilinear", align_corners=False)
              if self.sapa is None else self.sapa(encoder_feature=skip_feat, decoder_feature=deep_feat))
        feature = self.fuse(torch.cat((up, skip_feat), dim=1))
        local_full = self.local_branch(feature)
        # Includes post-activation residual behavior without changing ResNet.
        delta_local = local_full - feature
        if self.use_hypergraph:
            # TRUE parallel branches: HG receives feature, not the ResNet output.
            delta_hg = self.hypergraph_branch(feature)
            update = self.branch_fusion(delta_local, delta_hg)
        else:
            update = delta_local
        output = self.post(feature + update)
        if output.shape != (skip_feat.shape[0], self.out_channels, *skip_feat.shape[-2:]):
            raise RuntimeError("RHDB output shape does not match the skip resolution.")
        return output
