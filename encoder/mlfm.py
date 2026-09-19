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
        learnable_weights: bool = True,
    ) -> None:
        super().__init__()
        in_channels_b = in_channels_a if in_channels_b is None else in_channels_b
        out_channels = in_channels_a if out_channels is None else out_channels

        self.align_a = nn.Conv2d(in_channels_a, out_channels, kernel_size=1, bias=False)
        self.align_b = nn.Conv2d(in_channels_b, out_channels, kernel_size=1, bias=False)
        self.add_branch = FusionConv(out_channels, out_channels)
        self.cat_branch = FusionConv(out_channels * 2, out_channels)

        initial_a = torch.tensor(0.5, dtype=torch.float32)
        initial_b = torch.tensor(0.5, dtype=torch.float32)
        if learnable_weights:
            self.a = nn.Parameter(initial_a)
            self.b = nn.Parameter(initial_b)
        else:
            self.register_buffer("a", initial_a)
            self.register_buffer("b", initial_b)

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


class HypergraphMLFM(MLFM):
    """Independent residual HG refinement BEFORE all existing MLFM operations.

    RegionHypergraphBranch returns a gamma-scaled increment, not a full feature.
    Each modality/stage owns its graph, HGConv and parameters; no joint graph.
    """

    def __init__(self, in_channels_a: int, in_channels_b: int | None = None,
                 out_channels: int | None = None, hypergraph_options: dict | None = None):
        super().__init__(in_channels_a, in_channels_b, out_channels)
        from decoder.hypergraph import RegionHypergraphBranch

        if hypergraph_options is not None and not isinstance(hypergraph_options, dict):
            raise ValueError("mlfm_hg_options must be a dict or None.")
        options = dict(hypergraph_options or {})
        self.force_fp32 = options.pop("force_fp32", False)
        if not isinstance(self.force_fp32, bool):
            raise ValueError("mlfm_hg_options.force_fp32 must be boolean.")
        allowed = {"hypergraph_hidden_dim", "node_grid", "k", "alpha", "gamma_init", "eps"}
        if set(options) - allowed:
            raise ValueError(f"Unknown mlfm_hg_options: {sorted(set(options) - allowed)}")
        in_channels_b = in_channels_a if in_channels_b is None else in_channels_b
        self.hypergraph_a = RegionHypergraphBranch(in_channels_a, **options)
        self.hypergraph_b = RegionHypergraphBranch(in_channels_b, **options)

    def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        if feature_a.ndim != 4 or feature_b.ndim != 4 or feature_a.shape[0] != feature_b.shape[0]:
            raise ValueError("HypergraphMLFM requires NCHW features with matching batch sizes.")
        if feature_a.device != feature_b.device:
            raise ValueError("HypergraphMLFM requires both modalities on the same device.")
        if self.force_fp32:
            # Protect reduce/restore, residual addition AND subsequent MLFM
            # convolutions. Do not cast the result back to FP16 here.
            with torch.autocast(device_type=feature_a.device.type, enabled=False):
                return self._fuse_refined(feature_a.float(), feature_b.float())
        return self._fuse_refined(feature_a, feature_b)

    def _fuse_refined(self, feature_a, feature_b):
        refined_a = feature_a + self.hypergraph_a(feature_a)
        refined_b = feature_b + self.hypergraph_b(feature_b)
        return super().forward(refined_a, refined_b)


class MeanFusion(nn.Module):
    """Parameter-free mean fusion used as the primary no-MLFM baseline."""

    def __init__(
        self,
        in_channels_a: int,
        in_channels_b: int | None = None,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        in_channels_b = in_channels_a if in_channels_b is None else in_channels_b
        out_channels = in_channels_a if out_channels is None else out_channels
        self.align_a = (
            nn.Identity()
            if in_channels_a == out_channels
            else nn.Conv2d(in_channels_a, out_channels, kernel_size=1, bias=False)
        )
        self.align_b = (
            nn.Identity()
            if in_channels_b == out_channels
            else nn.Conv2d(in_channels_b, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        target_size = feature_a.shape[-2:]
        if feature_b.shape[-2:] != target_size:
            feature_b = F.interpolate(
                feature_b, size=target_size, mode="bilinear", align_corners=False
            )
        return 0.5 * (self.align_a(feature_a) + self.align_b(feature_b))


class SingleBranchFusion(nn.Module):
    """Retain only the Add or Cat refinement branch of MLFM."""

    def __init__(
        self,
        in_channels_a: int,
        in_channels_b: int | None = None,
        out_channels: int | None = None,
        branch: str = "add",
    ) -> None:
        super().__init__()
        in_channels_b = in_channels_a if in_channels_b is None else in_channels_b
        out_channels = in_channels_a if out_channels is None else out_channels
        if branch not in {"add", "cat"}:
            raise ValueError("branch must be either 'add' or 'cat'.")
        self.branch = branch
        self.align_a = nn.Conv2d(in_channels_a, out_channels, kernel_size=1, bias=False)
        self.align_b = nn.Conv2d(in_channels_b, out_channels, kernel_size=1, bias=False)
        branch_channels = out_channels if branch == "add" else out_channels * 2
        self.refine = FusionConv(branch_channels, out_channels)

    def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        target_size = feature_a.shape[-2:]
        if feature_b.shape[-2:] != target_size:
            feature_b = F.interpolate(
                feature_b, size=target_size, mode="bilinear", align_corners=False
            )
        aligned_a = self.align_a(feature_a)
        aligned_b = self.align_b(feature_b)
        if self.branch == "add":
            return self.refine(aligned_a + aligned_b)
        return self.refine(torch.cat((aligned_a, aligned_b), dim=1))


FUSION_MODES = ("mlfm", "mlfm_hg", "mean", "add_only", "cat_only", "dual_fixed")


def build_fusion(
    mode: str,
    in_channels_a: int,
    in_channels_b: int | None = None,
    out_channels: int | None = None,
    mlfm_hg_options: dict | None = None,
) -> nn.Module:
    """Build one fusion variant for controlled MLFM ablation studies."""
    normalized_mode = str(mode).lower().strip()
    if mlfm_hg_options is not None and normalized_mode != "mlfm_hg":
        raise ValueError("mlfm_hg_options requires fusion_mode='mlfm_hg'.")
    if normalized_mode == "mlfm_hg":
        return HypergraphMLFM(in_channels_a, in_channels_b, out_channels, mlfm_hg_options)
    if normalized_mode == "mlfm":
        return MLFM(in_channels_a, in_channels_b, out_channels)
    if normalized_mode == "dual_fixed":
        return MLFM(
            in_channels_a,
            in_channels_b,
            out_channels,
            learnable_weights=False,
        )
    if normalized_mode == "mean":
        return MeanFusion(in_channels_a, in_channels_b, out_channels)
    if normalized_mode == "add_only":
        return SingleBranchFusion(
            in_channels_a, in_channels_b, out_channels, branch="add"
        )
    if normalized_mode == "cat_only":
        return SingleBranchFusion(
            in_channels_a, in_channels_b, out_channels, branch="cat"
        )
    raise ValueError(
        f"Unknown fusion mode {mode!r}; expected one of {FUSION_MODES}."
    )
