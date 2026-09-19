"""Region-node KNN hypergraph and symmetric HGNN propagation (W = I)."""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class KNNHypergraph(nn.Module):
    """H[b, vertex, edge]: each edge contains its center and K OTHER nodes.

    Hard top-k topology is deliberately non-differentiable. Feature propagation
    still differentiates through X. All construction is per-sample and FP32.
    """

    def __init__(self, k: int = 8, alpha: float = 0.7, eps: float = 1e-6):
        super().__init__()
        if not isinstance(k, int) or isinstance(k, bool) or k < 0:
            raise ValueError("k must be a nonnegative integer (excluding the center).")
        if not 0 <= alpha <= 1 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("Require alpha in [0,1] and finite eps > 0.")
        self.k, self.alpha, self.eps = k, float(alpha), float(eps)

    @torch.no_grad()
    def forward(self, nodes: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if nodes.ndim != 3 or nodes.shape[1] < 1 or nodes.shape[2] < 1:
            raise ValueError("nodes must have shape [B,N,C] with N,C > 0.")
        batch, count, _ = nodes.shape
        if positions.shape != (count, 2) or positions.device != nodes.device:
            raise ValueError("positions must have shape [N,2] on the nodes' device.")
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            normalized = F.normalize(nodes.float(), p=2, dim=-1, eps=self.eps)
            semantic = 1 - torch.bmm(normalized, normalized.transpose(1, 2)).clamp(-1, 1)
            spatial = torch.cdist(positions.float(), positions.float(), p=2)
            distance = self.alpha * semantic + (1 - self.alpha) * spatial.unsqueeze(0)
            eye = torch.eye(count, device=nodes.device, dtype=torch.bool)
            distance.masked_fill_(eye.unsqueeze(0), float("inf"))
            neighbors = distance.topk(min(self.k, count - 1), dim=-1, largest=False).indices
            centers = torch.arange(count, device=nodes.device).view(1, count, 1).expand(batch, -1, -1)
            members = torch.cat((centers, neighbors), dim=-1)  # [B, edge, member]
            incidence = torch.zeros(batch, count, count, device=nodes.device, dtype=torch.float32)
            # Scatter along vertex axis; columns are hyperedges, NOT adjacency.
            incidence.scatter_(1, members.transpose(1, 2), 1.0)
            return incidence


def hypergraph_propagate(nodes: torch.Tensor, incidence: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dv^-1/2 H De^-1 H^T Dv^-1/2 X, without dense diagonal matrices or G."""
    if nodes.ndim != 3 or incidence.ndim != 3 or nodes.shape[:2] != incidence.shape[:2]:
        raise ValueError("Require nodes [B,N,C] and incidence [B,N,E].")
    if nodes.device != incidence.device or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Require matching devices and finite eps > 0.")
    with torch.autocast(device_type=nodes.device.type, enabled=False):
        h, x = incidence.float(), nodes.float()
        inv_vertex = h.sum(dim=2).clamp_min(eps).rsqrt()
        inv_edge = h.sum(dim=1).clamp_min(eps).reciprocal()
        edge_features = torch.bmm(h.transpose(1, 2), x * inv_vertex.unsqueeze(-1))
        edge_features = edge_features * inv_edge.unsqueeze(-1)
        return torch.bmm(h, edge_features) * inv_vertex.unsqueeze(-1)


class HypergraphConv(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.projection = nn.Linear(channels, channels)
        self.eps = eps

    def forward(self, nodes: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            propagated = hypergraph_propagate(nodes, incidence, self.eps)
            # FP32 even when the outer model is autocast or explicitly half().
            projected = F.linear(propagated, self.projection.weight.float(), self.projection.bias.float())
            return F.gelu(projected)


class RegionHypergraphBranch(nn.Module):
    """Return only the HG increment; gamma=0 yields an exactly zero output."""
    def __init__(
        self, channels: int, hypergraph_hidden_dim: int | None = None,
        node_grid: int | Sequence[int] = 16, k: int = 8, alpha: float = 0.7,
        gamma_init: float = 0.0, eps: float = 1e-6,
    ):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive.")
        hidden = max(1, min(64, channels // 2)) if hypergraph_hidden_dim is None else hypergraph_hidden_dim
        if not isinstance(hidden, int) or hidden < 1:
            raise ValueError("hypergraph_hidden_dim must be a positive integer.")
        grid = (node_grid, node_grid) if isinstance(node_grid, int) else tuple(node_grid)
        if len(grid) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in grid):
            raise ValueError("node_grid must be a positive integer or a pair of positive integers.")
        if not math.isfinite(gamma_init):
            raise ValueError("gamma_init must be finite.")
        self.channels, self.hidden_channels, self.node_grid = channels, hidden, grid
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1)
        self.builder = KNNHypergraph(k=k, alpha=alpha, eps=eps)
        self.hgconv = HypergraphConv(hidden, eps=eps)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        # A bias would inject a nonzero feature even when gamma is zero.
        self.restore = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.channels or min(feature.shape[-2:]) < 1:
            raise ValueError(f"Expected [B,{self.channels},H,W] with positive spatial sizes.")
        batch, _, height, width = feature.shape
        gh, gw = min(height, self.node_grid[0]), min(width, self.node_grid[1])
        reduced = self.reduce(feature)
        # Avoid reduced-precision node averages, distances and degree operations.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            pooled = F.adaptive_avg_pool2d(reduced.float(), (gh, gw))
            nodes = pooled.flatten(2).transpose(1, 2)
            y = (torch.arange(gh, device=feature.device, dtype=torch.float32) + 0.5) / gh
            x = (torch.arange(gw, device=feature.device, dtype=torch.float32) + 0.5) / gw
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            positions = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
            incidence = self.builder(nodes, positions)
            hg_update = self.hgconv(nodes, incidence)
            delta_nodes = self.gamma.float() * hg_update
            spatial = delta_nodes.transpose(1, 2).reshape(batch, self.hidden_channels, gh, gw)
            spatial = F.interpolate(spatial, size=(height, width), mode="bilinear", align_corners=False)
        return self.restore(spatial.to(reduced.dtype))
