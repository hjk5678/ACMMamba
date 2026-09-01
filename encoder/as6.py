"""Adaptor-S6 used by every directional scan path.

The selective S6 recurrence reuses the CUDA/PyTorch implementation bundled in
``vmamba.py``. A content-dependent memory adaptor retrieves several historical
sequence offsets, while the spatial adaptor restores local 2-D continuity after
the directional sequence is mapped back to image coordinates.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from vmamba import mamba_init, selective_scan_fn


class MemoryAdaptor(nn.Module):
    """Retrieve positive and negative correlations from earlier sequence states.

    The fused VMamba CUDA kernel does not expose every internal hidden state.
    Therefore, the adaptor operates on the S6 output state sequence and uses a
    configurable set of causal history offsets as its candidate memory bank.
    """

    def __init__(self, channels: int, history_offsets: Sequence[int]) -> None:
        super().__init__()
        offsets = tuple(sorted({int(offset) for offset in history_offsets if offset > 0}))
        if not offsets:
            raise ValueError("At least one positive history offset is required.")
        self.history_offsets = offsets
        self.query = nn.Conv1d(channels, channels, kernel_size=1, groups=channels)
        self.gate = nn.Conv1d(channels, channels, kernel_size=1, groups=channels)
        nn.init.constant_(self.gate.bias, -2.0)

    @staticmethod
    def _shift_right(x: torch.Tensor, offset: int) -> torch.Tensor:
        length = x.shape[-1]
        if offset >= length:
            return torch.zeros_like(x)
        return F.pad(x[..., :-offset], (offset, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        memories = torch.stack(
            [self._shift_right(x, offset) for offset in self.history_offsets], dim=1
        )  # [B, M, C, L]
        query = self.query(x).unsqueeze(1)
        correlation = (query * memories).sum(dim=2) / math.sqrt(x.shape[1])

        positive_weights = torch.softmax(F.relu(correlation), dim=1)
        negative_weights = torch.softmax(F.relu(-correlation), dim=1)
        weights = 0.5 * (positive_weights + negative_weights)
        recalled = (weights.unsqueeze(2) * memories).sum(dim=1)
        return x + torch.sigmoid(self.gate(x)) * recalled


class SpatialAdaptor(nn.Module):
    """Depthwise-separable spatial recovery with a learnable residual scale."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapted = self.pointwise(self.activation(self.depthwise(x)))
        return x + self.scale * adapted


class AdaptorS6(nn.Module):
    """One directional selective state-space path with memory adaptation."""

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        expand: float = 1.0,
        dt_rank: int | str = "auto",
        history_offsets: Sequence[int] = (1, 4, 16, 64),
        scan_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.inner_channels = int(channels * expand)
        self.d_state = d_state
        self.dt_rank = math.ceil(channels / 16) if dt_rank == "auto" else int(dt_rank)
        self.scan_backend = scan_backend

        self.input_projection = nn.Conv1d(channels, self.inner_channels * 2, kernel_size=1)
        self.local_conv = nn.Conv1d(
            self.inner_channels,
            self.inner_channels,
            kernel_size=3,
            padding=1,
            groups=self.inner_channels,
        )
        self.parameter_projection = nn.Conv1d(
            self.inner_channels, self.dt_rank + 2 * d_state, kernel_size=1, bias=False
        )

        dt_projection = mamba_init.dt_init(self.dt_rank, self.inner_channels)
        self.dt_weight = nn.Parameter(dt_projection.weight.detach().clone())
        self.dt_bias = nn.Parameter(dt_projection.bias.detach().clone())
        self.A_logs = mamba_init.A_log_init(d_state, self.inner_channels)
        self.Ds = mamba_init.D_init(self.inner_channels)

        self.memory_adaptor = MemoryAdaptor(self.inner_channels, history_offsets)
        self.output_norm = nn.LayerNorm(self.inner_channels)
        self.output_projection = nn.Conv1d(self.inner_channels, channels, kernel_size=1)
        self.spatial_adaptor = SpatialAdaptor(channels)
        self.activation = nn.SiLU()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Process a directional sequence shaped ``[B, C, L]``."""
        input_dtype = sequence.dtype
        projected, gate = self.input_projection(sequence).chunk(2, dim=1)
        projected = self.activation(self.local_conv(projected))

        parameters = self.parameter_projection(projected)
        delta, B_parameter, C_parameter = torch.split(
            parameters, (self.dt_rank, self.d_state, self.d_state), dim=1
        )
        delta = torch.einsum("b r l, d r -> b d l", delta, self.dt_weight)

        # The CUDA selective-scan kernel requires the sequence dimension to be
        # contiguous (stride(-1) == 1). Einsum/split views do not guarantee it.
        scan_input = projected.float().contiguous()
        scan_delta = delta.float().contiguous()
        A_parameter = -self.A_logs.float().exp().contiguous()
        B_parameter = B_parameter.float().unsqueeze(1).contiguous()
        C_parameter = C_parameter.float().unsqueeze(1).contiguous()
        output = selective_scan_fn(
            scan_input,
            scan_delta,
            A_parameter,
            B_parameter,
            C_parameter,
            self.Ds.float().contiguous(),
            self.dt_bias.float().contiguous(),
            delta_softplus=True,
            oflex=True,
            backend=self.scan_backend,
        )
        output = output.to(input_dtype)
        output = self.memory_adaptor(output)
        output = self.output_norm(output.transpose(1, 2)).transpose(1, 2)
        output = output * self.activation(gate)
        return self.output_projection(output)

    def recover_spatial(self, feature: torch.Tensor) -> torch.Tensor:
        """Apply the spatial part of AS6 after inverse directional mapping."""
        return self.spatial_adaptor(feature)
