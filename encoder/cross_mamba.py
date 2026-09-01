"""Self/Cross four-direction scan blocks for paired image modalities."""

from __future__ import annotations

from typing import Literal, Sequence, Tuple

import torch
import torch.nn as nn

from .as6 import AdaptorS6
from .layers import ConvFFN, DropPath, LayerNorm2d


StageMode = Literal["self", "cross"]


def four_direction_scan(x: torch.Tensor) -> torch.Tensor:
    """Return row, column, reversed-row and reversed-column paths.

    Output shape is ``[B, 4, C, H*W]``.
    """
    batch, channels, height, width = x.shape
    row = x.reshape(batch, channels, height * width)
    column = x.transpose(2, 3).contiguous().reshape(batch, channels, height * width)
    return torch.stack((row, column, row.flip(-1), column.flip(-1)), dim=1)


def inverse_directional_path(
    sequence: torch.Tensor, direction: int, height: int, width: int
) -> torch.Tensor:
    """Map one directional sequence back to canonical ``[B,C,H,W]`` layout."""
    batch, channels, length = sequence.shape
    if length != height * width:
        raise ValueError(
            f"Sequence length {length} does not match spatial size {height}x{width}."
        )
    if direction == 0:
        return sequence.reshape(batch, channels, height, width)
    if direction == 1:
        return sequence.reshape(batch, channels, width, height).transpose(2, 3).contiguous()
    if direction == 2:
        return sequence.flip(-1).reshape(batch, channels, height, width)
    if direction == 3:
        return (
            sequence.flip(-1)
            .reshape(batch, channels, width, height)
            .transpose(2, 3)
            .contiguous()
        )
    raise ValueError(f"Direction must be in [0, 3], got {direction}.")


class ModalityPreprocessor(nn.Module):
    """The branch-specific Linear-DWConv-SiLU preparation used before scanning."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DualScanMambaBlock(nn.Module):
    """A paired VMamba-style block with an explicit self or cross route table."""

    def __init__(
        self,
        channels: int,
        mode: StageMode,
        d_state: int = 16,
        as6_expand: float = 1.0,
        history_offsets: Sequence[int] = (1, 4, 16, 64),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        scan_backend: str | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"self", "cross"}:
            raise ValueError(f"Unsupported scan mode: {mode}")
        self.mode = mode

        # Every modality owns all branch parameters, including normalization,
        # preprocessing, four AS6 paths, output projection and FFN.
        self.norm_a = LayerNorm2d(channels)
        self.norm_b = LayerNorm2d(channels)
        self.pre_a = ModalityPreprocessor(channels)
        self.pre_b = ModalityPreprocessor(channels)
        self.as6_a = nn.ModuleList(
            [
                AdaptorS6(
                    channels,
                    d_state=d_state,
                    expand=as6_expand,
                    history_offsets=history_offsets,
                    scan_backend=scan_backend,
                )
                for _ in range(4)
            ]
        )
        self.as6_b = nn.ModuleList(
            [
                AdaptorS6(
                    channels,
                    d_state=d_state,
                    expand=as6_expand,
                    history_offsets=history_offsets,
                    scan_backend=scan_backend,
                )
                for _ in range(4)
            ]
        )
        self.projection_a = nn.Conv2d(channels, channels, kernel_size=1)
        self.projection_b = nn.Conv2d(channels, channels, kernel_size=1)
        self.drop_path = DropPath(drop_path)
        self.scan_scale_a = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))
        self.scan_scale_b = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))

        self.ffn_norm_a = LayerNorm2d(channels)
        self.ffn_norm_b = LayerNorm2d(channels)
        self.ffn_a = ConvFFN(channels, expansion=mlp_ratio, dropout=dropout)
        self.ffn_b = ConvFFN(channels, expansion=mlp_ratio, dropout=dropout)
        self.ffn_scale_a = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))
        self.ffn_scale_b = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))

    def route_paths(
        self, paths_a: torch.Tensor, paths_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "self":
            return paths_a, paths_b

        # CrossMamba: only reverse directions 3 and 4 are exchanged.
        routed_a = torch.cat((paths_a[:, :2], paths_b[:, 2:]), dim=1)
        routed_b = torch.cat((paths_b[:, :2], paths_a[:, 2:]), dim=1)
        return routed_a, routed_b

    @staticmethod
    def _run_paths(
        routed_paths: torch.Tensor,
        processors: nn.ModuleList,
        height: int,
        width: int,
    ) -> torch.Tensor:
        restored_paths = []
        for direction, processor in enumerate(processors):
            sequence = processor(routed_paths[:, direction])
            feature = inverse_directional_path(sequence, direction, height, width)
            restored_paths.append(processor.recover_spatial(feature))
        # The requested aggregation is an element-wise sum, not an average.
        return torch.stack(restored_paths, dim=0).sum(dim=0)

    def forward(
        self, feature_a: torch.Tensor, feature_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if feature_a.shape != feature_b.shape:
            raise ValueError(
                f"A/B stage features must have equal shapes, got "
                f"{tuple(feature_a.shape)} and {tuple(feature_b.shape)}."
            )
        height, width = feature_a.shape[-2:]
        paths_a = four_direction_scan(self.pre_a(self.norm_a(feature_a)))
        paths_b = four_direction_scan(self.pre_b(self.norm_b(feature_b)))
        routed_a, routed_b = self.route_paths(paths_a, paths_b)

        scan_a = self._run_paths(routed_a, self.as6_a, height, width)
        scan_b = self._run_paths(routed_b, self.as6_b, height, width)
        feature_a = feature_a + self.drop_path(
            self.scan_scale_a * self.projection_a(scan_a)
        )
        feature_b = feature_b + self.drop_path(
            self.scan_scale_b * self.projection_b(scan_b)
        )

        feature_a = feature_a + self.drop_path(
            self.ffn_scale_a * self.ffn_a(self.ffn_norm_a(feature_a))
        )
        feature_b = feature_b + self.drop_path(
            self.ffn_scale_b * self.ffn_b(self.ffn_norm_b(feature_b))
        )
        return feature_a, feature_b


class DualScanStage(nn.Module):
    """Stack blocks that all follow the interaction policy of one stage."""

    def __init__(
        self,
        channels: int,
        depth: int,
        mode: StageMode,
        drop_path_rates: Sequence[float],
        **block_kwargs,
    ) -> None:
        super().__init__()
        if depth != len(drop_path_rates):
            raise ValueError("Each block requires one drop-path rate.")
        self.mode = mode
        self.blocks = nn.ModuleList(
            [
                DualScanMambaBlock(
                    channels=channels,
                    mode=mode,
                    drop_path=drop_path_rates[index],
                    **block_kwargs,
                )
                for index in range(depth)
            ]
        )

    def forward(
        self, feature_a: torch.Tensor, feature_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            feature_a, feature_b = block(feature_a, feature_b)
        return feature_a, feature_b
