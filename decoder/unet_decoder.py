"""U-Net decoder for the four fused encoder features S1-S4."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 32) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualBlock(nn.Module):
    """ResNet basic block using GroupNorm for small segmentation batches."""

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(groups, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return self.activation(x + identity)


class ResidualStage(nn.Module):
    """Stack multiple residual units inside one decoder stage."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError("A residual stage must contain at least one block.")
        blocks = [ResidualBlock(in_channels, out_channels, dropout)]
        blocks.extend(
            ResidualBlock(out_channels, out_channels, dropout)
            for _ in range(num_blocks - 1)
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.refine = ResidualStage(
            out_channels + skip_channels,
            out_channels,
            num_blocks=num_residual_blocks,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.refine(torch.cat((x, skip), dim=1))


class UNetDecoder(nn.Module):
    """Symmetric four-stage decoder for ``[S1,S2,S3,S4]``.

    The dataflow is ``Bottleneck(S4)->S4'``, followed by
    ``D1(S4',S4)``, ``D2(D1,S3)``, ``D3(D2,S2)`` and ``D4(D3,S1)``.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int] = (96, 192, 384, 768),
        num_classes: int = 5,
        dropout: float = 0.1,
        blocks_per_stage: int = 3,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 4:
            raise ValueError("U-Net decoder expects exactly four encoder scales.")
        c1, c2, c3, c4 = (int(channels) for channels in encoder_channels)
        # S4 is retained as a skip. A separate downsampled representation S4'
        # enters Decoder1, giving four decoder stages symmetric to the encoder.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(c4), c4),
            nn.GELU(),
            ConvBlock(c4, c4, dropout),
        )
        self.decoder1 = UpBlock(c4, c4, c4, dropout, blocks_per_stage)
        self.decoder2 = UpBlock(c4, c3, c3, dropout, blocks_per_stage)
        self.decoder3 = UpBlock(c3, c2, c2, dropout, blocks_per_stage)
        self.decoder4 = UpBlock(c2, c1, c1, dropout, blocks_per_stage)
        self.segmentation_head = nn.Sequential(
            ConvBlock(c1, c1, dropout=0.0),
            nn.Conv2d(c1, num_classes, kernel_size=1),
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        output_size: Tuple[int, int],
        return_features: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor | List[torch.Tensor]]]:
        if len(features) != 4:
            raise ValueError(f"Expected [S1,S2,S3,S4], got {len(features)} features.")
        s1, s2, s3, s4 = features
        s4_prime = self.bottleneck(s4)
        d1 = self.decoder1(s4_prime, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)
        logits = self.segmentation_head(d4)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        if return_features:
            return logits, {"bottleneck": s4_prime, "stages": [d1, d2, d3, d4]}
        return logits
