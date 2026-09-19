"""Four-stage dual-modal VMamba-style hierarchical encoder."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from .cross_mamba import DualScanStage
from .layers import PatchEmbed, PatchMerging
from .mlfm import FUSION_MODES, build_fusion


class DualModalVMambaEncoder(nn.Module):
    """Independent A/B branches with Self-Cross-Self-Cross interaction.

    Stage 3 receives Stage 2's updated outputs, so prior cross-modal information
    remains present even though Stage 3 introduces no new exchanged paths.
    """

    STAGE_MODES = ("self", "cross", "self", "cross")

    def __init__(
        self,
        in_channels_a: int = 3,
        in_channels_b: int = 1,
        dims: Sequence[int] = (96, 192, 384, 768),
        depths: Sequence[int] = (1, 1, 1, 1),
        d_state: int = 16,
        as6_expand: float = 1.0,
        history_offsets: Sequence[int] = (1, 4, 16, 64),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.2,
        scan_backend: str | None = None,
        fusion_mode: str = "mlfm",
        stage_modes: Sequence[str] = STAGE_MODES,
        cross_mode: str = "hard",
        soft_cross_init: float = 0.5,
        cross_frequency: str = "every_block",
        mlfm_hg_options: dict | None = None,
    ) -> None:
        super().__init__()
        if len(dims) != 4 or len(depths) != 4:
            raise ValueError("The encoder requires exactly four dims and four depths.")
        self.dims = tuple(int(dim) for dim in dims)
        self.cross_mode = cross_mode
        self.cross_frequency = cross_frequency
        self.depths = tuple(int(depth) for depth in depths)
        self.stage_modes = tuple(str(mode).lower().strip() for mode in stage_modes)
        if len(self.stage_modes) != 4:
            raise ValueError(
                f"The encoder requires exactly four stage modes, got "
                f"{self.stage_modes}."
            )
        invalid_modes = tuple(
            mode for mode in self.stage_modes if mode not in {"self", "cross"}
        )
        if invalid_modes:
            raise ValueError(
                f"Unsupported stage modes {invalid_modes}; each mode must be "
                "'self' or 'cross'."
            )
        self.fusion_mode = str(fusion_mode).lower().strip()
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(
                f"Unknown fusion_mode {fusion_mode!r}; expected one of {FUSION_MODES}."
            )

        self.patch_embed_a = PatchEmbed(in_channels_a, self.dims[0])
        self.patch_embed_b = PatchEmbed(in_channels_b, self.dims[0])
        self.downsamples_a = nn.ModuleList(
            [PatchMerging(self.dims[index], self.dims[index + 1]) for index in range(3)]
        )
        self.downsamples_b = nn.ModuleList(
            [PatchMerging(self.dims[index], self.dims[index + 1]) for index in range(3)]
        )

        total_depth = sum(self.depths)
        drop_path_rates = torch.linspace(0, drop_path_rate, total_depth).tolist()
        self.stages = nn.ModuleList()
        offset = 0
        for channels, depth, mode in zip(
            self.dims, self.depths, self.stage_modes
        ):
            rates = drop_path_rates[offset : offset + depth]
            offset += depth
            self.stages.append(
                DualScanStage(
                    channels=channels,
                    depth=depth,
                    mode=mode,
                    drop_path_rates=rates,
                    d_state=d_state,
                    as6_expand=as6_expand,
                    history_offsets=history_offsets,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    scan_backend=scan_backend,
                    cross_mode=cross_mode,
                    soft_cross_init=soft_cross_init,
                    cross_frequency=cross_frequency,
                )
            )

        self.fusions = nn.ModuleList(
            [
                build_fusion(self.fusion_mode, channels, channels, channels, mlfm_hg_options)
                for channels in self.dims
            ]
        )

    def hypergraph_fusion_weights(self) -> Dict[str, List[float]]:
        """Detached independent [gamma_A, gamma_B] for HG-MLFM stages."""
        return {
            f"stage{index}": torch.stack((fusion.hypergraph_a.gamma, fusion.hypergraph_b.gamma)).detach().cpu().tolist()
            for index, fusion in enumerate(self.fusions, 1)
            if hasattr(fusion, "hypergraph_a")
        }

    def soft_cross_weights(self) -> Dict[str, List[float]]:
        """Detached exchange fractions in [A3, A4, B3, B4] order per block."""
        weights = {}
        for stage_index, stage in enumerate(self.stages, 1):
            for block_index, block in enumerate(stage.blocks, 1):
                if hasattr(block, "cross_logits_a"):
                    weights[f"stage{stage_index}.block{block_index}"] = torch.cat(
                        (block.cross_logits_a, block.cross_logits_b)
                    ).detach().sigmoid().cpu().tolist()
        return weights

    def forward(
        self,
        modality_a: torch.Tensor,
        modality_b: torch.Tensor,
        return_modal_features: bool = False,
    ) -> List[torch.Tensor] | Tuple[List[torch.Tensor], Dict[str, List[torch.Tensor]]]:
        if modality_a.shape[0] != modality_b.shape[0]:
            raise ValueError("A and B must have the same batch size.")
        if modality_a.shape[-2:] != modality_b.shape[-2:]:
            raise ValueError("A and B must have the same spatial resolution.")

        feature_a = self.patch_embed_a(modality_a)
        feature_b = self.patch_embed_b(modality_b)
        fused_features: List[torch.Tensor] = []
        features_a: List[torch.Tensor] = []
        features_b: List[torch.Tensor] = []

        for stage_index, (stage, fusion) in enumerate(zip(self.stages, self.fusions)):
            if stage_index > 0:
                feature_a = self.downsamples_a[stage_index - 1](feature_a)
                feature_b = self.downsamples_b[stage_index - 1](feature_b)
            feature_a, feature_b = stage(feature_a, feature_b)
            fused_features.append(fusion(feature_a, feature_b))
            features_a.append(feature_a)
            features_b.append(feature_b)

        if return_modal_features:
            return fused_features, {"a": features_a, "b": features_b}
        return fused_features
