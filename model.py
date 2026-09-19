"""Complete dual-modal Self-Cross-Self-Cross VMamba U-Net."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

from decoder import UNetDecoder
from encoder import DualModalVMambaEncoder


class DualModalMambaUNet(nn.Module):
    """Semantic segmentation network for paired modalities A and B."""

    def __init__(
        self,
        in_channels_a: int = 3,
        in_channels_b: int = 1,
        num_classes: int = 5,
        dims: Sequence[int] = (96, 192, 384, 768),
        depths: Sequence[int] = (1, 1, 1, 1),
        d_state: int = 16,
        as6_expand: float = 1.0,
        history_offsets: Sequence[int] = (1, 4, 16, 64),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        decoder_dropout: float = 0.1,
        decoder_blocks_per_stage: int = 3,
        drop_path_rate: float = 0.2,
        scan_backend: str | None = None,
        fusion_mode: str = "mlfm",
        stage_modes: Sequence[str] = ("self", "cross", "self", "cross"),
        decoder_type: str = "unet",
        cross_mode: str = "hard",
        soft_cross_init: float = 0.5,
        cross_frequency: str = "every_block",
        rhdb_options: dict | None = None,
        mscan_options: dict | None = None,
        upsample_mode: str = "bilinear",
        sapa_options: dict | None = None,
        mlfm_hg_options: dict | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder = DualModalVMambaEncoder(
            in_channels_a=in_channels_a,
            in_channels_b=in_channels_b,
            dims=dims,
            depths=depths,
            d_state=d_state,
            as6_expand=as6_expand,
            history_offsets=history_offsets,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            scan_backend=scan_backend,
            fusion_mode=fusion_mode,
            stage_modes=stage_modes,
            cross_mode=cross_mode,
            soft_cross_init=soft_cross_init,
            cross_frequency=cross_frequency,
            mlfm_hg_options=mlfm_hg_options,
        )
        self.decoder = UNetDecoder(
            encoder_channels=dims,
            num_classes=num_classes,
            dropout=decoder_dropout,
            blocks_per_stage=decoder_blocks_per_stage,
            decoder_type=decoder_type,
            rhdb_options=rhdb_options,
            mscan_options=mscan_options,
            upsample_mode=upsample_mode,
            sapa_options=sapa_options,
        )

    def forward(
        self,
        modality_a: torch.Tensor,
        modality_b: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | Dict[str, object]:
        output_size = modality_a.shape[-2:]
        if return_features:
            fused, modal = self.encoder(
                modality_a, modality_b, return_modal_features=True
            )
            logits, decoder_features = self.decoder(
                fused, output_size, return_features=True
            )
            return {
                "logits": logits,
                "fused": fused,
                "modal": modal,
                "decoder": decoder_features,
            }

        fused = self.encoder(modality_a, modality_b)
        return self.decoder(fused, output_size)


# Project-friendly alias.
ACMMamba = DualModalMambaUNet


def build_model(**kwargs) -> DualModalMambaUNet:
    return DualModalMambaUNet(**kwargs)
