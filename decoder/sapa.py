"""Pure-PyTorch SAPA-B-style reference following the project specification.

High-resolution skip queries guide a softmax-weighted sum of ORIGINAL decoder
values. No value projection, skip-value fusion, sqrt(D) scaling or interpolation
is performed here. Unfold uses zero padding (including those padded neighbors
in softmax), exactly as in the requested reference algorithm.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


class ChannelLayerNorm(nn.Module):
    """Normalize channels independently at each NCHW pixel (not over H/W)."""

    def __init__(self, channels: int):
        super().__init__()
        _positive_integer(channels, "channels")
        self.channels = channels
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"ChannelLayerNorm requires [B,{self.channels},H,W].")
        x = x.permute(0, 2, 3, 1)                 # NCHW -> NHWC
        x = self.norm(x)                         # normalize last (channel) axis
        return x.permute(0, 3, 1, 2).contiguous() # NHWC -> NCHW


class SAPA(nn.Module):
    """SAPA(encoder_feature=skip, decoder_feature=deep) -> upsampled deep.

    Output channels equal decoder_channels. Callers must still concatenate the
    skip and run their existing stage. Materialized K*K neighborhoods can use
    substantial memory; this intentionally uses no third-party CUDA extensions.
    """

    def __init__(self, encoder_channels: int, decoder_channels: int,
                 embedding_dim: int = 64, up_factor: int = 2,
                 kernel_size: int = 5, qkv_bias: bool = True):
        super().__init__()
        for name, value in (("encoder_channels", encoder_channels),
                            ("decoder_channels", decoder_channels),
                            ("embedding_dim", embedding_dim),
                            ("up_factor", up_factor), ("kernel_size", kernel_size)):
            _positive_integer(value, name)
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd.")
        if not isinstance(qkv_bias, bool):
            raise ValueError("qkv_bias must be boolean.")
        self.encoder_channels, self.decoder_channels = encoder_channels, decoder_channels
        self.embedding_dim, self.up_factor, self.kernel_size = embedding_dim, up_factor, kernel_size
        self.norm_encoder = ChannelLayerNorm(encoder_channels)
        self.norm_decoder = ChannelLayerNorm(decoder_channels)
        self.q_proj = nn.Conv2d(encoder_channels, embedding_dim, 1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(decoder_channels, embedding_dim, 1, bias=qkv_bias)
        # These 1x1 convolutions implement the official SAPA Linear Q/K maps.
        # Initialize like those Linear maps, not like a generic spatial Conv.
        for projection in (self.q_proj, self.k_proj):
            nn.init.trunc_normal_(projection.weight, std=0.02)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

    def _neighborhoods(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        # [B,C,H,W] -> [B,C*K*K,H*W] -> [B,C,K*K,H,W]
        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        return patches.reshape(batch, channels, self.kernel_size ** 2, height, width)

    def forward(self, encoder_feature: torch.Tensor, decoder_feature: torch.Tensor) -> torch.Tensor:
        y, x = encoder_feature, decoder_feature
        if y.ndim != 4 or x.ndim != 4:
            raise ValueError("SAPA inputs must both be NCHW tensors.")
        if y.shape[0] != x.shape[0] or x.shape[0] < 1:
            raise ValueError("SAPA requires equal positive batch sizes.")
        if y.shape[1] != self.encoder_channels or x.shape[1] != self.decoder_channels:
            raise ValueError("SAPA input channels do not match encoder_channels/decoder_channels.")
        if min(*y.shape[-2:], *x.shape[-2:]) < 1:
            raise ValueError("SAPA requires positive spatial dimensions.")
        expected_size = (x.shape[-2] * self.up_factor, x.shape[-1] * self.up_factor)
        if y.shape[-2:] != expected_size:
            raise ValueError(
                f"SAPA requires strict x{self.up_factor} spatial alignment: "
                f"decoder={tuple(x.shape[-2:])}, skip={tuple(y.shape[-2:])}, "
                f"expected skip={expected_size}. No resize/crop fallback is applied.")
        if y.device != x.device or not y.is_floating_point() or not x.is_floating_point():
            raise ValueError("SAPA requires floating-point inputs on the same device.")

        q = self.q_proj(self.norm_encoder(y))    # [B,D,sH,sW], high-res queries
        k = self.k_proj(self.norm_decoder(x))    # [B,D,H,W], low-res keys
        k_patch = self._neighborhoods(k)         # [B,D,K*K,H,W], no HR copies
        batch, channels, height, width = x.shape
        scale, neighbors = self.up_factor, self.kernel_size ** 2
        # Dot products and softmax accumulate in FP32 for AMP stability. Retain
        # FP64 for numerical gradchecks. No sqrt(D) or cosine normalization.
        work_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            # Split each HR axis into (LR center, subpixel offset). Singleton
            # offset axes broadcast the same LR neighborhood to its s*s pixels.
            q = q.to(work_dtype).reshape(batch, self.embedding_dim, 1, height, scale, width, scale)
            k_patch = k_patch.to(work_dtype).reshape(batch, self.embedding_dim, neighbors, height, 1, width, 1)
            similarity = (q * k_patch).sum(dim=1)
            attention = F.softmax(similarity, dim=1)  # [B,K*K,H,s,W,s], neighbor axis
            v_patch = self._neighborhoods(x)          # ORIGINAL x, never normalized
            v_patch = v_patch.to(work_dtype).reshape(batch, channels, neighbors, height, 1, width, 1)
            # Weighted reduction in FP32, returned in the decoder input dtype.
            out = (v_patch * attention.unsqueeze(1)).sum(dim=2)
            out = out.reshape(batch, channels, *expected_size)
        out = out.to(x.dtype)                    # [B,C_decoder,sH,sW]
        if out.shape != (x.shape[0], self.decoder_channels, *y.shape[-2:]):
            raise RuntimeError("SAPA output must preserve batch/channels and match skip resolution.")
        return out


def build_skip_upsampler(encoder_channels, decoder_channels, upsample_mode="bilinear", sapa_options=None):
    """None retains the exact existing bilinear path and old state-dict keys."""
    if upsample_mode not in {"bilinear", "sapa"}:
        raise ValueError("upsample_mode must be 'bilinear' or 'sapa'.")
    if sapa_options is not None and not isinstance(sapa_options, dict):
        raise ValueError("sapa_options must be a dict or None.")
    if upsample_mode == "bilinear":
        if sapa_options is not None:
            raise ValueError("sapa_options requires upsample_mode='sapa'.")
        return None
    options = dict(sapa_options or {})
    allowed = {"embedding_dim", "up_factor", "kernel_size", "qkv_bias"}
    if set(options) - allowed:
        raise ValueError(f"Unknown SAPA options: {sorted(set(options) - allowed)}")
    return SAPA(encoder_channels, decoder_channels, **options)
