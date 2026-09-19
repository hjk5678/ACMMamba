"""Segmentation decoders."""

from .unet_decoder import ResidualBlock, ResidualStage, UNetDecoder

__all__ = ["ResidualBlock", "ResidualStage", "UNetDecoder", "RHDBBlock", "MSCANStage", "SAPA", "ChannelLayerNorm"]


def __getattr__(name):
    if name in {"SAPA", "ChannelLayerNorm"}:
        from .sapa import SAPA, ChannelLayerNorm
        return {"SAPA": SAPA, "ChannelLayerNorm": ChannelLayerNorm}[name]
    if name == "MSCANStage":
        from .mscan import MSCANStage
        return MSCANStage
    if name == "RHDBBlock":
        from .rhdb import RHDBBlock
        return RHDBBlock
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
