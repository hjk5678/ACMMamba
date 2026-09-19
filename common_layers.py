"""Dependency-light PyTorch layers shared by encoder and decoder."""
from __future__ import annotations

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Per-sample stochastic depth (shared, unchanged encoder implementation)."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("Drop-path probability must be in [0, 1).")
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_probability)
        return x * mask / keep_probability
