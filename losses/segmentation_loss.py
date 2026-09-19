"""Losses for five-class multimodal remote-sensing segmentation."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# Pixel frequencies measured over all valid pixels in the Shandong GT set.
SHANDONG_CLASS_FREQUENCIES = (
    0.2169974384,
    0.2045685843,
    0.0863595045,
    0.3627266184,
    0.1293478544,
)


def class_weights_from_frequencies(
    frequencies: Sequence[float], power: float = 0.5
) -> torch.Tensor:
    """Create mean-one inverse-frequency weights.

    ``power=0.5`` uses inverse square-root balancing, which compensates for the
    underrepresented classes without the instability of full inverse frequency.
    """
    values = torch.as_tensor(frequencies, dtype=torch.float32)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("Class frequencies must be a non-empty 1-D sequence.")
    if torch.any(values <= 0):
        raise ValueError("Every class frequency must be positive.")
    weights = values.pow(-power)
    return weights / weights.mean()


class MulticlassDiceLoss(nn.Module):
    """Soft multiclass Dice loss with an ignored target value."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        smooth: float = 1.0,
        include_background: bool = True,
        ignore_empty_classes: bool = True,
        class_weights: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.include_background = include_background
        self.ignore_empty_classes = ignore_empty_classes

        if class_weights is None:
            weights = torch.empty(0, dtype=torch.float32)
        else:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights.numel() != num_classes:
                raise ValueError(
                    f"Expected {num_classes} class weights, got {weights.numel()}."
                )
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Logits must have shape [B,C,H,W], got {logits.shape}.")
        if target.ndim != 3:
            raise ValueError(f"Target must have shape [B,H,W], got {target.shape}.")
        if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
            raise ValueError("Logits and target batch/spatial dimensions must match.")
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} logit channels, got {logits.shape[1]}."
            )

        valid_mask = target != self.ignore_index
        if not torch.any(valid_mask):
            return logits.sum() * 0.0

        invalid_labels = valid_mask & ((target < 0) | (target >= self.num_classes))
        if torch.any(invalid_labels):
            invalid_values = torch.unique(target[invalid_labels]).detach().cpu().tolist()
            raise ValueError(f"Target contains invalid class values: {invalid_values}")

        safe_target = torch.where(valid_mask, target, torch.zeros_like(target)).long()
        probabilities = torch.softmax(logits.float(), dim=1)
        target_one_hot = F.one_hot(safe_target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).to(probabilities.dtype)
        valid_mask_float = valid_mask.unsqueeze(1).to(probabilities.dtype)

        probabilities = probabilities * valid_mask_float
        target_one_hot = target_one_hot * valid_mask_float
        reduce_dims = (0, 2, 3)
        intersection = (probabilities * target_one_hot).sum(dim=reduce_dims)
        prediction_volume = probabilities.sum(dim=reduce_dims)
        target_volume = target_one_hot.sum(dim=reduce_dims)
        dice = (2.0 * intersection + self.smooth) / (
            prediction_volume + target_volume + self.smooth
        )

        selected = torch.ones(self.num_classes, dtype=torch.bool, device=logits.device)
        if not self.include_background:
            selected[0] = False
        if self.ignore_empty_classes:
            selected &= target_volume > 0
        if not torch.any(selected):
            return logits.sum() * 0.0

        per_class_loss = 1.0 - dice[selected]
        if self.class_weights.numel() > 0:
            weights = self.class_weights.to(
                device=logits.device, dtype=probabilities.dtype
            )
            weights = weights[selected]
            weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
            return (per_class_loss * weights).sum()
        return per_class_loss.mean()


class CombinedSegmentationLoss(nn.Module):
    """Weighted cross-entropy plus multiclass Soft Dice."""

    def __init__(
        self,
        num_classes: int = 5,
        ignore_index: int = 255,
        ce_coefficient: float = 1.0,
        dice_coefficient: float = 1.0,
        class_weights: Sequence[float] | torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        dice_smooth: float = 1.0,
        include_background: bool = True,
    ) -> None:
        super().__init__()
        if ce_coefficient < 0 or dice_coefficient < 0:
            raise ValueError("Loss coefficients must be non-negative.")
        if ce_coefficient == 0 and dice_coefficient == 0:
            raise ValueError("At least one loss coefficient must be positive.")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_coefficient = ce_coefficient
        self.dice_coefficient = dice_coefficient
        self.label_smoothing = label_smoothing

        if class_weights is None:
            weights = torch.empty(0, dtype=torch.float32)
        else:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights.numel() != num_classes:
                raise ValueError(
                    f"Expected {num_classes} class weights, got {weights.numel()}."
                )
        self.register_buffer("class_weights", weights)
        self.dice = MulticlassDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            smooth=dice_smooth,
            include_background=include_background,
            class_weights=weights if weights.numel() > 0 else None,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        valid_mask = target != self.ignore_index
        if not torch.any(valid_mask):
            zero = logits.float().sum() * 0.0
            if return_components:
                return {"loss": zero, "cross_entropy": zero, "dice": zero}
            return zero

        weight = self.class_weights if self.class_weights.numel() > 0 else None
        cross_entropy = F.cross_entropy(
            logits,
            target.long(),
            weight=weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        dice = self.dice(logits, target)
        total = self.ce_coefficient * cross_entropy + self.dice_coefficient * dice
        if return_components:
            return {"loss": total, "cross_entropy": cross_entropy, "dice": dice}
        return total


def build_segmentation_loss(
    num_classes: int = 5,
    ignore_index: int = 255,
    use_dataset_class_weights: bool = True,
    class_frequencies: Sequence[float] | None = None,
    class_weight_power: float = 0.5,
    ce_coefficient: float = 1.0,
    dice_coefficient: float = 1.0,
    label_smoothing: float = 0.0,
) -> CombinedSegmentationLoss:
    """Build the default loss used by this project."""
    class_weights = None
    if use_dataset_class_weights:
        frequencies = (
            SHANDONG_CLASS_FREQUENCIES
            if class_frequencies is None
            else tuple(float(value) for value in class_frequencies)
        )
        if num_classes != len(frequencies):
            raise ValueError(
                f"Expected {num_classes} class frequencies, got {len(frequencies)}."
            )
        class_weights = class_weights_from_frequencies(
            frequencies, power=class_weight_power
        )
    return CombinedSegmentationLoss(
        num_classes=num_classes,
        ignore_index=ignore_index,
        ce_coefficient=ce_coefficient,
        dice_coefficient=dice_coefficient,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
    )
