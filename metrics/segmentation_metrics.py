"""Streaming metrics for multiclass semantic segmentation."""

from __future__ import annotations

from typing import Dict

import torch
import torch.distributed as dist


class SegmentationConfusionMatrix:
    """Accumulate a confusion matrix without retaining model predictions."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros(
            (num_classes, num_classes), dtype=torch.float64, device=device
        )

    @torch.no_grad()
    def update(self, logits_or_prediction: torch.Tensor, target: torch.Tensor) -> None:
        if logits_or_prediction.ndim == 4:
            if not torch.isfinite(logits_or_prediction).all():
                raise FloatingPointError("Non-finite logits: refusing argmax/segmentation metrics")
            prediction = logits_or_prediction.argmax(dim=1)
        elif logits_or_prediction.ndim == 3:
            prediction = logits_or_prediction
        else:
            raise ValueError("Prediction must have shape [B,C,H,W] or [B,H,W].")
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shapes differ: {prediction.shape} vs {target.shape}."
            )

        valid = target != self.ignore_index
        valid &= target >= 0
        valid &= target < self.num_classes
        if not torch.any(valid):
            return

        target_valid = target[valid].long()
        prediction_valid = prediction[valid].long()
        prediction_valid = prediction_valid.clamp(0, self.num_classes - 1)
        encoded = target_valid * self.num_classes + prediction_valid
        counts = torch.bincount(
            encoded, minlength=self.num_classes * self.num_classes
        )
        self.matrix += counts.reshape(self.num_classes, self.num_classes).to(
            self.matrix.dtype
        )

    def synchronize(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.matrix, op=dist.ReduceOp.SUM)

    def reset(self) -> None:
        self.matrix.zero_()

    def compute(self) -> Dict[str, torch.Tensor]:
        true_positive = self.matrix.diag()
        target_pixels = self.matrix.sum(dim=1)
        predicted_pixels = self.matrix.sum(dim=0)
        union = target_pixels + predicted_pixels - true_positive

        class_iou = torch.full_like(true_positive, float("nan"))
        valid_iou = union > 0
        class_iou[valid_iou] = true_positive[valid_iou] / union[valid_iou]

        class_f1 = torch.full_like(true_positive, float("nan"))
        f1_denominator = target_pixels + predicted_pixels
        valid_f1 = f1_denominator > 0
        class_f1[valid_f1] = (
            2.0 * true_positive[valid_f1] / f1_denominator[valid_f1]
        )

        class_accuracy = torch.full_like(true_positive, float("nan"))
        valid_accuracy = target_pixels > 0
        class_accuracy[valid_accuracy] = (
            true_positive[valid_accuracy] / target_pixels[valid_accuracy]
        )

        total = self.matrix.sum()
        false_positive = predicted_pixels - true_positive
        false_negative = target_pixels - true_positive
        true_negative = total - true_positive - false_positive - false_negative
        class_oa = (
            (true_positive + true_negative) / total
            if total > 0
            else torch.zeros_like(true_positive)
        )
        overall_accuracy = true_positive.sum() / total if total > 0 else total * 0.0
        mean_iou = class_iou[valid_iou].mean() if torch.any(valid_iou) else total * 0.0
        mean_f1 = class_f1[valid_f1].mean() if torch.any(valid_f1) else total * 0.0
        mean_accuracy = (
            class_accuracy[valid_accuracy].mean()
            if torch.any(valid_accuracy)
            else total * 0.0
        )
        frequency = target_pixels / total.clamp_min(1.0)
        frequency_weighted_iou = torch.nansum(frequency * class_iou)

        return {
            "mIoU": mean_iou,
            "mF1": mean_f1,
            "mAcc": mean_accuracy,
            "OA": overall_accuracy,
            "FWIoU": frequency_weighted_iou,
            "class_iou": class_iou,
            "class_f1": class_f1,
            "class_accuracy": class_accuracy,
            "class_oa": class_oa,
            "confusion_matrix": self.matrix.clone(),
        }
