"""Check loss values, ignore-label handling and backward gradients."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses import build_segmentation_loss


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = build_segmentation_loss().to(device)
    logits = torch.randn(2, 5, 32, 32, device=device, requires_grad=True)
    target = torch.randint(0, 5, (2, 32, 32), device=device)
    target[:, :4, :4] = 255

    components = criterion(logits, target, return_components=True)
    components["loss"].backward()

    print("device:", device)
    print("class weights:", criterion.class_weights.detach().cpu().tolist())
    print("cross entropy:", components["cross_entropy"].item())
    print("dice:", components["dice"].item())
    print("total loss:", components["loss"].item())
    print("gradient finite:", bool(torch.isfinite(logits.grad).all()))


if __name__ == "__main__":
    main()
