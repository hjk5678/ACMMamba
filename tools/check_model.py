"""Run a forward shape check on the server environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import DualModalMambaUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ACMMamba tensor shapes.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Use small channels and torch scan for a quick CPU-compatible check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    kwargs = {}
    if args.light:
        kwargs.update(
            dims=(8, 16, 32, 64),
            d_state=4,
            mlp_ratio=2.0,
            history_offsets=(1, 4, 16),
            scan_backend="torch",
        )
    model = DualModalMambaUNet(**kwargs).to(device).eval()
    modality_a = torch.randn(
        args.batch_size, 3, args.image_size, args.image_size, device=device
    )
    modality_b = torch.randn(
        args.batch_size, 1, args.image_size, args.image_size, device=device
    )

    with torch.inference_mode():
        output = model(modality_a, modality_b, return_features=True)

    print("device:", device)
    print("stage modes:", model.encoder.STAGE_MODES)
    print("parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    print(
        "MLFM weights:",
        [
            (fusion.a.detach().item(), fusion.b.detach().item())
            for fusion in model.encoder.fusions
        ],
    )
    print(
        "decoder residual blocks:",
        [
            len(getattr(model.decoder, f"decoder{index}").refine.blocks)
            for index in range(1, 5)
        ],
    )
    for index, (a, b, fused) in enumerate(
        zip(output["modal"]["a"], output["modal"]["b"], output["fused"]), start=1
    ):
        print(
            f"Stage{index}: A={tuple(a.shape)}, B={tuple(b.shape)}, "
            f"S{index}={tuple(fused.shape)}"
        )
    print("Bottleneck S4':", tuple(output["decoder"]["bottleneck"].shape))
    for index, feature in enumerate(output["decoder"]["stages"], start=1):
        print(f"Decoder{index}: {tuple(feature.shape)}")
    print("logits:", tuple(output["logits"].shape))


if __name__ == "__main__":
    main()
