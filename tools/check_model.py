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
from encoder.mlfm import FUSION_MODES


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
    parser.add_argument(
        "--fusion-mode",
        choices=FUSION_MODES,
        default="mlfm",
    )
    parser.add_argument(
        "--stage-modes",
        nargs=4,
        choices=("self", "cross"),
        default=("self", "cross", "self", "cross"),
        metavar=("STAGE1", "STAGE2", "STAGE3", "STAGE4"),
        help="Four encoder routing modes in stage order.",
    )
    parser.add_argument('--decoder-type', choices=('unet', 'rhdb'), default='unet')
    parser.add_argument('--mscan', action='store_true', help='Use MSCAN depth=2 in decoder3/4; requires --decoder-type rhdb')
    parser.add_argument('--upsample-mode', choices=('bilinear', 'sapa'), default='bilinear')
    parser.add_argument('--cross-mode', choices=('hard', 'soft'), default='hard')
    parser.add_argument('--soft-cross-init', type=float, default=0.5)
    parser.add_argument('--depths', type=int, nargs=4, default=(1, 1, 1, 1))
    parser.add_argument('--cross-frequency', choices=('every_block', 'once_per_stage'), default='every_block')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    kwargs = {
        "fusion_mode": args.fusion_mode,
        "stage_modes": tuple(args.stage_modes),
        "decoder_type": args.decoder_type,
        "upsample_mode": args.upsample_mode,
        "cross_mode": args.cross_mode,
        "soft_cross_init": args.soft_cross_init,
        "depths": tuple(args.depths),
        "cross_frequency": args.cross_frequency,
    }
    if args.mscan:
        kwargs['mscan_options'] = {'depth': 2}
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
    print("stage modes:", model.encoder.stage_modes)
    print("encoder depths:", model.encoder.depths)
    print("cross mode:", model.encoder.cross_mode)
    print("block routing:", [stage.block_modes for stage in model.encoder.stages])
    print("soft cross weights [A3,A4,B3,B4]:", model.encoder.soft_cross_weights())
    print("fusion mode:", model.encoder.fusion_mode)
    print("decoder type:", model.decoder.decoder_type)
    print("decoder upsampling:", model.decoder.upsample_mode)
    print("parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    print(
        "fusion weights:",
        [
            (
                (fusion.a.detach().item(), fusion.b.detach().item())
                if hasattr(fusion, "a") and hasattr(fusion, "b")
                else None
            )
            for fusion in model.encoder.fusions
        ],
    )
    print(
        "decoder refinement blocks (ResNet or MSCAN):",
        [
            (len(stage.refine.blocks) if hasattr(stage, "refine") else
             len(stage.local_branch.blocks))
            for index in range(1, 5)
            for stage in [getattr(model.decoder, f"decoder{index}")]
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
