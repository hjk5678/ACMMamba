"""Check learnable routing, legacy compatibility and full-model gradients."""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encoder.cross_mamba import DualScanMambaBlock, DualScanStage
from model import DualModalMambaUNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cross-frequency", choices=("every_block", "once_per_stage"), default="every_block")
    parser.add_argument("--depths", type=int, nargs=4, default=(2, 2, 4, 2))
    args = parser.parse_args()
    if any(depth < 1 for depth in args.depths):
        parser.error("All stage depths must be positive")
    if args.amp and args.device != "cuda":
        parser.error("--amp requires --device cuda")
    device = torch.device(args.device)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    block_options = dict(channels=4, mode="cross", d_state=2, history_offsets=(1,))
    soft = DualScanMambaBlock(**block_options, cross_mode="soft").to(device)
    a = torch.randn(2, 4, 4, 15, device=device, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    a_before, b_before = a.detach().clone(), b.detach().clone()
    ra, rb = soft.route_paths(a, b)
    torch.testing.assert_close(ra[:, 2:], (a[:, 2:] + b[:, 2:]) / 2)
    torch.testing.assert_close(rb[:, 2:], ra[:, 2:])
    expected = torch.tensor([0.2, 0.7, 0.35, 0.85], device=device)
    with torch.no_grad():
        soft.cross_logits_a.copy_(torch.logit(expected[:2]))
        soft.cross_logits_b.copy_(torch.logit(expected[2:]))
    ra, rb = soft.route_paths(a, b)
    for i in range(2):
        torch.testing.assert_close(ra[:, i], a[:, i], rtol=0, atol=0)
        torch.testing.assert_close(rb[:, i], b[:, i], rtol=0, atol=0)
        torch.testing.assert_close(ra[:, i + 2], (1 - expected[i]) * a[:, i + 2] + expected[i] * b[:, i + 2])
        torch.testing.assert_close(rb[:, i + 2], (1 - expected[i + 2]) * b[:, i + 2] + expected[i + 2] * a[:, i + 2])
    torch.testing.assert_close(a, a_before, rtol=0, atol=0)
    torch.testing.assert_close(b, b_before, rtol=0, atol=0)
    (ra.square().mean() + 2 * rb.square().mean()).backward()
    for tensor in (a, b, soft.cross_logits_a, soft.cross_logits_b):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) == tensor.numel()
    print("PASS: independent directional formulas, unchanged paths 1/2, no in-place mutation, gradients")

    for logit, swapped in ((-30., False), (30., True)):
        with torch.no_grad():
            soft.cross_logits_a.fill_(logit)
            soft.cross_logits_b.fill_(logit)
        ra, rb = soft.route_paths(a, b)
        torch.testing.assert_close(ra[:, 2:], (b if swapped else a)[:, 2:])
        torch.testing.assert_close(rb[:, 2:], (a if swapped else b)[:, 2:])
    for mode in ("self", "cross"):
        hard = DualScanMambaBlock(**{**block_options, "mode": mode}).to(device)
        explicit = DualScanMambaBlock(**{**block_options, "mode": mode}, cross_mode="hard").to(device)
        explicit.load_state_dict(hard.state_dict(), strict=True)
        assert not any("cross_logits" in k for k in hard.state_dict())
        ra, rb = hard.route_paths(a, b)
        torch.testing.assert_close(ra[:, :2], a[:, :2])
        torch.testing.assert_close(rb[:, :2], b[:, :2])
        torch.testing.assert_close(ra[:, 2:], (b if mode == "cross" else a)[:, 2:])
        torch.testing.assert_close(rb[:, 2:], (a if mode == "cross" else b)[:, 2:])
    self_soft = DualScanMambaBlock(**{**block_options, "mode": "self"}, cross_mode="soft")
    assert not any("cross_logits" in k for k in self_soft.state_dict())
    for initial in (0., 1., -0.1, float("nan")):
        try:
            DualScanMambaBlock(**block_options, cross_mode="soft", soft_cross_init=initial)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid initialization accepted")
    try:
        soft.route_paths(a, b[:, :3])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid path shape accepted")
    stacked = DualScanStage(**block_options, depth=2, drop_path_rates=(0., 0.), cross_mode="soft")
    assert stacked.blocks[0].cross_logits_a is not stacked.blocks[1].cross_logits_a
    for stage_mode in ("self", "cross"):
        once = DualScanStage(
            **{**block_options, "mode": stage_mode}, depth=4, drop_path_rates=(0.,) * 4,
            cross_mode="soft", cross_frequency="once_per_stage",
        )
        assert once.block_modes == (stage_mode, "self", "self", "self")
        assert sum(hasattr(block, "cross_logits_a") for block in once.blocks) == int(stage_mode == "cross")
    default_stage = DualScanStage(**block_options, depth=2, drop_path_rates=(0., 0.), cross_mode="soft")
    default_stage.load_state_dict(stacked.state_dict(), strict=True)
    try:
        DualScanStage(**block_options, depth=2, drop_path_rates=(0., 0.), cross_frequency="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid cross frequency accepted")
    print("PASS: soft limits, legacy hard/self routing, validation and independent stacked blocks")

    for channels_b, classes in ((1, 5), (3, 9)):
        config = dict(
            in_channels_b=channels_b, num_classes=classes, dims=(8, 16, 32, 64),
            depths=tuple(args.depths),
            d_state=2, history_offsets=(1, 4), mlp_ratio=2., drop_path_rate=0.,
            dropout=0., decoder_dropout=0., stage_modes=("cross",) * 4,
            cross_mode="soft", soft_cross_init=0.5, fusion_mode="mlfm",
            cross_frequency=args.cross_frequency,
            decoder_type="unet", decoder_blocks_per_stage=3,
            scan_backend="torch" if args.device == "cpu" else None,
        )
        model = DualModalMambaUNet(**config).to(device).train()
        gates = [p for name, p in model.named_parameters() if "cross_logits" in name]
        total_blocks = sum(args.depths)
        cross_blocks = total_blocks if args.cross_frequency == "every_block" else 4
        assert [len(stage.blocks) for stage in model.encoder.stages] == list(args.depths)
        for stage in model.encoder.stages:
            expected_modes = ("cross",) * len(stage.blocks) if args.cross_frequency == "every_block" else ("cross",) + ("self",) * (len(stage.blocks) - 1)
            assert tuple(block.mode for block in stage.blocks) == expected_modes
        assert sum(p.numel() for p in gates) == 4 * cross_blocks
        assert len({p.data_ptr() for p in gates}) == 2 * cross_blocks
        assert len(model.encoder.soft_cross_weights()) == cross_blocks
        processors = [
            processor for stage in model.encoder.stages for block in stage.blocks
            for branch in (block.as6_a, block.as6_b) for processor in branch
        ]
        assert len(processors) == 8 * total_blocks
        assert len({id(processor) for processor in processors}) == 8 * total_blocks
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        rgb = torch.randn(1, 3, 64, 128, device=device)
        second = torch.randn(1, channels_b, 64, 128, device=device)
        target = torch.randint(classes, (1, 64, 128), device=device)
        with torch.autocast(device_type=args.device, enabled=args.amp):
            output = model(rgb, second)
            assert output.shape == (1, classes, 64, 128)
            loss = torch.nn.functional.cross_entropy(output.float(), target)
        # Fixed small scaling checks AMP gradients without startup overflow.
        scaler = torch.amp.GradScaler("cuda", init_scale=16., enabled=args.amp)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for p in gates:
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert (p.grad != 0).all(), "Every directional gate must receive a gradient"
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        before = [p.detach().clone() for p in gates]
        scaler.step(optimizer)
        scaler.update()
        assert all(not torch.equal(p, old) for p, old in zip(gates, before))
        model.eval()
        with torch.no_grad():
            reference = model(rgb, second)
        buffer = io.BytesIO()
        torch.save({"config": config, "model": model.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True, map_location=device)
        restored = DualModalMambaUNet(**checkpoint["config"]).to(device).eval()
        restored.load_state_dict(checkpoint["model"], strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(rgb, second), reference, rtol=0, atol=0)
        print(f"PASS: CCCC + MLFM + ResNet3 full forward/backward/update/checkpoint ({classes} classes)")
    print(f"All soft-cross checks passed: device={device}, amp={args.amp}, depths={args.depths}, cross_frequency={args.cross_frequency}")


if __name__ == "__main__":
    main()
