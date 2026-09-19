"""RHDB numerical and integration checks; no dataset or custom HG library needed."""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decoder import RHDBBlock, UNetDecoder
from decoder.hypergraph import KNNHypergraph, RegionHypergraphBranch, hypergraph_propagate
from decoder.rhdb import AdaptiveBranchFusion
from decoder.unet_decoder import UpBlock


def finite_gradients(module, inputs):
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, f"Unused parameter: {name}"
        assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}"
    for value in inputs:
        assert value.grad is not None and torch.isfinite(value.grad).all()


def backward_for_check(loss, optimizer, scaler):
    # Match training: autocast alone does not protect small FP16 gradients.
    # Keep backward outside autocast and inspect UNscaled parameter gradients.
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)


def check_math(device):
    x = torch.randn(2, 5, 4, device=device, requires_grad=True)
    # Non-symmetric incidence, deliberately unequal degrees and E != N.
    h = torch.tensor([
        [[1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 0]],
    ], dtype=torch.float32, device=device)
    result = hypergraph_propagate(x, h)
    expected = []
    for sample in range(2):
        dv = torch.diag(h[sample].sum(1).rsqrt())
        de = torch.diag(h[sample].sum(0).reciprocal())
        g = dv @ h[sample] @ de @ h[sample].T @ dv
        torch.testing.assert_close(g, g.T)
        expected.append(g @ x[sample])
    torch.testing.assert_close(result, torch.stack(expected), rtol=1e-5, atol=1e-6)
    result.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(hypergraph_propagate(x, torch.zeros_like(h))).all()

    positions = torch.tensor([[0, 0], [0.1, 0.1], [0.5, 0.6], [0.7, 0.8], [1, 1]], device=device)
    builder = KNNHypergraph(k=2)
    actual = builder(x, positions)
    assert actual.shape == (2, 5, 5) and actual.dtype == torch.float32
    assert not actual.requires_grad
    assert (actual.diagonal(dim1=1, dim2=2) == 1).all()
    assert (actual.sum(1) == 3).all()  # center + TWO distinct other nodes
    for sample in range(2):
        norm = F.normalize(x[sample].detach(), dim=-1)
        distance = 0.7 * (1 - norm @ norm.T) + 0.3 * torch.cdist(positions, positions)
        distance.fill_diagonal_(float('inf'))
        for center in range(5):
            members = {center, *distance[center].topk(2, largest=False).indices.tolist()}
            assert set(actual[sample, :, center].nonzero().flatten().tolist()) == members
        torch.testing.assert_close(actual[sample:sample+1], builder(x[sample:sample+1], positions))
    identity = KNNHypergraph(k=0)(x, positions)
    torch.testing.assert_close(hypergraph_propagate(x, identity), x)
    assert (KNNHypergraph(k=99)(x, positions) == 1).all()
    singleton = builder(torch.zeros(1, 1, 3, device=device), torch.zeros(1, 2, device=device))
    assert singleton.item() == 1
    assert torch.isfinite(builder(torch.zeros_like(x), positions)).all()
    print("PASS: HGNN explicit-matrix reference, incidence orientation, K other neighbors, batch isolation, edge cases")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--full-model', action='store_true', help='Also check full VMamba encoder + RHDB model')
    parser.add_argument('--all-rhdb-stages', action='store_true', help='Check RHDB at all four decoder stages instead of the deepest two')
    args = parser.parse_args()
    if args.amp and args.device != 'cuda':
        parser.error('--amp requires --device cuda')
    device = torch.device(args.device)
    rhdb_stages = [1, 2, 3, 4] if args.all_rhdb_stages else [1, 2]
    torch.manual_seed(42)
    torch.set_num_threads(2)
    check_math(device)

    # Zero gamma suppresses the entire HG increment, including restoration.
    branch = RegionHypergraphBranch(8, node_grid=16).to(device)
    assert branch.restore.bias is None
    observed = {}
    hooks = [branch.builder.register_forward_hook(lambda m, inputs, output: observed.update(h=output)),
             branch.hgconv.register_forward_hook(lambda m, inputs, output: observed.update(nodes=inputs[0], hg=output))]
    for height, width in ((1, 1), (2, 3), (8, 8), (16, 20), (19, 25)):
        inp = torch.randn(1, 8, height, width, device=device)
        # CPU BF16 checks autocast isolation too; it is not a CUDA FP16 test.
        for mixed in (False, True):
            with torch.no_grad(), torch.autocast(device_type=args.device, enabled=mixed):
                result = branch(inp)
            assert result.abs().max().item() == 0.0
            assert all(value.dtype == torch.float32 for value in observed.values())
            assert observed['nodes'].shape[1] == min(16, height) * min(16, width)
    for hook in hooks:
        hook.remove()
    for alpha in (0., 1.):
        layer = RegionHypergraphBranch(8, hypergraph_hidden_dim=3, node_grid=(3, 5), k=99, alpha=alpha, gamma_init=0.1).to(device)
        assert layer(torch.randn(2, 8, 5, 7, device=device)).shape == (2, 8, 5, 7)
    print("PASS: zero-gamma HG max_abs=0, FP32 graph/HG under autocast, clamped grids and options")

    fusion = AdaptiveBranchFusion(8, 'gate').to(device)
    local, hg = torch.randn(2, 8, 5, 7, device=device), torch.randn(2, 8, 5, 7, device=device)
    assert torch.count_nonzero(fusion.gate_conv.weight) == 0
    assert fusion.gate_conv.bias.item() == -2.0
    torch.testing.assert_close(fusion(local, hg), local + torch.sigmoid(local.new_tensor(-2.)) * hg)
    torch.testing.assert_close(fusion(local, torch.zeros_like(hg)), local, rtol=0, atol=0)
    assert fusion.gate_conv(torch.cat((local, hg), 1)).shape == (2, 1, 5, 7)
    torch.testing.assert_close(AdaptiveBranchFusion(8, 'add')(local, hg), local + hg)
    concat = AdaptiveBranchFusion(8, 'concat').to(device)
    torch.testing.assert_close(concat(local, hg), concat.projection(torch.cat((local, hg), 1)))
    print('PASS: delta fusion formulas; spatial gate starts at sigmoid(-2) and never attenuates local')

    for mode in ('gate', 'add', 'concat'):
        for enabled in (True, False):
            block = RHDBBlock(8, 16, 8, use_hypergraph=enabled, fusion_mode=mode, node_grid=4).to(device)
            deep = torch.randn(2, 16, 3, 4, device=device, requires_grad=True)
            skip = torch.randn(2, 8, 7, 9, device=device, requires_grad=True)
            optimizer = torch.optim.SGD(block.parameters(), lr=0.01)
            scaler = torch.amp.GradScaler('cuda', enabled=args.amp, init_scale=1024.)
            # A spatially varying target probes the branch rather than only
            # minimizing the energy of a normalized activation.
            target = torch.randn(2, 8, 7, 9, device=device)
            seen = {}
            hooks = [block.local_branch.register_forward_pre_hook(lambda m, inputs: seen.update(local=inputs[0]))]
            if enabled:
                hooks.append(block.hypergraph_branch.register_forward_pre_hook(lambda m, inputs: seen.update(hg=inputs[0])))
            with torch.autocast(device_type=args.device, enabled=args.amp):
                out = block(deep, skip)
                assert out.shape == (2, 8, 7, 9) and torch.isfinite(out).all()
                loss = F.mse_loss(out.float(), target)
            if enabled:
                assert seen['local'] is seen['hg'], 'Branches must receive the SAME fused input'
            else:
                assert not any('hypergraph' in n or 'branch_fusion' in n for n, _ in block.named_parameters())
            backward_for_check(loss, optimizer, scaler)
            finite_gradients(block, (deep, skip))
            if enabled:
                assert block.hypergraph_branch.hgconv.projection.weight.grad.abs().sum() == 0
                assert block.hypergraph_branch.gamma.grad.abs() > 0
                if mode == 'gate':
                    assert block.branch_fusion.gate_conv.weight.grad.abs().sum() == 0
                scaler.step(optimizer)
                scaler.update()
                assert block.hypergraph_branch.gamma.detach().abs() > 0
                optimizer.zero_grad(set_to_none=True)
                deep.grad = skip.grad = None
                with torch.autocast(device_type=args.device, enabled=args.amp):
                    loss = F.mse_loss(block(deep, skip).float(), target)
                backward_for_check(loss, optimizer, scaler)
                finite_gradients(block, (deep, skip))
                hg_grad = block.hypergraph_branch.hgconv.projection.weight.grad.abs().sum().item()
                gamma = block.hypergraph_branch.gamma.detach().item()
                print(f'Gradient probe: fusion={mode}, gamma={gamma:.8e}, '
                      f'HGConv_grad_L1={hg_grad:.8e}, scale={scaler.get_scale():.0f}')
                assert hg_grad > 0, (
                    f'HGConv gradient is zero after gamma update: gamma={gamma:.8e}, '
                    f'amp={args.amp}, scale={scaler.get_scale()}; compare the --device cuda FP32 run')
            for hook in hooks:
                hook.remove()
            print(f"PASS: parallel RHDB shape/backward, fusion={mode}, hypergraph={enabled}")

    block = RHDBBlock(8, 16, 8, gamma_init=0.2).to(device).eval()
    deep, skip = torch.randn(1, 16, 3, 4, device=device), torch.randn(1, 8, 7, 9, device=device)
    with torch.no_grad():
        feature = block.fuse(torch.cat((F.interpolate(deep, size=(7, 9), mode='bilinear', align_corners=False), skip), 1))
        delta_local = block.local_branch(feature) - feature
        expected = block.post(feature + block.branch_fusion(delta_local, block.hypergraph_branch(feature)))
        torch.testing.assert_close(block(deep, skip), expected)
        max_error = 0.
        for mode in ('gate', 'add', 'concat'):
            disabled = RHDBBlock(8, 16, 8, use_hypergraph=False, fusion_mode=mode).to(device).eval()
            feature = disabled.fuse(torch.cat((F.interpolate(deep, size=(7, 9), mode='bilinear', align_corners=False), skip), 1))
            expected = disabled.post(disabled.local_branch(feature))
            actual = disabled(deep, skip)
            max_error = max(max_error, (actual - expected).abs().max().item())
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
            # With an identity local path and identity Post, the answer is F, NOT 2F.
            disabled.local_branch = nn.Identity()
            disabled.post = nn.Identity()
            torch.testing.assert_close(disabled(deep, skip), feature, rtol=0, atol=0)
        print(f'PASS: no-HG equals Post(local_full), max_abs_error={max_error:.3g}; identity added once')
    for invalid in (lambda: RHDBBlock(8, 16, 8, fusion_mode='bad'), lambda: KNNHypergraph(k=-1),
                    lambda: KNNHypergraph(alpha=1.1), lambda: RegionHypergraphBranch(8, node_grid=0),
                    lambda: block(deep, skip[:, :7]),
                    lambda: UNetDecoder(decoder_type='rhdb', rhdb_options={'hypergraph_stages': [1, 1]})):
        try:
            invalid()
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid arguments accepted')
    print("PASS: outer residual formula, input/config validation")

    for classes in (2, 5, 9):
        dec = UNetDecoder((8, 16, 32, 64), num_classes=classes, dropout=0., decoder_type='rhdb',
                         rhdb_options={'hypergraph_stages': rhdb_stages}).to(device)
        for index in range(1, 5):
            stage = getattr(dec, f'decoder{index}')
            if index in rhdb_stages:
                assert isinstance(stage, RHDBBlock) and len(stage.local_branch.blocks) == 1
            else:
                assert isinstance(stage, UpBlock) and len(stage.refine.blocks) == 3
        assert len(dec.hypergraph_gammas()) == len(rhdb_stages)
        assert len({id(getattr(dec, f'decoder{i}').hypergraph_branch.gamma) for i in rhdb_stages}) == len(rhdb_stages)
        features = [torch.randn(1, c, h, w, device=device, requires_grad=True) for c, h, w in
                    ((8, 32, 40), (16, 16, 20), (32, 8, 10), (64, 4, 5))]
        optimizer = torch.optim.SGD(dec.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler('cuda', enabled=args.amp, init_scale=1024.)
        with torch.autocast(device_type=args.device, enabled=args.amp):
            out, info = dec(features, (128, 160), return_features=True)
            assert out.shape == (1, classes, 128, 160)
            for result, source in zip(info['stages'], reversed(features)):
                assert result.shape == source.shape
            loss = out.float().square().mean()
        backward_for_check(loss, optimizer, scaler)
        finite_gradients(dec, features)
        print(f"PASS: four-stage decoder + bottleneck + head, RHDB stages={rhdb_stages}, classes={classes}")

    options = dict(encoder_channels=(8, 16, 32, 64), num_classes=5, dropout=0.)
    legacy = UNetDecoder(**options).to(device).eval()
    explicit = UNetDecoder(**options, decoder_type='unet').to(device).eval()
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    empty = UNetDecoder(**options, decoder_type='rhdb', rhdb_options={'hypergraph_stages': []}).to(device).eval()
    empty.load_state_dict(legacy.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(legacy(features, (128, 160)), empty(features, (128, 160)), rtol=0, atol=0)
        torch.testing.assert_close(legacy(features, (128, 160)), explicit(features, (128, 160)), rtol=0, atol=0)
    print("PASS: old decoder strict state-dict compatibility and unchanged outputs")

    if args.full_model:
        from model import DualModalMambaUNet
        cfg = dict(in_channels_b=3, num_classes=9, dims=(8, 16, 32, 64), depths=(2, 2, 4, 2),
                   d_state=2, history_offsets=(1, 4), mlp_ratio=2., drop_path_rate=0., decoder_dropout=0.,
                   stage_modes=('cross',) * 4, cross_mode='soft', cross_frequency='once_per_stage',
                   decoder_type='rhdb', rhdb_options={'hypergraph_stages': rhdb_stages},
                   scan_backend='torch' if args.device == 'cpu' else None)
        model = DualModalMambaUNet(**cfg).to(device).eval()
        a = torch.randn(1, 3, 64, 128, device=device, requires_grad=True)
        b = torch.randn_like(a, requires_grad=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler('cuda', enabled=args.amp, init_scale=1024.)
        with torch.autocast(device_type=args.device, enabled=args.amp):
            logits = model(a, b)
            loss = F.cross_entropy(logits.float(), torch.randint(9, (1, 64, 128), device=device))
        backward_for_check(loss, optimizer, scaler)
        finite_gradients(model, (a, b))
        buffer = io.BytesIO()
        torch.save({'config': cfg, 'model': model.state_dict()}, buffer)
        buffer.seek(0)
        state = torch.load(buffer, map_location=device, weights_only=True)
        restored = DualModalMambaUNet(**state['config']).to(device).eval()
        restored.load_state_dict(state['model'], strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(a, b), model(a, b), rtol=0, atol=0)
        print('PASS: full soft-cross-2242 encoder + RHDB forward/backward and config/weight round trip')
    print(f'All RHDB checks passed: device={device}, amp={args.amp}')


if __name__ == '__main__':
    main()
