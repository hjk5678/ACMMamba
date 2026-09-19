"""MSCAN formula, gradient, compatibility and RHDB+MSCAN integration checks."""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common_layers import DropPath
from decoder.mscan import (make_group_norm, MSCANSpatialAttention, MSCANAttention,
                           MSCANMLP, MSCANBlock, MSCANStage)
from decoder.rhdb import RHDBBlock
from decoder.unet_decoder import UNetDecoder, ResidualStage, UpBlock


def check_gradients(module, loss, amp):
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=1024.)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, f'Unused parameter: {name}'
        assert torch.isfinite(parameter.grad).all(), f'Nonfinite gradient: {name}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--full-model', action='store_true')
    args = parser.parse_args()
    if args.amp and args.device != 'cuda':
        parser.error('--amp requires --device cuda')
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device(args.device)

    # Requested demonstration uses the exact channels and dimensions, not a stub.
    stage = MSCANStage(384, 256, depth=2).to(device).eval()
    with torch.no_grad(), torch.autocast(args.device, enabled=args.amp):
        y = stage(torch.randn(2, 384, 64, 64, device=device))
    assert y.shape == (2, 256, 64, 64) and torch.isfinite(y).all()
    print(f'PASS: [2,384,64,64] -> {list(y.shape)}, depth=2, parameters={sum(p.numel() for p in stage.parameters()):,}')
    del stage, y

    spatial = MSCANSpatialAttention(7).to(device)
    x = torch.randn(2, 7, 9, 13, device=device)
    base = spatial.conv0(x)
    expected = x * spatial.proj(base + sum(branch(base) for branch in spatial.branches))
    torch.testing.assert_close(spatial(x), expected)
    for branch, kernel in zip(spatial.branches, (7, 11, 21)):
        assert branch[0].kernel_size == (1, kernel) and branch[1].kernel_size == (kernel, 1)
        assert branch[0].groups == branch[1].groups == 7
    with torch.no_grad():
        spatial.proj.weight.zero_()
        spatial.proj.bias.fill_(2.)
    torch.testing.assert_close(spatial(x), x * 2)  # Would fail if sigmoid is added.
    attention = MSCANAttention(7).to(device)
    expected = attention.proj2(attention.spatial_attention(F.gelu(attention.proj1(x)))) + x
    torch.testing.assert_close(attention(x), expected)
    block = MSCANBlock(7).to(device).eval()
    x1 = x + block.gamma1.view(1, -1, 1, 1) * block.attention(block.norm1(x))
    expected = x1 + block.gamma2.view(1, -1, 1, 1) * block.mlp(block.norm2(x1))
    torch.testing.assert_close(block(x), expected)
    with torch.no_grad():
        block.gamma1.zero_(); block.gamma2.zero_()
    torch.testing.assert_close(block(x), x, rtol=0, atol=0)
    assert not any(isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.Linear, nn.Sigmoid)) for m in block.modules())
    mlp = MSCANMLP(7, mlp_ratio=3.)
    assert mlp.fc1.out_channels == mlp.dwconv.conv.groups == 21
    print('PASS: multi-scale DW kernels, no sigmoid/BN/Linear, internal shortcut and both LayerScale residuals')

    for cin, cout, height, width in ((7, 7, 1, 1), (1, 1, 1, 1), (3, 7, 1, 9),
                                    (7, 11, 13, 1), (12, 10, 9, 13)):
        stage = MSCANStage(cin, cout).to(device)
        x = torch.randn(1, cin, height, width, device=device, requires_grad=True)
        assert isinstance(stage.proj, nn.Identity) == (cin == cout)
        with torch.autocast(args.device, enabled=args.amp):
            y = stage(x)
            assert y.shape == (1, cout, height, width)
            loss = y.float().square().mean()
        check_gradients(stage, loss, args.amp)
        assert x.grad is not None and torch.isfinite(x.grad).all()
    assert make_group_norm(37).num_groups == 1
    drop = DropPath(0.5).to(device).train()
    ones = torch.ones(64, 3, 2, 2, device=device)
    result = drop(ones)
    assert (result == result[:, :1, :1, :1]).all()
    assert set(result.unique().tolist()) == {0., 2.}
    torch.testing.assert_close(drop.eval()(ones), ones, rtol=0, atol=0)
    assert isinstance(MSCANBlock(7, drop_path=0.).drop_path, nn.Identity)
    stochastic = MSCANStage(7, 7, drop=0.1, drop_path=0.2).to(device).train()
    check_gradients(stochastic, stochastic(torch.randn(8, 7, 5, 9, device=device)).square().mean(), False)
    print('PASS: arbitrary/prime channels, singleton and non-square shapes, finite gradients and per-sample DropPath')

    for bad in (lambda: MSCANStage(0, 8), lambda: MSCANStage(8, 8, depth=0),
                lambda: MSCANStage(8, 8, mlp_ratio=0), lambda: MSCANStage(8, 8, drop_path=1),
                lambda: MSCANStage(8, 8, drop=-0.1), lambda: make_group_norm(8, 0),
                lambda: UNetDecoder(mscan_options={}),
                lambda: UNetDecoder(decoder_type='rhdb', mscan_options={'unknown': 1}),
                lambda: UNetDecoder(decoder_type='rhdb', rhdb_options={'hypergraph_stages': [1,2,3,4]}, mscan_options={})):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid configuration accepted')
    print('PASS: invalid dimensions/options and conflicting RHDB/MSCAN placement rejected')

    for classes in (2, 5, 9):
        options = dict(encoder_channels=(8,16,32,64), num_classes=classes,
                       decoder_type='rhdb', dropout=0.)
        baseline = UNetDecoder(**options).to(device).eval()
        unchanged = UNetDecoder(**options, mscan_options=None).to(device).eval()
        unchanged.load_state_dict(baseline.state_dict(), strict=True)
        decoder = UNetDecoder(**options, mscan_options={'depth': 2}).to(device).eval()
        assert isinstance(decoder.decoder1, RHDBBlock) and isinstance(decoder.decoder2, RHDBBlock)
        for index in (3, 4):
            up = getattr(decoder, f'decoder{index}')
            assert isinstance(up, UpBlock) and isinstance(up.refine, MSCANStage)
            assert len(up.refine.blocks) == 2
        # Prove old full decoder strict loading and outputs remain unchanged.
        unchanged.load_state_dict(baseline.state_dict(), strict=True)
        features = [torch.randn(1,c,h,w,device=device,requires_grad=True) for c,h,w in
                    ((8,32,40),(16,16,20),(32,8,10),(64,4,5))]
        with torch.no_grad():
            torch.testing.assert_close(baseline(features,(128,160)), unchanged(features,(128,160)), rtol=0,atol=0)
        # Outside shallow refine, every state key/shape remains compatible.
        before = {k: v.shape for k,v in baseline.state_dict().items() if not k.startswith(('decoder3.refine.', 'decoder4.refine.'))}
        after = {k: v.shape for k,v in decoder.state_dict().items() if not k.startswith(('decoder3.refine.', 'decoder4.refine.'))}
        assert before == after
        with torch.autocast(args.device, enabled=args.amp):
            out, info = decoder(features, (128,160), return_features=True)
            assert out.shape == (1,classes,128,160)
            for tensor, source in zip(info['stages'], reversed(features)):
                assert tensor.shape == source.shape
            loss = out.float().square().mean()
        check_gradients(decoder, loss, args.amp)
        # Explicitly check original shallow upsample/reduce/concat ordering.
        with torch.no_grad():
            up = decoder.decoder3
            deep, skip = info['stages'][1].float(), features[1].float()
            expected = up.refine(torch.cat((up.reduce(F.interpolate(deep, size=skip.shape[-2:], mode='bilinear', align_corners=False)),skip),1))
            torch.testing.assert_close(up(deep,skip), expected)
        print(f'PASS: RHDB1/2 + MSCAN3/4, classes={classes}, shape/backward and old baseline compatibility')

    if args.full_model:
        from model import DualModalMambaUNet
        from encoder.layers import DropPath as EncoderDropPath
        assert EncoderDropPath is DropPath  # one shared implementation
        cfg = dict(in_channels_b=1, num_classes=2, dims=(8,16,32,64), depths=(2,2,4,2),
                   d_state=2, history_offsets=(1,4), mlp_ratio=2., drop_path_rate=0., decoder_dropout=0.,
                   stage_modes=('cross',)*4, cross_mode='soft', cross_frequency='once_per_stage',
                   decoder_type='rhdb', rhdb_options={'hypergraph_stages':[1,2]},
                   mscan_options={'depth':2}, scan_backend='torch' if args.device=='cpu' else None)
        model = DualModalMambaUNet(**cfg).to(device).eval()
        a, b = torch.randn(1,3,64,96,device=device), torch.randn(1,1,64,96,device=device)
        with torch.autocast(args.device, enabled=args.amp):
            out = model(a,b)
            assert out.shape == (1,2,64,96)
            loss = F.cross_entropy(out.float(), torch.randint(2,(1,64,96),device=device))
        check_gradients(model, loss, args.amp)
        buffer = io.BytesIO()
        torch.save({'config':cfg,'model':model.state_dict()},buffer)
        buffer.seek(0)
        saved = torch.load(buffer,weights_only=True,map_location=device)
        restored = DualModalMambaUNet(**saved['config']).to(device).eval()
        restored.load_state_dict(saved['model'],strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(a,b), model(a,b), rtol=0,atol=0)
        print('PASS: full soft-once/2242 + RHDB + MSCAN forward/backward and config/checkpoint round trip')
    print(f'All MSCAN checks passed: device={device}, amp={args.amp}')


if __name__ == '__main__':
    main()
