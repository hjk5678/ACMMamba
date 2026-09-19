"""Pure-PyTorch SAPA numerical reference, AMP and decoder integration checks."""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decoder.sapa import SAPA, ChannelLayerNorm
from decoder.unet_decoder import UNetDecoder, UpBlock, ResidualStage
from decoder.rhdb import RHDBBlock


def repeat_reference(module, y, x):
    """Frozen pre-refactor formula; share weights so initialization cannot mask differences."""
    def neighborhoods(tensor):
        b, c, h, w = tensor.shape
        patches = F.unfold(tensor, module.kernel_size, padding=module.kernel_size // 2)
        patches = patches.reshape(b, c, module.kernel_size ** 2, h, w)
        return patches.repeat_interleave(module.up_factor, -2).repeat_interleave(module.up_factor, -1)

    q = module.q_proj(module.norm_encoder(y))
    k_patch = neighborhoods(module.k_proj(module.norm_decoder(x)))
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        similarity = (q.to(dtype).unsqueeze(2) * k_patch.to(dtype)).sum(dim=1)
        attention = F.softmax(similarity, dim=1)
        out = (neighborhoods(x).to(dtype) * attention.unsqueeze(1)).sum(dim=2)
    return out.to(x.dtype)


def check_refactor(device, amp):
    for dtype in (torch.float32, torch.float64):
        active_amp = amp and dtype == torch.float32
        max_output_error = max_grad_error = 0.0
        for kernel, factor, height, width in ((5,2,2,3), (3,3,1,2), (1,2,3,2), (3,1,2,3)):
            module = SAPA(3,4,embedding_dim=5,up_factor=factor,kernel_size=kernel).to(device=device,dtype=dtype)
            # Noncontiguous inputs also exercise the split-axis reshape layout.
            y = torch.randn(2,3,width*factor,height*factor,device=device,dtype=dtype).transpose(-1,-2).requires_grad_()
            x = torch.randn(2,4,width,height,device=device,dtype=dtype).transpose(-1,-2).requires_grad_()
            patches = module._neighborhoods(x)
            assert patches.shape == (2,4,kernel*kernel,height,width)
            with torch.autocast(device.type,enabled=active_amp):
                expected = repeat_reference(module,y,x)
                with patch.object(torch.Tensor,'repeat_interleave',side_effect=AssertionError('HR neighborhood repetition')):
                    actual = module(y,x)
            atol, rtol = ((1e-11,1e-9) if dtype == torch.float64 else
                          (2e-5,2e-3) if active_amp else (5e-6,1e-4))
            torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
            probe = torch.randn_like(actual)
            # Scale AMP probes to protect tiny Q/K gradients from FP16 underflow.
            scale = 128.0 if active_amp else 1.0
            targets = (y,x,*module.parameters())
            actual_grads = torch.autograd.grad((actual*probe).sum()*scale,targets)
            reference_grads = torch.autograd.grad((expected*probe).sum()*scale,targets)
            for a,b in zip(actual_grads,reference_grads):
                a,b = a/scale,b/scale
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
                max_grad_error = max(max_grad_error,(a-b).abs().max().item())
            max_output_error = max(max_output_error,(actual-expected).abs().max().item())
        print(f'PASS: broadcast vs old repeat, outputs + all input/parameter gradients; dtype={dtype}, amp={active_amp}, max_output_error={max_output_error:.3g}, max_grad_error={max_grad_error:.3g}')


def loop_reference(module, y, x):
    """Independent per-pixel implementation: no unfold or repeat_interleave."""
    q = module.q_proj(module.norm_encoder(y))
    k = module.k_proj(module.norm_decoder(x))
    radius, weights, rows = module.kernel_size // 2, [], []
    for row in range(y.shape[-2]):
        outputs = []
        for col in range(y.shape[-1]):
            keys, values = [], []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    i, j = row // module.up_factor + dy, col // module.up_factor + dx
                    valid = 0 <= i < x.shape[-2] and 0 <= j < x.shape[-1]
                    keys.append(k[:, :, i, j] if valid else torch.zeros_like(k[:, :, 0, 0]))
                    values.append(x[:, :, i, j] if valid else torch.zeros_like(x[:, :, 0, 0]))
            scores = (q[:, :, row, col].unsqueeze(-1) * torch.stack(keys, -1)).sum(1)
            weight = scores.softmax(-1)  # [B,K*K]
            weights.append(weight)
            outputs.append((torch.stack(values, -1) * weight.unsqueeze(1)).sum(-1))
        rows.append(torch.stack(outputs, -1))
    return torch.stack(rows, -2), torch.stack(weights, -1)


def backward_check(module, loss, amp):
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
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = torch.device(args.device)
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    with patch.object(nn.init,'trunc_normal_',wraps=nn.init.trunc_normal_) as initialize:
        initialized = SAPA(128,256)
        assert initialize.call_count == 2
        for call, projection in zip(initialize.call_args_list,(initialized.q_proj,initialized.k_proj)):
            assert call.args[0] is projection.weight and call.kwargs == {'std':0.02}
            assert torch.count_nonzero(projection.bias) == 0
            assert abs(projection.weight.std().item()-0.02) < 0.001
        for norm in (initialized.norm_encoder.norm,initialized.norm_decoder.norm):
            assert torch.all(norm.weight == 1) and torch.all(norm.bias == 0)
    # Existing checkpoints overwrite initialization, with unchanged state keys.
    restored = SAPA(128,256)
    restored.load_state_dict(initialized.state_dict(),strict=True)
    for name,value in initialized.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)
    del initialized, restored
    print('PASS: Q/K truncated-normal std=0.02, zero biases, LayerNorm identity initialization, strict weight reload')
    check_refactor(device,args.amp)

    # Exact requested example, with real channels. No loss/backward on this large
    # materialized reference example; gradients are exercised separately below.
    module = SAPA(128, 256).to(device).eval()
    skip = torch.randn(2, 128, 64, 64, device=device)
    deep = torch.randn(2, 256, 32, 32, device=device)
    with torch.no_grad(), torch.autocast(args.device, enabled=args.amp):
        out = module(encoder_feature=skip, decoder_feature=deep)
    assert out.shape == (2, 256, 64, 64) and torch.isfinite(out).all()
    assert torch.cat((out, skip), 1).shape == (2, 384, 64, 64)
    print(f'PASS: inputs decoder={tuple(deep.shape)}, skip={tuple(skip.shape)}, output={tuple(out.shape)}; parameters={sum(p.numel() for p in module.parameters()):,}')
    del module, skip, deep, out

    norm = ChannelLayerNorm(3).to(device)
    x = torch.randn(2, 3, 5, 7, device=device)
    expected = F.layer_norm(x.permute(0,2,3,1), (3,), norm.norm.weight, norm.norm.bias, norm.norm.eps).permute(0,3,1,2)
    torch.testing.assert_close(norm(x), expected)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in norm.modules())
    for kernel, factor, height, width in ((5,2,2,3), (3,3,1,2), (1,2,3,2)):
        module = SAPA(3, 4, embedding_dim=5, up_factor=factor, kernel_size=kernel).to(device)
        y = torch.randn(2,3,height*factor,width*factor,device=device,requires_grad=True)
        x = torch.randn(2,4,height,width,device=device,requires_grad=True)
        actual = module(y,x)
        expected, weights = loop_reference(module,y,x)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(weights.sum(1),torch.ones_like(weights[:,0]))
        probe = torch.randn_like(actual)
        actual_grads = torch.autograd.grad((actual*probe).sum(), (y,x), retain_graph=True)
        reference_grads = torch.autograd.grad((expected*probe).sum(), (y,x))
        for a,b in zip(actual_grads, reference_grads):
            torch.testing.assert_close(a,b,rtol=1e-4,atol=5e-6)
        with torch.autocast(args.device, enabled=args.amp):
            loss = module(y,x).float().square().mean()
        backward_check(module,loss,args.amp)
        assert torch.isfinite(y.grad).all() and torch.isfinite(x.grad).all()
    print(f'PASS: channel LayerNorm; independent local-dot-product reference, neighbor softmax, input gradients, backward (amp={args.amp})')

    # K=1 must return raw decoder values, NOT their normalization/projection.
    nearest = SAPA(3,4,kernel_size=1,qkv_bias=False).to(device)
    x = torch.randn(1,4,2,3,device=device) * 7 + 10
    y = torch.randn(1,3,4,6,device=device)
    torch.testing.assert_close(nearest(y,x), x.repeat_interleave(2,-2).repeat_interleave(2,-1),rtol=0,atol=0)
    module = SAPA(3,4).to(device)
    torch.testing.assert_close(module(y,torch.zeros_like(x)),torch.zeros(1,4,4,6,device=device),rtol=0,atol=0)
    # Uniform scores and all-one values make padded zero participation explicit.
    with torch.no_grad():
        module.q_proj.weight.zero_(); module.q_proj.bias.zero_()
    actual = module(torch.randn(1,3,6,6,device=device),torch.ones(1,4,3,3,device=device))
    torch.testing.assert_close(actual[0,:,0,0],torch.full((4,),9/25,device=device))
    print('PASS: raw decoder V, no skip values injected, K=1 identity-nearest and zero-padding edge behavior')

    for bad in (lambda: SAPA(3,4,kernel_size=4), lambda: SAPA(3,4,up_factor=0),
                lambda: SAPA(3,4,embedding_dim=0),
                lambda: module(y[:,:,:-1],x), lambda: module(y[:,:2],x),
                lambda: module(y.expand(2,-1,-1,-1),x),
                lambda: UNetDecoder(upsample_mode='invalid'),
                lambda: UNetDecoder(sapa_options={}),
                lambda: UNetDecoder(upsample_mode='sapa',sapa_options={'unknown':1})):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid SAPA config/shape accepted')
    tiny = SAPA(2,2,embedding_dim=2,kernel_size=3).double()
    inputs = (torch.randn(1,2,2,4,dtype=torch.double,requires_grad=True),
              torch.randn(1,2,1,2,dtype=torch.double,requires_grad=True))
    assert torch.autograd.gradcheck(tiny,inputs,fast_mode=True)
    print('PASS: explicit shape/config rejection and FP64 gradcheck')

    for classes in (2,5,9):
        cfg = dict(encoder_channels=(8,16,32,64),num_classes=classes,dropout=0.,decoder_type='rhdb')
        baseline = UNetDecoder(**cfg).to(device).eval()
        explicit = UNetDecoder(**cfg,upsample_mode='bilinear').to(device).eval()
        explicit.load_state_dict(baseline.state_dict(),strict=True)
        decoder = UNetDecoder(**cfg,upsample_mode='sapa',sapa_options={'embedding_dim':8}).to(device).eval()
        state = {k:v for k,v in decoder.state_dict().items() if '.sapa.' not in k}
        assert {k:v.shape for k,v in state.items()} == {k:v.shape for k,v in baseline.state_dict().items()}
        assert isinstance(decoder.decoder1,RHDBBlock) and isinstance(decoder.decoder2,RHDBBlock)
        assert isinstance(decoder.decoder3,UpBlock) and isinstance(decoder.decoder3.refine,ResidualStage)
        assert len(decoder.decoder3.refine.blocks)==len(decoder.decoder4.refine.blocks)==3
        assert len({id(getattr(decoder,f'decoder{i}').sapa) for i in range(1,5)})==4
        features = [torch.randn(1,c,h,w,device=device) for c,h,w in ((8,32,64),(16,16,32),(32,8,16),(64,4,8))]
        with torch.no_grad():
            torch.testing.assert_close(baseline(features,(128,256)),explicit(features,(128,256)),rtol=0,atol=0)
        calls = []
        hooks = [getattr(decoder,f'decoder{i}').sapa.register_forward_hook(lambda m,inp,out: calls.append((inp,out))) for i in range(1,5)]
        with torch.autocast(args.device,enabled=args.amp):
            out,info = decoder(features,(128,256),return_features=True)
            assert out.shape==(1,classes,128,256)
            for tensor,skip in zip(info['stages'],reversed(features)):
                assert tensor.shape==skip.shape
            loss = out.float().square().mean()
        assert len(calls)==4  # one and only one SAPA per stage
        for hook in hooks:
            hook.remove()
        backward_check(decoder,loss,args.amp)
        with torch.no_grad():
            # The head remains the original head followed by final bilinear resize.
            expected = F.interpolate(decoder.segmentation_head(info['stages'][-1].float()),size=(128,256),mode='bilinear',align_corners=False)
            if not args.amp:
                torch.testing.assert_close(out,expected)
        print(f'PASS: four independent SAPA stages + unchanged RHDB/ResNet/head; classes={classes}; legacy strict compatibility')

    if args.full_model:
        from model import DualModalMambaUNet
        cfg = dict(in_channels_b=3,num_classes=9,dims=(8,16,32,64),depths=(2,2,4,2),d_state=2,
                   history_offsets=(1,4),mlp_ratio=2.,drop_path_rate=0.,decoder_dropout=0.,
                   stage_modes=('cross',)*4,cross_mode='soft',cross_frequency='once_per_stage',
                   decoder_type='rhdb',rhdb_options={'hypergraph_stages':[1,2]},
                   upsample_mode='sapa',sapa_options={'embedding_dim':8},
                   scan_backend='torch' if args.device=='cpu' else None)
        model = DualModalMambaUNet(**cfg).to(device).eval()
        a,b = torch.randn(1,3,64,128,device=device),torch.randn(1,3,64,128,device=device)
        with torch.autocast(args.device,enabled=args.amp):
            logits = model(a,b)
            assert logits.shape==(1,9,64,128)
            loss = F.cross_entropy(logits.float(),torch.randint(9,(1,64,128),device=device))
        backward_check(model,loss,args.amp)
        buffer = io.BytesIO()
        torch.save({'config':cfg,'model':model.state_dict()},buffer); buffer.seek(0)
        saved = torch.load(buffer,map_location=device,weights_only=True)
        restored = DualModalMambaUNet(**saved['config']).to(device).eval()
        restored.load_state_dict(saved['model'],strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(a,b),model(a,b),rtol=0,atol=0)
        print('PASS: RGB/TIR full encoder + RHDB/ResNet + SAPA backward and config/checkpoint round trip')
    if args.device=='cuda':
        print(f'Peak test allocated GPU memory: {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GiB')
    print(f'All SAPA checks passed: device={device}, amp={args.amp}')


if __name__=='__main__':
    main()
