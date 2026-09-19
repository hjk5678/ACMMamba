"""Independent pre-fusion HG, legacy MLFM, full model and optional CPU DDP checks."""
from __future__ import annotations

import argparse
from datetime import timedelta
import io
from pathlib import Path
import socket
import sys

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encoder.mlfm import MLFM, HypergraphMLFM, build_fusion
from decoder.hypergraph import HypergraphConv


def finite_gradients(module):
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, f'Unused: {name}'
        assert torch.isfinite(parameter.grad).all(), f'Nonfinite: {name}'


def ddp_worker(rank, port):
    torch.set_num_threads(1)
    torch.manual_seed(42 + rank)
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        module = DistributedDataParallel(HypergraphMLFM(8, hypergraph_options={'node_grid':4}), find_unused_parameters=False)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            module(torch.randn(2,8,5,7),torch.randn(2,8,5,7)).square().mean().backward()
            finite_gradients(module)
            for branch in (module.module.hypergraph_a,module.module.hypergraph_b):
                assert branch.gamma.grad.abs() > 0
                if step == 0:
                    assert branch.hgconv.projection.weight.grad.abs().sum() == 0
                else:
                    assert branch.hgconv.projection.weight.grad.abs().sum() > 0
            optimizer.step()
        flat = torch.cat([p.detach().flatten() for p in module.parameters()])
        copies = [torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(copies,flat)
        torch.testing.assert_close(*copies,rtol=0,atol=0)
        if rank == 0:
            print('PASS: 2-rank Gloo, 3 optimizer updates, no unused parameters, synchronized weights',flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--amp',action='store_true')
    parser.add_argument('--full-model',action='store_true')
    parser.add_argument('--force-fp32',action='store_true',help='Use FP32 for the entire HG-MLFM module')
    parser.add_argument('--ddp',action='store_true',help='Also run two-process CPU/Gloo regression')
    args = parser.parse_args()
    if args.amp and args.device != 'cuda':
        parser.error('--amp requires CUDA')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device = torch.device(args.device)
    old = MLFM(8,12,8).to(device).eval()
    module = HypergraphMLFM(8,12,8,{'node_grid':4,'force_fp32':args.force_fp32}).to(device).eval()
    missing, unexpected = module.load_state_dict(old.state_dict(),strict=False)
    assert not unexpected and missing and all(n.startswith(('hypergraph_a.','hypergraph_b.')) for n in missing)
    copy = MLFM(8,12,8).to(device).eval()
    copy.load_state_dict(old.state_dict(),strict=True)
    a,b = torch.randn(2,8,5,7,device=device),torch.randn(2,12,3,4,device=device)
    with torch.no_grad():
        torch.testing.assert_close(old(a,b),copy(a,b),rtol=0,atol=0)
        torch.testing.assert_close(module(a,b),old(a,b),rtol=0,atol=0)
    assert not ({id(p) for p in module.hypergraph_a.parameters()} & {id(p) for p in module.hypergraph_b.parameters()})
    assert sum(isinstance(m,HypergraphConv) for m in module.modules()) == 2
    print('PASS: legacy MLFM unchanged, zero-gamma equivalence, independent A/B parameters, one HGConv per modality')

    for branch in (module.hypergraph_a,module.hypergraph_b):
        nn.init.constant_(branch.gamma,0.2)
    calls = []
    handles = [branch.hgconv.register_forward_hook(lambda m,i,o: calls.append(o)) for branch in (module.hypergraph_a,module.hypergraph_b)]
    actual = module(a,b)
    assert len(calls) == 2
    for handle in handles:
        handle.remove()
    expected = MLFM.forward(module,a+module.hypergraph_a(a),b+module.hypergraph_b(b))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    # Per-sample construction, including rectangular and smaller-than-grid inputs.
    with torch.no_grad():
        torch.testing.assert_close(actual[:1],module(a[:1],b[:1]),rtol=2e-5,atol=2e-6)
    for size in ((1,1),(2,3),(8,8)):
        assert module(torch.randn(1,8,*size,device=device),torch.randn(1,12,*size,device=device)).shape == (1,8,*size)
    print('PASS: exact HG -> residual -> align -> Add/Cat formula, sample isolation and small grids')

    for gamma in (0.,0.2):
        module.zero_grad(set_to_none=True)
        for branch in (module.hypergraph_a,module.hypergraph_b):
            nn.init.constant_(branch.gamma,gamma)
        aa,bb = a.clone().requires_grad_(),b.clone().requires_grad_()
        optimizer = torch.optim.SGD(module.parameters(),lr=0.01)
        scaler = torch.amp.GradScaler('cuda',enabled=args.amp,init_scale=1024)
        with torch.autocast(device.type,enabled=args.amp):
            loss = module(aa,bb).float().square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite_gradients(module)
        assert torch.isfinite(aa.grad).all() and torch.isfinite(bb.grad).all()
        for branch in (module.hypergraph_a,module.hypergraph_b):
            assert branch.gamma.grad.abs() > 0
            hg_grad = branch.hgconv.projection.weight.grad.abs().sum()
            assert (hg_grad == 0) if gamma == 0 else (hg_grad > 0)
    print(f'PASS: backward amp={args.amp}, zero gamma gates HG weights but not gamma gradients; nonzero gamma trains HG')
    for bad in (lambda: build_fusion('mean',8,mlfm_hg_options={}),
                lambda: HypergraphMLFM(8,hypergraph_options=[]),
                lambda: HypergraphMLFM(8,hypergraph_options={'unknown':1}),
                lambda: HypergraphMLFM(8,hypergraph_options={'node_grid':0}),
                lambda: module(a,b[:1])):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid configuration/shape accepted')

    if args.full_model:
        from model import DualModalMambaUNet
        cfg = dict(in_channels_b=3,num_classes=9,dims=(8,16,32,64),depths=(2,2,4,2),
                   d_state=2,history_offsets=(1,4),mlp_ratio=2.,drop_path_rate=0.,decoder_dropout=0.,
                   stage_modes=('cross',)*4,cross_mode='soft',cross_frequency='once_per_stage',
                   fusion_mode='mlfm_hg',mlfm_hg_options={'node_grid':4,'force_fp32':args.force_fp32},decoder_type='unet',
                   decoder_blocks_per_stage=3,scan_backend='torch' if device.type=='cpu' else None)
        model = DualModalMambaUNet(**cfg).to(device).eval()
        from decoder.unet_decoder import UpBlock, ResidualStage
        assert model.decoder.hypergraph_gammas() == {}
        assert not any(isinstance(m,HypergraphConv) for m in model.decoder.modules())
        for index in range(1,5):
            stage = getattr(model.decoder,f'decoder{index}')
            assert isinstance(stage,UpBlock) and isinstance(stage.refine,ResidualStage)
            assert len(stage.refine.blocks) == 3 and stage.sapa is None
        print('PASS: decoder ResNet blocks=[3,3,3,3], bilinear upsampling, no decoder HG/RHDB')
        assert len(model.encoder.hypergraph_fusion_weights()) == 4
        branches = [branch for fusion in model.encoder.fusions for branch in (fusion.hypergraph_a,fusion.hypergraph_b)]
        assert len({id(p) for branch in branches for p in branch.parameters()}) == sum(len(list(branch.parameters())) for branch in branches)
        rgb,tir = torch.randn(1,3,64,128,device=device),torch.randn(1,3,64,128,device=device)
        with torch.no_grad():
            before = model.encoder(rgb,tir,return_modal_features=True)
            for branch in branches:
                branch.gamma.fill_(0.2)
            after = model.encoder(rgb,tir,return_modal_features=True)
            for modality in ('a','b'):
                for first,second in zip(before[1][modality],after[1][modality]):
                    torch.testing.assert_close(first,second,rtol=0,atol=0)
            assert all(not torch.equal(first,second) for first,second in zip(before[0],after[0]))
        optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
        scaler = torch.amp.GradScaler('cuda',enabled=args.amp,init_scale=1024)
        with torch.autocast(device.type,enabled=args.amp):
            logits = model(rgb,tir)
            assert logits.shape == (1,9,64,128)
            loss = torch.nn.functional.cross_entropy(logits.float(),torch.randint(9,(1,64,128),device=device))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite_gradients(model)
        buffer = io.BytesIO()
        torch.save({'config':cfg,'model':model.state_dict()},buffer)
        buffer.seek(0)
        saved = torch.load(buffer,map_location=device,weights_only=True)
        restored = DualModalMambaUNet(**saved['config']).to(device).eval()
        restored.load_state_dict(saved['model'],strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(rgb,tir),model(rgb,tir),rtol=0,atol=0)
        print('PASS: 8 independent HG branches, encoder propagation unchanged, fused skips changed, full model backward/config round trip')
    if args.ddp:
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0))
            port = listener.getsockname()[1]
        mp.spawn(ddp_worker,args=(port,),nprocs=2,join=True)
    print(f'All HG-MLFM checks passed: device={device}, amp={args.amp}')


if __name__ == '__main__':
    main()
