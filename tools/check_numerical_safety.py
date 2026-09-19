"""Failure injection: AMP, gradient clipping, checkpoint safety and DDP rendezvous."""
from __future__ import annotations
import argparse
from datetime import timedelta
from pathlib import Path
import socket
import sys
import tempfile

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.numerics import SafeOptimizerStep, require_finite
from utils.training import save_model_weights, load_checkpoint
from metrics import SegmentationConfusionMatrix
from encoder.mlfm import HypergraphMLFM


def expect_failure(fn, text):
    try:
        fn()
    except FloatingPointError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f'Expected numerical failure: {text}')


def setup(device):
    model = nn.Linear(3,2).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    scaler = torch.amp.GradScaler(device.type,init_scale=128.)
    return model,opt,scheduler,scaler


def scaled_backward(model,scaler,device):
    scaler.scale(model(torch.ones(2,3,device=device)).square().mean()).backward()


class ToyModel(nn.Module):
    def __init__(self, bad=False):
        super().__init__()
        self.conv = nn.Conv2d(2,2,1)
        self.bad = bad
    def forward(self,a,b):
        out = self.conv(torch.cat((a,b),1))
        return out * float('nan') if self.bad else out


class ToyLoss(nn.Module):
    def forward(self,x,y,return_components=False):
        loss = nn.functional.cross_entropy(x.float(),y)
        return dict(loss=loss,cross_entropy=loss,dice=loss*0)


def batch():
    return dict(rgb=torch.randn(1,1,3,4),sar=torch.randn(1,1,3,4),
                label=torch.zeros(1,3,4,dtype=torch.long),id=['injected-sample'])


def ddp_worker(rank,port):
    from train import train_one_epoch,validate
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2,timeout=timedelta(seconds=60))
    try:
        device=torch.device('cpu')
        model=DDP(ToyModel(bad=rank==1),broadcast_buffers=False)
        opt=torch.optim.SGD(model.parameters(),lr=.01)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        scaler=torch.amp.GradScaler('cpu',init_scale=128.)
        expect_failure(lambda:train_one_epoch(model,ToyLoss(),[batch()],opt,sched,scaler,
            device,0,2,False,1.,20,None),'logits')
        # Per-rank unequal val lengths, one rank has NaN: synchronize only at end.
        model=DDP(ToyModel(bad=rank==1),broadcast_buffers=False)
        expect_failure(lambda:validate(model,ToyLoss(),[batch()]*(2 if rank==0 else 1),device,
            2,255,False,['background','object'],0),'logits')
        # Only rank 1 has invalid gradients; neither rank may update.
        model=DDP(nn.Linear(3,2))
        opt=torch.optim.SGD(model.parameters(),lr=.01)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
        scaler=torch.amp.GradScaler('cpu',init_scale=128.)
        guard=SafeOptimizerStep()
        before=[p.detach().clone() for p in model.parameters()]
        scaled_backward(model,scaler,device)
        if rank==1:
            next(model.parameters()).grad.fill_(float('inf'))
        assert not guard.step(model,opt,sched,scaler,1.,device,'rank overflow')[0]
        for a,b in zip(before,model.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert scaler.get_scale()==64.
        scaled_backward(model,scaler,device)
        assert guard.step(model,opt,sched,scaler,1.,device,'recovered')[0]
        if rank==0:
            print('PASS: 2-rank forward abort, uneven validation abort, synchronized AMP skip and recovery',flush=True)
    finally:
        dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--ddp',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device=torch.device(args.device)
    # Old half-precision reduce overflows on large but finite weights/features.
    hg=HypergraphMLFM(8,hypergraph_options={'node_grid':2,'gamma_init':.2,'force_fp32':True}).to(device)
    with torch.no_grad():
        hg.hypergraph_a.reduce.weight.fill_(40000.)
    a=torch.ones(1,8,3,4,device=device,requires_grad=True)
    b=torch.randn_like(a,requires_grad=True)
    with torch.autocast(device.type,dtype=torch.float16):
        safe=hg(a,b)
    assert safe.dtype==torch.float32 and torch.isfinite(safe).all()
    safe.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in hg.parameters())
    with torch.no_grad():
        torch.testing.assert_close(safe,hg(a,b),rtol=0,atol=0)
        hg.force_fp32=False
        with torch.autocast(device.type,dtype=torch.float16):
            unsafe=hg(a,b)
        assert not torch.isfinite(unsafe).all()
    print('PASS: reproduced HG-MLFM FP16 overflow; full FP32 island stays finite, same FP32 formula and backward')

    m,o,s,g=setup(device)
    guard=SafeOptimizerStep()
    scaled_backward(m,g,device)
    initial=s.last_epoch
    assert guard.step(m,o,s,g,1.,device,'valid')[0] and s.last_epoch==initial+1
    before=[p.detach().clone() for p in m.parameters()]
    scaled_backward(m,g,device)
    next(m.parameters()).grad.fill_(float('nan'))
    assert not guard.step(m,o,s,g,1.,device,'backward overflow')[0]
    assert s.last_epoch==initial+1 and g.get_scale()==64.
    for a,b in zip(before,m.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    scaled_backward(m,g,device)
    assert guard.step(m,o,s,g,1.,device,'recovery')[0]
    assert s.last_epoch==initial+2
    g.update(new_scale=0.)
    expect_failure(lambda:guard.check_scale(g,device,'zero scale'),'scale')
    print('PASS: actual optimizer/scheduler progress, AMP skip/recovery and zero-scale rejection')

    m,o,s,g=setup(device)
    guard=SafeOptimizerStep(max_consecutive_skips=2)
    for index in range(2):
        scaled_backward(m,g,device)
        next(m.parameters()).grad.fill_(float('inf'))
        if index==0:
            assert not guard.step(m,o,s,g,1.,device,'persistent')[0]
        else:
            expect_failure(lambda:guard.step(m,o,s,g,1.,device,'persistent'),'persistent')
    m,o,s,g=setup(device)
    scaled_backward(m,g,device)
    next(m.parameters()).grad.fill_(1e30)
    expect_failure(lambda:SafeOptimizerStep().step(m,o,s,g,1.,device,'huge norm'),'gradient_norm')
    m,o,s,g=setup(device)
    scaled_backward(m,g,device)
    def corrupt(opt,*_):
        next(iter(opt.state.values()))['exp_avg_sq'].fill_(float('inf'))
    handle=o.register_step_post_hook(corrupt)
    expect_failure(lambda:SafeOptimizerStep().step(m,o,s,g,1.,device,'bad state'),'state')
    handle.remove()
    print('PASS: persistent overflow, finite-gradient norm overflow and corrupted Adam state rejected')

    cm=SegmentationConfusionMatrix(2,device=device)
    expect_failure(lambda:cm.update(torch.full((1,2,2,2),float('nan'),device=device),torch.zeros(1,2,2,dtype=torch.long,device=device)),'logits')
    with tempfile.TemporaryDirectory(prefix='acmmamba-numerics-') as directory:
        m,o,s,g=setup(device)
        path=Path(directory)/'best.pt'
        save_model_weights(path,m,0,{'mIoU':.5,'loss':1.},{})
        original=path.read_bytes()
        expect_failure(lambda:save_model_weights(path,m,1,{'mIoU':float('nan')},{}),'mIoU')
        assert path.read_bytes()==original
        with torch.no_grad(): next(m.parameters()).fill_(float('nan'))
        expect_failure(lambda:save_model_weights(path,m,1,{'mIoU':.6},{}),'non-finite')
        assert path.read_bytes()==original
        bad_path=Path(directory)/'bad.pt'
        torch.save({'model':m.state_dict()},bad_path)
        expect_failure(lambda:load_checkpoint(bad_path,m),'resume')
    print('PASS: invalid metrics/resume rejected; existing best weights never overwritten')

    from train import train_one_epoch
    model=ToyModel()
    opt=torch.optim.SGD(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    scaler=torch.amp.GradScaler('cpu',init_scale=128.)
    start=scheduler.last_epoch
    train_one_epoch(model,ToyLoss(),[batch() for _ in range(3)],opt,scheduler,scaler,
                    torch.device('cpu'),0,2,False,1.,20,None)
    assert scheduler.last_epoch==start+2  # includes partial accumulation group
    print('PASS: real train loop, accumulation=2 and partial final group')
    class BadLoss(ToyLoss):
        def forward(self,x,y,return_components=False):
            result=super().forward(x,y,return_components)
            result['loss']=result['loss']*float('nan')
            return result
    expect_failure(lambda:train_one_epoch(model,BadLoss(),[batch()],opt,scheduler,scaler,
                    torch.device('cpu'),0,1,False,1.,20,None),'loss')
    print('PASS: finite logits but non-finite loss aborts before backward/update')
    if args.ddp:
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0))
            port=listener.getsockname()[1]
        mp.spawn(ddp_worker,args=(port,),nprocs=2,join=True)
    print(f'All numerical safety checks passed: device={device}')


if __name__=='__main__': main()
