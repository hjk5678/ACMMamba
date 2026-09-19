"""Two-process CPU/Gloo RHDB regression test, including zero-gamma gradients."""
from datetime import timedelta
from pathlib import Path
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decoder import RHDBBlock
from check_rhdb import finite_gradients


def worker(rank, port):
    torch.set_num_threads(1)
    torch.manual_seed(42 + rank)
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        for mode in ('gate', 'add', 'concat'):
            for enabled in (True, False):
                model = DistributedDataParallel(
                    RHDBBlock(8, 16, 8, use_hypergraph=enabled, fusion_mode=mode, node_grid=4),
                    find_unused_parameters=False)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                for step in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    deep = torch.randn(2, 16, 3, 4, requires_grad=True)
                    skip = torch.randn(2, 8, 7, 9, requires_grad=True)
                    model(deep, skip).square().mean().backward()
                    finite_gradients(model, (deep, skip))
                    if enabled and step == 0:
                        assert model.module.hypergraph_branch.hgconv.projection.weight.grad.abs().sum() == 0
                        assert model.module.hypergraph_branch.gamma.grad.abs() > 0
                    optimizer.step()
                # Check all parameters remain synchronized, not just absence of errors.
                values = torch.cat([p.detach().flatten() for p in model.parameters()])
                copies = [torch.empty_like(values) for _ in range(2)]
                dist.all_gather(copies, values)
                torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
                if rank == 0:
                    print(f'PASS: 2-rank Gloo, 3 updates, no unused parameters, synchronized: {mode}, HG={enabled}', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    if not dist.is_gloo_available():
        raise RuntimeError('This check requires a PyTorch build with Gloo support.')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=2, join=True)
