"""Reusable training, distributed and checkpoint helpers."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def initialize_distributed() -> Tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def build_adamw_parameter_groups(
    model: nn.Module, weight_decay: float
) -> list[Dict[str, object]]:
    decay_parameters = []
    no_decay_parameters = []
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        no_decay = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or getattr(parameter, "_no_weight_decay", False)
            or "A_logs" in name
            or "Ds" in name
        )
        (no_decay_parameters if no_decay else decay_parameters).append(parameter)
    return [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]


def build_warmup_poly_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_steps: int,
    power: float = 0.9,
    warmup_start_factor: float = 0.01,
) -> LambdaLR:
    if total_steps < 1:
        raise ValueError("total_steps must be positive.")

    def learning_rate_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            progress = step / max(1, warmup_steps)
            return warmup_start_factor + progress * (1.0 - warmup_start_factor)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress) ** power

    return LambdaLR(optimizer, learning_rate_factor)


def reduce_loss_sums(
    loss_sums: Mapping[str, float], sample_count: int, device: torch.device
) -> Dict[str, float]:
    keys = sorted(loss_sums)
    values = [float(loss_sums[key]) for key in keys] + [float(sample_count)]
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_samples = max(tensor[-1].item(), 1.0)
    return {key: tensor[index].item() / total_samples for index, key in enumerate(keys)}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_miou: float,
    config: Mapping[str, object],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_miou": best_miou,
            "config": dict(config),
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def save_model_weights(
    path: str | Path,
    model: nn.Module,
    epoch: int,
    metrics: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "metrics": dict(metrics),
            "config": dict(config),
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> Tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", -1)) + 1, float(checkpoint.get("best_miou", 0.0))


def append_jsonl(path: str | Path, record: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def optimizer_steps_per_epoch(loader_length: int, accumulation_steps: int) -> int:
    return math.ceil(loader_length / accumulation_steps)
