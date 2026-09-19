"""Fail-closed numerical guards; collectives only at equal-work rendezvous points."""
from __future__ import annotations

import math
import torch
import torch.distributed as dist


def finite_named(tensors, device):
    tensors = [(name, value) for name, value in tensors
               if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex())]
    checks = [torch.isfinite(value.detach()).all().to(device) for _, value in tensors]
    if not checks:
        return []
    good = torch.stack(checks).cpu().tolist()
    return [name for (name, _), ok in zip(tensors, good) if not ok]


def any_rank_bad(bad, device, synchronize=True):
    flag = torch.tensor(int(bool(bad)), device=device, dtype=torch.int32)
    if synchronize and dist.is_available() and dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def raise_if_bad(message, device, synchronize=True):
    """All ranks must call this, including ranks without a local error."""
    if not any_rank_bad(bool(message), device, synchronize):
        return
    messages = [message]
    if synchronize and dist.is_available() and dist.is_initialized():
        messages = [None] * dist.get_world_size()
        dist.all_gather_object(messages, message)
    detail = "; ".join(f"rank={i}: {text}" for i, text in enumerate(messages) if text)
    raise FloatingPointError(detail or "Non-finite value detected on another rank")


def require_finite(tensors, context, device, synchronize=False):
    bad = finite_named(tensors, device)
    raise_if_bad(f"{context}: non-finite {bad[:8]}" if bad else "", device, synchronize)


def optimizer_tensors(optimizer):
    for index, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                yield f"optimizer[{index}].{key}", value


class SafeOptimizerStep:
    """Allow bounded AMP backward overflows, never update on invalid gradients.

    Forward NaNs must be rejected separately, BEFORE backward. Scheduler progress
    follows the optimizer's actual post-step hook, not a scale comparison.
    """
    def __init__(self, min_scale=2.0**-16, max_consecutive_skips=8):
        if not math.isfinite(min_scale) or min_scale <= 0:
            raise ValueError("amp_min_scale must be finite and positive")
        if not isinstance(max_consecutive_skips, int) or max_consecutive_skips < 1:
            raise ValueError("amp_max_consecutive_skips must be a positive integer")
        self.min_scale = min_scale
        self.max_consecutive_skips = max_consecutive_skips
        self.consecutive_skips = 0

    def check_scale(self, scaler, device, context):
        scale = scaler.get_scale()
        bad = scaler.is_enabled() and (not math.isfinite(scale) or scale < self.min_scale)
        raise_if_bad(f"{context}: invalid/exhausted AMP scale={scale:.9g}" if bad else "", device)

    def step(self, model, optimizer, scheduler, scaler, max_grad_norm, device, context):
        self.check_scale(scaler, device, context)
        before = scaler.get_scale()
        scaler.unscale_(optimizer)
        named_grads = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
        bad_grads = finite_named(named_grads, device)
        if any_rank_bad(bad_grads, device):
            # Skip on EVERY rank, even if only one rank saw an overflow.
            optimizer.zero_grad(set_to_none=True)
            self.consecutive_skips += 1
            after = before * scaler.get_backoff_factor() if scaler.is_enabled() else before
            fatal = (not scaler.is_enabled() or after < self.min_scale or
                     self.consecutive_skips >= self.max_consecutive_skips)
            raise_if_bad(
                f"{context}: persistent/non-AMP non-finite gradients; scale={before:.9g}->{after:.9g}, "
                f"consecutive_skips={self.consecutive_skips}, local_bad={bad_grads[:8]}" if fatal else "", device)
            scaler.update(new_scale=after)
            return False, before, after
        if max_grad_norm > 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=False)
            require_finite([("gradient_norm", norm), *[(n,p.grad) for n,p in model.named_parameters() if p.grad is not None]],
                           context + " after gradient clipping", device, synchronize=True)
        steps = []
        hook = optimizer.register_step_post_hook(lambda *_: steps.append(True))
        try:
            scaler.step(optimizer)
        finally:
            hook.remove()
        scaler.update()
        raise_if_bad(f"{context}: optimizer unexpectedly skipped finite gradients" if not steps else "", device)
        require_finite(model.named_parameters(), context + " after optimizer step (parameters)", device, synchronize=True)
        require_finite(optimizer_tensors(optimizer), context + " after optimizer step (state)", device, synchronize=True)
        self.check_scale(scaler, device, context)
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        self.consecutive_skips = 0
        return True, before, scaler.get_scale()
