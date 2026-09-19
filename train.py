"""Single-GPU and DistributedDataParallel training entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data import build_datasets
from losses import build_segmentation_loss
from metrics import SegmentationConfusionMatrix
from model import DualModalMambaUNet
from utils import (
    append_jsonl,
    build_adamw_parameter_groups,
    build_warmup_poly_scheduler,
    cleanup_distributed,
    initialize_distributed,
    is_main_process,
    load_checkpoint,
    optimizer_steps_per_epoch,
    reduce_loss_sums,
    save_checkpoint,
    save_model_weights,
    seed_everything,
    seed_worker,
)
from utils.numerics import SafeOptimizerStep, require_finite, finite_named, raise_if_bad


LOGGER = logging.getLogger("ACMMamba.train")


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation data across ranks without padding or duplication."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int) -> None:
        if rank < 0 or rank >= world_size:
            raise ValueError(f"Invalid rank {rank} for world size {world_size}.")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.world_size - 1) // self.world_size


class TqdmLoggingHandler(logging.StreamHandler):
    """Write console logs without corrupting an active tqdm progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def configure_logging(log_dir: Path, main_process: bool) -> None:
    handlers: list[logging.Handler] = []
    if main_process:
        console_handler = TqdmLoggingHandler(sys.stdout)
        file_handler = logging.FileHandler(
            log_dir / "train.log", mode="a", encoding="utf-8"
        )
        handlers.extend((console_handler, file_handler))
    else:
        handlers.append(logging.NullHandler())

    logging.basicConfig(
        level=logging.INFO if main_process else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACMMamba on a paired dataset.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train_shandong.yaml")
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one train batch and one validation batch without saving checkpoints.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")
    return config


def create_data_loaders(
    config: Mapping[str, object],
    distributed: bool,
    rank: int,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    data_config = config["data"]
    datasets = build_datasets(
        rgb_dir=data_config["rgb_dir"],
        sar_dir=data_config["sar_dir"],
        label_dir=data_config["label_dir"],
        split_dir=data_config["split_dir"],
        rgb_mean=data_config.get("rgb_mean", (0.2278617560, 0.2390318349, 0.2416281091)),
        rgb_std=data_config.get("rgb_std", (0.1460782053, 0.1307787441, 0.1309645726)),
        sar_mean=data_config.get("sar_mean", (0.2589521446,)),
        sar_std=data_config.get("sar_std", (0.2057200670,)),
        modality_b_channels=int(
            data_config.get("modality_b_channels", config["model"]["in_channels_b"])
        ),
        train_horizontal_flip_probability=float(
            data_config.get("train_horizontal_flip_probability", 0.5)
        ),
        train_vertical_flip_probability=float(
            data_config.get("train_vertical_flip_probability", 0.5)
        ),
        train_random_90_rotation=bool(
            data_config.get("train_random_90_rotation", True)
        ),
        rgb_filename_template=data_config.get("rgb_filename_template"),
        modality_b_filename_template=data_config.get(
            "modality_b_filename_template"
        ),
        label_filename_template=data_config.get("label_filename_template"),
        label_threshold=data_config.get("label_threshold"),
        test_rgb_dir=data_config.get("test_rgb_dir"),
        test_sar_dir=data_config.get("test_sar_dir"),
        test_label_dir=data_config.get("test_label_dir"),
    )

    train_sampler = (
        DistributedSampler(
            datasets["train"], shuffle=True, seed=int(config["seed"]), drop_last=False
        )
        if distributed
        else None
    )
    val_sampler = (
        DistributedEvalSampler(datasets["val"], rank=rank, world_size=dist.get_world_size())
        if distributed
        else None
    )
    num_workers = int(data_config["num_workers"])
    common_loader_args = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        common_loader_args["prefetch_factor"] = int(data_config.get("prefetch_factor", 2))

    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]) + rank)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(data_config["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        generator=generator,
        **common_loader_args,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=int(data_config.get("val_batch_size", data_config["batch_size"])),
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **common_loader_args,
    )
    return train_loader, val_loader, train_sampler


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    accumulation_steps: int,
    amp_enabled: bool,
    max_grad_norm: float,
    log_interval: int,
    writer: SummaryWriter | None,
    max_batches: int | None = None,
    step_guard: SafeOptimizerStep | None = None,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step_guard = step_guard or SafeOptimizerStep()
    loss_sums = {"loss": 0.0, "cross_entropy": 0.0, "dice": 0.0}
    sample_count = 0
    effective_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    remainder = effective_batches % accumulation_steps
    epoch_start = time.perf_counter()
    progress = tqdm(
        total=effective_batches,
        desc=f"Train {epoch + 1:03d}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False,
        disable=not is_main_process(),
    )

    try:
        for step, batch in enumerate(loader):
            if step >= effective_batches:
                break
            is_last_batch = step + 1 == effective_batches
            should_update = (step + 1) % accumulation_steps == 0 or is_last_batch
            in_partial_group = remainder > 0 and step >= effective_batches - remainder
            loss_divisor = remainder if in_partial_group else accumulation_steps

            modality_a = batch["rgb"].to(device, non_blocking=True)
            modality_b = batch["sar"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            context = f"train epoch={epoch+1} step={step+1} ids={batch.get('id', '?')}"
            step_guard.check_scale(scaler, device, context)
            require_finite([("input_A", modality_a), ("input_B", modality_b)], context, device, synchronize=True)

            synchronization_context = contextlib.nullcontext()
            if isinstance(model, DistributedDataParallel) and not should_update:
                synchronization_context = model.no_sync()

            with synchronization_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = model(modality_a, modality_b)
                    require_finite([("logits", logits)], context, device, synchronize=True)
                    components = criterion(logits, target, return_components=True)
                    backward_loss = components["loss"] / loss_divisor
                require_finite(components.items(), context, device, synchronize=True)
                scaler.scale(backward_loss).backward()

            if should_update:
                optimizer_updated, scale_before_update, scale_after_update = step_guard.step(
                    model, optimizer, scheduler, scaler, max_grad_norm, device, context)
                if not optimizer_updated and is_main_process():
                    LOGGER.warning(
                        "AMP skipped optimizer update at epoch=%d step=%d "
                        "because non-finite gradients were detected; "
                        "the LR scheduler was not advanced (scale %.9g -> %.9g)",
                        epoch + 1,
                        step + 1,
                        scale_before_update,
                        scale_after_update,
                    )

            batch_size = target.shape[0]
            sample_count += batch_size
            for key in loss_sums:
                loss_sums[key] += components[key].detach().item() * batch_size

            current_lr = optimizer.param_groups[0]["lr"]
            if is_main_process():
                progress.set_postfix(
                    loss=f"{components['loss'].item():.4f}",
                    ce=f"{components['cross_entropy'].item():.4f}",
                    dice=f"{components['dice'].item():.4f}",
                    lr=f"{current_lr:.2e}",
                )
                progress.update(1)

            if is_main_process() and (
                (step + 1) % log_interval == 0 or step == 0 or is_last_batch
            ):
                elapsed = time.perf_counter() - epoch_start
                memory_gib = (
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                    if device.type == "cuda"
                    else 0.0
                )
                LOGGER.info(
                    "epoch=%d step=%d/%d loss=%.4f ce=%.4f dice=%.4f "
                    "lr=%.3e memory=%.2fGiB time=%.1fs",
                    epoch + 1,
                    step + 1,
                    effective_batches,
                    components["loss"].item(),
                    components["cross_entropy"].item(),
                    components["dice"].item(),
                    current_lr,
                    memory_gib,
                    elapsed,
                )
                if writer is not None:
                    global_step = epoch * len(loader) + step
                    writer.add_scalar(
                        "train/step_loss", components["loss"].item(), global_step
                    )
                    writer.add_scalar("train/learning_rate", current_lr, global_step)
    finally:
        progress.close()

    averages = reduce_loss_sums(loss_sums, sample_count, device)
    averages["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    averages["epoch_seconds"] = time.perf_counter() - epoch_start
    return averages


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    amp_enabled: bool,
    class_names: Sequence[str],
    epoch: int,
    max_batches: int | None = None,
) -> Dict[str, object]:
    model.eval()
    validation_error = ""
    confusion = SegmentationConfusionMatrix(num_classes, ignore_index, device)
    loss_sums = {"loss": 0.0, "cross_entropy": 0.0, "dice": 0.0}
    sample_count = 0
    effective_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    progress = tqdm(
        total=effective_batches,
        desc=f"Val   {epoch + 1:03d}",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=False,
        disable=not is_main_process(),
    )

    try:
        for step, batch in enumerate(loader):
            if step >= effective_batches:
                break
            modality_a = batch["rgb"].to(device, non_blocking=True)
            modality_b = batch["sar"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            context = f"validation epoch={epoch+1} step={step+1} ids={batch.get('id', '?')}"
            bad = finite_named([("input_A", modality_a), ("input_B", modality_b)], device)
            if bad:
                validation_error = validation_error or f"{context}: non-finite {bad}"
                continue
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(modality_a, modality_b)
                bad = finite_named([("logits", logits)], device)
                if bad:
                    validation_error = validation_error or f"{context}: non-finite {bad}"
                    continue
                components = criterion(logits, target, return_components=True)

            bad = finite_named(components.items(), device)
            if bad:
                validation_error = validation_error or f"{context}: non-finite {bad}"
                continue

            batch_size = target.shape[0]
            sample_count += batch_size
            for key in loss_sums:
                loss_sums[key] += components[key].detach().item() * batch_size
            confusion.update(logits, target)
            if is_main_process():
                progress.set_postfix(loss=f"{components['loss'].item():.4f}")
                progress.update(1)
    finally:
        progress.close()

    # Eval shards have unequal lengths: NEVER all-reduce per validation batch.
    # Reject the entire validation before publishing metrics or saving weights.
    raise_if_bad(validation_error, device)
    confusion.synchronize()
    averages = reduce_loss_sums(loss_sums, sample_count, device)
    metric_tensors = confusion.compute()
    result: Dict[str, object] = dict(averages)
    for key in ("mIoU", "mF1", "mAcc", "OA", "FWIoU"):
        result[key] = metric_tensors[key].item()
    result["class_iou"] = metric_tensors["class_iou"].detach().cpu().tolist()
    result["class_f1"] = metric_tensors["class_f1"].detach().cpu().tolist()
    result["class_accuracy"] = (
        metric_tensors["class_accuracy"].detach().cpu().tolist()
    )
    result["class_oa"] = metric_tensors["class_oa"].detach().cpu().tolist()
    result["per_class"] = {
        name: {
            "IoU": result["class_iou"][index],
            "F1": result["class_f1"][index],
            "OA": result["class_oa"][index],
            "Acc": result["class_accuracy"][index],
        }
        for index, name in enumerate(class_names)
    }
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    class_names = tuple(str(name) for name in config.get("class_names", ()))
    num_classes = int(config["model"]["num_classes"])
    if len(class_names) != num_classes:
        raise ValueError(
            f"Expected {num_classes} class names, got {len(class_names)}: "
            f"{class_names}."
        )
    distributed, rank, local_rank, world_size, device = initialize_distributed()
    main_process = rank == 0
    legacy_output_dir = Path(config.get("output_dir", "outputs/acmmamba_shandong"))
    log_dir = Path(config.get("log_dir", legacy_output_dir))
    checkpoint_dir = Path(config.get("checkpoint_dir", legacy_output_dir))
    checkpoint_prefix = str(config.get("checkpoint_prefix", "")).strip()
    if checkpoint_prefix and Path(checkpoint_prefix).name != checkpoint_prefix:
        raise ValueError("checkpoint_prefix must be a filename prefix, not a path.")
    checkpoint_stem = f"{checkpoint_prefix}_" if checkpoint_prefix else ""
    best_checkpoint_path = checkpoint_dir / f"{checkpoint_stem}best_miou.pt"
    last_checkpoint_path = checkpoint_dir / f"{checkpoint_stem}last.pt"
    if main_process:
        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier(device_ids=[local_rank])
    configure_logging(log_dir, main_process)
    try:
        if device.type != "cuda":
            raise RuntimeError("Full ACMMamba training requires a CUDA device.")
        seed_everything(int(config["seed"]), rank)
        torch.backends.cudnn.benchmark = bool(config["train"].get("cudnn_benchmark", True))
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(device)

        if main_process:
            with (log_dir / "resolved_config.yaml").open(
                "w", encoding="utf-8"
            ) as stream:
                yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)

        train_loader, val_loader, train_sampler = create_data_loaders(
            config, distributed, rank
        )
        model = DualModalMambaUNet(**config["model"]).to(device)
        criterion = build_segmentation_loss(**config["loss"]).to(device)

        optimizer_config = config["optimizer"]
        parameter_groups = build_adamw_parameter_groups(
            model, float(optimizer_config["weight_decay"])
        )
        optimizer = AdamW(
            parameter_groups,
            lr=float(optimizer_config["learning_rate"]),
            betas=tuple(optimizer_config.get("betas", (0.9, 0.999))),
            eps=float(optimizer_config.get("eps", 1e-8)),
        )

        train_config = config["train"]
        accumulation_steps = int(train_config["accumulation_steps"])
        if args.smoke_test:
            accumulation_steps = 1
        steps_per_epoch = optimizer_steps_per_epoch(len(train_loader), accumulation_steps)
        total_steps = int(train_config["epochs"]) * steps_per_epoch
        warmup_steps = int(train_config["warmup_epochs"]) * steps_per_epoch
        scheduler = build_warmup_poly_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            power=float(train_config.get("poly_power", 0.9)),
        )
        amp_enabled = bool(train_config.get("amp", True))
        init_scale = float(train_config.get("amp_init_scale", 65536.0))
        if not math.isfinite(init_scale) or init_scale <= 0:
            raise ValueError("amp_init_scale must be finite and positive")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled, init_scale=init_scale)
        step_guard = SafeOptimizerStep(
            min_scale=float(train_config.get("amp_min_scale", 2.0**-16)),
            max_consecutive_skips=int(train_config.get("amp_max_consecutive_skips", 8)))

        resume_path = args.resume or train_config.get("resume")
        start_epoch = 0
        # A valid mIoU is always >= 0, so the first validation must become the
        # initial best checkpoint even if the untrained model scores exactly 0.
        best_miou = -1.0
        if resume_path:
            start_epoch, best_miou = load_checkpoint(
                resume_path, model, optimizer, scheduler, scaler
            )
            LOGGER.info("resumed from %s at epoch %d", resume_path, start_epoch + 1)

        step_guard.check_scale(scaler, device, "initial/resumed scaler")
        require_finite(model.state_dict().items(), "initial/resumed model", device, synchronize=True)

        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

        writer = SummaryWriter(log_dir / "tensorboard") if main_process else None
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        LOGGER.info(
            "device=%s world_size=%d parameters=%s train=%d val=%d "
            "batch_per_gpu=%d accum=%d fusion=%s stage_modes=%s decoder=%s cross_mode=%s depths=%s cross_frequency=%s",
            device,
            world_size,
            f"{parameter_count:,}",
            len(train_loader.dataset),
            len(val_loader.dataset),
            int(config["data"]["batch_size"]),
            accumulation_steps,
            str(config["model"].get("fusion_mode", "mlfm")),
            "-".join(
                str(mode)
                for mode in config["model"].get(
                    "stage_modes", ("self", "cross", "self", "cross")
                )
            ),
            str(config["model"].get("decoder_type", "unet")),
            str(config["model"].get("cross_mode", "hard")),
            tuple(config["model"].get("depths", (1, 1, 1, 1))),
            str(config["model"].get("cross_frequency", "every_block")),
        )
        if main_process:
            raw_model = model.module if distributed else model
            LOGGER.info("numerical guards enabled: scaler=%.9g min_scale=%.9g max_consecutive_skips=%d; non-finite forward/validation aborts",
                        scaler.get_scale(), step_guard.min_scale, step_guard.max_consecutive_skips)
            LOGGER.info("encoder block routing: %s", [stage.block_modes for stage in raw_model.encoder.stages])
            if raw_model.encoder.fusion_mode == "mlfm_hg":
                LOGGER.info("MLFM pre-fusion HG: independent A/B, one HGConv each, stages=1,2,3,4 options=%s", config["model"].get("mlfm_hg_options", {}))
            if raw_model.decoder.decoder_type == "rhdb":
                LOGGER.info("RHDB stages (deep first)=%s options=%s", raw_model.decoder.hypergraph_stages, config["model"].get("rhdb_options", {}))
            if raw_model.decoder.mscan_stages:
                LOGGER.info("MSCAN stages (deep first)=%s options=%s", raw_model.decoder.mscan_stages, config["model"].get("mscan_options", {}))
            LOGGER.info("decoder upsampling=%s options=%s (segmentation-head resize unchanged)", raw_model.decoder.upsample_mode, config["model"].get("sapa_options", {}))

        epochs = int(train_config["epochs"])
        for epoch in range(start_epoch, epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_result = train_one_epoch(
                model=model,
                criterion=criterion,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                epoch=epoch,
                accumulation_steps=accumulation_steps,
                amp_enabled=amp_enabled,
                max_grad_norm=float(train_config.get("max_grad_norm", 1.0)),
                log_interval=int(train_config.get("log_interval", 20)),
                writer=writer,
                max_batches=1 if args.smoke_test else None,
                step_guard=step_guard,
            )

            should_validate = (
                (epoch + 1) % int(train_config.get("val_interval", 1)) == 0
                or epoch + 1 == epochs
                or args.smoke_test
            )
            val_result = None
            if should_validate:
                val_result = validate(
                    model=model,
                    criterion=criterion,
                    loader=val_loader,
                    device=device,
                    num_classes=num_classes,
                    ignore_index=int(config["loss"]["ignore_index"]),
                    amp_enabled=amp_enabled,
                    class_names=class_names,
                    epoch=epoch,
                    max_batches=1 if args.smoke_test else None,
                )

            if main_process:
                LOGGER.info(
                    "epoch=%d train_loss=%.4f val_loss=%s mIoU=%s mF1=%s OA=%s",
                    epoch + 1,
                    train_result["loss"],
                    f"{val_result['loss']:.4f}" if val_result else "-",
                    f"{val_result['mIoU']:.4f}" if val_result else "-",
                    f"{val_result['mF1']:.4f}" if val_result else "-",
                    f"{val_result['OA']:.4f}" if val_result else "-",
                )
                record = {"epoch": epoch + 1, "train": train_result, "val": val_result}
                unwrapped_model = model.module if distributed else model
                cross_weights = unwrapped_model.encoder.soft_cross_weights()
                if cross_weights:
                    record["soft_cross_weights"] = cross_weights
                    for block_name, values in cross_weights.items():
                        LOGGER.info(
                            "soft_cross epoch=%d %s A3=%.6f A4=%.6f B3=%.6f B4=%.6f",
                            epoch + 1, block_name, *values,
                        )
                        if writer is not None:
                            for direction, value in zip(("A3", "A4", "B3", "B4"), values):
                                writer.add_scalar(
                                    f"soft_cross/{block_name}/{direction}", value, epoch + 1
                                )
                fusion_gammas = unwrapped_model.encoder.hypergraph_fusion_weights()
                if fusion_gammas:
                    record["mlfm_hg_gammas"] = fusion_gammas
                    for stage_name, values in fusion_gammas.items():
                        LOGGER.info("mlfm_hg epoch=%d %s gamma_A=%.8f gamma_B=%.8f", epoch + 1, stage_name, *values)
                        if writer is not None:
                            for modality, value in zip(("A", "B"), values):
                                writer.add_scalar(f"mlfm_hg/{stage_name}/gamma_{modality}", value, epoch + 1)
                hg_gammas = unwrapped_model.decoder.hypergraph_gammas()
                if hg_gammas:
                    record["hypergraph_gammas"] = hg_gammas
                    for stage_name, gamma in hg_gammas.items():
                        LOGGER.info("rhdb epoch=%d %s gamma=%.6f", epoch + 1, stage_name, gamma)
                        if writer is not None:
                            writer.add_scalar(f"rhdb/{stage_name}/gamma", gamma, epoch + 1)
                append_jsonl(log_dir / "history.jsonl", record)
                if writer is not None:
                    for key, value in train_result.items():
                        writer.add_scalar(f"train_epoch/{key}", value, epoch + 1)
                    if val_result is not None:
                        for key in ("loss", "cross_entropy", "dice", "mIoU", "mF1", "mAcc", "OA", "FWIoU"):
                            writer.add_scalar(f"val/{key}", val_result[key], epoch + 1)
                        for class_name, class_metrics in val_result["per_class"].items():
                            for metric_name, metric_value in class_metrics.items():
                                writer.add_scalar(
                                    f"val_class/{class_name}_{metric_name}",
                                    metric_value,
                                    epoch + 1,
                                )

                if val_result is not None:
                    LOGGER.info(
                        "validation total: mIoU=%.4f mF1=%.4f OA=%.4f mAcc=%.4f",
                        val_result["mIoU"],
                        val_result["mF1"],
                        val_result["OA"],
                        val_result["mAcc"],
                    )
                    for class_name, class_metrics in val_result["per_class"].items():
                        LOGGER.info(
                            "validation class=%-9s IoU=%.4f F1=%.4f OA=%.4f Acc=%.4f",
                            class_name,
                            class_metrics["IoU"],
                            class_metrics["F1"],
                            class_metrics["OA"],
                            class_metrics["Acc"],
                        )

                if not args.smoke_test:
                    current_miou = float(val_result["mIoU"]) if val_result else best_miou
                    if val_result is not None and current_miou > best_miou:
                        best_miou = current_miou
                        save_model_weights(
                            best_checkpoint_path,
                            model,
                            epoch,
                            val_result,
                            config,
                        )
                        LOGGER.info(
                            "new best mIoU=%.4f; saved to %s",
                            best_miou,
                            best_checkpoint_path,
                        )
                    save_checkpoint(
                        last_checkpoint_path,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        best_miou,
                        config,
                    )
                else:
                    LOGGER.info("smoke test completed; checkpoints were not written")

            if writer is not None:
                writer.flush()
            if distributed:
                dist.barrier(device_ids=[local_rank])
            if args.smoke_test:
                break

        if writer is not None:
            writer.close()
    except FloatingPointError:
        LOGGER.exception("Numerical failure: aborting; existing best/last checkpoints were not overwritten by this epoch.")
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
