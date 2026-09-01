"""Evaluate ACMMamba checkpoints and export semantic-segmentation predictions."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import PairedRemoteSensingDataset
from losses import build_segmentation_loss
from metrics import SegmentationConfusionMatrix
from model import DualModalMambaUNet


LOGGER = logging.getLogger("ACMMamba.infer")

# The first five entries preserve the palette used by the original datasets.
# Additional entries make the default visualization work for larger label sets.
CLASS_COLORS = (
    (255, 215, 0),
    (0, 160, 0),
    (128, 128, 128),
    (220, 30, 30),
    (30, 100, 230),
    (255, 140, 0),
    (0, 190, 190),
    (190, 0, 190),
    (120, 70, 20),
    (0, 120, 255),
    (140, 200, 60),
    (255, 105, 180),
    (70, 70, 210),
    (210, 180, 140),
    (0, 100, 70),
    (180, 80, 255),
)
IGNORE_COLOR = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ACMMamba inference on a configured dataset split."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_xian_cloud.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to the best checkpoint derived from the config.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/<checkpoint-prefix>/<split>.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--max-comparisons",
        type=int,
        default=-1,
        help="Maximum four-panel visualizations; -1 saves all and 0 disables them.",
    )
    parser.add_argument(
        "--visual-scale",
        type=int,
        default=2,
        help="Integer enlargement factor for comparison panels.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")
    return config


def resolve_checkpoint(config: Mapping[str, Any], requested: Path | None) -> Path:
    if requested is not None:
        return requested
    checkpoint_dir = Path(config["checkpoint_dir"])
    prefix = str(config.get("checkpoint_prefix", "")).strip()
    filename = f"{prefix}_best_miou.pt" if prefix else "best_miou.pt"
    return checkpoint_dir / filename


def resolve_output_dir(
    config: Mapping[str, Any], split: str, requested: Path | None
) -> Path:
    if requested is not None:
        return requested
    experiment = str(config.get("checkpoint_prefix", "acmmamba")).strip() or "acmmamba"
    return Path("results") / experiment / split


def configure_logging(output_dir: Path) -> None:
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "inference.log", mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def create_dataset(config: Mapping[str, Any], split: str) -> PairedRemoteSensingDataset:
    data_config = config["data"]
    return PairedRemoteSensingDataset(
        rgb_dir=data_config.get(f"{split}_rgb_dir", data_config["rgb_dir"]),
        sar_dir=data_config.get(f"{split}_sar_dir", data_config["sar_dir"]),
        label_dir=data_config.get(f"{split}_label_dir", data_config["label_dir"]),
        split_file=Path(data_config["split_dir"]) / f"{split}.txt",
        augment=False,
        rgb_mean=data_config.get("rgb_mean", (0.2278617560, 0.2390318349, 0.2416281091)),
        rgb_std=data_config.get("rgb_std", (0.1460782053, 0.1307787441, 0.1309645726)),
        sar_mean=data_config.get("sar_mean", (0.2589521446,)),
        sar_std=data_config.get("sar_std", (0.2057200670,)),
        modality_b_channels=int(
            data_config.get("modality_b_channels", config["model"]["in_channels_b"])
        ),
        rgb_filename_template=data_config.get("rgb_filename_template"),
        modality_b_filename_template=data_config.get(
            "modality_b_filename_template"
        ),
        label_filename_template=data_config.get("label_filename_template"),
        label_threshold=data_config.get("label_threshold"),
    )


def load_model(
    config: Mapping[str, Any], checkpoint_path: Path, device: torch.device
) -> tuple[DualModalMambaUNet, Mapping[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a mapping.")
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint does not contain a valid model state.")
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}

    model = DualModalMambaUNet(**config["model"])
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, checkpoint


def denormalize_rgb(
    tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]
) -> np.ndarray:
    array = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    array = array * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def denormalize_modality_b(
    tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]
) -> np.ndarray:
    array = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
    if array.shape[2] != mean_array.shape[2] or array.shape[2] != std_array.shape[2]:
        raise ValueError(
            "Modality-B channel count does not match its normalization statistics: "
            f"tensor={array.shape[2]}, mean={mean_array.shape[2]}, std={std_array.shape[2]}."
        )
    array = array * std_array + mean_array
    array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return array[..., 0] if array.shape[2] == 1 else array


# Backward-compatible name used by older project code.
denormalize_sar = denormalize_modality_b


def _parse_color(value: Sequence[int], name: str) -> tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three RGB values.")
    color = tuple(int(channel) for channel in value)
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError(f"{name} RGB values must be between 0 and 255.")
    return color


def resolve_class_colors(
    config: Mapping[str, Any], num_classes: int
) -> tuple[tuple[int, int, int], ...]:
    visualization = config.get("visualization", {})
    configured = visualization.get("class_colors") if isinstance(visualization, Mapping) else None
    if configured is None:
        if num_classes > len(CLASS_COLORS):
            raise ValueError(
                f"The default palette supports at most {len(CLASS_COLORS)} classes; "
                "define visualization.class_colors in the config."
            )
        return tuple(CLASS_COLORS[:num_classes])
    if len(configured) != num_classes:
        raise ValueError(
            f"Expected {num_classes} visualization colors, got {len(configured)}."
        )
    return tuple(
        _parse_color(value, f"visualization.class_colors[{index}]")
        for index, value in enumerate(configured)
    )


def resolve_ignore_color(config: Mapping[str, Any]) -> tuple[int, int, int]:
    visualization = config.get("visualization", {})
    if not isinstance(visualization, Mapping):
        return IGNORE_COLOR
    return _parse_color(
        visualization.get("ignore_color", IGNORE_COLOR),
        "visualization.ignore_color",
    )


def colorize_mask(
    mask: np.ndarray,
    class_colors: Sequence[Sequence[int]] = CLASS_COLORS,
    ignore_color: Sequence[int] = IGNORE_COLOR,
) -> np.ndarray:
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored[...] = _parse_color(ignore_color, "ignore_color")
    for class_index, color in enumerate(class_colors):
        colored[mask == class_index] = color
    return colored


def save_comparison(
    path: Path,
    rgb: np.ndarray,
    modality_b: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    scale: int,
    class_colors: Sequence[Sequence[int]] = CLASS_COLORS,
    ignore_color: Sequence[int] = IGNORE_COLOR,
    modality_a_title: str = "RGB",
    modality_b_title: str = "SAR",
) -> None:
    if scale < 1:
        raise ValueError("visual_scale must be at least 1.")
    modality_b_image = (
        Image.fromarray(modality_b, mode="L").convert("RGB")
        if modality_b.ndim == 2
        else Image.fromarray(modality_b, mode="RGB")
    )
    panels = (
        (modality_a_title, Image.fromarray(rgb, mode="RGB")),
        (modality_b_title, modality_b_image),
        ("GT", Image.fromarray(colorize_mask(target, class_colors, ignore_color), mode="RGB")),
        (
            "Prediction",
            Image.fromarray(
                colorize_mask(prediction, class_colors, ignore_color), mode="RGB"
            ),
        ),
    )
    width, height = panels[0][1].size
    target_size = (width * scale, height * scale)
    header_height = 24
    canvas = Image.new("RGB", (target_size[0] * len(panels), target_size[1] + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        resampling = (
            Image.Resampling.NEAREST
            if title in {"GT", "Prediction"}
            else Image.Resampling.BILINEAR
        )
        panel = panel.resize(target_size, resampling)
        x = index * target_size[0]
        canvas.paste(panel, (x, header_height))
        draw.text((x + 6, 5), title, fill="black")
    canvas.save(path)


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def build_result(
    computed: Mapping[str, torch.Tensor],
    class_names: Sequence[str],
    losses: Mapping[str, float],
    sample_count: int,
    checkpoint_path: Path,
    checkpoint_epoch: int | None,
    split: str,
    inference_seconds: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "split": split,
        "samples": sample_count,
        "loss": losses["loss"],
        "cross_entropy": losses["cross_entropy"],
        "dice": losses["dice"],
        "mIoU": float(computed["mIoU"].item()),
        "mF1": float(computed["mF1"].item()),
        "mAcc": float(computed["mAcc"].item()),
        "OA": float(computed["OA"].item()),
        "FWIoU": float(computed["FWIoU"].item()),
        "inference_seconds": inference_seconds,
        "milliseconds_per_image": 1000.0 * inference_seconds / max(sample_count, 1),
        "per_class": {},
        "confusion_matrix": computed["confusion_matrix"].long().cpu().tolist(),
    }
    for index, class_name in enumerate(class_names):
        result["per_class"][class_name] = {
            "IoU": finite_or_none(float(computed["class_iou"][index].item())),
            "F1": finite_or_none(float(computed["class_f1"][index].item())),
            "Acc": finite_or_none(float(computed["class_accuracy"][index].item())),
            "OA": finite_or_none(float(computed["class_oa"][index].item())),
        }
    return result


def write_confusion_matrix(
    path: Path, matrix: Sequence[Sequence[int]], class_names: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])


def write_summary(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        f"config: {result.get('config', 'unknown')}",
        f"checkpoint: {result['checkpoint']}",
        f"checkpoint epoch: {result['checkpoint_epoch']}",
        f"split: {result['split']}",
        f"samples: {result['samples']}",
        f"loss: {result['loss']:.6f}",
        f"cross entropy: {result['cross_entropy']:.6f}",
        f"dice: {result['dice']:.6f}",
        f"mIoU: {result['mIoU']:.6f}",
        f"mF1: {result['mF1']:.6f}",
        f"mAcc: {result['mAcc']:.6f}",
        f"OA: {result['OA']:.6f}",
        f"FWIoU: {result['FWIoU']:.6f}",
        f"inference time: {result['inference_seconds']:.3f}s",
        f"milliseconds/image: {result['milliseconds_per_image']:.3f}",
        "",
        "per-class metrics:",
    ]
    for class_name, metrics in result["per_class"].items():
        formatted = " ".join(
            f"{name}={value:.6f}" if value is not None else f"{name}=nan"
            for name, value in metrics.items()
        )
        lines.append(f"{class_name}: {formatted}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    config = load_config(args.config)
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    output_dir = resolve_output_dir(config, args.split, args.output_dir)
    if output_dir.is_dir() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose another directory or pass --overwrite."
        )

    raw_dir = output_dir / "raw_masks"
    color_dir = output_dir / "color_masks"
    comparison_dir = output_dir / "comparisons"
    raw_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)
    if args.max_comparisons != 0:
        comparison_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    class_names = tuple(str(name) for name in config["class_names"])
    num_classes = int(config["model"]["num_classes"])
    if len(class_names) != num_classes:
        raise ValueError(
            f"Expected {num_classes} class names, got {len(class_names)}."
        )
    class_colors = resolve_class_colors(config, num_classes)
    ignore_color = resolve_ignore_color(config)
    visualization = config.get("visualization", {})
    modality_a_title = str(
        visualization.get("modality_a_name", "RGB")
        if isinstance(visualization, Mapping)
        else "RGB"
    )
    modality_b_title = str(
        visualization.get("modality_b_name", "SAR")
        if isinstance(visualization, Mapping)
        else "SAR"
    )

    dataset = create_dataset(config, args.split)
    num_workers = (
        int(config["data"].get("num_workers", 4))
        if args.num_workers is None
        else args.num_workers
    )
    loader_args: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_args["prefetch_factor"] = int(config["data"].get("prefetch_factor", 2))
    loader = DataLoader(dataset, **loader_args)

    model, checkpoint = load_model(config, checkpoint_path, device)
    checkpoint_epoch_value = checkpoint.get("epoch")
    checkpoint_epoch = (
        int(checkpoint_epoch_value) + 1
        if checkpoint_epoch_value is not None
        else None
    )
    criterion = build_segmentation_loss(**config["loss"]).to(device).eval()
    metrics = SegmentationConfusionMatrix(
        num_classes=num_classes,
        ignore_index=int(config["loss"].get("ignore_index", 255)),
        device=device,
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    data_config = config["data"]
    rgb_mean = data_config["rgb_mean"]
    rgb_std = data_config["rgb_std"]
    sar_mean = data_config["sar_mean"]
    sar_std = data_config["sar_std"]

    LOGGER.info(
        "device=%s checkpoint=%s epoch=%s split=%s samples=%d batch_size=%d amp=%s",
        device,
        checkpoint_path,
        checkpoint_epoch if checkpoint_epoch is not None else "unknown",
        args.split,
        len(dataset),
        args.batch_size,
        amp_enabled,
    )

    loss_sums = {"loss": 0.0, "cross_entropy": 0.0, "dice": 0.0}
    sample_count = 0
    comparison_count = 0
    inference_seconds = 0.0

    with torch.inference_mode():
        progress = tqdm(loader, desc=f"Inference {args.split}", dynamic_ncols=True)
        for batch in progress:
            rgb_cpu = batch["rgb"]
            sar_cpu = batch["sar"]
            target_cpu = batch["label"]
            sample_ids = batch["id"]
            rgb = rgb_cpu.to(device, non_blocking=True)
            sar = sar_cpu.to(device, non_blocking=True)
            target = target_cpu.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                logits = model(rgb, sar)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - start_time

            # Keep loss computation under autocast as in training. In
            # particular, weighted cross entropy must not mix half-precision
            # logits with float32 class weights outside an autocast context.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                components = criterion(logits, target, return_components=True)
            prediction = logits.argmax(dim=1)
            metrics.update(prediction, target)
            current_batch_size = rgb.shape[0]
            sample_count += current_batch_size
            for name in loss_sums:
                loss_sums[name] += float(components[name].item()) * current_batch_size

            prediction_cpu = prediction.cpu().numpy().astype(np.uint8)
            target_numpy = target_cpu.numpy().astype(np.uint8)
            for index, sample_id in enumerate(sample_ids):
                sample_id = str(sample_id)
                Image.fromarray(prediction_cpu[index], mode="L").save(raw_dir / f"{sample_id}.png")
                Image.fromarray(
                    colorize_mask(prediction_cpu[index], class_colors, ignore_color),
                    mode="RGB",
                ).save(
                    color_dir / f"{sample_id}.png"
                )
                save_this_comparison = (
                    args.max_comparisons < 0
                    or comparison_count < args.max_comparisons
                )
                if args.max_comparisons != 0 and save_this_comparison:
                    save_comparison(
                        comparison_dir / f"{sample_id}.png",
                        denormalize_rgb(rgb_cpu[index], rgb_mean, rgb_std),
                        denormalize_modality_b(sar_cpu[index], sar_mean, sar_std),
                        target_numpy[index],
                        prediction_cpu[index],
                        args.visual_scale,
                        class_colors=class_colors,
                        ignore_color=ignore_color,
                        modality_a_title=modality_a_title,
                        modality_b_title=modality_b_title,
                    )
                    comparison_count += 1
            progress.set_postfix(loss=f"{components['loss'].item():.4f}")

    computed = metrics.compute()
    averaged_losses = {
        name: value / max(sample_count, 1) for name, value in loss_sums.items()
    }
    result = build_result(
        computed=computed,
        class_names=class_names,
        losses=averaged_losses,
        sample_count=sample_count,
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=checkpoint_epoch,
        split=args.split,
        inference_seconds=inference_seconds,
    )
    result.update(
        {
            "config": str(args.config.resolve()),
            "class_names": list(class_names),
            "class_colors": [list(color) for color in class_colors],
            "ignore_color": list(ignore_color),
            "modality_a_name": modality_a_title,
            "modality_b_name": modality_b_title,
        }
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    write_confusion_matrix(
        output_dir / "confusion_matrix.csv",
        result["confusion_matrix"],
        class_names,
    )
    write_summary(output_dir / "summary.txt", result)

    LOGGER.info(
        "%s total: loss=%.4f mIoU=%.4f mF1=%.4f OA=%.4f mAcc=%.4f FWIoU=%.4f",
        args.split,
        result["loss"],
        result["mIoU"],
        result["mF1"],
        result["OA"],
        result["mAcc"],
        result["FWIoU"],
    )
    for class_name, class_metrics in result["per_class"].items():
        LOGGER.info(
            "%s class=%-16s IoU=%s F1=%s OA=%s Acc=%s",
            args.split,
            class_name,
            f"{class_metrics['IoU']:.4f}" if class_metrics["IoU"] is not None else "nan",
            f"{class_metrics['F1']:.4f}" if class_metrics["F1"] is not None else "nan",
            f"{class_metrics['OA']:.4f}" if class_metrics["OA"] is not None else "nan",
            f"{class_metrics['Acc']:.4f}" if class_metrics["Acc"] is not None else "nan",
        )
    LOGGER.info("raw masks: %s", raw_dir.resolve())
    LOGGER.info("color masks: %s", color_dir.resolve())
    LOGGER.info("comparisons: %s (%d saved)", comparison_dir.resolve(), comparison_count)
    LOGGER.info("metrics: %s", (output_dir / "metrics.json").resolve())


if __name__ == "__main__":
    main()
