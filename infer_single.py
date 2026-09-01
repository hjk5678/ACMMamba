"""Run ACMMamba inference on one RGB/SAR pair and create a four-panel image."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from infer import (
    colorize_mask,
    load_config,
    load_model,
    resolve_checkpoint,
    resolve_class_colors,
    resolve_ignore_color,
)
from metrics import SegmentationConfusionMatrix


GT_CANDIDATES = ("GT.png", "GT.tif", "GT.tiff", "GT.jpg", "GT.jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer one paired RGB/SAR sample with an optional GT mask."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train_shandong.yaml")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to the best checkpoint derived from the config.",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--rgb-name", default="RGB.jpg")
    parser.add_argument(
        "--modality-b-name",
        "--sar-name",
        dest="modality_b_name",
        default="SAR.jpg",
        help="Second-modality filename; --sar-name remains as a compatible alias.",
    )
    parser.add_argument(
        "--gt-name",
        default=None,
        help="When omitted, GT.png/tif/tiff/jpg/jpeg are searched in that order.",
    )
    parser.add_argument(
        "--output-prefix",
        default="shandong",
        help="Prefix for the four-panel image, predictions and metrics files.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional four-panel filename override; other files still use output-prefix.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visual-scale", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_gt(input_dir: Path, requested_name: str | None) -> Path:
    if requested_name:
        path = input_dir / requested_name
        if not path.is_file():
            raise FileNotFoundError(f"GT does not exist: {path}")
        return path
    for name in GT_CANDIDATES:
        path = input_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No GT file was found in {input_dir}; tried {', '.join(GT_CANDIDATES)}."
    )


def read_inputs(
    rgb_path: Path,
    modality_b_path: Path,
    gt_path: Path,
    num_classes: int,
    modality_b_channels: int,
    label_threshold: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not rgb_path.is_file():
        raise FileNotFoundError(f"RGB image does not exist: {rgb_path}")
    if not modality_b_path.is_file():
        raise FileNotFoundError(
            f"Second-modality image does not exist: {modality_b_path}"
        )
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(modality_b_path) as image:
        modality_b_mode = "L" if modality_b_channels == 1 else "RGB"
        modality_b = np.asarray(image.convert(modality_b_mode), dtype=np.uint8)
        if modality_b.ndim == 2:
            modality_b = modality_b[..., None]
    with Image.open(gt_path) as image:
        target = np.asarray(image)

    if target.ndim == 3:
        if target.shape[2] in (3, 4) and np.all(target[..., :3] == target[..., :1]):
            target = target[..., 0]
        else:
            raise ValueError(
                f"GT must be a class-index mask, not an RGB color image: {gt_path}"
            )
    if target.ndim != 2:
        raise ValueError(f"Unsupported GT shape {target.shape}: {gt_path}")
    if label_threshold is not None:
        threshold = int(label_threshold)
        if not 0 <= threshold <= 255:
            raise ValueError("label_threshold must be between 0 and 255.")
        target = (target >= threshold).astype(np.uint8)
    if rgb.shape[:2] != modality_b.shape[:2] or modality_b.shape[:2] != target.shape:
        raise ValueError(
            f"Spatial sizes differ: RGB={rgb.shape}, modality-B={modality_b.shape}, "
            f"GT={target.shape}."
        )

    target = target.astype(np.int64, copy=False)
    valid_values = set(range(num_classes)) | {255}
    unique_values = {int(value) for value in np.unique(target)}
    invalid_values = sorted(unique_values - valid_values)
    if invalid_values:
        jpeg_note = (
            " JPEG is lossy and must not be used for class-index masks; use the original PNG."
            if gt_path.suffix.lower() in {".jpg", ".jpeg"}
            else ""
        )
        raise ValueError(
            f"GT contains invalid class values {invalid_values[:30]}; expected "
            f"0..{num_classes - 1} or 255.{jpeg_note}"
        )
    return rgb, modality_b, target.astype(np.uint8)


def normalize_inputs(
    rgb: np.ndarray, modality_b: np.ndarray, data_config: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    rgb_mean = np.asarray(data_config["rgb_mean"], dtype=np.float32).reshape(1, 1, 3)
    rgb_std = np.asarray(data_config["rgb_std"], dtype=np.float32).reshape(1, 1, 3)
    modality_b_channels = modality_b.shape[2]
    sar_mean = np.asarray(data_config["sar_mean"], dtype=np.float32).reshape(
        1, 1, modality_b_channels
    )
    sar_std = np.asarray(data_config["sar_std"], dtype=np.float32).reshape(
        1, 1, modality_b_channels
    )
    rgb_float = (rgb.astype(np.float32) / 255.0 - rgb_mean) / rgb_std
    sar_float = (modality_b.astype(np.float32) / 255.0 - sar_mean) / sar_std
    rgb_tensor = torch.from_numpy(rgb_float.transpose(2, 0, 1).copy()).unsqueeze(0)
    sar_tensor = torch.from_numpy(sar_float.transpose(2, 0, 1).copy()).unsqueeze(0)
    return rgb_tensor, sar_tensor


def save_four_panel(
    path: Path,
    rgb: np.ndarray,
    modality_b: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    class_names: Sequence[str],
    scale: int,
    class_colors: Sequence[Sequence[int]],
    ignore_color: Sequence[int],
    modality_a_title: str,
    modality_b_title: str,
) -> None:
    if scale < 1:
        raise ValueError("visual_scale must be at least 1.")
    modality_b_display = modality_b[..., 0] if modality_b.shape[2] == 1 else modality_b
    modality_b_image = (
        Image.fromarray(modality_b_display, mode="L").convert("RGB")
        if modality_b_display.ndim == 2
        else Image.fromarray(modality_b_display, mode="RGB")
    )
    panels = (
        (modality_a_title, Image.fromarray(rgb, mode="RGB")),
        (modality_b_title, modality_b_image),
        (
            "GT (colorized)",
            Image.fromarray(
                colorize_mask(target, class_colors, ignore_color), mode="RGB"
            ),
        ),
        (
            "Prediction",
            Image.fromarray(
                colorize_mask(prediction, class_colors, ignore_color), mode="RGB"
            ),
        ),
    )
    source_width, source_height = panels[0][1].size
    panel_size = (source_width * scale, source_height * scale)
    header_height = 28
    legend_height = 42
    canvas = Image.new(
        "RGB",
        (panel_size[0] * len(panels), panel_size[1] + header_height + legend_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        resampling = (
            Image.Resampling.BILINEAR
            if title not in {"GT (colorized)", "Prediction"}
            else Image.Resampling.NEAREST
        )
        panel = panel.resize(panel_size, resampling)
        x = index * panel_size[0]
        canvas.paste(panel, (x, header_height))
        draw.text((x + 8, 7), title, fill="black")

    legend_y = header_height + panel_size[1] + 12
    legend_entries = list(zip(class_names, class_colors)) + [("ignore", ignore_color)]
    item_width = canvas.width // len(legend_entries)
    for index, (name, color) in enumerate(legend_entries):
        x = index * item_width + 8
        draw.rectangle((x, legend_y, x + 18, legend_y + 18), fill=color, outline="black")
        draw.text((x + 25, legend_y + 2), str(name), fill="black")
    canvas.save(path)


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def single_image_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_names: Sequence[str],
    ignore_index: int,
) -> dict[str, Any]:
    meter = SegmentationConfusionMatrix(
        num_classes=len(class_names), ignore_index=ignore_index, device=prediction.device
    )
    meter.update(prediction, target)
    computed = meter.compute()
    result: dict[str, Any] = {
        name: float(computed[name].item())
        for name in ("mIoU", "mF1", "mAcc", "OA", "FWIoU")
    }
    result["per_class"] = {}
    for index, class_name in enumerate(class_names):
        result["per_class"][class_name] = {
            "IoU": finite_or_none(float(computed["class_iou"][index].item())),
            "F1": finite_or_none(float(computed["class_f1"][index].item())),
            "Acc": finite_or_none(float(computed["class_accuracy"][index].item())),
            "OA": finite_or_none(float(computed["class_oa"][index].item())),
        }
    result["confusion_matrix"] = computed["confusion_matrix"].long().cpu().tolist()
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    input_dir = args.input_dir
    output_prefix = str(args.output_prefix).strip()
    if not output_prefix or Path(output_prefix).name != output_prefix:
        raise ValueError("output_prefix must be a non-empty filename prefix, not a path.")
    rgb_path = input_dir / args.rgb_name
    modality_b_path = input_dir / args.modality_b_name
    gt_path = find_gt(input_dir, args.gt_name)
    output_name = args.output_name or f"{output_prefix}_4panel_result.png"
    if Path(output_name).name != output_name:
        raise ValueError("output_name must be a filename, not a path.")
    output_path = input_dir / output_name
    raw_prediction_path = input_dir / f"{output_prefix}_prediction_raw.png"
    color_prediction_path = input_dir / f"{output_prefix}_prediction_color.png"
    metrics_path = input_dir / f"{output_prefix}_single_metrics.json"
    generated_paths = (
        output_path,
        raw_prediction_path,
        color_prediction_path,
        metrics_path,
    )
    existing = [path for path in generated_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {existing[0]}. Pass --overwrite to replace generated results."
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")

    class_names = tuple(str(name) for name in config["class_names"])
    num_classes = int(config["model"]["num_classes"])
    if len(class_names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, got {len(class_names)}.")
    modality_b_channels = int(
        config["data"].get(
            "modality_b_channels", config["model"]["in_channels_b"]
        )
    )
    rgb, modality_b, target = read_inputs(
        rgb_path,
        modality_b_path,
        gt_path,
        num_classes,
        modality_b_channels,
        config["data"].get("label_threshold"),
    )
    rgb_tensor, sar_tensor = normalize_inputs(rgb, modality_b, config["data"])
    rgb_tensor = rgb_tensor.to(device)
    sar_tensor = sar_tensor.to(device)
    target_tensor = torch.from_numpy(target.astype(np.int64)).unsqueeze(0).to(device)

    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    model, checkpoint = load_model(config, checkpoint_path, device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=amp_enabled,
    ):
        logits = model(rgb_tensor, sar_tensor)
        prediction_tensor = logits.argmax(dim=1)
    prediction = prediction_tensor[0].cpu().numpy().astype(np.uint8)

    Image.fromarray(prediction, mode="L").save(raw_prediction_path)
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
    Image.fromarray(
        colorize_mask(prediction, class_colors, ignore_color), mode="RGB"
    ).save(color_prediction_path)
    save_four_panel(
        output_path,
        rgb,
        modality_b,
        target,
        prediction,
        class_names,
        args.visual_scale,
        class_colors,
        ignore_color,
        modality_a_title,
        modality_b_title,
    )
    metrics = single_image_metrics(
        prediction_tensor,
        target_tensor,
        class_names,
        int(config["loss"].get("ignore_index", 255)),
    )
    metrics.update(
        {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_epoch": (
                int(checkpoint["epoch"]) + 1 if "epoch" in checkpoint else None
            ),
            "rgb": str(rgb_path.resolve()),
            "modality_b": str(modality_b_path.resolve()),
            "gt": str(gt_path.resolve()),
            "prediction": str(raw_prediction_path.resolve()),
            "four_panel": str(output_path.resolve()),
        }
    )
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint epoch: {metrics['checkpoint_epoch']}")
    print(f"input size: {rgb.shape[1]}x{rgb.shape[0]}")
    print(f"GT: {gt_path}; values={sorted(int(value) for value in np.unique(target))}")
    print(f"prediction values: {sorted(int(value) for value in np.unique(prediction))}")
    print(
        f"single-image metrics: mIoU={metrics['mIoU']:.4f} "
        f"mF1={metrics['mF1']:.4f} OA={metrics['OA']:.4f} "
        f"mAcc={metrics['mAcc']:.4f}"
    )
    print(f"four-panel result: {output_path.resolve()}")
    print(f"raw prediction: {raw_prediction_path.resolve()}")
    print(f"color prediction: {color_prediction_path.resolve()}")
    print(f"metrics: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
