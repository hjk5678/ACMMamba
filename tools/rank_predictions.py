"""Rank saved segmentation predictions by per-image class IoU."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank existing raw prediction masks by per-image class IoU."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument(
        "--class-index",
        type=int,
        default=1,
        help="Class whose per-image IoU determines the ranking.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Defaults to <prediction-dir parent>/per_image_metrics.csv.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config does not exist: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")
    return config


def read_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")
    sample_ids: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = line.replace(",", " ").split()
        stems = {Path(field.replace("\\", "/")).stem for field in fields}
        if len(stems) != 1:
            raise ValueError(f"Ambiguous manifest row: {raw_line!r}")
        sample_ids.append(stems.pop())
    if not sample_ids:
        raise RuntimeError(f"Split manifest is empty: {path}")
    return sample_ids


def resolve_file(
    directory: Path, sample_id: str, template: str | None, suffixes: Sequence[str]
) -> Path:
    if template is not None:
        path = directory / template.format(id=sample_id)
        if path.is_file():
            return path
        raise FileNotFoundError(f"File does not exist: {path}")
    for suffix in suffixes:
        path = directory / f"{sample_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find sample {sample_id!r} in {directory}")


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else float("nan")


def per_image_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int,
    ranking_class: int,
) -> dict[str, float | int]:
    valid = target != ignore_index
    target_valid = target[valid]
    prediction_valid = prediction[valid]
    if target_valid.size == 0:
        raise ValueError("The sample contains no valid target pixels.")
    invalid_prediction = (prediction_valid < 0) | (prediction_valid >= num_classes)
    if np.any(invalid_prediction):
        values = np.unique(prediction_valid[invalid_prediction]).tolist()
        raise ValueError(f"Prediction contains invalid class values: {values[:20]}")

    encoded = target_valid.astype(np.int64) * num_classes + prediction_valid.astype(
        np.int64
    )
    matrix = np.bincount(encoded, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )
    true_positive = np.diag(matrix)
    target_pixels = matrix.sum(axis=1)
    predicted_pixels = matrix.sum(axis=0)
    union = target_pixels + predicted_pixels - true_positive
    class_iou = np.full(num_classes, np.nan, dtype=np.float64)
    nonempty_union = union > 0
    class_iou[nonempty_union] = true_positive[nonempty_union] / union[nonempty_union]

    tp = int(true_positive[ranking_class])
    fp = int(predicted_pixels[ranking_class] - tp)
    fn = int(target_pixels[ranking_class] - tp)
    tn = int(matrix.sum() - tp - fp - fn)
    return {
        "class_iou": safe_ratio(tp, tp + fp + fn),
        "class_f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
        "class_precision": safe_ratio(tp, tp + fp),
        "class_recall": safe_ratio(tp, tp + fn),
        "mIoU": float(np.nanmean(class_iou)),
        "OA": safe_ratio(tp + tn, int(matrix.sum())),
        "gt_class_pixels": int(target_pixels[ranking_class]),
        "pred_class_pixels": int(predicted_pixels[ranking_class]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def format_metric(value: float) -> str:
    return f"{value:.6f}" if math.isfinite(value) else "nan"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config: Mapping[str, Any] = config["data"]
    class_names = tuple(str(name) for name in config["class_names"])
    num_classes = int(config["model"]["num_classes"])
    if len(class_names) != num_classes:
        raise ValueError("class_names and model.num_classes do not match.")
    if not 0 <= args.class_index < num_classes:
        raise ValueError(f"class-index must be between 0 and {num_classes - 1}.")
    if args.top_k < 1:
        raise ValueError("top-k must be positive.")

    split_dir = Path(data_config["split_dir"])
    sample_ids = read_manifest(split_dir / f"{args.split}.txt")
    label_dir = Path(
        data_config.get(f"{args.split}_label_dir", data_config["label_dir"])
    )
    rgb_dir = Path(data_config.get(f"{args.split}_rgb_dir", data_config["rgb_dir"]))
    modality_b_dir = Path(
        data_config.get(f"{args.split}_sar_dir", data_config["sar_dir"])
    )
    label_template = data_config.get("label_filename_template")
    rgb_template = data_config.get("rgb_filename_template")
    modality_b_template = data_config.get("modality_b_filename_template")
    label_threshold = data_config.get("label_threshold")
    ignore_index = int(config["loss"].get("ignore_index", 255))

    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        prediction_path = resolve_file(
            args.prediction_dir, sample_id, None, IMAGE_SUFFIXES
        )
        label_path = resolve_file(
            label_dir, sample_id, label_template, (".png", ".tif", ".tiff")
        )
        with Image.open(prediction_path) as image:
            prediction = np.asarray(image.convert("L"), dtype=np.int64)
        with Image.open(label_path) as image:
            target = np.asarray(image.convert("L"), dtype=np.int64)
        if label_threshold is not None:
            target = (target >= int(label_threshold)).astype(np.int64)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {sample_id}: prediction={prediction.shape}, "
                f"target={target.shape}."
            )

        metrics = per_image_metrics(
            prediction, target, num_classes, ignore_index, args.class_index
        )
        rows.append(
            {
                "sample_id": sample_id,
                "class_name": class_names[args.class_index],
                **metrics,
                "rgb": str(
                    resolve_file(rgb_dir, sample_id, rgb_template, IMAGE_SUFFIXES)
                ),
                "modality_b": str(
                    resolve_file(
                        modality_b_dir,
                        sample_id,
                        modality_b_template,
                        IMAGE_SUFFIXES,
                    )
                ),
                "gt": str(label_path),
                "prediction": str(prediction_path),
            }
        )

    rows.sort(
        key=lambda row: (
            math.isfinite(float(row["class_iou"])),
            float(row["class_iou"])
            if math.isfinite(float(row["class_iou"]))
            else -1.0,
        ),
        reverse=True,
    )
    output_csv = args.output_csv or args.prediction_dir.parent / "per_image_metrics.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    class_name = class_names[args.class_index]
    print(f"Ranked samples: {len(rows)}")
    print(f"Ranking class: {args.class_index} ({class_name})")
    print(f"Top {min(args.top_k, len(rows))}:")
    for rank, row in enumerate(rows[: args.top_k], start=1):
        print(
            f"{rank:2d}. {row['sample_id']}: "
            f"IoU={format_metric(float(row['class_iou']))} "
            f"F1={format_metric(float(row['class_f1']))} "
            f"Precision={format_metric(float(row['class_precision']))} "
            f"Recall={format_metric(float(row['class_recall']))}"
        )
    print(f"CSV: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
