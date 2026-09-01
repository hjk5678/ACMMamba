"""Inspect a paired RGB/SAR/label semantic-segmentation dataset.

The script is read-only.  It validates filename pairing and spatial sizes,
then computes exact image normalization statistics and label frequencies.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
LABEL_SUFFIXES = {".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect paired RGB, SAR and semantic-segmentation labels."
    )
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--sar-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=500,
        help="Print progress every N paired samples; use 0 to disable.",
    )
    return parser.parse_args()


def collect_files(directory: Path, suffixes: Iterable[str]) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    allowed = {suffix.lower() for suffix in suffixes}
    files: Dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        if path.stem in files:
            raise RuntimeError(
                f"Duplicate sample ID '{path.stem}' in {directory}: "
                f"{files[path.stem].name}, {path.name}"
            )
        files[path.stem] = path
    return files


def print_group(name: str, directory: Path, files: Dict[str, Path]) -> None:
    suffix_counts = Counter(path.suffix.lower() for path in files.values())
    print(f"[{name}]")
    print(f"path: {directory}")
    print(f"files: {len(files)}")
    print(f"suffixes: {dict(sorted(suffix_counts.items()))}")


def main() -> None:
    args = parse_args()
    groups = {
        "RGB": collect_files(args.rgb_dir, IMAGE_SUFFIXES),
        "SAR": collect_files(args.sar_dir, IMAGE_SUFFIXES),
        "LABEL": collect_files(args.label_dir, LABEL_SUFFIXES),
    }

    print("\n=== Paths and file counts ===")
    print_group("RGB", args.rgb_dir, groups["RGB"])
    print_group("SAR", args.sar_dir, groups["SAR"])
    print_group("LABEL", args.label_dir, groups["LABEL"])

    all_ids = set().union(*(set(files) for files in groups.values()))
    paired_ids = sorted(set.intersection(*(set(files) for files in groups.values())))
    print("\n=== Pairing ===")
    print(f"paired samples: {len(paired_ids)}")
    for name, files in groups.items():
        missing = sorted(all_ids - set(files))
        print(f"missing from {name}: {len(missing)}; examples: {missing[:10]}")
    if not paired_ids:
        raise RuntimeError("No paired samples were found.")

    shape_counts: Counter[tuple] = Counter()
    rgb_mode_counts: Counter[str] = Counter()
    sar_mode_counts: Counter[str] = Counter()
    label_mode_counts: Counter[str] = Counter()
    label_value_counts: Counter[int] = Counter()
    label_color_counts: Counter[tuple] = Counter()
    failures = []

    rgb_sum = np.zeros(3, dtype=np.float64)
    rgb_square_sum = np.zeros(3, dtype=np.float64)
    rgb_pixels = 0
    sar_sum = 0.0
    sar_square_sum = 0.0
    sar_pixels = 0

    print("\n=== Reading paired samples ===")
    for index, sample_id in enumerate(paired_ids, start=1):
        try:
            with Image.open(groups["RGB"][sample_id]) as image:
                rgb_mode_counts[image.mode] += 1
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            with Image.open(groups["SAR"][sample_id]) as image:
                sar_mode_counts[image.mode] += 1
                sar = np.asarray(image.convert("L"), dtype=np.uint8)
            with Image.open(groups["LABEL"][sample_id]) as image:
                label_mode_counts[image.mode] += 1
                label = np.asarray(image)

            shape_counts[(rgb.shape, sar.shape, label.shape)] += 1
            if rgb.shape[:2] != sar.shape or sar.shape != label.shape[:2]:
                failures.append(
                    f"{sample_id}: spatial mismatch RGB={rgb.shape}, "
                    f"SAR={sar.shape}, LABEL={label.shape}"
                )
                continue

            rgb_float = rgb.astype(np.float64) / 255.0
            sar_float = sar.astype(np.float64) / 255.0
            rgb_sum += rgb_float.sum(axis=(0, 1))
            rgb_square_sum += np.square(rgb_float).sum(axis=(0, 1))
            rgb_pixels += rgb.shape[0] * rgb.shape[1]
            sar_sum += float(sar_float.sum())
            sar_square_sum += float(np.square(sar_float).sum())
            sar_pixels += sar.size

            if label.ndim == 2:
                label_flat = label.reshape(-1).astype(np.int64, copy=False)
                if label_flat.size and int(label_flat.min()) < 0:
                    raise ValueError("Negative class values are not supported.")
                # Class-index masks normally contain small unsigned integers.
                # bincount is linear and substantially faster than sorting all
                # pixels with np.unique on large remote-sensing masks.
                counts = np.bincount(label_flat, minlength=256)
                label_value_counts.update(
                    {
                        int(value): int(count)
                        for value, count in enumerate(counts)
                        if count > 0
                    }
                )
            elif label.ndim == 3:
                colors, counts = np.unique(label.reshape(-1, label.shape[-1]), axis=0, return_counts=True)
                label_color_counts.update(
                    {tuple(int(channel) for channel in color): int(count) for color, count in zip(colors, counts)}
                )
            else:
                failures.append(f"{sample_id}: unsupported label shape {label.shape}")
        except Exception as error:  # continue to report all bad files
            failures.append(f"{sample_id}: {error!r}")

        if args.progress_interval > 0 and index % args.progress_interval == 0:
            print(f"processed {index}/{len(paired_ids)}", flush=True)

    print("\n=== Shapes and label modes ===")
    for shapes, count in shape_counts.most_common():
        print(f"{shapes} -> {count}")
    print(f"RGB image modes: {dict(rgb_mode_counts)}")
    print(f"second-modality image modes: {dict(sar_mode_counts)}")
    print(f"label image modes: {dict(label_mode_counts)}")
    print(f"failed/anomalous samples: {len(failures)}")
    for failure in failures[:30]:
        print(f"  {failure}")

    if rgb_pixels and sar_pixels:
        rgb_mean = rgb_sum / rgb_pixels
        rgb_variance = np.maximum(rgb_square_sum / rgb_pixels - np.square(rgb_mean), 0.0)
        rgb_std = np.sqrt(rgb_variance)
        sar_mean = sar_sum / sar_pixels
        sar_variance = max(sar_square_sum / sar_pixels - sar_mean**2, 0.0)
        sar_std = sar_variance**0.5
        print("\n=== Normalization statistics (image values divided by 255) ===")
        print(f"RGB mean: {rgb_mean.tolist()}")
        print(f"RGB std:  {rgb_std.tolist()}")
        print(f"SAR mean: {sar_mean}")
        print(f"SAR std:  {sar_std}")

    print("\n=== Label distribution ===")
    if label_value_counts:
        total = sum(label_value_counts.values())
        valid_total = total - label_value_counts.get(255, 0)
        for value in sorted(label_value_counts):
            count = label_value_counts[value]
            print(f"value={value:>5}, pixels={count:>14}, ratio={count / total:.8%}")
        if valid_total > 0 and 255 in label_value_counts:
            print("excluding ignore value 255:")
            for value in sorted(value for value in label_value_counts if value != 255):
                count = label_value_counts[value]
                print(
                    f"class={value:>5}, pixels={count:>14}, "
                    f"ratio={count / valid_total:.8%}"
                )
    elif label_color_counts:
        total = sum(label_color_counts.values())
        print("Labels are color encoded. Unique colors:")
        for color, count in label_color_counts.most_common():
            print(f"color={color}, pixels={count:>14}, ratio={count / total:.8%}")
    else:
        print("No readable label pixels were found.")


if __name__ == "__main__":
    main()
