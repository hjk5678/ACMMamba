"""Inspect channel redundancy and statistics of RGB-encoded TIR images."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect RGB-encoded TIR channels.")
    parser.add_argument("--tir-dir", type=Path, required=True)
    parser.add_argument("--progress-interval", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tir_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {args.tir_dir}")
    paths = sorted(
        path
        for path in args.tir_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"No images found in {args.tir_dir}")

    mode_counts: Counter[str] = Counter()
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    average_sum = 0.0
    average_square_sum = 0.0
    pixel_count = 0
    equal_pixel_count = 0
    fully_equal_images = 0
    shape_counts: Counter[tuple[int, ...]] = Counter()

    for index, path in enumerate(paths, start=1):
        with Image.open(path) as image:
            mode_counts[image.mode] += 1
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        shape_counts[array.shape] += 1

        values = array.astype(np.float64) / 255.0
        channel_sum += values.sum(axis=(0, 1))
        channel_square_sum += np.square(values).sum(axis=(0, 1))
        channel_average = values.mean(axis=2)
        average_sum += float(channel_average.sum())
        average_square_sum += float(np.square(channel_average).sum())

        equal_pixels = (array[..., 0] == array[..., 1]) & (
            array[..., 1] == array[..., 2]
        )
        current_pixels = array.shape[0] * array.shape[1]
        equal_count = int(equal_pixels.sum())
        pixel_count += current_pixels
        equal_pixel_count += equal_count
        fully_equal_images += int(equal_count == current_pixels)

        if args.progress_interval > 0 and index % args.progress_interval == 0:
            print(f"processed {index}/{len(paths)}", flush=True)

    channel_mean = channel_sum / pixel_count
    channel_variance = np.maximum(
        channel_square_sum / pixel_count - np.square(channel_mean), 0.0
    )
    channel_std = np.sqrt(channel_variance)
    average_mean = average_sum / pixel_count
    average_variance = max(
        average_square_sum / pixel_count - average_mean**2, 0.0
    )

    print("\n=== TIR channel inspection ===")
    print(f"directory: {args.tir_dir}")
    print(f"images: {len(paths)}")
    print(f"modes: {dict(mode_counts)}")
    print(f"shapes: {dict(shape_counts)}")
    print(f"fully channel-identical images: {fully_equal_images}/{len(paths)}")
    print(
        f"channel-identical pixels: {equal_pixel_count}/{pixel_count} "
        f"({equal_pixel_count / pixel_count:.8%})"
    )
    print("statistics after dividing values by 255:")
    print(f"TIR RGB mean: {channel_mean.tolist()}")
    print(f"TIR RGB std:  {channel_std.tolist()}")
    print(f"TIR channel-average mean: {average_mean}")
    print(f"TIR channel-average std:  {average_variance**0.5}")


if __name__ == "__main__":
    main()
