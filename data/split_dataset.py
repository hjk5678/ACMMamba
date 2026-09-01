"""Create reproducible train/test/validation manifests for GF2, GF3 and GT.

Only sample identifiers are written. The original images are never copied,
moved or modified.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Set


DEFAULT_RGB_DIR = Path("/data/BUAS/HJK/ACMMamba/data/DDHRNet/shandong/GF2")
DEFAULT_SAR_DIR = Path("/data/BUAS/HJK/ACMMamba/data/DDHRNet/shandong/GF3")
DEFAULT_LABEL_DIR = Path("/data/BUAS/HJK/ACMMamba/data/DDHRNet/shandong/label")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split paired GF2/GF3/label samples with a 7:2:1 ratio."
    )
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB_DIR)
    parser.add_argument("--sar-dir", type=Path, default=DEFAULT_SAR_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_stems(directory: Path, suffixes: Iterable[str]) -> Set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    allowed = {suffix.lower() for suffix in suffixes}
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    }


def validate_pairs(rgb_dir: Path, sar_dir: Path, label_dir: Path) -> List[str]:
    groups = {
        "GF2 RGB": collect_stems(rgb_dir, {".jpg", ".jpeg", ".png", ".tif", ".tiff"}),
        "GF3 SAR": collect_stems(sar_dir, {".jpg", ".jpeg", ".png", ".tif", ".tiff"}),
        "label": collect_stems(label_dir, {".png", ".tif", ".tiff"}),
    }

    common = set.intersection(*groups.values())
    if not common:
        raise RuntimeError("No paired samples were found in the three directories.")

    problems = []
    for name, stems in groups.items():
        missing = sorted(set.union(*groups.values()) - stems)
        if missing:
            problems.append(f"{name} is missing {len(missing)} samples, e.g. {missing[:5]}")

    if problems:
        raise RuntimeError("The dataset is not fully paired:\n" + "\n".join(problems))

    return sorted(common)


def split_ids(sample_ids: List[str], seed: int) -> Dict[str, List[str]]:
    """Shuffle and split IDs into train/test/val using a 7:2:1 ratio."""
    shuffled = sample_ids.copy()
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * 0.7)
    test_end = train_end + int(total * 0.2)

    return {
        "train": shuffled[:train_end],
        "test": shuffled[train_end:test_end],
        "val": shuffled[test_end:],
    }


def write_manifests(splits: Dict[str, List[str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, sample_ids in splits.items():
        manifest = output_dir / f"{split_name}.txt"
        manifest.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")

    all_ids = [sample_id for ids in splits.values() for sample_id in ids]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Duplicate sample IDs occurred across split files.")


def main() -> None:
    args = parse_args()
    sample_ids = validate_pairs(args.rgb_dir, args.sar_dir, args.label_dir)
    splits = split_ids(sample_ids, args.seed)
    write_manifests(splits, args.output_dir)

    total = len(sample_ids)
    print(f"Total paired samples: {total}")
    for name in ("train", "test", "val"):
        count = len(splits[name])
        print(f"{name:>5}: {count:4d} ({count / total:.2%})")
    print(f"Seed: {args.seed}")
    print(f"Manifests saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
