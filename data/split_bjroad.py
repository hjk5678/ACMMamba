"""Create leakage-safe BJRoad train/validation/test manifests.

The provided ``train_val`` directory contains seven augmented variants for
each original tile.  Variants are grouped by removing the hundreds digit from
the second coordinate (for example, 10_8, 10_108, ..., 10_608).  A whole group
is assigned to either train or validation so augmented copies cannot leak.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split BJRoad train_val by source tile.")
    parser.add_argument(
        "--root", type=Path, default=Path("/data/BUAS/HJK/ACMMamba/data/BJRoad")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/BUAS/HJK/ACMMamba/data/splits_bjroad"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-groups",
        type=int,
        default=35,
        help="35 of 278 source tiles gives the published approximately 70/10/20 split.",
    )
    return parser.parse_args()


def collect_ids(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    ids = {
        path.name[: -len(suffix)]
        for path in directory.iterdir()
        if path.is_file() and path.name.endswith(suffix)
    }
    if not ids:
        raise RuntimeError(f"No *{suffix} files found in {directory}")
    return ids


def paired_ids(root: Path, split: str) -> set[str]:
    mappings = {
        "image": collect_ids(root / split / "image", "_sat.png"),
        "gps": collect_ids(root / split / "gps", "_gps.jpg"),
        "mask": collect_ids(root / split / "mask", "_mask.png"),
    }
    reference = mappings["image"]
    for name, ids in mappings.items():
        if ids != reference:
            raise RuntimeError(
                f"Unpaired {split}/{name}: missing={sorted(reference - ids)[:10]}, "
                f"extra={sorted(ids - reference)[:10]}"
            )
    return reference


def source_group(sample_id: str) -> str:
    try:
        first, second = sample_id.rsplit("_", 1)
        second_number = int(second)
    except ValueError as error:
        raise ValueError(f"Unexpected BJRoad sample ID: {sample_id!r}") from error
    return f"{first}_{second_number % 100}"


def numeric_id_key(sample_id: str) -> tuple[int, int]:
    first, second = sample_id.rsplit("_", 1)
    return int(first), int(second)


def write_manifest(path: Path, sample_ids: list[str]) -> None:
    path.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_val_ids = paired_ids(args.root, "train_val")
    test_ids = paired_ids(args.root, "test")
    if train_val_ids & test_ids:
        raise RuntimeError("train_val and test contain overlapping sample IDs.")

    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in train_val_ids:
        groups[source_group(sample_id)].append(sample_id)
    invalid_groups = {
        group: sorted(ids, key=numeric_id_key)
        for group, ids in groups.items()
        if len(ids) != 7
    }
    if invalid_groups:
        examples = dict(list(invalid_groups.items())[:5])
        raise RuntimeError(f"Expected seven variants per source tile; examples: {examples}")
    if not 0 < args.val_groups < len(groups):
        raise ValueError(f"val_groups must be between 1 and {len(groups) - 1}.")

    group_ids = sorted(groups, key=numeric_id_key)
    random.Random(args.seed).shuffle(group_ids)
    val_group_ids = set(group_ids[: args.val_groups])
    train_group_ids = set(group_ids[args.val_groups :])

    train_ids = sorted(
        (sample for group in train_group_ids for sample in groups[group]),
        key=numeric_id_key,
    )
    val_ids = sorted(
        (sample for group in val_group_ids for sample in groups[group]),
        key=numeric_id_key,
    )
    test_ids_sorted = sorted(test_ids, key=numeric_id_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_dir / "train.txt", train_ids)
    write_manifest(args.output_dir / "val.txt", val_ids)
    write_manifest(args.output_dir / "test.txt", test_ids_sorted)
    metadata = {
        "seed": args.seed,
        "source_groups": len(groups),
        "train_groups": len(train_group_ids),
        "val_groups": len(val_group_ids),
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "test_samples": len(test_ids_sorted),
        "train_group_ids": sorted(train_group_ids, key=numeric_id_key),
        "val_group_ids": sorted(val_group_ids, key=numeric_id_key),
    }
    (args.output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Source train/val groups: {len(groups)} (7 variants each)")
    print(f"train: {len(train_ids)} samples from {len(train_group_ids)} groups")
    print(f"  val: {len(val_ids)} samples from {len(val_group_ids)} groups")
    print(f" test: {len(test_ids_sorted)} fixed samples")
    print(f"Seed: {args.seed}")
    print(f"Manifests saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
