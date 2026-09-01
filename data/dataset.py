"""PyTorch dataset for paired GF2 RGB, GF3 SAR and segmentation labels."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


RGB_MEAN = (0.22786175604401648, 0.23903183493110483, 0.2416281091275199)
RGB_STD = (0.14607820531457671, 0.13077874406935705, 0.13096457260040756)
SAR_MEAN = (0.2589521446425652,)
SAR_STD = (0.20572006698576253,)


def read_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Split file does not exist: {path}. Run data/split_dataset.py first."
        )
    sample_ids: list[str] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue

        # Accept a bare ID, a filename/path, or a row containing the three
        # paired paths.  In the latter case all columns must refer to the same
        # sample stem so a malformed split cannot silently mis-pair images.
        fields = line.replace(",", " ").split()
        stems = {
            Path(field.replace("\\", "/")).stem
            for field in fields
            if field
        }
        if len(stems) != 1:
            raise ValueError(
                f"Ambiguous split entry at {path}:{line_number}: {raw_line!r}"
            )
        sample_ids.append(stems.pop())

    if not sample_ids:
        raise RuntimeError(f"Split file is empty: {path}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Split file contains duplicate sample IDs: {path}")
    return sample_ids


def find_sample(directory: Path, sample_id: str, suffixes: Sequence[str]) -> Path:
    for suffix in suffixes:
        path = directory / f"{sample_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find sample '{sample_id}' in {directory}")


def resolve_sample(
    directory: Path,
    sample_id: str,
    suffixes: Sequence[str],
    filename_template: str | None,
) -> Path:
    if filename_template is None:
        return find_sample(directory, sample_id, suffixes)
    try:
        filename = filename_template.format(id=sample_id)
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"Invalid filename template {filename_template!r}; use '{{id}}' for the sample ID."
        ) from error
    if "{id}" not in filename_template or Path(filename).name != filename:
        raise ValueError(
            f"Filename template must contain '{{id}}' and produce a filename: "
            f"{filename_template!r}"
        )
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find sample '{sample_id}': {path}")
    return path


class PairedRemoteSensingDataset(Dataset):
    """Load paired RGB, SAR and semantic-segmentation targets.

    RGB and SAR are returned as normalized float tensors. Labels are returned
    as int64 tensors and preserve the ignore value 255.
    """

    def __init__(
        self,
        rgb_dir: str | Path,
        sar_dir: str | Path,
        label_dir: str | Path,
        split_file: str | Path,
        augment: bool = False,
        rgb_mean: Sequence[float] = RGB_MEAN,
        rgb_std: Sequence[float] = RGB_STD,
        sar_mean: Sequence[float] = SAR_MEAN,
        sar_std: Sequence[float] = SAR_STD,
        modality_b_channels: int = 1,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        random_90_rotation: bool = True,
        rgb_filename_template: str | None = None,
        modality_b_filename_template: str | None = None,
        label_filename_template: str | None = None,
        label_threshold: int | None = None,
    ) -> None:
        self.rgb_dir = Path(rgb_dir)
        self.sar_dir = Path(sar_dir)
        self.label_dir = Path(label_dir)
        self.sample_ids = read_manifest(Path(split_file))
        self.augment = augment
        self.modality_b_channels = int(modality_b_channels)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.vertical_flip_probability = float(vertical_flip_probability)
        self.random_90_rotation = bool(random_90_rotation)
        self.rgb_filename_template = rgb_filename_template
        self.modality_b_filename_template = modality_b_filename_template
        self.label_filename_template = label_filename_template
        self.label_threshold = None if label_threshold is None else int(label_threshold)

        if self.modality_b_channels not in (1, 3):
            raise ValueError("modality_b_channels must be either 1 or 3.")
        if self.label_threshold is not None and not 0 <= self.label_threshold <= 255:
            raise ValueError("label_threshold must be between 0 and 255.")
        for name, probability in (
            ("horizontal_flip_probability", self.horizontal_flip_probability),
            ("vertical_flip_probability", self.vertical_flip_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        self.rgb_mean = np.asarray(rgb_mean, dtype=np.float32).reshape(1, 1, 3)
        self.rgb_std = np.asarray(rgb_std, dtype=np.float32).reshape(1, 1, 3)
        self.sar_mean = np.asarray(sar_mean, dtype=np.float32).reshape(
            1, 1, self.modality_b_channels
        )
        self.sar_std = np.asarray(sar_std, dtype=np.float32).reshape(
            1, 1, self.modality_b_channels
        )

        if np.any(self.rgb_std <= 0) or np.any(self.sar_std <= 0):
            raise ValueError("Normalization standard deviations must be positive.")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _joint_augment(
        self,
        rgb: np.ndarray, sar: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < self.horizontal_flip_probability:
            rgb = np.flip(rgb, axis=1)
            sar = np.flip(sar, axis=1)
            label = np.flip(label, axis=1)

        if random.random() < self.vertical_flip_probability:
            rgb = np.flip(rgb, axis=0)
            sar = np.flip(sar, axis=0)
            label = np.flip(label, axis=0)

        if self.random_90_rotation:
            rotations = random.randint(0, 3)
            if rotations:
                rgb = np.rot90(rgb, rotations, axes=(0, 1))
                sar = np.rot90(sar, rotations, axes=(0, 1))
                label = np.rot90(label, rotations, axes=(0, 1))

        return rgb.copy(), sar.copy(), label.copy()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample_id = self.sample_ids[index]
        rgb_path = resolve_sample(
            self.rgb_dir,
            sample_id,
            (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
            self.rgb_filename_template,
        )
        sar_path = resolve_sample(
            self.sar_dir,
            sample_id,
            (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
            self.modality_b_filename_template,
        )
        label_path = resolve_sample(
            self.label_dir,
            sample_id,
            (".png", ".tif", ".tiff"),
            self.label_filename_template,
        )

        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(sar_path) as image:
            modality_b_mode = "L" if self.modality_b_channels == 1 else "RGB"
            sar = np.asarray(image.convert(modality_b_mode), dtype=np.uint8)
            if sar.ndim == 2:
                sar = sar[..., None]
        with Image.open(label_path) as image:
            label = np.asarray(image, dtype=np.uint8)

        if label.ndim != 2:
            raise ValueError(
                f"Label must be a single-channel class-index mask for '{sample_id}', "
                f"got shape {label.shape}."
            )
        if self.label_threshold is not None:
            label = (label >= self.label_threshold).astype(np.uint8)

        if rgb.shape[:2] != sar.shape[:2] or sar.shape[:2] != label.shape:
            raise ValueError(
                f"Spatial shape mismatch for '{sample_id}': "
                f"RGB={rgb.shape}, SAR={sar.shape}, label={label.shape}"
            )

        if self.augment:
            rgb, sar, label = self._joint_augment(rgb, sar, label)

        rgb_float = rgb.astype(np.float32) / 255.0
        sar_float = sar.astype(np.float32) / 255.0
        rgb_float = (rgb_float - self.rgb_mean) / self.rgb_std
        sar_float = (sar_float - self.sar_mean) / self.sar_std

        rgb_tensor = torch.from_numpy(rgb_float.transpose(2, 0, 1).copy())
        sar_tensor = torch.from_numpy(sar_float.transpose(2, 0, 1).copy())
        label_tensor = torch.from_numpy(label.astype(np.int64, copy=True))

        return {
            "rgb": rgb_tensor,
            "sar": sar_tensor,
            "label": label_tensor,
            "id": sample_id,
        }


def build_datasets(
    rgb_dir: str | Path,
    sar_dir: str | Path,
    label_dir: str | Path,
    split_dir: str | Path,
    rgb_mean: Sequence[float] = RGB_MEAN,
    rgb_std: Sequence[float] = RGB_STD,
    sar_mean: Sequence[float] = SAR_MEAN,
    sar_std: Sequence[float] = SAR_STD,
    modality_b_channels: int = 1,
    train_horizontal_flip_probability: float = 0.5,
    train_vertical_flip_probability: float = 0.5,
    train_random_90_rotation: bool = True,
    rgb_filename_template: str | None = None,
    modality_b_filename_template: str | None = None,
    label_filename_template: str | None = None,
    label_threshold: int | None = None,
    test_rgb_dir: str | Path | None = None,
    test_sar_dir: str | Path | None = None,
    test_label_dir: str | Path | None = None,
) -> Dict[str, PairedRemoteSensingDataset]:
    """Build the standard train/test/validation datasets."""
    split_dir = Path(split_dir)
    common_args = {
        "rgb_dir": rgb_dir,
        "sar_dir": sar_dir,
        "label_dir": label_dir,
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
        "sar_mean": sar_mean,
        "sar_std": sar_std,
        "modality_b_channels": modality_b_channels,
        "rgb_filename_template": rgb_filename_template,
        "modality_b_filename_template": modality_b_filename_template,
        "label_filename_template": label_filename_template,
        "label_threshold": label_threshold,
    }
    return {
        "train": PairedRemoteSensingDataset(
            split_file=split_dir / "train.txt",
            augment=True,
            horizontal_flip_probability=train_horizontal_flip_probability,
            vertical_flip_probability=train_vertical_flip_probability,
            random_90_rotation=train_random_90_rotation,
            **common_args,
        ),
        "test": PairedRemoteSensingDataset(
            split_file=split_dir / "test.txt",
            augment=False,
            **{
                **common_args,
                "rgb_dir": test_rgb_dir or rgb_dir,
                "sar_dir": test_sar_dir or sar_dir,
                "label_dir": test_label_dir or label_dir,
            },
        ),
        "val": PairedRemoteSensingDataset(
            split_file=split_dir / "val.txt", augment=False, **common_args
        ),
    }
