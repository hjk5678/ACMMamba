"""Generate isolated MLFM-ablation configs from an existing baseline config."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path, PurePosixPath

import yaml


FUSION_MODES = ("mean", "add_only", "cat_only", "dual_fixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fusion-ablation YAML files from one baseline config."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("configs/ablations")
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=FUSION_MODES,
        default=list(FUSION_MODES),
    )
    return parser.parse_args()


def experiment_name(config: dict, base_path: Path) -> str:
    configured = str(config.get("checkpoint_prefix", "")).strip()
    if configured:
        return configured
    stem = base_path.stem
    return stem[len("train_") :] if stem.startswith("train_") else stem


def output_root(configured_path: str, experiment: str) -> PurePosixPath:
    # Training configs target the Linux server even when this generator runs
    # from a synced Windows workspace, so never reinterpret them as Windows paths.
    path = PurePosixPath(configured_path)
    return path.parent if path.name == experiment else path


def main() -> None:
    args = parse_args()
    if not args.base.is_file():
        raise FileNotFoundError(f"Base config does not exist: {args.base}")
    base_config = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    if not isinstance(base_config, dict):
        raise ValueError("The base YAML root must be a mapping.")
    if "model" not in base_config or "train" not in base_config:
        raise ValueError("The base config must contain model and train sections.")

    experiment = experiment_name(base_config, args.base)
    log_root = output_root(str(base_config["log_dir"]), experiment)
    checkpoint_root = output_root(
        str(base_config["checkpoint_dir"]), experiment
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        config = deepcopy(base_config)
        run_name = f"{experiment}_fusion_{mode}"
        config["model"]["fusion_mode"] = mode
        config["log_dir"] = str(log_root / "ablations" / experiment / mode)
        config["checkpoint_dir"] = str(
            checkpoint_root / "ablations" / experiment / mode
        )
        config["checkpoint_prefix"] = run_name
        config["train"]["resume"] = None
        output_path = args.output_dir / f"train_{run_name}.yaml"
        output_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(
            f"{mode:10s} -> {output_path} | "
            f"checkpoints={config['checkpoint_dir']}"
        )


if __name__ == "__main__":
    main()
