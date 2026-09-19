"""Generate SSSS and CCCC CrossMamba-ablation configs.

Only the four-stage routing policy and isolated output paths are changed. All
model dimensions, AS6 blocks, fusion modules and training settings remain
identical to the selected baseline configuration.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path, PurePosixPath

import yaml


STAGE_MODE_ABLATIONS = {
    "ssss": ["self", "self", "self", "self"],
    "cccc": ["cross", "cross", "cross", "cross"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CrossMamba stage-routing ablation configurations."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("configs/ablations")
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(STAGE_MODE_ABLATIONS),
        default=list(STAGE_MODE_ABLATIONS),
    )
    return parser.parse_args()


def experiment_name(config: dict, base_path: Path) -> str:
    configured = str(config.get("checkpoint_prefix", "")).strip()
    if configured:
        return configured
    stem = base_path.stem
    return stem[len("train_") :] if stem.startswith("train_") else stem


def output_root(configured_path: str, experiment: str) -> PurePosixPath:
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

    for abbreviation in args.modes:
        config = deepcopy(base_config)
        run_name = f"{experiment}_encoder_{abbreviation}"
        config["model"]["fusion_mode"] = "mlfm"
        config["model"]["stage_modes"] = STAGE_MODE_ABLATIONS[abbreviation]
        config["log_dir"] = str(
            log_root / "ablations" / experiment / "encoder" / abbreviation
        )
        config["checkpoint_dir"] = str(
            checkpoint_root
            / "ablations"
            / experiment
            / "encoder"
            / abbreviation
        )
        config["checkpoint_prefix"] = run_name
        config["train"]["resume"] = None
        output_path = args.output_dir / f"train_{run_name}.yaml"
        output_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(
            f"{abbreviation}: {config['model']['stage_modes']} -> "
            f"{output_path}"
        )


if __name__ == "__main__":
    main()
