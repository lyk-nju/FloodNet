"""Train the online residual frontier NoiseInitializer on one debug sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.initialize import load_config  # noqa: E402


def _parse_overrides(items: list[str] | None) -> dict[str, str]:
    overrides = {}
    for item in items or []:
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/noise_initializer_overfit.yaml",
        help="NoiseInitializer overfit config.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=None,
        help="Config overrides as key=value.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_obj = load_config(
        config_path=args.config,
        override_args=_parse_overrides(args.override),
    )
    cfg = OmegaConf.to_container(cfg_obj.config, resolve=True)
    from utils.training.noise_initializer.overfit_runner import run_single_sample_overfit

    summary = run_single_sample_overfit(cfg)
    printable = dict(summary)
    printable["loss_curve"] = summary.get("loss_curve", [])[-5:]
    print(json.dumps(printable, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
