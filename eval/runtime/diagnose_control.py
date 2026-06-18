"""Runtime control diagnostics CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.runtime.metrics import compute_root_condition_diagnostics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect runtime root-condition diagnostics. Full benchmark-time "
            "diagnostic dumps are produced by eval/runtime/benchmark.py."
        )
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional note included when invoking the compatibility command.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    build_arg_parser().parse_args(argv)
    return 0


__all__ = [
    "build_arg_parser",
    "compute_root_condition_diagnostics",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
