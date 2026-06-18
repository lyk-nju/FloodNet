"""Rename state-dict key prefixes in a PyTorch checkpoint."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def rename_checkpoint_keys(
    ckpt_path: str | Path,
    save_path: str | Path,
    old_prefix: str,
    new_prefix: str = "",
    *,
    dry_run: bool = False,
) -> int:
    """Rename checkpoint `state_dict` keys that start with `old_prefix`.

    Returns the number of renamed keys. Non-state-dict payload fields are
    preserved exactly.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"checkpoint has no dict state_dict: {ckpt_path}")

    renamed = 0
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(old_prefix):
            suffix = key[len(old_prefix) :]
            key = f"{new_prefix}{suffix}" if new_prefix else suffix
            renamed += 1
        new_state_dict[key] = value

    checkpoint["state_dict"] = new_state_dict
    if not dry_run:
        torch.save(checkpoint, save_path)
    return renamed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", required=True, help="Input checkpoint path.")
    parser.add_argument("--save_path", required=True, help="Output checkpoint path.")
    parser.add_argument("--old_prefix", required=True, help="Prefix to replace.")
    parser.add_argument(
        "--new_prefix",
        default="",
        help="Replacement prefix. Empty string removes the old prefix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the rename count without writing the output checkpoint.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    renamed = rename_checkpoint_keys(
        ckpt_path=args.ckpt_path,
        save_path=args.save_path,
        old_prefix=args.old_prefix,
        new_prefix=args.new_prefix,
        dry_run=args.dry_run,
    )
    action = "would rename" if args.dry_run else "renamed"
    print(
        f"{action} {renamed} keys: "
        f"{args.old_prefix!r} -> {args.new_prefix!r}"
    )
    if not args.dry_run:
        print(f"saved modified checkpoint to: {args.save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
