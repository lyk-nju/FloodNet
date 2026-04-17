"""
Data alignment sanity check — no model required.

Verifies that after the causal-VAE crop fixes, token / feature / traj are
temporally consistent for every sample in the dataset:

  Check 1  token_length == 1 + (feature_length - 1) // 4
  Check 2  traj_length  == feature_length
  Check 3  traj_mask non-zero coverage matches token_mask causal expansion
           (token 0 → frame 0 ; token k≥1 → frames [4k-3, 4k])

Usage:
    python tools/check_data_alignment.py --config configs/ldf.yaml
    python tools/check_data_alignment.py --config configs/ldf.yaml --split train --num_samples 200
"""

import argparse
import os
import random
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.initialize import Config, instantiate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/ldf.yaml")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--num_samples", type=int, default=100,
                   help="How many samples to check (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true",
                   help="Print per-frame mask detail for failures")
    return p.parse_args()


def _token_length_expected(feature_length: int) -> int:
    """Causal encoder formula: iter_ = 1 + (t - 1) // 4"""
    return 1 + (feature_length - 1) // 4


def _frames_for_token(k: int) -> range:
    """Frame range for token k under causal convention."""
    if k == 0:
        return range(0, 1)
    return range(4 * k - 3, 4 * k + 1)


def check_sample(sample, verbose=False):
    """Returns list of (check_name, passed, detail_str)."""
    results = []
    name = sample.get("name", "?")

    feat_len = int(sample.get("feature_length", -1))
    tok_len = int(sample.get("token_length", -1))
    traj_len = int(sample.get("traj_length", -1))
    token_mask = sample.get("token_mask", None)
    traj_mask = sample.get("traj_mask", None)

    # ── Check 1: token_length vs feature_length ──────────────────────────────
    if feat_len >= 0 and tok_len >= 0:
        expected_tok = _token_length_expected(feat_len)
        ok = (tok_len == expected_tok)
        results.append((
            "token/feature length",
            ok,
            f"token_len={tok_len}  feature_len={feat_len}  "
            f"expected_token_len={expected_tok}  diff={tok_len - expected_tok:+d}",
        ))

    # ── Check 2: traj_length vs feature_length ────────────────────────────────
    if feat_len >= 0 and traj_len >= 0:
        ok = (traj_len == feat_len)
        results.append((
            "traj/feature length",
            ok,
            f"traj_len={traj_len}  feature_len={feat_len}  diff={traj_len - feat_len:+d}",
        ))

    # ── Check 3: traj_mask causal expansion ──────────────────────────────────
    if token_mask is not None and traj_mask is not None and traj_len >= 0 and tok_len >= 0:
        token_mask_arr = np.asarray(token_mask, dtype=np.float32)
        traj_mask_arr = np.asarray(traj_mask, dtype=np.float32)

        errors = []
        for k in range(len(token_mask_arr)):
            for f in _frames_for_token(k):
                if f >= traj_len:
                    break
                expected_val = token_mask_arr[k]
                actual_val = traj_mask_arr[f] if f < len(traj_mask_arr) else 0.0
                if abs(actual_val - expected_val) > 1e-4:
                    errors.append(
                        f"token[{k}]={expected_val:.0f} but traj_mask[{f}]={actual_val:.2f}"
                    )

        ok = len(errors) == 0
        detail = f"token_len={tok_len}  traj_len={traj_len}"
        if errors:
            detail += f"  ERRORS({len(errors)}): " + "; ".join(errors[:5])
            if len(errors) > 5:
                detail += f" ... (+{len(errors)-5} more)"
            if verbose:
                detail += f"\n      token_mask = {token_mask_arr.tolist()}"
                detail += f"\n      traj_mask  = {traj_mask_arr.tolist()}"
        results.append(("traj_mask causal expansion", ok, detail))

    return name, results


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config(args.config).config
    dataset = instantiate(
        cfg.data.get("test_target", cfg.data.target), cfg=cfg, split=args.split
    )

    indices = list(range(len(dataset)))
    if args.num_samples > 0 and args.num_samples < len(indices):
        indices = random.sample(indices, args.num_samples)
    indices.sort()

    print(f"===== Data Alignment Check =====")
    print(f"config: {args.config}")
    print(f"split:  {args.split}")
    print(f"samples checked: {len(indices)} / {len(dataset)}")
    print()

    total_checks = 0
    failed_checks = 0
    failed_samples = []

    for idx in indices:
        sample = dataset[idx]
        name, results = check_sample(sample, verbose=args.verbose)
        sample_fail = False
        for check_name, ok, detail in results:
            total_checks += 1
            if not ok:
                failed_checks += 1
                sample_fail = True
                print(f"  [FAIL] {name}  |  {check_name}")
                print(f"         {detail}")
        if sample_fail:
            failed_samples.append(name)

    print()
    print(f"===== Summary =====")
    print(f"Total checks : {total_checks}")
    print(f"Passed       : {total_checks - failed_checks}")
    print(f"Failed       : {failed_checks}")
    if failed_samples:
        print(f"Failed samples ({len(failed_samples)}): {failed_samples[:10]}", end="")
        if len(failed_samples) > 10:
            print(f"  ... (+{len(failed_samples)-10} more)")
        else:
            print()
    else:
        print("All checks passed ✓")


if __name__ == "__main__":
    main()
