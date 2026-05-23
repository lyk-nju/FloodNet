"""Compute per-channel mean / std for RefinerDataset outputs (T_A_06).

Walks `RefinerDataset` with `normalize=False` and accumulates Welford
statistics over **canonicalized local-frame** values, separately for
`current_motion` (5-dim) and `target_waypoints` (7-dim). Only positions where
the corresponding mask (`history_mask` / `target_mask`) is True are counted.

Stats are produced for ALL channels (Welford is uniform), but z-score
application is selective: cos / sin heading channels [3], [4] are NOT
normalized (they live on the unit circle by construction). The selected
channels are saved as `*_norm_indices.npy` alongside the mean / std files:

    current_motion_mean.npy            [5] float32
    current_motion_std.npy             [5] float32
    current_motion_norm_indices.npy    [3] int64  = [0, 1, 2]    (xyz only)
    waypoint_mean.npy                  [7] float32
    waypoint_std.npy                   [7] float32
    waypoint_norm_indices.npy          [5] int64  = [0, 1, 2, 5, 6]
                                                   (xyz + fwd_delta + yaw_delta)

References:
- docs/TODO.md §T_A_06 lines 1211-1277 — full spec.
- docs/TODO.md §T_A_04 T13 — cos/sin invariance under normalize (lock-in test).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Insert repo root so `python scripts/compute_5d_stats.py` can `import datasets.*`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.refiner_dataset import RefinerDataset   # noqa: E402

log = logging.getLogger(__name__)

CURRENT_MOTION_NORM_INDICES = np.array([0, 1, 2], dtype=np.int64)
WAYPOINT_NORM_INDICES = np.array([0, 1, 2, 5, 6], dtype=np.int64)


class WelfordAccumulator:
    """Online per-channel mean + variance (parallel Welford / Chan).

    All channels share a single sample count `n` (consistent with how we feed
    valid frames as a [n_valid, D] block per update — every channel in a valid
    frame is observed simultaneously).
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)

    def update_batch(self, values: np.ndarray) -> None:
        """`values`: shape [n_batch, dim]. Empty / 0-row updates are no-ops."""
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError(
                f"WelfordAccumulator(dim={self.dim}) update_batch expected "
                f"[n, {self.dim}], got {values.shape}"
            )
        n_batch = values.shape[0]
        if n_batch == 0:
            return
        # Parallel Welford merge (numerically stable for very large n).
        new_n = self.n + n_batch
        batch_mean = values.mean(axis=0)
        delta = batch_mean - self.mean
        batch_var_sum = ((values - batch_mean) ** 2).sum(axis=0)
        # M2_combined = M2_a + M2_b + delta^2 * n_a * n_b / (n_a + n_b)
        if self.n == 0:
            self.M2 = batch_var_sum
        else:
            self.M2 = self.M2 + batch_var_sum + (delta ** 2) * (self.n * n_batch / new_n)
        # mean_combined = (n_a * mean_a + n_b * mean_b) / new_n
        self.mean = self.mean + delta * (n_batch / new_n)
        self.n = new_n

    def finalize(self, *, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std). std is clamped to >= eps so downstream divide
        is safe even on constant channels."""
        if self.n == 0:
            return self.mean.copy(), np.full(self.dim, eps, dtype=np.float64)
        var = self.M2 / self.n
        std = np.sqrt(np.maximum(var, eps * eps))
        return self.mean.copy(), std


def compute_stats(dataset: RefinerDataset,
                   max_samples: int = -1,
                   progress: bool = False) -> dict:
    """Walk `dataset` and accumulate Welford stats over masked positions.

    Returns dict with keys:
      current_motion_mean, current_motion_std (each [5])
      waypoint_mean, waypoint_std (each [7])
      n_current_motion, n_waypoint (sample counts)
    """
    cm_acc = WelfordAccumulator(dim=5)
    wp_acc = WelfordAccumulator(dim=7)

    n_iter = len(dataset) if max_samples < 0 else min(max_samples, len(dataset))
    log.info("computing stats over %d samples", n_iter)

    progress_iter = range(n_iter)
    try:
        from tqdm import tqdm
        if progress:
            progress_iter = tqdm(progress_iter, desc="compute_5d_stats")
    except ImportError:
        pass

    for idx in progress_iter:
        sample = dataset[idx]
        cm = sample["current_motion"]       # [n_hist, 5]
        cm_mask = sample["history_mask"]    # [n_hist] bool
        wp = sample["target_waypoints"]     # [max_frames, 7]
        wp_mask = sample["target_mask"]     # [max_frames] bool

        cm_valid = cm[cm_mask].detach().cpu().numpy().astype(np.float64)
        wp_valid = wp[wp_mask].detach().cpu().numpy().astype(np.float64)
        cm_acc.update_batch(cm_valid)
        wp_acc.update_batch(wp_valid)

    cm_mean, cm_std = cm_acc.finalize()
    wp_mean, wp_std = wp_acc.finalize()
    return {
        "current_motion_mean": cm_mean,
        "current_motion_std": cm_std,
        "waypoint_mean": wp_mean,
        "waypoint_std": wp_std,
        "n_current_motion": cm_acc.n,
        "n_waypoint": wp_acc.n,
    }


def save_stats(stats: dict, output_dir: str | Path) -> None:
    """Persist stats + norm_indices files. Creates `output_dir` if needed."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "current_motion_mean.npy", stats["current_motion_mean"].astype(np.float32))
    np.save(out / "current_motion_std.npy", stats["current_motion_std"].astype(np.float32))
    np.save(out / "current_motion_norm_indices.npy", CURRENT_MOTION_NORM_INDICES)
    np.save(out / "waypoint_mean.npy", stats["waypoint_mean"].astype(np.float32))
    np.save(out / "waypoint_std.npy", stats["waypoint_std"].astype(np.float32))
    np.save(out / "waypoint_norm_indices.npy", WAYPOINT_NORM_INDICES)
    log.info("saved stats to %s (n_current=%d, n_waypoint=%d)",
             out, stats["n_current_motion"], stats["n_waypoint"])


def load_clips_from_dir(raw_data_dir: str | Path) -> list[dict]:
    """Load HumanML3D-style clips from `raw_data_dir`.

    Expected layout (extend at deployment):
      raw_data_dir/
        clips_meta.txt           — one clip name per line
        motions/<name>.npy       — (T, 263) float32 motion features
        texts/<name>.txt         — one or more text lines (first non-empty used)

    This is a permissive stub: production deployments often have a specific
    layout (per-dataset). Override / replace this function at integration time.
    """
    root = Path(raw_data_dir)
    meta = root / "clips_meta.txt"
    if not meta.is_file():
        raise FileNotFoundError(
            f"expected meta file {meta} (one clip name per line)"
        )
    clips: list[dict] = []
    with meta.open() as f:
        names = [ln.strip() for ln in f if ln.strip()]
    for name in names:
        motion_path = root / "motions" / f"{name}.npy"
        text_path = root / "texts" / f"{name}.txt"
        if not motion_path.is_file():
            log.warning("missing motion file %s, skipping clip", motion_path)
            continue
        motion = np.load(motion_path).astype(np.float32)
        if motion.ndim != 2 or motion.shape[1] != 263:
            log.warning("clip %s has unexpected shape %s, skipping", name, motion.shape)
            continue
        text = ""
        if text_path.is_file():
            with text_path.open() as tf:
                for line in tf:
                    line = line.strip()
                    if line:
                        text = line
                        break
        clips.append({"motion_263": torch.from_numpy(motion), "text": text})
    return clips


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="deps/refiner_stats")
    parser.add_argument("--max_samples", type=int, default=-1,
                         help="-1 = all; positive integer = first N (dry-run)")
    parser.add_argument("--seed", type=int, default=0,
                         help="Dataset RNG seed for reproducibility.")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level)

    log.info("loading clips from %s", args.raw_data_dir)
    clips = load_clips_from_dir(args.raw_data_dir)
    log.info("loaded %d clips", len(clips))

    dataset = RefinerDataset(clips, normalize=False, seed=args.seed)
    log.info("dataset has %d eligible samples", len(dataset))

    stats = compute_stats(dataset, max_samples=args.max_samples, progress=True)
    save_stats(stats, args.output_dir)

    # Sanity logging.
    log.info("current_motion_mean=%s", stats["current_motion_mean"])
    log.info("current_motion_std=%s", stats["current_motion_std"])
    log.info("waypoint_mean=%s", stats["waypoint_mean"])
    log.info("waypoint_std=%s", stats["waypoint_std"])


if __name__ == "__main__":
    main()
