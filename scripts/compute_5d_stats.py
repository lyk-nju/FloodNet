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

from datasets.humanml3d_refiner import HumanML3DRefinerDataset as RefinerDataset   # noqa: E402

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


# Per-dataset layout defaults, derived from the real repo structure
# (configs/ldf.yaml + datasets/humanml3d.py / datasets/babel.py):
#   <dataset_dir>/<split_file>           — one sample id per line
#   <dataset_dir>/<feature_path>/<id>.npy — (T, 263) motion features
#   <dataset_dir>/<text_path>/<id>.txt    — '#'-delimited caption lines
# HumanML3D feature dir is `new_joint_vecs`; BABEL_streamed is `motions`.
DATASET_DEFAULTS: dict[str, dict] = {
    "humanml3d": {
        "subdir": "HumanML3D",
        "feature_path": "new_joint_vecs",
        "text_path": "texts",
        "split_file": "train.txt",
    },
    "babel": {
        "subdir": "BABEL_streamed",
        "feature_path": "motions",        # ⚠ NOT new_joint_vecs (HumanML3D only)
        "text_path": "texts",
        "split_file": "train_processed.txt",
    },
}


def resolve_dataset_dir(raw_data_dir: str | Path, dataset: str) -> Path:
    """Resolve the concrete dataset directory.

    Accepts either the raw_data ROOT (which contains a `HumanML3D` /
    `BABEL_streamed` subdir) OR the dataset directory itself. If
    `<raw_data_dir>/<subdir>` exists it is used; otherwise `raw_data_dir` is
    assumed to already be the dataset dir.
    """
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {list(DATASET_DEFAULTS)}")
    root = Path(raw_data_dir)
    candidate = root / DATASET_DEFAULTS[dataset]["subdir"]
    if candidate.is_dir():
        return candidate
    return root


def _read_all_captions(text_file: Path) -> list[str]:
    """All distinct non-empty captions from a HumanML3D/BABEL text file.

    Text files are `#`-delimited: `caption#tokens#f_tag#to_tag` (see
    datasets/humanml3d.py:load_text). Each line contributes the part before the
    first `#` (or the whole line if no `#`). Duplicates are dropped while
    preserving first-seen order, so `result[0]` is the first distinct caption.
    Missing/empty file → empty list.
    """
    if not text_file.is_file():
        return []
    seen: set[str] = set()
    caps: list[str] = []
    with text_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cap = line.split("#", 1)[0].strip()
            if cap and cap not in seen:
                seen.add(cap)
                caps.append(cap)
    return caps


def load_clips_from_dir(raw_data_dir: str | Path,
                         *,
                         dataset: str = "humanml3d",
                         split_file: str | None = None,
                         feature_path: str | None = None,
                         text_path: str | None = None,
                         max_samples: int = -1) -> list[dict]:
    """Load clips from the real HumanML3D / BABEL layout.

    Returns a list of `{"motion_263": Tensor[T, 263], "text": str, "texts": list[str]}`.
    `text` is the first caption (backward compatible); `texts` holds every
    distinct caption so RefinerDataset can randomly pick one per epoch (text
    augmentation). The precomputed T5 cache contains all captions (see
    tools/pretokenize_t5_text.py), so the encoder lookup always hits.

    Layout (per dataset):
        <dataset_dir>/<split_file>            one sample id per line
        <dataset_dir>/<feature_path>/<id>.npy  (T, 263) motion features
        <dataset_dir>/<text_path>/<id>.txt     '#'-delimited caption lines

    `raw_data_dir` may be the raw_data root (containing `HumanML3D` /
    `BABEL_streamed`) or the dataset dir itself (see `resolve_dataset_dir`).
    `split_file` / `feature_path` / `text_path` default per-dataset
    (DATASET_DEFAULTS). `max_samples > 0` stops after that many valid clips
    (dry-run / stats speed-up). Samples with a missing / malformed motion file
    are skipped with a warning; a missing text file yields an empty caption.
    """
    defaults = DATASET_DEFAULTS.get(dataset)
    if defaults is None:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {list(DATASET_DEFAULTS)}")
    split_file = split_file or defaults["split_file"]
    feature_path = feature_path or defaults["feature_path"]
    text_path = text_path or defaults["text_path"]

    dataset_dir = resolve_dataset_dir(raw_data_dir, dataset)
    split_path = dataset_dir / split_file
    if not split_path.is_file():
        raise FileNotFoundError(
            f"split file not found: {split_path} "
            f"(dataset={dataset}, dataset_dir={dataset_dir})"
        )

    with split_path.open() as f:
        names = [ln.strip() for ln in f if ln.strip()]

    feature_dir = dataset_dir / feature_path
    text_dir = dataset_dir / text_path

    clips: list[dict] = []
    n_missing_motion = 0
    n_bad_shape = 0
    for name in names:
        if max_samples > 0 and len(clips) >= max_samples:
            break
        motion_file = feature_dir / f"{name}.npy"
        if not motion_file.is_file():
            n_missing_motion += 1
            log.warning("missing motion file %s, skipping clip", motion_file)
            continue
        motion = np.load(motion_file).astype(np.float32)
        if motion.ndim != 2 or motion.shape[1] != 263:
            n_bad_shape += 1
            log.warning("clip %s has unexpected shape %s (expected [T, 263]), skipping",
                        name, motion.shape)
            continue
        texts = _read_all_captions(text_dir / f"{name}.txt")
        clips.append({
            "motion_263": torch.from_numpy(motion),
            "text": texts[0] if texts else "",   # first caption (backward compat)
            "texts": texts,                       # all captions (RefinerDataset random pick)
        })

    if not clips:
        log.warning(
            "load_clips_from_dir produced 0 clips (dataset=%s, split=%s, "
            "dataset_dir=%s, names_in_split=%d, missing_motion=%d, bad_shape=%d)",
            dataset, split_file, dataset_dir, len(names), n_missing_motion, n_bad_shape,
        )
    else:
        log.info("loaded %d clips (dataset=%s, split=%s, missing_motion=%d, bad_shape=%d)",
                 len(clips), dataset, split_file, n_missing_motion, n_bad_shape)
    return clips


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_data_dir", type=str, required=True,
                         help="raw_data root (contains HumanML3D / BABEL_streamed) "
                              "OR the dataset dir itself.")
    parser.add_argument("--dataset", type=str, default="humanml3d",
                         choices=sorted(DATASET_DEFAULTS.keys()))
    parser.add_argument("--split_file", type=str, default=None,
                         help="e.g. train.txt (humanml3d) / train_processed.txt (babel). "
                              "Default per-dataset.")
    parser.add_argument("--feature_path", type=str, default=None,
                         help="feature subdir; default new_joint_vecs (humanml3d) / motions (babel).")
    parser.add_argument("--text_path", type=str, default=None,
                         help="text subdir; default texts.")
    parser.add_argument("--output_dir", type=str, default="deps/refiner_stats")
    parser.add_argument("--max_samples", type=int, default=-1,
                         help="-1 = all; positive integer = first N (dry-run)")
    parser.add_argument("--seed", type=int, default=0,
                         help="Dataset RNG seed for reproducibility.")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level)

    log.info("loading clips from %s (dataset=%s)", args.raw_data_dir, args.dataset)
    clips = load_clips_from_dir(
        args.raw_data_dir,
        dataset=args.dataset,
        split_file=args.split_file,
        feature_path=args.feature_path,
        text_path=args.text_path,
        max_samples=args.max_samples,
    )
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
