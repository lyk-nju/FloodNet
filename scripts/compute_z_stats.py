"""Compute VAE latent z stats (z_mean, z_std) for history corruption (T_B_02).

Walks a pretokenize cache of per-clip VAE latents and accumulates per-channel
Welford mean/std over all latent vectors. Output:

    deps/body_stats/z_mean.npy   [D] float32
    deps/body_stats/z_std.npy    [D] float32

These are loaded into the Wan model via `WanModel.load_z_stats(stats_dir)` and
used (with `mask_emb`) by the T_B_03 history-corruption augmentation.

CLI:
    python scripts/compute_z_stats.py \
        --pretokenize_cache <dir of *.npy latents> \
        --output_dir deps/body_stats/ \
        [--channel_axis -1] [--max_files -1]

Each cached file is a numpy array whose `channel_axis` is the latent channel
dim D (default last axis); all other axes are flattened into the sample count.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.compute_5d_stats import WelfordAccumulator   # noqa: E402  (reuse)

log = logging.getLogger(__name__)


def iter_latent_files(cache_dir: str | Path) -> list[Path]:
    """Sorted list of *.npy latent files under `cache_dir` (non-recursive)."""
    return sorted(Path(cache_dir).glob("*.npy"))


def compute_z_stats(cache_dir: str | Path,
                    *,
                    channel_axis: int = -1,
                    max_files: int = -1) -> tuple[np.ndarray, np.ndarray, int]:
    """Accumulate per-channel Welford stats over all latent vectors.

    Returns (z_mean [D], z_std [D], n_vectors). Raises FileNotFoundError if the
    cache has no .npy files.
    """
    files = iter_latent_files(cache_dir)
    if not files:
        raise FileNotFoundError(f"no .npy latent files under {cache_dir}")

    acc: WelfordAccumulator | None = None
    n_files = 0
    for f in files:
        if max_files > 0 and n_files >= max_files:
            break
        arr = np.load(f).astype(np.float64)
        # Move the channel axis to last, flatten everything else to [N, D].
        arr = np.moveaxis(arr, channel_axis, -1)
        D = arr.shape[-1]
        flat = arr.reshape(-1, D)
        if acc is None:
            acc = WelfordAccumulator(dim=D)
        elif acc.dim != D:
            raise ValueError(
                f"inconsistent latent channel dim: file {f} has D={D}, "
                f"expected {acc.dim}"
            )
        acc.update_batch(flat)
        n_files += 1

    mean, std = acc.finalize()
    log.info("z stats over %d files / %d vectors: mean=%s std=%s",
             n_files, acc.n, mean, std)
    return mean, std, acc.n


def save_z_stats(z_mean: np.ndarray, z_std: np.ndarray, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "z_mean.npy", z_mean.astype(np.float32))
    np.save(out / "z_std.npy", z_std.astype(np.float32))
    log.info("saved z_mean/z_std to %s", out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretokenize_cache", type=str, required=True,
                         help="Directory of per-clip VAE latent *.npy files.")
    parser.add_argument("--output_dir", type=str, default="deps/body_stats")
    parser.add_argument("--channel_axis", type=int, default=-1,
                         help="Axis that holds the latent channel dim D (default last).")
    parser.add_argument("--max_files", type=int, default=-1,
                         help="-1 = all; positive = first N files (dry-run).")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level)
    z_mean, z_std, n = compute_z_stats(
        args.pretokenize_cache,
        channel_axis=args.channel_axis,
        max_files=args.max_files,
    )
    save_z_stats(z_mean, z_std, args.output_dir)
    log.info("done: %d vectors, D=%d", n, z_mean.shape[0])


if __name__ == "__main__":
    main()
