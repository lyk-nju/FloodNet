"""Artifacts for LDF condition-source experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch


def _to_traj7_array(value) -> np.ndarray:
    if value is None:
        return np.zeros((0, 7), dtype=np.float32)
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def plot_7d_xz_heading(
    output_path: str | Path,
    series: Mapping[str, torch.Tensor | np.ndarray],
    *,
    update_frames: list[int] | tuple[int, ...] | None = None,
    title: str,
    arrow_count: int = 28,
) -> Path:
    """Plot XZ positions with heading arrows for one or more 7D trajectories."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    for label, traj in series.items():
        arr = _to_traj7_array(traj)
        if arr.ndim != 2 or arr.shape[0] <= 0:
            continue
        if arr.shape[1] < 5:
            raise ValueError(f"{label} must have at least 5 columns, got {arr.shape}")
        xz = arr[:, [0, 2]]
        ax.plot(xz[:, 0], xz[:, 1], linewidth=1.4, label=label)
        ax.scatter(xz[:, 0], xz[:, 1], s=8, alpha=0.55)
        ax.scatter(xz[0, 0], xz[0, 1], s=36, marker="o")
        ax.scatter(xz[-1, 0], xz[-1, 1], s=42, marker="s")
        step = max(1, int(arr.shape[0]) // max(1, int(arrow_count)))
        idx = np.arange(0, arr.shape[0], step, dtype=np.int64)
        if idx.size == 0 or idx[-1] != arr.shape[0] - 1:
            idx = np.concatenate([idx, np.asarray([arr.shape[0] - 1], dtype=np.int64)])
        yaw = np.arctan2(arr[idx, 4], arr[idx, 3])
        scale = 0.12
        ax.quiver(
            xz[idx, 0],
            xz[idx, 1],
            np.sin(yaw) * scale,
            np.cos(yaw) * scale,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
            alpha=0.8,
        )
    for frame in update_frames or []:
        frame_idx = int(frame)
        drawn = False
        for traj in series.values():
            arr = _to_traj7_array(traj)
            if arr.ndim == 2 and 0 <= frame_idx < arr.shape[0] and arr.shape[1] >= 3:
                ax.scatter(
                    arr[frame_idx, 0],
                    arr[frame_idx, 2],
                    s=70,
                    marker="x",
                    color="black",
                    zorder=6,
                )
                drawn = True
                break
        if drawn:
            ax.annotate(
                f"update {frame_idx}",
                xy=(0.02, 0.96),
                xycoords="axes fraction",
                fontsize=8,
                alpha=0.75,
            )
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


__all__ = ["plot_7d_xz_heading"]
