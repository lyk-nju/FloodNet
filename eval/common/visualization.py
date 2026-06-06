"""Shared visualization helpers for eval artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from eval.common.artifacts import ensure_dir


def _as_numpy(value) -> np.ndarray:
    if value is None:
        return np.asarray([])
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except Exception:  # pragma: no cover - torch is available in normal eval.
        pass
    return np.asarray(value)


def xz_from_path(value) -> np.ndarray:
    arr = _as_numpy(value).astype(np.float32, copy=False)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] >= 3:
        return arr[..., [0, 2]].reshape(-1, 2)
    if arr.shape[-1] >= 2:
        return arr[..., :2].reshape(-1, 2)
    return np.zeros((0, 2), dtype=np.float32)


def yaw_from_7d(value) -> np.ndarray:
    arr = _as_numpy(value).astype(np.float32, copy=False)
    if arr.size == 0 or arr.ndim == 0 or arr.shape[-1] < 5:
        return np.zeros((0,), dtype=np.float32)
    return np.arctan2(arr[..., 4], arr[..., 3]).reshape(-1).astype(np.float32)


def yaw_from_root_path(value) -> np.ndarray:
    xz = xz_from_path(value)
    if xz.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if xz.shape[0] == 1:
        return np.zeros((1,), dtype=np.float32)
    vel = np.diff(xz, axis=0, prepend=xz[:1])
    yaw = np.arctan2(vel[:, 0], vel[:, 1])
    if yaw.shape[0] > 1:
        yaw[0] = yaw[1]
    return yaw.astype(np.float32)


def plot_xz_trajectories(
    output_path: str | Path,
    series: Mapping[str, object],
    *,
    point_series: Mapping[str, object] | None = None,
    title: str | None = None,
    boundary_frames: Sequence[int] | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out = Path(output_path)
    ensure_dir(out.parent)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    any_points = False
    for label, value in series.items():
        xz = xz_from_path(value)
        if xz.shape[0] == 0:
            continue
        any_points = True
        ax.plot(xz[:, 0], xz[:, 1], linewidth=1.6, label=str(label))
        ax.scatter(xz[0, 0], xz[0, 1], s=18)
        if boundary_frames:
            for frame in boundary_frames:
                idx = int(frame)
                if 0 <= idx < xz.shape[0]:
                    ax.scatter(
                        xz[idx, 0],
                        xz[idx, 1],
                        s=28,
                        marker="x",
                        color="0.25",
                    )
    if point_series:
        for label, value in point_series.items():
            xz = xz_from_path(value)
            if xz.shape[0] == 0:
                continue
            any_points = True
            ax.scatter(
                xz[:, 0],
                xz[:, 1],
                s=34,
                marker="x",
                linewidths=1.4,
                label=str(label),
            )
    if not any_points:
        ax.text(0.5, 0.5, "no trajectory", ha="center", va="center")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="datalim")
    if title:
        ax.set_title(title)
    if any_points:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_yaw_series(
    output_path: str | Path,
    series: Mapping[str, object],
    *,
    title: str | None = None,
    boundary_frames: Sequence[int] | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out = Path(output_path)
    ensure_dir(out.parent)
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    any_points = False
    for label, value in series.items():
        yaw = _as_numpy(value).astype(np.float32, copy=False).reshape(-1)
        if yaw.shape[0] == 0:
            continue
        any_points = True
        ax.plot(np.arange(yaw.shape[0]), yaw, linewidth=1.4, label=str(label))
    if boundary_frames:
        for frame in boundary_frames:
            ax.axvline(int(frame), color="0.75", linestyle="--", linewidth=0.8)
    if not any_points:
        ax.text(0.5, 0.5, "no yaw", ha="center", va="center")
    ax.set_xlabel("frame")
    ax.set_ylabel("yaw rad")
    if title:
        ax.set_title(title)
    if any_points:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def render_motion_video(
    motion,
    output_path: str | Path,
    *,
    dim: int = 263,
    render_setting: dict | None = None,
    traj_xz=None,
    traj_mask=None,
) -> Path:
    from utils.visualize import render_single_video

    out = Path(output_path)
    ensure_dir(out.parent)
    render_single_video(
        motion=_as_numpy(motion),
        save_path=str(out),
        dim=dim,
        render_setting=render_setting or {},
        traj_xz=None if traj_xz is None else _as_numpy(traj_xz),
        traj_mask=None if traj_mask is None else _as_numpy(traj_mask),
    )
    return out


__all__ = [
    "plot_xz_trajectories",
    "plot_yaw_series",
    "render_motion_video",
    "yaw_from_7d",
    "yaw_from_root_path",
    "xz_from_path",
]
