"""Visualization helpers for runtime stream benchmarks."""

from __future__ import annotations

import shutil

import numpy as np

from eval.common.artifacts import ensure_dir
from eval.common.visualization import (
    plot_xz_trajectories,
    plot_yaw_series,
    xz_from_path,
    yaw_from_root_path,
)
from eval.runtime.metrics import estimate_body_yaw
from utils.motion_process import convert_motion_to_joints
from utils.token_frame import token_start_frame
from utils.visualization.skeleton import get_humanml3d_chains, render_simple_skeleton_video


def _motion_video_overlay_kwargs(target_root):
    """Build trajectory overlay kwargs for the skeleton action video."""
    render_setting = {
        "cond_traj_show_full": True,
        "traj_mask_point_radius": 3,
        "cond_traj_point_radius": 4,
    }
    kwargs = {"render_setting": render_setting}
    if target_root is None:
        return kwargs
    target_root = np.asarray(target_root, dtype=np.float32)
    if target_root.ndim != 2 or target_root.shape[0] == 0 or target_root.shape[1] < 3:
        return kwargs
    kwargs["traj_xz"] = target_root[:, [0, 2]]
    kwargs["traj_mask"] = np.ones((target_root.shape[0],), dtype=np.float32)
    return kwargs


def _transform_joints_to_runtime_world(
    joint_positions,
    *,
    pred_root,
    pred_yaw_offset: float = 0.0,
) -> np.ndarray:
    """Put rendered joints in the same world frame as runtime metrics."""

    joints = np.asarray(joint_positions, dtype=np.float32).copy()
    pred = np.asarray(pred_root, dtype=np.float32)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"joint_positions must be [T,J,3], got {joints.shape}")
    if pred.ndim != 2 or pred.shape[-1] < 3:
        raise ValueError(f"pred_root must be [T,>=3], got {pred.shape}")
    n = min(len(joints), len(pred))
    joints = joints[:n]
    pred = pred[:n]
    yaw = float(pred_yaw_offset)
    if abs(yaw) > 1e-8:
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        x = joints[..., 0].copy()
        z = joints[..., 2].copy()
        joints[..., 0] = c * x + s * z
        joints[..., 2] = -s * x + c * z
    root_xz = joints[:, 0, [0, 2]]
    offset_xz = pred[:, [0, 2]] - root_xz
    offset_y = pred[:, 1] - joints[:, 0, 1]
    joints[:, :, 0] += offset_xz[:, 0, None]
    joints[:, :, 2] += offset_xz[:, 1, None]
    joints[:, :, 1] += offset_y[:, None]
    return joints.astype(np.float32)


def _render_runtime_case_video(
    *,
    motion_263,
    pred_root,
    target_root,
    save_path,
    pred_yaw_offset: float = 0.0,
):
    overlay = _motion_video_overlay_kwargs(target_root)
    render_setting = overlay.get("render_setting", {})
    joints = convert_motion_to_joints(np.asarray(motion_263), dim=263)
    joints_world = _transform_joints_to_runtime_world(
        joints,
        pred_root=pred_root,
        pred_yaw_offset=pred_yaw_offset,
    )
    traj_xz = overlay.get("traj_xz")
    if traj_xz is not None:
        traj_xz = np.asarray(traj_xz, dtype=np.float32)[: len(joints_world)]
    traj_mask = overlay.get("traj_mask")
    if traj_mask is not None:
        traj_mask = np.asarray(traj_mask, dtype=np.float32)[: len(joints_world)]
    render_simple_skeleton_video(
        data=joints_world,
        chains=get_humanml3d_chains(),
        out_path=str(save_path),
        fps=render_setting.get("fps", 20),
        traj_mask=traj_mask,
        traj_mask_point_radius=int(render_setting.get("traj_mask_point_radius", 3)),
        traj_xz=traj_xz,
        cond_traj_mask=traj_mask,
        cond_traj_point_radius=int(render_setting.get("cond_traj_point_radius", 4)),
        cond_traj_show_full=bool(render_setting.get("cond_traj_show_full", True)),
    )


def _render_traj_video(pred_root, target_root, out_path, title, *, split_tok=None):
    """Render animated XZ trajectory comparison video."""
    _n = min(len(pred_root), len(target_root))
    if _n <= 1:
        return
    import matplotlib
    matplotlib.use("Agg")
    if shutil.which("ffmpeg") is None:
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    _f2, _a2 = plt.subplots(figsize=(7, 7))
    _all_x = [target_root[:_n, 0], pred_root[:_n, 0]]
    _all_z = [target_root[:_n, 2], pred_root[:_n, 2]]
    _xl = (min(a.min() for a in _all_x) - 0.5, max(a.max() for a in _all_x) + 0.5)
    _zl = (min(a.min() for a in _all_z) - 0.5, max(a.max() for a in _all_z) + 0.5)
    _wr = FFMpegWriter(fps=20)
    _sf = max(1, _n // 150)
    with _wr.saving(_f2, out_path, dpi=100):
        for _f in range(1, _n + 1, _sf):
            _a2.clear()
            _a2.plot(target_root[:min(_f, _n), 0], target_root[:min(_f, _n), 2],
                     "g-", lw=1.5, alpha=0.7, label="target")
            _a2.plot(pred_root[:min(_f, _n), 0], pred_root[:min(_f, _n), 2],
                     "r-", lw=1.5, alpha=0.7, label="pred")
            _a2.plot(target_root[0, 0], target_root[0, 2], "go", ms=6)
            if split_tok is not None:
                _sb = max(1, 1 + 4 * (split_tok - 1))
                if 0 < _sb < _n:
                    _a2.axvline(x=target_root[min(_sb, _n - 1), 0],
                                color="gray", ls="--", alpha=0.5, label="split")
            _a2.set_xlim(_xl); _a2.set_ylim(_zl)
            _a2.set_aspect("equal")
            _a2.legend(loc="upper right")
            _a2.set_title(f"{title}  f{min(_f,_n)}/{_n}")
            _wr.grab_frame()
    plt.close(_f2)


def _sample_label_frames(*paths) -> list[int]:
    max_len = max((len(xz_from_path(path)) for path in paths if path is not None), default=0)
    if max_len <= 0:
        return []
    candidates = [0, 20, 45, 70, 93, 124, max_len - 1]
    return sorted({int(idx) for idx in candidates if 0 <= int(idx) < max_len})


def _write_path_debug_xz(
    output_dir,
    *,
    case_name: str,
    input_path_root=None,
    root_condition_root=None,
    pred_root=None,
    input_waypoints_xyz=None,
    gt_root=None,
    boundary_frames=None,
) -> None:
    """Write a static XZ comparison of path input, LDF condition, and output."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out = ensure_dir(output_dir)
    input_path_xz = xz_from_path(input_path_root)
    root_condition_xz = xz_from_path(root_condition_root)
    pred_motion_xz = xz_from_path(pred_root)
    input_waypoints_xz = xz_from_path(input_waypoints_xyz)
    gt_motion_xz = xz_from_path(gt_root)

    np.savez(
        out / f"{case_name}_path_debug_xz.npz",
        input_path_xz=input_path_xz.astype(np.float32, copy=False),
        root_condition_xz=root_condition_xz.astype(np.float32, copy=False),
        pred_motion_xz=pred_motion_xz.astype(np.float32, copy=False),
        input_waypoints_xz=input_waypoints_xz.astype(np.float32, copy=False),
        gt_motion_xz=gt_motion_xz.astype(np.float32, copy=False),
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    any_points = False

    def _plot_line(label, xz, *, color, marker, linestyle="-", alpha=0.9, linewidth=1.6):
        nonlocal any_points
        if xz.shape[0] <= 0:
            return
        any_points = True
        ax.plot(
            xz[:, 0],
            xz[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
        )
        ax.scatter(
            xz[:, 0],
            xz[:, 1],
            s=14,
            color=color,
            marker=marker,
            alpha=min(1.0, alpha + 0.05),
        )

    _plot_line("input_path_xz", input_path_xz, color="#d62728", marker="o", alpha=0.65)
    _plot_line(
        "root_condition_xz",
        root_condition_xz,
        color="#1f77b4",
        marker="^",
        linestyle="--",
        alpha=0.78,
    )
    _plot_line("pred_motion_xz", pred_motion_xz, color="#2ca02c", marker=".", alpha=0.78)
    if gt_motion_xz.shape[0] > 0:
        _plot_line(
            "gt_motion_xz",
            gt_motion_xz,
            color="0.35",
            marker="x",
            linestyle=":",
            alpha=0.55,
            linewidth=1.1,
        )
    if input_waypoints_xz.shape[0] > 0:
        any_points = True
        ax.scatter(
            input_waypoints_xz[:, 0],
            input_waypoints_xz[:, 1],
            s=34,
            marker="x",
            linewidths=1.4,
            color="#d62728",
            label="input_waypoints_xz",
        )

    label_frames = _sample_label_frames(input_path_xz, root_condition_xz, pred_motion_xz)
    for idx in label_frames:
        if idx < root_condition_xz.shape[0]:
            ax.text(
                root_condition_xz[idx, 0],
                root_condition_xz[idx, 1],
                f"c{idx}",
                color="#1f77b4",
                fontsize=7,
            )
        if idx < pred_motion_xz.shape[0]:
            ax.text(
                pred_motion_xz[idx, 0],
                pred_motion_xz[idx, 1],
                f"p{idx}",
                color="#2ca02c",
                fontsize=7,
            )

    for frame in boundary_frames or []:
        idx = int(frame)
        if 0 <= idx < root_condition_xz.shape[0]:
            ax.scatter(
                root_condition_xz[idx, 0],
                root_condition_xz[idx, 1],
                s=52,
                marker="s",
                facecolors="none",
                edgecolors="#1f77b4",
                linewidths=1.2,
            )

    if not any_points:
        ax.text(0.5, 0.5, "no trajectory", ha="center", va="center")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"{case_name} path debug")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    if any_points:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{case_name}_path_debug_xz.png", dpi=140)
    plt.close(fig)


def _write_runtime_case_visuals(
    output_dir,
    *,
    case_name: str,
    pred_root,
    target_root,
    input_path_root=None,
    root_condition_root=None,
    input_waypoints_xyz=None,
    gt_root=None,
    motion_263=None,
    split_tok: int | None = None,
    pred_yaw_offset: float = 0.0,
) -> None:
    out = ensure_dir(output_dir)
    split_frame = None if split_tok is None else int(token_start_frame(int(split_tok)))
    boundary_frames = [] if split_frame is None else [split_frame]
    plot_xz_trajectories(
        out / f"{case_name}_plot_world_xz.png",
        {
            "target": target_root,
            "pred": pred_root,
        },
        title=str(case_name),
        boundary_frames=boundary_frames,
    )
    _write_path_debug_xz(
        out,
        case_name=case_name,
        input_path_root=target_root if input_path_root is None else input_path_root,
        root_condition_root=root_condition_root,
        pred_root=pred_root,
        input_waypoints_xyz=input_waypoints_xyz,
        gt_root=gt_root,
        boundary_frames=boundary_frames,
    )
    if motion_263 is not None:
        try:
            pred_yaw = estimate_body_yaw(np.asarray(motion_263)) + float(pred_yaw_offset)
        except Exception:
            pred_yaw = yaw_from_root_path(pred_root)
    else:
        pred_yaw = yaw_from_root_path(pred_root)
    plot_yaw_series(
        out / f"{case_name}_plot_yaw.png",
        {
            "target_yaw": yaw_from_root_path(target_root),
            "pred_yaw": pred_yaw,
        },
        title=str(case_name),
        boundary_frames=boundary_frames,
    )



__all__ = [
    "_render_runtime_case_video",
    "_render_traj_video",
    "_write_path_debug_xz",
    "_write_runtime_case_visuals",
]
