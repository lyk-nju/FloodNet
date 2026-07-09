"""Plot a turn-update root condition without running LDF generation.

This is a geometry-only debug tool. It composes a new route suffix by treating
the update pose as the suffix local-frame origin, then writes trajectory plots
with heading arrows so the route condition can be inspected before inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

for _key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_key, "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from eval.ldf.stream_setup import build_eval_dataloader
from metrics.traj import _slice_single_sample_batch
from tools.run_stream_turn_update_debug import (
    _select_caption,
    compose_anchor_local_updated_traj7,
    compose_center_symmetric_updated_traj7,
)
from utils.initialize import load_config
from utils.local_frame import heading_dir_xz, wrap_angle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ldf_test.yaml")
    parser.add_argument("--meta_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_name", default="000021")
    parser.add_argument("--caption_index", type=int, default=0)
    parser.add_argument("--update_frame", type=int, default=None)
    parser.add_argument(
        "--update_lead_tokens",
        type=int,
        default=0,
        help="Move the update point this many VAE tokens before the final frame.",
    )
    parser.add_argument("--frames_per_token", type=int, default=4)
    parser.add_argument(
        "--composition_mode",
        choices=["anchor_local", "center_symmetric"],
        default="anchor_local",
    )
    parser.add_argument("--suffix_frames", type=int, default=None)
    parser.add_argument("--source_start_frame", type=int, default=0)
    parser.add_argument(
        "--transition_frames",
        type=int,
        default=0,
        help="Blend the first N suffix frames from straight entry to source route curvature.",
    )
    parser.add_argument(
        "--transition_output_frames",
        type=int,
        default=0,
        help="Resample the transition interval to this many output frames.",
    )
    parser.add_argument(
        "--anchor_yaw_policy",
        choices=["path_tangent", "heading"],
        default="path_tangent",
        help="Use incoming route tangent or source root heading as the new local frame yaw.",
    )
    parser.add_argument(
        "--derive_heading_from_path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute updated condition heading from the composed route tangent.",
    )
    parser.add_argument("--arrow_stride", type=int, default=8)
    parser.add_argument("--arrow_scale", type=float, default=0.12)
    parser.add_argument("--zoom_window", type=int, default=40)
    return parser.parse_args()


def _load_sample(args: argparse.Namespace, cfg):
    _, dataloader = build_eval_dataloader(
        cfg,
        meta_paths=[args.meta_path],
        batch_size=1,
        num_workers=0,
        group_present_segments=False,
    )
    batch = next(iter(dataloader))
    sample_batch = _slice_single_sample_batch(batch, 0)
    name = str(sample_batch["name"][0])
    if args.sample_name and name != str(args.sample_name):
        raise ValueError(f"Expected sample {args.sample_name!r}, got {name!r}")
    _select_caption(sample_batch, int(args.caption_index))
    return sample_batch


def _yaw_from_7d(traj7: torch.Tensor) -> torch.Tensor:
    return torch.atan2(traj7[:, 4], traj7[:, 3])


def _xz(traj7: torch.Tensor) -> torch.Tensor:
    return traj7[:, [0, 2]].detach().cpu()


def _angle_between_xz(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = torch.linalg.norm(a).clamp_min(1e-8)
    b_norm = torch.linalg.norm(b).clamp_min(1e-8)
    cos_v = ((a * b).sum() / (a_norm * b_norm)).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cos_v)).item())


def _boundary_metrics(original: torch.Tensor, updated: torch.Tensor, update_frame: int) -> dict:
    update = int(update_frame)
    yaw = _yaw_from_7d(updated)
    original_yaw = _yaw_from_7d(original)
    pos_jump = torch.linalg.norm(updated[update, [0, 2]] - original[update, [0, 2]])
    yaw_jump = wrap_angle(yaw[update] - original_yaw[update])
    pre_delta = updated[update, [0, 2]] - updated[update - 1, [0, 2]]
    post_delta = updated[update + 1, [0, 2]] - updated[update, [0, 2]]
    head = heading_dir_xz(yaw[update])
    return {
        "position_jump": float(pos_jump.item()),
        "yaw_jump_deg": float(torch.rad2deg(yaw_jump).item()),
        "pre_post_tangent_angle_deg": _angle_between_xz(pre_delta, post_delta),
        "heading_to_post_delta_angle_deg": _angle_between_xz(head, post_delta),
    }


def _plot_route_with_heading(
    path: Path,
    *,
    original: torch.Tensor,
    updated: torch.Tensor,
    update_frame: int,
    title: str,
    arrow_stride: int,
    arrow_scale: float,
    zoom_window: int | None = None,
) -> None:
    original_xz = _xz(original)
    updated_xz = _xz(updated)
    updated_yaw = _yaw_from_7d(updated).detach().cpu()
    heading = heading_dir_xz(updated_yaw).detach().cpu()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(original_xz[:, 0], original_xz[:, 1], color="#1f77b4", linewidth=2, label="original")
    ax.plot(updated_xz[:, 0], updated_xz[:, 1], color="#d62728", linewidth=2, label="updated_condition")
    ax.scatter(
        [updated_xz[update_frame, 0]],
        [updated_xz[update_frame, 1]],
        color="#111111",
        s=60,
        zorder=5,
        label="update anchor",
    )

    start = 0
    end = int(updated_xz.shape[0])
    if zoom_window is not None:
        start = max(0, update_frame - int(zoom_window))
        end = min(end, update_frame + int(zoom_window) + 1)
    arrow_idx = torch.arange(start, end, max(1, int(arrow_stride)))
    ax.quiver(
        updated_xz[arrow_idx, 0],
        updated_xz[arrow_idx, 1],
        heading[arrow_idx, 0] * float(arrow_scale),
        heading[arrow_idx, 1] * float(arrow_scale),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#ff7f0e",
        width=0.004,
        label="updated heading",
    )

    if zoom_window is not None:
        pad = 0.4
        view = updated_xz[start:end]
        ax.set_xlim(float(view[:, 0].min()) - pad, float(view[:, 0].max()) + pad)
        ax.set_ylim(float(view[:, 1].min()) - pad, float(view[:, 1].max()) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path=args.config)
    sample_batch = _load_sample(args, cfg)
    original_frames = int(sample_batch["feature_length"][0].item())
    original = sample_batch["traj_cond_7d"][0, :original_frames].float().cpu()
    if args.update_frame is None:
        update_frame = original_frames - 1 - int(args.update_lead_tokens) * int(args.frames_per_token)
    else:
        update_frame = int(args.update_frame)
    update_frame = max(1, min(update_frame, original_frames - 1))
    suffix_frames = int(args.suffix_frames) if args.suffix_frames is not None else original_frames
    if str(args.composition_mode) == "center_symmetric":
        updated = compose_center_symmetric_updated_traj7(
            original,
            update_frame=update_frame,
            suffix_frames=suffix_frames,
            source_start_frame=int(args.source_start_frame),
            derive_heading_from_path=bool(args.derive_heading_from_path),
            transition_frames=int(args.transition_frames),
            transition_output_frames=int(args.transition_output_frames),
        )
    else:
        updated = compose_anchor_local_updated_traj7(
            original,
            update_frame=update_frame,
            suffix_frames=suffix_frames,
            source_start_frame=int(args.source_start_frame),
            anchor_yaw_policy=str(args.anchor_yaw_policy),
            derive_heading_from_path=bool(args.derive_heading_from_path),
            transition_frames=int(args.transition_frames),
            transition_output_frames=int(args.transition_output_frames),
        )

    full_plot = out_dir / "condition_route_with_heading.png"
    zoom_plot = out_dir / "condition_route_boundary_zoom.png"
    title = f"anchor-local route update: {sample_batch['name'][0]} frame {update_frame}"
    _plot_route_with_heading(
        full_plot,
        original=original,
        updated=updated,
        update_frame=update_frame,
        title=title,
        arrow_stride=int(args.arrow_stride),
        arrow_scale=float(args.arrow_scale),
    )
    _plot_route_with_heading(
        zoom_plot,
        original=original,
        updated=updated,
        update_frame=update_frame,
        title=f"{title} boundary zoom",
        arrow_stride=max(1, int(args.arrow_stride) // 2),
        arrow_scale=float(args.arrow_scale),
        zoom_window=int(args.zoom_window),
    )
    summary = {
        "sample_name": str(sample_batch["name"][0]),
        "caption_index": sample_batch.get("_caption_index"),
        "caption_text": sample_batch.get("_caption_text"),
        "original_frames": original_frames,
        "updated_frames": int(updated.shape[0]),
        "update_frame": int(update_frame),
        "composition_mode": str(args.composition_mode),
        "suffix_frames": int(suffix_frames),
        "source_start_frame": int(args.source_start_frame),
        "transition_frames": int(args.transition_frames),
        "transition_output_frames": int(args.transition_output_frames),
        "anchor_yaw_policy": str(args.anchor_yaw_policy),
        "derive_heading_from_path": bool(args.derive_heading_from_path),
        "full_plot": str(full_plot),
        "zoom_plot": str(zoom_plot),
        "boundary_metrics": _boundary_metrics(original, updated, update_frame),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
