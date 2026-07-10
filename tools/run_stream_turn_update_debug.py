"""Run a single-sample stream update debug experiment.

This simulates a web-demo style route update: near the end of the original
route, the condition switches to a new route that turns 180 degrees and runs
another loop. It can either feed the composed 7D condition directly or route it
through RootRefiner, including a second RootRefiner call at the update point.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np

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

import torch

from eval.common.visualization import plot_xz_trajectories, render_motion_video
from eval.ldf.conditioning import LdfEvalStreamConditioner
from eval.ldf.stream_generation import (
    StreamTextRolloutController,
    _decode_latent_chunk,
    _decode_raw_chunk_preserving_feedback_cache,
    _encode_corrected_chunk_token,
    _replace_chunk_root_from_condition,
    _write_committed_latent_to_model,
    run_stream_generate_step_sample,
)
from eval.root_refiner.benchmark import _load_model_from_ckpt as _load_root_refiner_from_ckpt
from eval.ldf.stream_setup import (
    _set_seed,
    build_eval_dataloader,
    enable_cpu_text_encoding,
    load_eval_model_and_vae,
)
from metrics.traj import (
    _compute_traj_metrics,
    _seed_eval_locally,
    _slice_single_sample_batch,
    _stable_eval_seed,
)
from utils.initialize import load_config
from utils.inference.route_condition import RoutePlan, reanchor_route_to_xz
from utils.inference.stream_generator import StreamGenerator
from utils.inference.timeline import RootFrameState, RootTimeline
from utils.local_frame import (
    canonicalize_7d,
    transform_xz_local_to_world,
    uncanonicalize_7d,
    wrap_angle,
)
from utils.motion_process import (
    StreamJointRecovery263,
    append_traj_deltas_5d_to_7d,
    extract_root_trajectory_263_torch,
    recover_root_rot_pos,
    root_to_traj_feats_7d,
)
from utils.token_frame import (
    frame_idx_to_token_idx,
    num_tokens_for_frame_len,
    token_start_frame,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ldf_test.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vae_ckpt", default=None)
    parser.add_argument("--meta_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample_name", default="000021")
    parser.add_argument("--caption_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--probe_tag", default="turn_update_180")
    parser.add_argument("--cfg_text", type=float, default=1.2)
    parser.add_argument("--cfg_traj", type=float, default=2.2)
    parser.add_argument("--history_length", type=int, default=30)
    parser.add_argument("--horizon_tokens", type=int, default=10)
    parser.add_argument(
        "--condition_source",
        choices=[
            "gt7d",
            "root_refiner",
            "root_refiner_update",
            "root_refiner_multi_update",
        ],
        default="gt7d",
        help=(
            "gt7d uses the manually composed 7D route directly; root_refiner "
            "routes the full condition through RootRefiner once; root_refiner_update "
            "calls RootRefiner once initially and once again at the update point; "
            "root_refiner_multi_update calls RootRefiner at multiple update points."
        ),
    )
    parser.add_argument("--root_refiner_ckpt", default=None)
    parser.add_argument(
        "--root_refiner_forced_frames",
        type=int,
        default=None,
        help="Future frame count forced into RootRefiner. Defaults to updated_frames - 1.",
    )
    parser.add_argument(
        "--root_refiner_heading_override",
        choices=["none", "path_tangent"],
        default="none",
        help="Optional postprocess for RootRefiner output heading before feeding LDF.",
    )
    parser.add_argument("--update_lead_tokens", type=int, default=6)
    parser.add_argument("--update_frame", type=int, default=None)
    parser.add_argument("--suffix_frames", type=int, default=None)
    parser.add_argument(
        "--composition_mode",
        choices=[
            "turn180",
            "anchor_local",
            "center_symmetric",
            "clean_s_curve",
            "legacy_s_turn",
            "reflected_history",
            "two_segment_arc",
            "forward_line",
            "four_segment_curve",
            "constant_arc",
        ],
        default="turn180",
        help="How to compose the updated route condition.",
    )
    parser.add_argument(
        "--condition_traj_display",
        choices=["future", "full"],
        default="future",
        help=(
            "Controls the red condition trajectory overlay in rendered videos. "
            "'future' hides the pre-update prefix so route-update clips show only "
            "the newly planned path; 'full' draws the complete 7D condition."
        ),
    )
    parser.add_argument("--source_start_frame", type=int, default=0)
    parser.add_argument(
        "--transition_frames",
        type=int,
        default=0,
        help="For anchor_local mode, blend first N suffix frames from straight entry to source curvature.",
    )
    parser.add_argument(
        "--transition_output_frames",
        type=int,
        default=0,
        help="For anchor_local mode, resample the transition interval to this many output frames.",
    )
    parser.add_argument(
        "--anchor_yaw_policy",
        choices=["heading", "path_tangent"],
        default="path_tangent",
    )
    parser.add_argument(
        "--derive_heading_from_path",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--turn_blend_frames",
        type=int,
        default=None,
        help="Frames used to smoothly ramp the updated route from 0 to 180 degrees.",
    )
    parser.add_argument(
        "--first_straight_frames",
        type=int,
        default=40,
        help="For two_segment_arc mode, frames to keep moving straight before the arc.",
    )
    parser.add_argument(
        "--arc_frames",
        type=int,
        default=48,
        help="For two_segment_arc mode, frames used by the smooth turn arc.",
    )
    parser.add_argument(
        "--arc_turn_degrees",
        type=float,
        default=35.0,
        help="For two_segment_arc mode, signed turn angle applied by the arc.",
    )
    parser.add_argument(
        "--step_lookback_frames",
        type=int,
        default=20,
        help="Frames used to estimate step length for synthetic route composition.",
    )
    parser.add_argument(
        "--forward_frames",
        type=int,
        default=180,
        help="For forward_line mode, number of frames in the hand-authored route.",
    )
    parser.add_argument(
        "--forward_step_length",
        type=float,
        default=0.015,
        help="For forward_line mode, root advance per frame in meters.",
    )
    parser.add_argument(
        "--four_segment_frames",
        type=int,
        default=240,
        help="For four_segment_curve mode, total frames in the hand-authored route.",
    )
    parser.add_argument(
        "--four_segment_forward",
        type=float,
        default=4.2,
        help="For four_segment_curve mode, total forward distance in meters.",
    )
    parser.add_argument(
        "--constant_arc_frames",
        type=int,
        default=493,
        help="For constant_arc mode, total frames in the hand-authored route.",
    )
    parser.add_argument(
        "--constant_arc_length",
        type=float,
        default=14.76,
        help="For constant_arc mode, total XZ arc length in meters.",
    )
    parser.add_argument(
        "--constant_arc_turn_degrees",
        type=float,
        default=20.0,
        help="For constant_arc mode, signed total heading change in degrees.",
    )
    parser.add_argument(
        "--multi_update_frames",
        default="60,120,180",
        help="Comma-separated frame indices where root_refiner_multi_update refreshes the route.",
    )
    parser.add_argument(
        "--root_update_reanchor_mode",
        choices=["route_boundary", "current_root_translate", "current_root_pose"],
        default="route_boundary",
        help=(
            "How RootRefiner update segments are reanchored. route_boundary keeps "
            "the legacy absolute-route splice; current_root_translate starts the "
            "new segment at the generated actor root; current_root_pose also "
            "rotates the future route into the actor's current yaw frame."
        ),
    )
    parser.add_argument(
        "--root_refiner_update_mask_front_ratio",
        type=float,
        default=0.0,
        help=(
            "Diagnostic RootRefiner update ablation. For update calls only, "
            "mask this front fraction of the path condition. Default 0 disables it."
        ),
    )
    parser.add_argument(
        "--root_refiner_update_mask_kind",
        choices=["control", "valid"],
        default="control",
        help=(
            "Mask type for --root_refiner_update_mask_front_ratio. control clears "
            "path_control_mask only; valid removes the points from path attention."
        ),
    )
    parser.add_argument(
        "--root_refiner_update_anchor_yaw_mode",
        choices=["runtime", "path_tangent"],
        default="runtime",
        help=(
            "Diagnostic update ablation. runtime uses the generated actor yaw; "
            "path_tangent uses the new update segment's front XZ tangent as the "
            "RootRefiner anchor yaw."
        ),
    )
    parser.add_argument(
        "--root_refiner_update_anchor_tangent_frames",
        type=int,
        default=20,
        help="Frames used to estimate path_tangent anchor yaw for update ablations.",
    )
    parser.add_argument(
        "--alphas",
        default="0.5,1.0",
        help="Comma-separated feedback XZ blend alphas to render.",
    )
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--num_denoise_steps", type=int, default=None)
    parser.add_argument("--frames_per_token", type=int, default=4)
    parser.add_argument("--token_dt", type=float, default=0.20)
    return parser.parse_args()


def _parse_alphas(raw: str) -> list[float]:
    raw_text = str(raw).strip()
    if raw_text.lower() in {"", "none", "null", "false", "no"}:
        return []
    values = []
    for item in raw_text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def _parse_int_list(raw: str) -> list[int]:
    raw_text = str(raw).strip()
    if raw_text.lower() in {"", "none", "null", "false", "no"}:
        return []
    values = []
    for item in raw_text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def _copy_tensor_batch_value(value, tensor):
    if torch.is_tensor(value):
        return tensor.unsqueeze(0).to(device=value.device, dtype=value.dtype)
    return tensor.unsqueeze(0)


def _select_caption(sample_batch: dict, caption_index: int) -> None:
    text_all = sample_batch.get("text_all")
    if not text_all:
        return
    captions = text_all[0] if isinstance(text_all, list) else text_all
    if captions and isinstance(captions[0], list):
        captions = captions[0]
    index = int(caption_index)
    if index < 0 or index >= len(captions):
        raise ValueError(f"caption_index={index} out of range for {len(captions)} captions")
    sample_batch["text"] = [str(captions[index])]
    sample_batch["_caption_index"] = index
    sample_batch["_caption_text"] = str(captions[index])


def _yaw_from_7d(traj7: torch.Tensor) -> torch.Tensor:
    return torch.atan2(traj7[:, 4], traj7[:, 3])


def _path_yaw_from_xz(xz: torch.Tensor, *, min_speed: float = 1e-3) -> torch.Tensor:
    if xz.dim() != 2 or xz.shape[-1] != 2:
        raise ValueError(f"Expected xz [T,2], got {tuple(xz.shape)}")
    if int(xz.shape[0]) <= 1:
        return torch.zeros((int(xz.shape[0]),), device=xz.device, dtype=xz.dtype)
    delta = torch.zeros_like(xz)
    delta[:-1] = xz[1:] - xz[:-1]
    delta[-1] = delta[-2]
    yaw = torch.atan2(delta[:, 0], delta[:, 1])
    speed = torch.linalg.norm(delta, dim=-1)
    out = yaw.clone()
    last = torch.zeros((), device=xz.device, dtype=xz.dtype)
    min_speed_t = torch.as_tensor(float(min_speed), device=xz.device, dtype=xz.dtype)
    for idx in range(int(xz.shape[0])):
        if bool(speed[idx] > min_speed_t):
            last = yaw[idx]
        out[idx] = last
    return out


def _rotate_xz_delta(delta_xz: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    x = delta_xz[:, 0] * cos_y + delta_xz[:, 1] * sin_y
    z = -delta_xz[:, 0] * sin_y + delta_xz[:, 1] * cos_y
    return torch.stack([x, z], dim=-1)


def _smooth_turn_offsets(num_deltas: int, turn_blend_frames: int, device, dtype) -> torch.Tensor:
    if num_deltas <= 0:
        return torch.zeros((0,), device=device, dtype=dtype)
    blend = max(1, int(turn_blend_frames))
    t = torch.arange(num_deltas, device=device, dtype=dtype) / float(blend)
    t = t.clamp(0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return smooth * math.pi


def _source_route_for_suffix(
    traj7: torch.Tensor,
    suffix_len: int,
    *,
    source_start_frame: int = 0,
) -> torch.Tensor:
    start = max(0, min(int(source_start_frame), int(traj7.shape[0]) - 1))
    source = traj7[start:]
    if int(source.shape[0]) >= suffix_len:
        return source[:suffix_len]
    tail = source[-1:].expand(suffix_len - int(source.shape[0]), -1)
    return torch.cat([source, tail], dim=0)


def _anchor_yaw_for_update(
    traj7: torch.Tensor,
    update: int,
    *,
    policy: str,
) -> torch.Tensor:
    policy = str(policy)
    if policy == "heading":
        return _yaw_from_7d(traj7[update:update + 1])[0]
    if policy != "path_tangent":
        raise ValueError("anchor_yaw_policy must be 'heading' or 'path_tangent'")
    xz = traj7[:, [0, 2]]
    if update > 0:
        delta = xz[update] - xz[update - 1]
    else:
        delta = xz[1] - xz[0]
    if bool(torch.linalg.norm(delta) <= 1e-8):
        return _yaw_from_7d(traj7[update:update + 1])[0]
    return torch.atan2(delta[0], delta[1])


def _resample_yaw_nearest(yaw: torch.Tensor, target_len: int) -> torch.Tensor:
    target_len = int(target_len)
    if int(yaw.shape[0]) == target_len:
        return yaw
    if target_len <= 0:
        return yaw.new_zeros((0,))
    if int(yaw.shape[0]) <= 0:
        return yaw.new_zeros((target_len,))
    if int(yaw.shape[0]) == 1:
        return yaw[:1].expand(target_len)
    index = torch.linspace(
        0,
        int(yaw.shape[0]) - 1,
        target_len,
        device=yaw.device,
        dtype=yaw.dtype,
    ).round().long()
    return yaw.index_select(0, index)


def _build_route_derived_7d_from_xyz(
    xyz: torch.Tensor,
    *,
    heading_xyz: torch.Tensor | None = None,
) -> torch.Tensor:
    heading_source = xyz if heading_xyz is None else heading_xyz
    yaw = _path_yaw_from_xz(heading_source[:, [0, 2]])
    yaw = _resample_yaw_nearest(yaw, int(xyz.shape[0]))
    traj5 = torch.cat(
        [xyz, torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]],
        dim=-1,
    )
    return append_traj_deltas_5d_to_7d(traj5)


def _override_heading_from_path_tangent(
    traj7: torch.Tensor,
    path_xyz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep xyz and replace heading channels with reference path-tangent heading."""
    if traj7.dim() != 2 or traj7.shape[-1] < 5:
        raise ValueError(f"Expected traj7 [T,>=5], got {tuple(traj7.shape)}")
    if path_xyz is not None and (path_xyz.dim() != 2 or path_xyz.shape[-1] < 3):
        raise ValueError(f"Expected path_xyz [T,>=3], got {tuple(path_xyz.shape)}")
    return _build_route_derived_7d_from_xyz(
        traj7[:, :3],
        heading_xyz=None if path_xyz is None else path_xyz[:, :3].to(
            device=traj7.device,
            dtype=traj7.dtype,
        ),
    )


def _maybe_override_rootrefiner_heading(
    traj7: torch.Tensor,
    mode: str,
    *,
    path_xyz: torch.Tensor | None = None,
) -> torch.Tensor:
    mode = str(mode)
    if mode == "none":
        return traj7
    if mode == "path_tangent":
        return _override_heading_from_path_tangent(traj7, path_xyz=path_xyz)
    raise ValueError(f"Unsupported root_refiner_heading_override={mode!r}")


def _smoothstep_weight(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp(0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _smooth_suffix_entry_xz(
    source_local_xz: torch.Tensor,
    *,
    transition_frames: int,
) -> torch.Tensor:
    transition = int(transition_frames)
    if transition <= 1 or int(source_local_xz.shape[0]) <= 2:
        return source_local_xz
    out = source_local_xz.clone()
    limit = min(transition, int(source_local_xz.shape[0]) - 1)
    step_len = torch.linalg.norm(
        source_local_xz[1:] - source_local_xz[:-1],
        dim=-1,
    )
    arc = torch.zeros(
        (int(source_local_xz.shape[0]),),
        device=source_local_xz.device,
        dtype=source_local_xz.dtype,
    )
    arc[1:] = torch.cumsum(step_len, dim=0)
    straight = torch.zeros_like(source_local_xz)
    straight[:, 1] = arc
    t = torch.arange(
        limit + 1,
        device=source_local_xz.device,
        dtype=source_local_xz.dtype,
    ) / float(limit)
    weight = _smoothstep_weight(t)[:, None]
    out[:limit + 1] = (1.0 - weight) * straight[:limit + 1] + weight * source_local_xz[:limit + 1]
    return out


def _resample_transition_interval(
    values: torch.Tensor,
    *,
    transition_frames: int,
    transition_output_frames: int,
) -> torch.Tensor:
    source_steps = int(transition_frames)
    target_steps = int(transition_output_frames)
    if (
        source_steps <= 0
        or target_steps <= 0
        or source_steps == target_steps
        or int(values.shape[0]) <= 1
    ):
        return values
    source_steps = min(source_steps, int(values.shape[0]) - 1)
    positions = torch.linspace(
        0.0,
        float(source_steps),
        target_steps + 1,
        device=values.device,
        dtype=values.dtype,
    )
    left = positions.floor().long().clamp(0, source_steps)
    right = (left + 1).clamp(max=source_steps)
    alpha = (positions - left.to(values.dtype)).unsqueeze(-1)
    segment = (1.0 - alpha) * values[left] + alpha * values[right]
    return torch.cat([segment, values[source_steps + 1:]], dim=0)


def compose_anchor_local_updated_traj7(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    suffix_frames: int | None = None,
    source_start_frame: int = 0,
    anchor_yaw_policy: str = "heading",
    derive_heading_from_path: bool = False,
    transition_frames: int = 0,
    transition_output_frames: int = 0,
) -> torch.Tensor:
    """Append a route suffix in the update pose's local coordinate frame.

    The source route is first canonicalized to its own first pose, so its first
    frame is local origin with local yaw zero. That local route is then
    uncanonicalized using ``traj7[update_frame]`` as the new origin and heading.
    """
    if traj7.dim() != 2 or traj7.shape[-1] < 5:
        raise ValueError(f"Expected traj7 [T,>=5], got {tuple(traj7.shape)}")
    total = int(traj7.shape[0])
    if total <= 1:
        raise ValueError("traj7 must contain at least two frames")
    update = max(1, min(int(update_frame), total - 1))
    suffix_len = int(suffix_frames) if suffix_frames is not None else total
    suffix_len = max(2, suffix_len)

    source = _source_route_for_suffix(
        traj7,
        suffix_len,
        source_start_frame=source_start_frame,
    )
    anchor = traj7[update]
    anchor_xz = anchor[[0, 2]]
    anchor_yaw = _anchor_yaw_for_update(
        traj7,
        update,
        policy=anchor_yaw_policy,
    )

    source_anchor_xz = source[0, [0, 2]]
    source_path_yaw = _path_yaw_from_xz(source[:, [0, 2]])[0]
    source_local = canonicalize_7d(source, source_anchor_xz, source_path_yaw)
    source_local[:, [0, 2]] = _smooth_suffix_entry_xz(
        source_local[:, [0, 2]],
        transition_frames=int(transition_frames),
    )
    source_local[:, 1] = anchor[1] + (source[:, 1] - source[0, 1])
    source_local_xyz = _resample_transition_interval(
        source_local[:, :3],
        transition_frames=int(transition_frames),
        transition_output_frames=int(transition_output_frames),
    )
    source_local_yaw = _path_yaw_from_xz(source_local_xyz[:, [0, 2]])
    source_local_5d = torch.cat(
        [
            source_local_xyz,
            torch.cos(source_local_yaw)[:, None],
            torch.sin(source_local_yaw)[:, None],
        ],
        dim=-1,
    )
    source_local = append_traj_deltas_5d_to_7d(source_local_5d)

    suffix_world_7d = uncanonicalize_7d(source_local, anchor_xz, anchor_yaw)
    if derive_heading_from_path:
        combined_xyz = torch.cat([traj7[:update, :3], suffix_world_7d[:, :3]], dim=0)
        return _build_route_derived_7d_from_xyz(combined_xyz)
    combined_5d = torch.cat([traj7[:update, :5], suffix_world_7d[:, :5]], dim=0)
    return append_traj_deltas_5d_to_7d(combined_5d)


def compose_center_symmetric_updated_traj7(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    suffix_frames: int | None = None,
    source_start_frame: int = 0,
    derive_heading_from_path: bool = True,
    transition_frames: int = 0,
    transition_output_frames: int = 0,
) -> torch.Tensor:
    """Continue by reflecting recent history through the update anchor.

    This builds an S-like continuation. If ``p_u`` is the update position, the
    suffix points are ``2 * p_u - p_{u-j}``, so the first post-update tangent is
    equal to the incoming tangent at ``update_frame``.
    """
    if traj7.dim() != 2 or traj7.shape[-1] < 5:
        raise ValueError(f"Expected traj7 [T,>=5], got {tuple(traj7.shape)}")
    total = int(traj7.shape[0])
    if total <= 1:
        raise ValueError("traj7 must contain at least two frames")
    update = max(1, min(int(update_frame), total - 1))
    start = max(0, min(int(source_start_frame), update))
    history = traj7[start:update + 1]
    if suffix_frames is not None:
        suffix_cap = max(1, int(suffix_frames))
        if int(history.shape[0]) > suffix_cap:
            history = history[-suffix_cap:]

    anchor = traj7[update]
    reflected_history = torch.flip(history, dims=[0])
    suffix_xyz = reflected_history[:, :3].clone()
    suffix_xyz[:, 0] = 2.0 * anchor[0] - reflected_history[:, 0]
    suffix_xyz[:, 2] = 2.0 * anchor[2] - reflected_history[:, 2]
    suffix_xyz = _resample_transition_interval(
        suffix_xyz,
        transition_frames=int(transition_frames),
        transition_output_frames=int(transition_output_frames),
    )
    if not derive_heading_from_path:
        suffix_yaw = _path_yaw_from_xz(suffix_xyz[:, [0, 2]])
        suffix_5d = torch.cat(
            [
                suffix_xyz,
                torch.cos(suffix_yaw)[:, None],
                torch.sin(suffix_yaw)[:, None],
            ],
            dim=-1,
        )
        suffix_7d = append_traj_deltas_5d_to_7d(suffix_5d)
        combined_5d = torch.cat([traj7[:update, :5], suffix_7d[:, :5]], dim=0)
        return append_traj_deltas_5d_to_7d(combined_5d)
    suffix_7d = _build_route_derived_7d_from_xyz(suffix_xyz)
    combined_5d = torch.cat([traj7[:update, :5], suffix_7d[:, :5]], dim=0)
    return append_traj_deltas_5d_to_7d(combined_5d)


def compose_legacy_s_turn_updated_traj7(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    suffix_frames: int | None = None,
    turn_blend_frames: int = 24,
    transition_frames: int = 0,
    transition_output_frames: int = 0,
) -> torch.Tensor:
    """Compose the old S-turn draft used by the 000021 history sweeps.

    This is the smooth 180-degree local-route update, with optional transition
    duration resampling.  It intentionally does not use reflected history; that
    reflected-history probe is available through ``reflected_history``.
    """
    if traj7.dim() != 2 or traj7.shape[-1] < 5:
        raise ValueError(f"Expected traj7 [T,>=5], got {tuple(traj7.shape)}")
    total = int(traj7.shape[0])
    if total <= 1:
        raise ValueError("traj7 must contain at least two frames")
    update = max(1, min(int(update_frame), total - 1))
    suffix_len = int(suffix_frames) if suffix_frames is not None else total
    suffix_len = max(2, suffix_len)

    if suffix_len <= total:
        source = traj7[:suffix_len]
    else:
        tail = traj7[-1:].expand(suffix_len - total, -1)
        source = torch.cat([traj7, tail], dim=0)
    anchor = traj7[update]
    anchor_xz = anchor[[0, 2]]
    anchor_yaw = _yaw_from_7d(anchor.unsqueeze(0))[0]

    source_anchor_xz = source[0, [0, 2]]
    source_anchor_yaw = _yaw_from_7d(source[:1])[0]
    source_local = canonicalize_7d(source, source_anchor_xz, source_anchor_yaw)
    source_local_xyz = _resample_transition_interval(
        source_local[:, :3],
        transition_frames=int(transition_frames),
        transition_output_frames=int(transition_output_frames),
    )

    source_local_xz = source_local_xyz[:, [0, 2]]
    source_delta = source_local_xz[1:] - source_local_xz[:-1]
    yaw_offsets = _smooth_turn_offsets(
        int(source_delta.shape[0]),
        int(turn_blend_frames),
        device=traj7.device,
        dtype=traj7.dtype,
    )
    rotated_local_delta = _rotate_xz_delta(source_delta, yaw_offsets)
    out_len = int(source_local_xyz.shape[0])
    suffix_local_xz = torch.zeros((out_len, 2), device=traj7.device, dtype=traj7.dtype)
    if out_len > 1:
        suffix_local_xz[1:] = torch.cumsum(rotated_local_delta, dim=0)
    suffix_y = anchor[1] + (source_local_xyz[:, 1] - source_local_xyz[0, 1])
    source_local_yaw = _path_yaw_from_xz(source_local_xz)
    yaw_frame_offsets = torch.zeros((out_len,), device=traj7.device, dtype=traj7.dtype)
    if out_len > 1:
        yaw_frame_offsets[1:] = yaw_offsets
    suffix_local_yaw = wrap_angle(source_local_yaw + yaw_frame_offsets)
    suffix_local_5d = torch.cat(
        [
            suffix_local_xz[:, :1],
            suffix_y[:, None],
            suffix_local_xz[:, 1:2],
            torch.cos(suffix_local_yaw)[:, None],
            torch.sin(suffix_local_yaw)[:, None],
        ],
        dim=-1,
    )
    suffix_world_7d = uncanonicalize_7d(
        append_traj_deltas_5d_to_7d(suffix_local_5d),
        anchor_xz,
        anchor_yaw,
    )
    combined_5d = torch.cat([traj7[:update, :5], suffix_world_7d[:, :5]], dim=0)
    return append_traj_deltas_5d_to_7d(combined_5d)


def compose_clean_s_curve_updated_traj7(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    suffix_frames: int | None = None,
    transition_frames: int = 0,
    transition_output_frames: int = 0,
    step_lookback_frames: int = 20,
) -> torch.Tensor:
    """Compose a single forward S-curve from the update pose.

    Unlike the legacy turn probe, this does not reuse the sample's circular
    route as the suffix. It authors a clean forward path in the update tangent
    frame and adds a smooth lateral S offset.
    """
    if traj7.dim() != 2 or traj7.shape[-1] < 5:
        raise ValueError(f"Expected traj7 [T,>=5], got {tuple(traj7.shape)}")
    total = int(traj7.shape[0])
    if total <= 1:
        raise ValueError("traj7 must contain at least two frames")
    update = max(1, min(int(update_frame), total - 1))
    suffix_len = int(suffix_frames) if suffix_frames is not None else total
    suffix_len = max(2, suffix_len)
    transition_extra = max(0, int(transition_output_frames) - int(transition_frames))
    suffix_len = suffix_len + transition_extra

    anchor = traj7[update]
    anchor_xz = anchor[[0, 2]]
    anchor_yaw = _estimate_route_anchor_yaw(
        traj7,
        update,
        lookback_frames=int(step_lookback_frames),
    )
    step_len = _estimate_route_step_length(
        traj7,
        update,
        lookback_frames=int(step_lookback_frames),
    )
    frame_idx = torch.arange(suffix_len, device=traj7.device, dtype=traj7.dtype)
    progress = frame_idx / float(max(suffix_len - 1, 1))
    forward = frame_idx * step_len
    total_forward = forward[-1].clamp(min=step_len)
    amplitude = (total_forward * 0.24).clamp(min=0.25, max=0.75)
    envelope = torch.sin(math.pi * progress).clamp(min=0.0)
    lateral = amplitude * torch.sin(2.0 * math.pi * progress) * envelope

    cos_y = torch.cos(anchor_yaw)
    sin_y = torch.sin(anchor_yaw)
    forward_dir = torch.stack([sin_y, cos_y])
    right_dir = torch.stack([cos_y, -sin_y])
    suffix_xz = anchor_xz[None, :] + forward[:, None] * forward_dir[None, :]
    suffix_xz = suffix_xz + lateral[:, None] * right_dir[None, :]
    suffix_y = anchor[1].expand(suffix_len)
    suffix_xyz = torch.stack([suffix_xz[:, 0], suffix_y, suffix_xz[:, 1]], dim=-1)
    combined_xyz = torch.cat([traj7[:update, :3], suffix_xyz], dim=0)
    return _build_route_derived_7d_from_xyz(combined_xyz)


def _estimate_route_step_length(
    traj7: torch.Tensor,
    update: int,
    *,
    lookback_frames: int,
) -> torch.Tensor:
    xz = traj7[:, [0, 2]]
    moving_threshold = torch.as_tensor(5e-3, device=traj7.device, dtype=traj7.dtype)
    if int(xz.shape[0]) <= 1:
        return torch.as_tensor(0.03, device=traj7.device, dtype=traj7.dtype)
    right = max(1, min(int(update), int(xz.shape[0]) - 1))
    left = max(1, right - max(1, int(lookback_frames)) + 1)
    deltas = xz[left:right + 1] - xz[left - 1:right]
    speed = torch.linalg.norm(deltas, dim=-1)
    valid = speed > moving_threshold
    if bool(valid.any()):
        return speed[valid].mean().clamp(min=1e-3)
    all_speed = torch.linalg.norm(xz[1:] - xz[:-1], dim=-1)
    valid_all = all_speed > moving_threshold
    if bool(valid_all.any()):
        return all_speed[valid_all].mean().clamp(min=1e-3)
    return torch.as_tensor(0.03, device=traj7.device, dtype=traj7.dtype)


def _estimate_route_anchor_yaw(
    traj7: torch.Tensor,
    update: int,
    *,
    lookback_frames: int,
) -> torch.Tensor:
    xz = traj7[:, [0, 2]]
    moving_threshold = torch.as_tensor(5e-3, device=traj7.device, dtype=traj7.dtype)
    if int(xz.shape[0]) <= 1:
        return _yaw_from_7d(traj7[update:update + 1])[0]

    right = max(1, min(int(update), int(xz.shape[0]) - 1))
    left = max(1, right - max(1, int(lookback_frames)) + 1)
    deltas = xz[left:right + 1] - xz[left - 1:right]
    speed = torch.linalg.norm(deltas, dim=-1)
    valid = speed > moving_threshold
    if bool(valid.any()):
        delta = deltas[valid].sum(dim=0)
        if bool(torch.linalg.norm(delta) > moving_threshold):
            return torch.atan2(delta[0], delta[1])

    all_deltas = xz[1:right + 1] - xz[:right]
    all_speed = torch.linalg.norm(all_deltas, dim=-1)
    valid_all = all_speed > moving_threshold
    if bool(valid_all.any()):
        delta = all_deltas[valid_all].sum(dim=0)
        if bool(torch.linalg.norm(delta) > moving_threshold):
            return torch.atan2(delta[0], delta[1])

    return _yaw_from_7d(traj7[update:update + 1])[0]


def compose_forward_line_traj7(
    traj7: torch.Tensor,
    *,
    num_frames: int,
    step_length: float,
) -> torch.Tensor:
    """Hand-author a straight forward route from the sample's first root pose."""
    if traj7.dim() != 2 or traj7.shape[-1] < 5:
        raise ValueError(f"Expected traj7 [T,>=5], got {tuple(traj7.shape)}")
    if int(traj7.shape[0]) <= 0:
        raise ValueError("traj7 must contain at least one frame")
    frames = max(2, int(num_frames))
    step = float(step_length)
    if step < 0.0:
        raise ValueError(f"step_length must be >= 0, got {step_length}")
    anchor = traj7[0]
    yaw = _yaw_from_7d(traj7[:1])[0]
    frame_idx = torch.arange(frames, device=traj7.device, dtype=traj7.dtype)
    distance = frame_idx * torch.as_tensor(step, device=traj7.device, dtype=traj7.dtype)
    x = anchor[0] + torch.sin(yaw) * distance
    y = anchor[1].expand_as(x)
    z = anchor[2] + torch.cos(yaw) * distance
    heading = yaw.expand_as(x)
    traj5 = torch.stack(
        [x, y, z, torch.cos(heading), torch.sin(heading)],
        dim=-1,
    )
    return append_traj_deltas_5d_to_7d(traj5)


def compose_four_segment_forward_curve_traj7(
    *,
    num_frames: int = 240,
    total_forward: float = 4.2,
    primary_amplitude: float = 0.22,
    secondary_amplitude: float = 0.08,
) -> torch.Tensor:
    """Hand-authored long forward route with four mild-curvature segments."""
    frames = max(2, int(num_frames))
    frame_idx = torch.arange(frames, dtype=torch.float32)
    progress = frame_idx / float(max(frames - 1, 1))
    z = torch.as_tensor(float(total_forward), dtype=torch.float32) * progress
    x = (
        torch.as_tensor(float(primary_amplitude), dtype=torch.float32)
        * torch.sin(2.0 * math.pi * (progress - 0.08))
        + torch.as_tensor(float(secondary_amplitude), dtype=torch.float32)
        * torch.sin(4.0 * math.pi * progress + 0.35)
    )
    x = x - x[:1]
    y = torch.zeros_like(x)
    xyz = torch.stack([x, y, z - z[:1]], dim=-1)
    return _build_route_derived_7d_from_xyz(xyz)


def compose_constant_arc_traj7(
    traj7: torch.Tensor,
    *,
    num_frames: int = 493,
    arc_length: float = 14.76,
    turn_degrees: float = 20.0,
) -> torch.Tensor:
    """Hand-author a uniformly sampled circular arc from the sample start pose."""
    if traj7.dim() != 2 or traj7.shape[-1] < 5:
        raise ValueError(f"Expected traj7 [T,>=5], got {tuple(traj7.shape)}")
    if int(traj7.shape[0]) <= 0:
        raise ValueError("traj7 must contain at least one frame")
    frames = max(2, int(num_frames))
    length = max(0.0, float(arc_length))
    turn = math.radians(float(turn_degrees))
    anchor = traj7[0]
    anchor_yaw = _yaw_from_7d(traj7[:1])[0]
    frame_idx = torch.arange(frames, device=traj7.device, dtype=traj7.dtype)
    progress = frame_idx / float(max(frames - 1, 1))
    if abs(turn) < 1e-7 or length <= 0.0:
        local_z = torch.as_tensor(length, device=traj7.device, dtype=traj7.dtype) * progress
        local_x = torch.zeros_like(local_z)
        local_yaw = torch.zeros_like(local_z)
    else:
        phi = torch.as_tensor(turn, device=traj7.device, dtype=traj7.dtype) * progress
        curvature = torch.as_tensor(turn / length, device=traj7.device, dtype=traj7.dtype)
        local_x = (1.0 - torch.cos(phi)) / curvature
        local_z = torch.sin(phi) / curvature
        local_yaw = phi
    local_xz = torch.stack([local_x, local_z], dim=-1)
    world_xz = anchor[[0, 2]] + _rotate_xz_delta(local_xz, anchor_yaw)
    y = anchor[1].expand_as(local_x)
    yaw = wrap_angle(anchor_yaw + local_yaw)
    traj5 = torch.stack(
        [world_xz[:, 0], y, world_xz[:, 1], torch.cos(yaw), torch.sin(yaw)],
        dim=-1,
    )
    return append_traj_deltas_5d_to_7d(traj5)


def compose_two_segment_arc_updated_traj7(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    suffix_frames: int | None = None,
    first_straight_frames: int = 40,
    arc_frames: int = 48,
    turn_degrees: float = 35.0,
    step_lookback_frames: int = 20,
) -> torch.Tensor:
    """Compose a clean straight-arc-straight route from the update pose.

    This is intended for runtime route-update probes where we want a plausible
    new path instead of reusing mirrored source motion. The generated suffix
    starts at ``update_frame``, continues along the incoming path tangent,
    follows a smooth signed arc, then keeps the final heading.
    """
    if traj7.dim() != 2 or traj7.shape[-1] < 5:
        raise ValueError(f"Expected traj7 [T,>=5], got {tuple(traj7.shape)}")
    total = int(traj7.shape[0])
    if total <= 1:
        raise ValueError("traj7 must contain at least two frames")
    update = max(1, min(int(update_frame), total - 1))
    suffix_len = int(suffix_frames) if suffix_frames is not None else total
    suffix_len = max(2, suffix_len)
    first_straight = max(0, int(first_straight_frames))
    arc_len = max(1, int(arc_frames))

    anchor = traj7[update]
    anchor_xz = anchor[[0, 2]]
    anchor_yaw = _estimate_route_anchor_yaw(
        traj7,
        update,
        lookback_frames=int(step_lookback_frames),
    )
    step_len = _estimate_route_step_length(
        traj7,
        update,
        lookback_frames=int(step_lookback_frames),
    )
    turn_radians = math.radians(float(turn_degrees))
    suffix_xyz = torch.zeros((suffix_len, 3), device=traj7.device, dtype=traj7.dtype)
    suffix_xyz[0] = anchor[:3]
    xz = anchor_xz.clone()
    for idx in range(1, suffix_len):
        if idx <= first_straight:
            progress = torch.as_tensor(0.0, device=traj7.device, dtype=traj7.dtype)
        elif idx >= first_straight + arc_len:
            progress = torch.as_tensor(1.0, device=traj7.device, dtype=traj7.dtype)
        else:
            raw = float(idx - first_straight) / float(arc_len)
            progress = _smoothstep_weight(
                torch.as_tensor(raw, device=traj7.device, dtype=traj7.dtype)
            )
        yaw = anchor_yaw + progress * turn_radians
        delta = torch.stack([torch.sin(yaw), torch.cos(yaw)]) * step_len
        xz = xz + delta
        suffix_xyz[idx, 0] = xz[0]
        suffix_xyz[idx, 1] = anchor[1]
        suffix_xyz[idx, 2] = xz[1]

    combined_xyz = torch.cat([traj7[:update, :3], suffix_xyz], dim=0)
    return _build_route_derived_7d_from_xyz(combined_xyz)


def compose_turn180_updated_traj7(
    traj7: torch.Tensor,
    *,
    update_frame: int,
    suffix_frames: int | None = None,
    turn_blend_frames: int = 24,
) -> torch.Tensor:
    """Compose ``prefix + smooth 180-degree updated loop`` as a physical 7D route.

    The updated suffix is authored in a new local frame whose first pose is the
    update pose: origin at ``traj7[update_frame]`` and local yaw zero aligned to
    the update heading. This matches the web-demo edit semantics where a newly
    drawn route starts from the current character pose instead of reusing a
    world-space tail from the old route.
    """
    if traj7.dim() != 2 or traj7.shape[-1] < 5:
        raise ValueError(f"Expected traj7 [T,>=5], got {tuple(traj7.shape)}")
    total = int(traj7.shape[0])
    if total <= 1:
        raise ValueError("traj7 must contain at least two frames")
    update = max(1, min(int(update_frame), total - 1))
    suffix_len = int(suffix_frames) if suffix_frames is not None else total
    suffix_len = max(2, suffix_len)

    if suffix_len <= total:
        source = traj7[:suffix_len]
    else:
        tail = traj7[-1:].expand(suffix_len - total, -1)
        source = torch.cat([traj7, tail], dim=0)
    anchor = traj7[update]
    anchor_xz = anchor[[0, 2]]
    anchor_yaw = _yaw_from_7d(anchor.unsqueeze(0))[0]

    source_anchor_xz = source[0, [0, 2]]
    source_anchor_yaw = _yaw_from_7d(source[:1])[0]
    source_local = canonicalize_7d(source, source_anchor_xz, source_anchor_yaw)

    source_local_xz = source_local[:, [0, 2]]
    source_delta = source_local_xz[1:] - source_local_xz[:-1]
    yaw_offsets = _smooth_turn_offsets(
        int(source_delta.shape[0]),
        int(turn_blend_frames),
        device=traj7.device,
        dtype=traj7.dtype,
    )
    rotated_local_delta = _rotate_xz_delta(source_delta, yaw_offsets)
    suffix_local_xz = torch.zeros((suffix_len, 2), device=traj7.device, dtype=traj7.dtype)
    if suffix_len > 1:
        suffix_local_xz[1:] = torch.cumsum(rotated_local_delta, dim=0)
    suffix_y = anchor[1] + (source[:, 1] - source[0, 1])
    source_local_yaw = _yaw_from_7d(source_local)
    yaw_frame_offsets = torch.zeros((suffix_len,), device=traj7.device, dtype=traj7.dtype)
    if suffix_len > 1:
        yaw_frame_offsets[1:] = yaw_offsets
    suffix_local_yaw = wrap_angle(source_local_yaw + yaw_frame_offsets)
    suffix_local_5d = torch.cat(
        [
            suffix_local_xz[:, :1],
            suffix_y[:, None],
            suffix_local_xz[:, 1:2],
            torch.cos(suffix_local_yaw)[:, None],
            torch.sin(suffix_local_yaw)[:, None],
        ],
        dim=-1,
    )
    suffix_local_7d = append_traj_deltas_5d_to_7d(suffix_local_5d)
    suffix_world_7d = uncanonicalize_7d(suffix_local_7d, anchor_xz, anchor_yaw)
    combined_5d = torch.cat([traj7[:update, :5], suffix_world_7d[:, :5]], dim=0)
    return append_traj_deltas_5d_to_7d(combined_5d)


def compose_updated_route_for_mode(
    traj7: torch.Tensor,
    *,
    composition_mode: str,
    update_frame: int,
    suffix_frames: int | None,
    source_start_frame: int,
    transition_frames: int,
    transition_output_frames: int,
    anchor_yaw_policy: str,
    derive_heading_from_path: bool,
    turn_blend_frames: int | None,
    update_lead_tokens: int,
    frames_per_token: int,
    first_straight_frames: int,
    arc_frames: int,
    arc_turn_degrees: float,
    step_lookback_frames: int,
    forward_frames: int,
    forward_step_length: float,
    four_segment_frames: int,
    four_segment_forward: float,
    constant_arc_frames: int = 493,
    constant_arc_length: float = 14.76,
    constant_arc_turn_degrees: float = 20.0,
) -> torch.Tensor:
    """Compose the route condition for the requested debug experiment mode."""
    mode = str(composition_mode)
    if mode == "forward_line":
        return compose_forward_line_traj7(
            traj7,
            num_frames=int(forward_frames),
            step_length=float(forward_step_length),
        )
    if mode == "four_segment_curve":
        return compose_four_segment_forward_curve_traj7(
            num_frames=int(four_segment_frames),
            total_forward=float(four_segment_forward),
        )
    if mode == "constant_arc":
        return compose_constant_arc_traj7(
            traj7,
            num_frames=int(constant_arc_frames),
            arc_length=float(constant_arc_length),
            turn_degrees=float(constant_arc_turn_degrees),
        )
    if mode == "anchor_local":
        return compose_anchor_local_updated_traj7(
            traj7,
            update_frame=update_frame,
            suffix_frames=suffix_frames,
            source_start_frame=int(source_start_frame),
            anchor_yaw_policy=str(anchor_yaw_policy),
            derive_heading_from_path=bool(derive_heading_from_path),
            transition_frames=int(transition_frames),
            transition_output_frames=int(transition_output_frames),
        )
    if mode in {"center_symmetric", "reflected_history"}:
        return compose_center_symmetric_updated_traj7(
            traj7,
            update_frame=update_frame,
            suffix_frames=suffix_frames,
            source_start_frame=int(source_start_frame),
            derive_heading_from_path=bool(derive_heading_from_path),
            transition_frames=int(transition_frames),
            transition_output_frames=int(transition_output_frames),
        )
    if mode == "clean_s_curve":
        return compose_clean_s_curve_updated_traj7(
            traj7,
            update_frame=update_frame,
            suffix_frames=suffix_frames,
            transition_frames=int(transition_frames),
            transition_output_frames=int(transition_output_frames),
            step_lookback_frames=int(step_lookback_frames),
        )
    if mode == "legacy_s_turn":
        return compose_legacy_s_turn_updated_traj7(
            traj7,
            update_frame=update_frame,
            suffix_frames=suffix_frames,
            turn_blend_frames=(
                int(turn_blend_frames)
                if turn_blend_frames is not None
                else int(update_lead_tokens) * int(frames_per_token)
            ),
            transition_frames=int(transition_frames),
            transition_output_frames=int(transition_output_frames),
        )
    if mode == "two_segment_arc":
        return compose_two_segment_arc_updated_traj7(
            traj7,
            update_frame=update_frame,
            suffix_frames=suffix_frames,
            first_straight_frames=int(first_straight_frames),
            arc_frames=int(arc_frames),
            turn_degrees=float(arc_turn_degrees),
            step_lookback_frames=int(step_lookback_frames),
        )
    if mode == "turn180":
        return compose_turn180_updated_traj7(
            traj7,
            update_frame=update_frame,
            suffix_frames=suffix_frames,
            turn_blend_frames=(
                int(turn_blend_frames)
                if turn_blend_frames is not None
                else int(update_lead_tokens) * int(frames_per_token)
            ),
        )
    raise ValueError(f"Unsupported composition_mode={composition_mode!r}")


def apply_updated_traj_to_sample_batch(sample_batch: dict, updated_traj7: torch.Tensor) -> dict:
    """Return a shallow-copied sample batch with trajectory fields replaced."""
    out = dict(sample_batch)
    device = (
        sample_batch["traj_cond_7d"].device
        if torch.is_tensor(sample_batch.get("traj_cond_7d"))
        else updated_traj7.device
    )
    dtype = (
        sample_batch["traj_cond_7d"].dtype
        if torch.is_tensor(sample_batch.get("traj_cond_7d"))
        else updated_traj7.dtype
    )
    updated = updated_traj7.to(device=device, dtype=dtype)
    valid_frames = int(updated.shape[0])
    valid_tokens = num_tokens_for_frame_len(valid_frames, 4)

    for key in ("traj_cond_7d",):
        out[key] = _copy_tensor_batch_value(sample_batch.get(key), updated)
    xyz = updated[:, :3]
    for key in ("traj", "traj_cond"):
        out[key] = _copy_tensor_batch_value(sample_batch.get(key), xyz)
    mask = torch.ones(valid_frames, device=device, dtype=torch.float32)
    for key in ("traj_mask", "traj_cond_mask", "traj_loss_mask"):
        if key in sample_batch:
            out[key] = mask.unsqueeze(0)
    out["traj_length"] = torch.tensor([valid_frames], device=device, dtype=torch.long)
    out["feature_length"] = torch.tensor([valid_frames], device=device, dtype=torch.long)
    out["token_length"] = torch.tensor([valid_tokens], device=device, dtype=torch.long)
    if "token_mask" in sample_batch:
        out["token_mask"] = torch.ones(1, valid_tokens, device=device, dtype=torch.float32)
    return out


def _root_plan_to_world_7d(root_plan) -> torch.Tensor:
    return uncanonicalize_7d(
        root_plan.waypoints_local_7d[: int(root_plan.valid_frames)],
        root_plan.anchor_world_xz,
        root_plan.anchor_world_yaw,
    )


def _rootrefiner_input_path_world_xz(condition, root_plan) -> torch.Tensor:
    """Return the exact RootRefiner path condition points in world XZ."""
    path = condition.path.detach().cpu().float()
    valid_mask = condition.path_valid_mask.detach().cpu().bool()
    path = path[valid_mask]
    if int(path.shape[0]) == 0:
        return path.new_zeros((0, 2))
    anchor_xz = root_plan.anchor_world_xz.detach().cpu().float()
    anchor_yaw = root_plan.anchor_world_yaw.detach().cpu().float()
    return transform_xz_local_to_world(path, anchor_xz, anchor_yaw).detach().cpu().float()


def _mask_rootrefiner_condition_front(
    condition,
    *,
    ratio: float,
    kind: str = "control",
):
    """Mask the front fraction of a RootRefiner path condition.

    This is a diagnostic ablation for route-update experiments. ``control``
    keeps the path tokens visible but removes the early control-point flag;
    ``valid`` removes those path tokens from attention as a stronger ablation.
    """
    ratio = max(0.0, min(float(ratio), 1.0))
    kind = str(kind)
    if kind not in {"control", "valid"}:
        raise ValueError(f"unknown RootRefiner mask kind {kind!r}")

    valid_mask = condition.path_valid_mask.detach().clone().bool()
    control_mask = condition.path_control_mask.detach().clone().bool()
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
    valid_count = int(valid_indices.numel())
    max_masked = max(0, valid_count - 1)
    masked_count = min(max_masked, int(math.ceil(valid_count * ratio)))
    if masked_count <= 0:
        debug = {
            "kind": kind,
            "ratio": float(ratio),
            "valid_points": valid_count,
            "masked_points": 0,
            "masked_indices": [],
        }
        return condition, debug

    masked_indices = valid_indices[:masked_count]
    if kind == "control":
        control_mask[masked_indices] = False
    else:
        valid_mask[masked_indices] = False
        control_mask = control_mask & valid_mask

    debug = {
        "kind": kind,
        "ratio": float(ratio),
        "valid_points": valid_count,
        "masked_points": int(masked_count),
        "masked_indices": [int(index) for index in masked_indices.tolist()],
    }
    return replace(
        condition,
        path_valid_mask=valid_mask,
        path_control_mask=control_mask,
    ), debug


def _build_route_plan_from_traj7(
    traj7: torch.Tensor,
    *,
    token_dt: float,
    frames_per_token: int,
    source: str,
    start_commit_index: int = 0,
) -> RoutePlan:
    frame_dt = float(token_dt) / float(frames_per_token)
    times = (
        torch.arange(int(traj7.shape[0]), dtype=torch.float32).numpy()
        * float(frame_dt)
    )
    return RoutePlan(
        times=times.astype("float32"),
        points_xyz=traj7[:, :3].detach().cpu().numpy().astype("float32"),
        start_commit_index=int(start_commit_index),
        version=1,
        source=str(source),
    )


def _build_rootrefiner_traj7(
    *,
    ldf_model,
    root_refiner,
    root_text_encoder,
    text: str,
    route_traj7: torch.Tensor,
    device: torch.device,
    token_dt: float,
    frames_per_token: int,
    forced_future_frames: int | None,
) -> tuple[torch.Tensor, object]:
    route_plan = _build_route_plan_from_traj7(
        route_traj7,
        token_dt=float(token_dt),
        frames_per_token=int(frames_per_token),
        source="turn_update_debug_route",
    )
    anchor = route_traj7[0]
    anchor_state = RootFrameState(
        commit_idx=0,
        world_xz=anchor[[0, 2]].to(device=device, dtype=torch.float32),
        world_yaw=_yaw_from_7d(route_traj7[:1])[0].to(device=device, dtype=torch.float32),
        source="turn_update_debug_anchor",
    )
    forced = (
        max(1, int(route_traj7.shape[0]) - 1)
        if forced_future_frames is None
        else int(forced_future_frames)
    )
    stream = StreamGenerator(
        ldf_model=ldf_model,
        root_refiner=root_refiner,
        root_text_encoder=root_text_encoder,
        device=device,
        token_dt=float(token_dt),
    )
    root_plan = stream.build_root_plan(
        text=str(text),
        route=route_plan,
        anchor_state=anchor_state,
        forced_num_frames=forced,
        anchor_world_y=anchor[1].to(device=device, dtype=torch.float32),
    )
    world_7d = _root_plan_to_world_7d(root_plan).detach().cpu().float()
    return world_7d, root_plan


def _clamp_refiner_future_frames(root_refiner, requested: int) -> int:
    requested = max(1, int(requested))
    min_frames = int(getattr(root_refiner, "min_frames", 1))
    max_frames = int(getattr(root_refiner, "max_frames", requested))
    return max(min_frames, min(max_frames, requested))


def _hold_last_traj7(traj7: torch.Tensor, length: int) -> torch.Tensor:
    length = max(0, int(length))
    if length <= 0:
        return traj7.new_zeros((0, 7))
    if int(traj7.shape[0]) >= length:
        return append_traj_deltas_5d_to_7d(traj7[:length, :5])
    if int(traj7.shape[0]) <= 0:
        raise ValueError("cannot hold-last an empty trajectory")
    tail = traj7[-1:, :5].expand(length - int(traj7.shape[0]), -1)
    return append_traj_deltas_5d_to_7d(torch.cat([traj7[:, :5], tail], dim=0))


def _compose_rootrefiner_condition_traj7(
    first_plan,
    second_plan,
    *,
    target_frames: int,
    frames_per_token: int,
    prefix_world_5d: torch.Tensor | None = None,
) -> torch.Tensor:
    """World-frame target used for metrics and optional root feedback.

    The actual LDF condition is still the active RootPlan payload. This dense
    array mirrors the currently active plans so the feedback path can replace
    the just-decoded root against the same target timeline.
    """
    first_world = _root_plan_to_world_7d(first_plan).detach().cpu().float()
    out = _hold_last_traj7(first_world, int(target_frames))
    if prefix_world_5d is not None:
        prefix = prefix_world_5d.detach().cpu().float()
        if prefix.dim() != 2 or prefix.shape[-1] < 5:
            raise ValueError(
                f"prefix_world_5d must be [T,>=5], got {tuple(prefix.shape)}"
            )
        prefix_len = min(int(prefix.shape[0]), int(target_frames))
        if prefix_len > 0:
            out = out.clone()
            out[:prefix_len, :5] = prefix[:prefix_len, :5]
    if second_plan is None:
        return append_traj_deltas_5d_to_7d(out[:, :5])

    second_world = _root_plan_to_world_7d(second_plan).detach().cpu().float()
    anchor_frame_idx = getattr(second_plan, "anchor_frame_idx", None)
    if anchor_frame_idx is None:
        switch_frame = token_start_frame(
            int(second_plan.anchor_commit_idx),
            int(frames_per_token),
        )
    else:
        switch_frame = int(anchor_frame_idx)
    if switch_frame >= int(target_frames):
        return out
    switch_frame = max(0, int(switch_frame))
    remaining = int(target_frames) - int(switch_frame)
    second_global = _hold_last_traj7(second_world, remaining)
    out = out.clone()
    out[switch_frame:, :5] = second_global[:, :5]
    return append_traj_deltas_5d_to_7d(out[:, :5])


def _plan_anchor_frame_idx(plan, *, frames_per_token: int) -> int:
    anchor_frame_idx = getattr(plan, "anchor_frame_idx", None)
    if anchor_frame_idx is not None:
        return int(anchor_frame_idx)
    return token_start_frame(
        int(getattr(plan, "anchor_commit_idx", 0)),
        int(frames_per_token),
    )


def _compose_multi_rootrefiner_condition_traj7(
    plans: list,
    *,
    target_frames: int,
    frames_per_token: int,
    prefix_world_5d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense world-frame condition assembled from sequential RootRefiner plans."""
    if not plans:
        raise ValueError("plans must contain at least one RootRefiner plan")
    target_frames = int(target_frames)
    first_world = _root_plan_to_world_7d(plans[0]).detach().cpu().float()
    out = _hold_last_traj7(first_world, target_frames)
    for plan in plans[1:]:
        switch_frame = _plan_anchor_frame_idx(
            plan,
            frames_per_token=int(frames_per_token),
        )
        if switch_frame >= target_frames:
            continue
        switch_frame = max(0, int(switch_frame))
        remaining = target_frames - switch_frame
        plan_world = _root_plan_to_world_7d(plan).detach().cpu().float()
        plan_global = _hold_last_traj7(plan_world, remaining)
        out = out.clone()
        out[switch_frame:, :5] = plan_global[:, :5]
    if prefix_world_5d is not None:
        prefix = prefix_world_5d.detach().cpu().float()
        if prefix.dim() != 2 or prefix.shape[-1] < 5:
            raise ValueError(
                f"prefix_world_5d must be [T,>=5], got {tuple(prefix.shape)}"
            )
        prefix_len = min(int(prefix.shape[0]), target_frames)
        if prefix_len > 0:
            out = out.clone()
            out[:prefix_len, :5] = prefix[:prefix_len, :5]
    return append_traj_deltas_5d_to_7d(out[:, :5])


def _decoded_anchor_frame_for_update(
    *,
    generated_frames: int,
    fallback_frame: int,
    target_frames: int,
) -> int:
    if int(generated_frames) > 0:
        frame = int(generated_frames) - 1
    else:
        frame = int(fallback_frame)
    return max(0, min(int(frame), max(0, int(target_frames) - 1)))


def _build_second_source_from_anchor_boundary(
    route_traj7: torch.Tensor,
    *,
    boundary_frame: int,
    anchor_state: RootFrameState,
    anchor_world_y,
) -> torch.Tensor:
    total = int(route_traj7.shape[0])
    boundary = max(0, min(int(boundary_frame), total - 1))
    second_source = route_traj7[boundary:].clone()
    if int(second_source.shape[0]) < 2:
        start = max(0, total - 2)
        second_source = route_traj7[start:].clone()
        boundary = start
    first_5d = second_source[:, :5].clone()
    anchor_xz = anchor_state.world_xz.detach().cpu().float().to(dtype=first_5d.dtype)
    anchor_y = torch.as_tensor(anchor_world_y, dtype=first_5d.dtype).detach().cpu()
    offset_x = anchor_xz[0] - first_5d[0, 0]
    offset_z = anchor_xz[1] - first_5d[0, 2]
    offset_y = anchor_y - first_5d[0, 1]
    first_5d[:, 0] = first_5d[:, 0] + offset_x
    first_5d[:, 1] = first_5d[:, 1] + offset_y
    first_5d[:, 2] = first_5d[:, 2] + offset_z
    yaw = anchor_state.world_yaw.detach().cpu().float()
    first_5d[0, 3] = torch.cos(yaw)
    first_5d[0, 4] = torch.sin(yaw)
    return append_traj_deltas_5d_to_7d(first_5d)


def _build_segment_source_from_anchor_boundary(
    route_traj7: torch.Tensor,
    *,
    boundary_frame: int,
    end_frame: int,
    anchor_state: RootFrameState,
    anchor_world_y,
) -> torch.Tensor:
    total = int(route_traj7.shape[0])
    if total < 2:
        raise ValueError("route_traj7 must contain at least two frames")
    boundary = max(0, min(int(boundary_frame), total - 1))
    end = max(boundary + 1, min(int(end_frame), total - 1))
    clipped = route_traj7[:end + 1]
    return _build_second_source_from_anchor_boundary(
        clipped,
        boundary_frame=boundary,
        anchor_state=anchor_state,
        anchor_world_y=anchor_world_y,
    )


def _jsonify_debug_record(record: dict[str, object]) -> dict[str, object]:
    def convert(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if tensor.numel() == 1:
                return float(tensor.reshape(-1)[0].item())
            return tensor.tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return {str(key): convert(value) for key, value in record.items()}


def _build_segment_source_from_runtime_anchor(
    route_traj7: torch.Tensor,
    *,
    boundary_frame: int,
    end_frame: int,
    anchor_state: RootFrameState,
    anchor_world_y,
    reanchor_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int | str]]:
    """Build a route segment that starts at the current generated actor root.

    ``route_boundary`` preserves the legacy behavior. The two current-root
    modes make the route update relative to the actor's actual runtime pose:
    translate keeps the authored world-space future direction, while pose also
    rotates the route-local future deltas into the actor's current yaw frame.
    """
    mode = str(reanchor_mode)
    if mode == "route_boundary":
        out = _build_segment_source_from_anchor_boundary(
            route_traj7,
            boundary_frame=int(boundary_frame),
            end_frame=int(end_frame),
            anchor_state=anchor_state,
            anchor_world_y=anchor_world_y,
        )
        route_boundary = route_traj7[
            max(0, min(int(boundary_frame), int(route_traj7.shape[0]) - 1))
        ]
        debug = {
            "mode": mode,
            "boundary_frame": int(boundary_frame),
            "route_boundary_xz": route_boundary[[0, 2]].detach().cpu().float(),
            "runtime_anchor_xz": anchor_state.world_xz.detach().cpu().float(),
            "xz_offset": (
                anchor_state.world_xz.detach().cpu().float()
                - route_boundary[[0, 2]].detach().cpu().float()
            ),
        }
        return out, debug
    if mode not in {"current_root_translate", "current_root_pose"}:
        raise ValueError(
            "reanchor_mode must be 'route_boundary', 'current_root_translate', "
            f"or 'current_root_pose', got {reanchor_mode!r}"
        )

    total = int(route_traj7.shape[0])
    if total < 2:
        raise ValueError("route_traj7 must contain at least two frames")
    boundary = max(0, min(int(boundary_frame), total - 1))
    end = max(boundary + 1, min(int(end_frame), total - 1))
    source = route_traj7[boundary:end + 1].clone()
    if int(source.shape[0]) < 2:
        start = max(0, total - 2)
        source = route_traj7[start:].clone()
        boundary = start

    source_5d = source[:, :5].clone()
    anchor_xz = anchor_state.world_xz.detach().cpu().float().to(dtype=source_5d.dtype)
    anchor_y = torch.as_tensor(anchor_world_y, dtype=source_5d.dtype).detach().cpu()
    anchor_yaw = anchor_state.world_yaw.detach().cpu().float().to(dtype=source_5d.dtype)

    route_boundary_xz = source_5d[0, [0, 2]].clone()
    route_boundary_y = source_5d[0, 1].clone()
    route_yaw = _yaw_from_7d(source)
    route_boundary_yaw = route_yaw[0].detach().cpu().float().to(dtype=source_5d.dtype)
    route_delta_xz = source_5d[:, [0, 2]] - route_boundary_xz[None, :]

    if mode == "current_root_pose":
        route_local_xz = _rotate_xz_delta(route_delta_xz, -route_boundary_yaw)
        new_xz = anchor_xz[None, :] + _rotate_xz_delta(route_local_xz, anchor_yaw)
    else:
        new_xz = anchor_xz[None, :] + route_delta_xz

    yaw_delta = wrap_angle(route_yaw.detach().cpu().float().to(dtype=source_5d.dtype) - route_boundary_yaw)
    new_yaw = wrap_angle(anchor_yaw + yaw_delta)
    source_5d[:, 0] = new_xz[:, 0]
    source_5d[:, 1] = anchor_y + (source_5d[:, 1] - route_boundary_y)
    source_5d[:, 2] = new_xz[:, 1]
    source_5d[:, 3] = torch.cos(new_yaw)
    source_5d[:, 4] = torch.sin(new_yaw)

    debug = {
        "mode": mode,
        "boundary_frame": int(boundary),
        "route_boundary_xz": route_boundary_xz.detach().cpu().float(),
        "runtime_anchor_xz": anchor_xz.detach().cpu().float(),
        "xz_offset": (anchor_xz - route_boundary_xz).detach().cpu().float(),
        "route_boundary_yaw": route_boundary_yaw.detach().cpu().float(),
        "runtime_anchor_yaw": anchor_yaw.detach().cpu().float(),
        "yaw_offset": wrap_angle(anchor_yaw - route_boundary_yaw).detach().cpu().float(),
    }
    return append_traj_deltas_5d_to_7d(source_5d), debug


def _resolve_update_anchor_yaw_state(
    anchor_state: RootFrameState,
    segment_source: torch.Tensor,
    *,
    mode: str,
    tangent_frames: int = 20,
) -> tuple[RootFrameState, dict[str, object]]:
    mode = str(mode)
    old_yaw = anchor_state.world_yaw.detach().cpu().float()
    if mode == "runtime":
        debug = {
            "mode": mode,
            "old_yaw": old_yaw,
            "new_yaw": old_yaw,
            "yaw_delta": old_yaw.new_tensor(0.0),
            "tangent_frames": int(tangent_frames),
        }
        return anchor_state, debug
    if mode != "path_tangent":
        raise ValueError(
            "root_refiner update anchor yaw mode must be 'runtime' or "
            f"'path_tangent', got {mode!r}"
        )

    source = segment_source.detach().cpu().float()
    if source.dim() != 2 or source.shape[-1] < 3 or int(source.shape[0]) < 2:
        new_yaw = old_yaw
    else:
        xz = source[:, [0, 2]]
        end = min(int(xz.shape[0]) - 1, max(1, int(tangent_frames)))
        delta = xz[end] - xz[0]
        if float(torch.linalg.norm(delta).item()) < 1e-6:
            yaw_series = _path_yaw_from_xz(xz)
            new_yaw = yaw_series[0].detach().cpu().float()
        else:
            new_yaw = torch.atan2(delta[0], delta[1]).detach().cpu().float()
    new_yaw = wrap_angle(new_yaw.to(dtype=old_yaw.dtype))
    out = RootFrameState(
        commit_idx=int(anchor_state.commit_idx),
        world_xz=anchor_state.world_xz.clone(),
        world_yaw=new_yaw.to(
            device=anchor_state.world_yaw.device,
            dtype=anchor_state.world_yaw.dtype,
        ),
        source=f"{anchor_state.source}:anchor_yaw_{mode}",
    )
    debug = {
        "mode": mode,
        "old_yaw": old_yaw,
        "new_yaw": new_yaw,
        "yaw_delta": wrap_angle(new_yaw - old_yaw),
        "tangent_frames": int(tangent_frames),
    }
    return out, debug


def _generated_history_world_5d(decoded_chunks: list[torch.Tensor], device: torch.device):
    if not decoded_chunks:
        return None
    decoded = torch.cat(decoded_chunks, dim=0).to(device=device).unsqueeze(0)
    root_quat, root_xyz = recover_root_rot_pos(decoded)
    traj7 = root_to_traj_feats_7d(root_quat, root_xyz)[0]
    return traj7[:, :5].detach()


def _build_rootrefiner_plan_for_route(
    *,
    stream: StreamGenerator,
    text: str,
    route_traj7: torch.Tensor,
    anchor_state: RootFrameState,
    forced_future_frames: int,
    token_dt: float,
    frames_per_token: int,
    source: str,
    anchor_world_y: float | torch.Tensor,
    history_motion_world_5d=None,
    reanchor_to_anchor_xz: bool = False,
    mask_front_ratio: float = 0.0,
    mask_kind: str = "control",
):
    route_plan = _build_route_plan_from_traj7(
        route_traj7,
        token_dt=float(token_dt),
        frames_per_token=int(frames_per_token),
        source=str(source),
        start_commit_index=int(anchor_state.commit_idx),
    )
    if bool(reanchor_to_anchor_xz):
        route_plan = reanchor_route_to_xz(
            route_plan,
            anchor_state.world_xz.detach().cpu().numpy(),
        )
    condition = None
    refiner = getattr(stream, "root_refiner", None)
    if refiner is not None:
        max_frames = int(refiner.max_frames)
        condition = stream.condition_manager.route.build_root_refiner_path_condition_for_route(
            route_plan,
            anchor_state=anchor_state,
            n_path=int(refiner.n_path),
            valid_frame_count=max_frames,
            max_frames=max_frames,
        )
        if condition is not None:
            condition, mask_debug = _mask_rootrefiner_condition_front(
                condition,
                ratio=float(mask_front_ratio),
                kind=str(mask_kind),
            )
        else:
            mask_debug = {
                "kind": str(mask_kind),
                "ratio": float(mask_front_ratio),
                "valid_points": 0,
                "masked_points": 0,
                "masked_indices": [],
            }
    else:
        mask_debug = {
            "kind": str(mask_kind),
            "ratio": float(mask_front_ratio),
            "valid_points": 0,
            "masked_points": 0,
            "masked_indices": [],
        }
    root_plan = stream.build_root_plan(
        text=str(text),
        route=route_plan,
        anchor_state=anchor_state,
        history_motion_world_5d=history_motion_world_5d,
        forced_num_frames=int(forced_future_frames),
        anchor_world_y=anchor_world_y,
        path_condition=condition,
    )
    if condition is not None:
        root_plan.debug_rootrefiner_mask_front = _jsonify_debug_record(mask_debug)
        root_plan.debug_rootrefiner_input_path_world_xz = (
            _rootrefiner_input_path_world_xz(condition, root_plan)
        )
        root_plan.debug_rootrefiner_input_path_local_xz = (
            condition.path[condition.path_valid_mask.bool()].detach().cpu().float()
        )
        root_plan.debug_route_points_world_xz = torch.as_tensor(
            route_plan.points_xyz[:, [0, 2]],
            dtype=torch.float32,
        )
    return root_plan


def _load_sample(args: argparse.Namespace, cfg):
    _, dataloader = build_eval_dataloader(
        cfg,
        meta_paths=[args.meta_path],
        batch_size=1,
        num_workers=0,
        group_present_segments=False,
    )
    last_name = None
    for batch in dataloader:
        sample_batch = _slice_single_sample_batch(batch, 0)
        name = str(sample_batch["name"][0])
        last_name = name
        if not args.sample_name or name == str(args.sample_name):
            _select_caption(sample_batch, int(args.caption_index))
            return sample_batch
    raise ValueError(
        f"Expected sample {args.sample_name!r}, but it was not found in "
        f"{args.meta_path!r}; last sample seen was {last_name!r}"
    )


def _root_xz(motion: torch.Tensor) -> torch.Tensor:
    return extract_root_trajectory_263_torch(motion.unsqueeze(0))[0][:, [0, 2]].cpu()


def _root_7d_from_motion(motion: torch.Tensor) -> torch.Tensor:
    root_quat, root_xyz = recover_root_rot_pos(motion.unsqueeze(0))
    return root_to_traj_feats_7d(root_quat, root_xyz)[0].detach().cpu().float()


def _numpy_7d(value) -> np.ndarray:
    if value is None:
        return np.zeros((0, 7), dtype=np.float32)
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _numpy_xz(value) -> np.ndarray:
    if value is None:
        return np.zeros((0, 2), dtype=np.float32)
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _plot_7d_heading_debug(
    output_path: Path,
    *,
    series: dict[str, torch.Tensor | np.ndarray],
    boundary_frames: list[int],
    title: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    for label, traj in series.items():
        arr = _numpy_7d(traj)
        if arr.shape[0] <= 0:
            continue
        xz = arr[:, [0, 2]]
        ax.plot(xz[:, 0], xz[:, 1], linewidth=1.5, label=label)
        ax.scatter(xz[0, 0], xz[0, 1], s=18)
        step = max(1, int(arr.shape[0]) // 24)
        idx = np.arange(0, arr.shape[0], step, dtype=np.int64)
        if idx[-1] != arr.shape[0] - 1:
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
            alpha=0.75,
        )
        for frame in boundary_frames:
            if 0 <= int(frame) < arr.shape[0]:
                ax.scatter(
                    xz[int(frame), 0],
                    xz[int(frame), 1],
                    s=52,
                    marker="x",
                    color="black",
                    zorder=5,
                )
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def _save_rootrefiner_update_debug_artifacts(
    artifacts_dir: Path,
    *,
    label: str,
    run_idx: int,
    route_traj7: torch.Tensor,
    condition_traj7: torch.Tensor,
    decoded_feature: torch.Tensor,
    run_out: dict,
) -> tuple[str, str]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{label}_run{int(run_idx)}"
    first_plan = run_out.get("first_root_plan")
    second_plan = run_out.get("second_root_plan")
    first_world = (
        _root_plan_to_world_7d(first_plan).detach().cpu().float()
        if first_plan is not None
        else torch.zeros((0, 7), dtype=torch.float32)
    )
    second_world = (
        _root_plan_to_world_7d(second_plan).detach().cpu().float()
        if second_plan is not None
        else torch.zeros((0, 7), dtype=torch.float32)
    )
    first_local = (
        first_plan.waypoints_local_7d[: int(first_plan.valid_frames)].detach().cpu().float()
        if first_plan is not None
        else torch.zeros((0, 7), dtype=torch.float32)
    )
    second_local = (
        second_plan.waypoints_local_7d[: int(second_plan.valid_frames)].detach().cpu().float()
        if second_plan is not None
        else torch.zeros((0, 7), dtype=torch.float32)
    )
    generated_root_7d = _root_7d_from_motion(decoded_feature)
    npz_path = artifacts_dir / f"root_debug_{suffix}.npz"
    np.savez_compressed(
        npz_path,
        manual_route_7d=_numpy_7d(route_traj7),
        condition_sent_world_7d=_numpy_7d(condition_traj7),
        first_rootrefiner_world_7d=_numpy_7d(first_world),
        second_rootrefiner_world_7d=_numpy_7d(second_world),
        first_rootrefiner_local_7d=_numpy_7d(first_local),
        second_rootrefiner_local_7d=_numpy_7d(second_local),
        first_rootrefiner_input_path_world_xz=_numpy_xz(
            getattr(first_plan, "debug_rootrefiner_input_path_world_xz", None)
        ),
        second_rootrefiner_input_path_world_xz=_numpy_xz(
            getattr(second_plan, "debug_rootrefiner_input_path_world_xz", None)
        ),
        first_route_points_world_xz=_numpy_xz(
            getattr(first_plan, "debug_route_points_world_xz", None)
        ),
        second_route_points_world_xz=_numpy_xz(
            getattr(second_plan, "debug_route_points_world_xz", None)
        ),
        generated_root_7d=_numpy_7d(generated_root_7d),
        update_commit=np.asarray([int(run_out.get("update_commit", -1))], dtype=np.int64),
        switch_frame=np.asarray([int(run_out.get("switch_frame", -1))], dtype=np.int64),
    )
    plot_path = artifacts_dir / f"root_debug_heading_{suffix}.png"
    _plot_7d_heading_debug(
        plot_path,
        series={
            "manual_route_7d": route_traj7,
            "condition_sent_world_7d": condition_traj7,
            "first_rootrefiner_world_7d": first_world,
            "second_rootrefiner_world_7d": second_world,
            "generated_root_7d": generated_root_7d,
        },
        boundary_frames=[
            int(run_out.get("switch_frame", -1)),
        ],
        title=f"{suffix}: sent condition vs RootRefiner decoded 7D",
    )
    return str(npz_path), str(plot_path)


def _run_one(
    model,
    vae,
    sample_batch: dict,
    args: argparse.Namespace,
    device: torch.device,
    *,
    alpha: float | None,
    run_idx: int = 0,
):
    seed = _stable_eval_seed(
        int(args.seed),
        str(args.probe_tag),
        str(sample_batch["name"][0]),
        int(run_idx),
    )
    _seed_eval_locally(seed)
    start = time.perf_counter()
    with torch.no_grad():
        out = run_stream_generate_step_sample(
            model=model,
            vae=vae,
            sample_batch=sample_batch,
            device=device,
            history_length=int(args.history_length),
            num_denoise_steps=args.num_denoise_steps,
            traj_horizon_tokens=int(args.horizon_tokens),
            token_dt=float(args.token_dt),
            frames_per_token=int(args.frames_per_token),
            root_replace_feedback=alpha is not None,
            root_feedback_xz_blend_alpha=1.0 if alpha is None else float(alpha),
        )
    elapsed = time.perf_counter() - start
    metrics = _compute_traj_metrics(out["decoded_feature"], sample_batch, 0, seg_size=20)
    return out, metrics, elapsed, seed


def _run_rootrefiner_update_one(
    model,
    vae,
    sample_batch: dict,
    *,
    original_traj7: torch.Tensor,
    route_traj7: torch.Tensor,
    update_frame: int,
    root_refiner,
    root_text_encoder,
    args: argparse.Namespace,
    device: torch.device,
    alpha: float | None,
    run_idx: int = 0,
):
    seed = _stable_eval_seed(
        int(args.seed),
        f"{args.probe_tag}_rootrefiner_update",
        str(sample_batch["name"][0]),
        int(run_idx),
    )
    _seed_eval_locally(seed)

    frames_per_token = int(args.frames_per_token)
    target_total_frames = int(route_traj7.shape[0])
    step_count = num_tokens_for_frame_len(target_total_frames, frames_per_token)
    update_commit = frame_idx_to_token_idx(int(update_frame), frames_per_token)
    token_switch_frame = token_start_frame(update_commit, frames_per_token)
    condition_switch_frame = int(token_switch_frame)
    text = str(sample_batch.get("_caption_text") or sample_batch["text"][0])
    first_future_frames = int(original_traj7.shape[0]) - 1
    if args.root_refiner_forced_frames is not None:
        first_future_frames = int(args.root_refiner_forced_frames)
    first_future_frames = _clamp_refiner_future_frames(
        root_refiner,
        first_future_frames,
    )

    initial_anchor = original_traj7[0].to(device=device, dtype=torch.float32)
    initial_state = RootFrameState(
        commit_idx=0,
        world_xz=initial_anchor[[0, 2]].clone(),
        world_yaw=_yaw_from_7d(original_traj7[:1])[0].to(device=device, dtype=torch.float32),
        source="root_refiner_update_initial_anchor",
    )

    stream = StreamGenerator(
        ldf_model=model,
        root_refiner=root_refiner,
        root_text_encoder=root_text_encoder,
        device=device,
        history_length=int(args.history_length),
        traj_horizon_tokens=int(args.horizon_tokens),
        token_dt=float(args.token_dt),
    )
    stream.reset(initial_state, text=text)
    stream.init_ldf_generation(
        history_length=int(args.history_length),
        batch_size=1,
        num_denoise_steps=(
            args.num_denoise_steps
            if args.num_denoise_steps is not None
            else int(getattr(model, "noise_steps"))
        ),
    )
    vae.clear_cache()

    first_plan = _build_rootrefiner_plan_for_route(
        stream=stream,
        text=text,
        route_traj7=original_traj7,
        anchor_state=initial_state,
        forced_future_frames=first_future_frames,
        token_dt=float(args.token_dt),
        frames_per_token=frames_per_token,
        source="root_refiner_update_first_route",
        anchor_world_y=initial_anchor[1],
        history_motion_world_5d=None,
        reanchor_to_anchor_xz=False,
    )
    stream_conditioner = LdfEvalStreamConditioner(
        sample_batch,
        history_length=int(args.history_length),
        traj_horizon_tokens=int(args.horizon_tokens),
        token_dt=float(args.token_dt),
        frames_per_token=frames_per_token,
        device=device,
    )
    stream_conditioner.timeline = RootTimeline(initial_state)
    stream_conditioner.root_plan = first_plan
    stream_conditioner._anchor_xz = initial_state.world_xz.clone()
    stream_conditioner._anchor_yaw = initial_state.world_yaw.clone()

    text_rollout = StreamTextRolloutController.from_sample_batch(sample_batch)
    stream_recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    condition_traj7 = _compose_rootrefiner_condition_traj7(
        first_plan,
        None,
        target_frames=target_total_frames,
        frames_per_token=frames_per_token,
    )
    feedback_batch = apply_updated_traj_to_sample_batch(sample_batch, condition_traj7)

    second_plan = None
    update_triggered = False
    first_chunk = True
    latent_tokens: list[torch.Tensor] = []
    decoded_chunks: list[torch.Tensor] = []
    chunk_frame_ends: list[int] = []
    generated_frames = 0
    start = time.perf_counter()
    try:
        for commit_index in range(step_count):
            if (
                not update_triggered
                and int(commit_index) >= int(update_commit)
                and stream_conditioner.timeline.has_exact_state(update_commit)
            ):
                anchor_state = stream_conditioner.timeline.at_commit(update_commit)
                history_5d = _generated_history_world_5d(decoded_chunks, device)
                condition_switch_frame = _decoded_anchor_frame_for_update(
                    generated_frames=int(generated_frames),
                    fallback_frame=int(token_switch_frame),
                    target_frames=int(target_total_frames),
                )
                if history_5d is not None and int(history_5d.shape[0]) > 0:
                    anchor_y = history_5d[-1, 1]
                else:
                    anchor_y = route_traj7[
                        min(condition_switch_frame, target_total_frames - 1),
                        1,
                    ]
                second_source = _build_second_source_from_anchor_boundary(
                    route_traj7,
                    boundary_frame=int(condition_switch_frame),
                    anchor_state=anchor_state,
                    anchor_world_y=anchor_y,
                )
                refiner_anchor_state, anchor_yaw_debug = _resolve_update_anchor_yaw_state(
                    anchor_state,
                    second_source,
                    mode=str(args.root_refiner_update_anchor_yaw_mode),
                    tangent_frames=int(args.root_refiner_update_anchor_tangent_frames),
                )
                if args.root_refiner_forced_frames is None:
                    second_future_frames = target_total_frames - condition_switch_frame - 1
                else:
                    second_future_frames = int(args.root_refiner_forced_frames)
                second_future_frames = _clamp_refiner_future_frames(
                    root_refiner,
                    second_future_frames,
                )
                second_plan = _build_rootrefiner_plan_for_route(
                    stream=stream,
                    text=text_rollout.get_text_for_commit_index(commit_index),
                    route_traj7=second_source,
                    anchor_state=refiner_anchor_state,
                    forced_future_frames=second_future_frames,
                    token_dt=float(args.token_dt),
                    frames_per_token=frames_per_token,
                    source="root_refiner_update_second_route",
                    anchor_world_y=anchor_y,
                    history_motion_world_5d=history_5d,
                    reanchor_to_anchor_xz=True,
                    mask_front_ratio=float(args.root_refiner_update_mask_front_ratio),
                    mask_kind=str(args.root_refiner_update_mask_kind),
                )
                second_plan.anchor_frame_idx = int(condition_switch_frame)
                second_plan.debug_rootrefiner_update_anchor_yaw = (
                    _jsonify_debug_record(anchor_yaw_debug)
                )
                stream_conditioner.root_plan = second_plan
                condition_traj7 = _compose_rootrefiner_condition_traj7(
                    first_plan,
                    second_plan,
                    target_frames=target_total_frames,
                    frames_per_token=frames_per_token,
                    prefix_world_5d=history_5d,
                )
                feedback_batch = apply_updated_traj_to_sample_batch(
                    sample_batch,
                    condition_traj7,
                )
                update_triggered = True

            current_text = text_rollout.get_text_for_commit_index(commit_index)
            local_commit_index = int(getattr(model, "commit_index", commit_index))
            chunk_size = int(getattr(model, "chunk_size", 1))
            traj_input = stream_conditioner.build_step_payload(
                local_commit_index=local_commit_index,
                absolute_commit_index=int(commit_index),
                chunk_size=chunk_size,
            )
            step_payload = stream.build_step_input(current_text, traj_input=traj_input)
            condition_provider = stream.build_ldf_condition_provider(
                step_payload,
                first_chunk=first_chunk,
                device=device,
            )
            output = model.stream_generate_step(
                step_payload,
                first_chunk=first_chunk,
                condition=condition_provider,
            )
            latent_token = output["generated"][0].detach().cpu()
            if alpha is not None:
                decoded_chunk_raw = _decode_raw_chunk_preserving_feedback_cache(
                    vae,
                    latent_token,
                    first_chunk=first_chunk,
                    device=device,
                )
                decoded_chunk = _replace_chunk_root_from_condition(
                    decoded_chunk_raw,
                    feedback_batch,
                    start_frame=int(generated_frames),
                    previous_decoded_chunks=decoded_chunks,
                    xz_blend_alpha=float(alpha),
                )
                corrected_latent = _encode_corrected_chunk_token(
                    vae,
                    decoded_chunk,
                    first_chunk=first_chunk,
                    device=device,
                )
                _decode_latent_chunk(
                    vae,
                    corrected_latent,
                    first_chunk=first_chunk,
                    device=device,
                )
                _write_committed_latent_to_model(
                    model,
                    corrected_latent,
                    local_commit_index,
                )
                latent_token = corrected_latent
            else:
                decoded_chunk = vae.stream_decode(
                    latent_token.to(device=device).unsqueeze(0),
                    first_chunk=first_chunk,
                )[0].float().detach().cpu()

            first_chunk = False
            latent_tokens.append(latent_token)
            decoded_chunks.append(decoded_chunk)
            generated_frames += int(decoded_chunk.shape[0])
            chunk_frame_ends.append(min(int(generated_frames), target_total_frames))
            stream_conditioner.append_decoded(
                decoded_chunk,
                commit_idx=int(commit_index) + 1,
                recovery=stream_recovery,
            )
            if generated_frames >= target_total_frames:
                break
    finally:
        vae.clear_cache()

    elapsed = time.perf_counter() - start
    decoded_feature = (
        torch.cat(decoded_chunks, dim=0)[:target_total_frames]
        if decoded_chunks
        else torch.zeros((0, 263), dtype=torch.float32)
    )
    latent_stream = (
        torch.cat(latent_tokens, dim=0)
        if latent_tokens
        else torch.zeros((0, model.input_dim), dtype=torch.float32)
    )
    metric_batch = apply_updated_traj_to_sample_batch(sample_batch, condition_traj7)
    condition_metrics = _compute_traj_metrics(
        decoded_feature,
        metric_batch,
        0,
        seg_size=20,
    )
    gt_route_batch = apply_updated_traj_to_sample_batch(
        sample_batch,
        _hold_last_traj7(route_traj7, target_total_frames),
    )
    gt_route_metrics = _compute_traj_metrics(
        decoded_feature,
        gt_route_batch,
        0,
        seg_size=20,
    )
    metrics = dict(condition_metrics)
    metrics.update({f"condition_{key}": value for key, value in condition_metrics.items()})
    metrics.update({f"gt_route_{key}": value for key, value in gt_route_metrics.items()})
    out = {
        "decoded_feature": decoded_feature,
        "decoded_chunks": decoded_chunks,
        "chunk_frame_ends": chunk_frame_ends,
        "latent_stream": latent_stream,
        "original_total_frames": int(original_traj7.shape[0]),
        "target_total_frames": int(target_total_frames),
        "extra_frames": 0,
        "root_replace_feedback": alpha is not None,
        "root_feedback_xz_blend_alpha": 1.0 if alpha is None else float(alpha),
        "condition_traj7": condition_traj7,
        "first_root_plan": first_plan,
        "second_root_plan": second_plan,
        "update_commit": int(update_commit),
        "switch_frame": int(condition_switch_frame),
        "token_switch_frame": int(token_switch_frame),
        "update_triggered": bool(update_triggered),
        "update_mask_front_ratio": float(args.root_refiner_update_mask_front_ratio),
        "update_mask_kind": str(args.root_refiner_update_mask_kind),
        "update_anchor_yaw_mode": str(args.root_refiner_update_anchor_yaw_mode),
        "update_anchor_tangent_frames": int(
            args.root_refiner_update_anchor_tangent_frames
        ),
    }
    return out, metrics, elapsed, seed


def _run_rootrefiner_multi_update_one(
    model,
    vae,
    sample_batch: dict,
    *,
    original_traj7: torch.Tensor,
    route_traj7: torch.Tensor,
    update_frames: list[int],
    root_refiner,
    root_text_encoder,
    args: argparse.Namespace,
    device: torch.device,
    alpha: float | None,
    run_idx: int = 0,
):
    seed = _stable_eval_seed(
        int(args.seed),
        f"{args.probe_tag}_rootrefiner_multi_update",
        str(sample_batch["name"][0]),
        int(run_idx),
    )
    _seed_eval_locally(seed)

    frames_per_token = int(args.frames_per_token)
    target_total_frames = int(route_traj7.shape[0])
    step_count = num_tokens_for_frame_len(target_total_frames, frames_per_token)
    update_frames = sorted(
        {
            max(1, min(int(frame), target_total_frames - 2))
            for frame in update_frames
        }
    )
    update_commits = [
        frame_idx_to_token_idx(frame, frames_per_token) for frame in update_frames
    ]
    text = str(sample_batch.get("_caption_text") or sample_batch["text"][0])

    initial_anchor = original_traj7[0].to(device=device, dtype=torch.float32)
    initial_state = RootFrameState(
        commit_idx=0,
        world_xz=initial_anchor[[0, 2]].clone(),
        world_yaw=_yaw_from_7d(original_traj7[:1])[0].to(device=device, dtype=torch.float32),
        source="root_refiner_multi_update_initial_anchor",
    )

    stream = StreamGenerator(
        ldf_model=model,
        root_refiner=root_refiner,
        root_text_encoder=root_text_encoder,
        device=device,
        history_length=int(args.history_length),
        traj_horizon_tokens=int(args.horizon_tokens),
        token_dt=float(args.token_dt),
    )
    stream.reset(initial_state, text=text)
    stream.init_ldf_generation(
        history_length=int(args.history_length),
        batch_size=1,
        num_denoise_steps=(
            args.num_denoise_steps
            if args.num_denoise_steps is not None
            else int(getattr(model, "noise_steps"))
        ),
    )
    vae.clear_cache()

    segment_ends = update_frames + [target_total_frames - 1]
    first_source = route_traj7[: segment_ends[0] + 1]
    first_future_frames = (
        int(args.root_refiner_forced_frames)
        if args.root_refiner_forced_frames is not None
        else max(1, int(first_source.shape[0]) - 1)
    )
    first_future_frames = _clamp_refiner_future_frames(
        root_refiner,
        first_future_frames,
    )
    first_plan = _build_rootrefiner_plan_for_route(
        stream=stream,
        text=text,
        route_traj7=first_source,
        anchor_state=initial_state,
        forced_future_frames=first_future_frames,
        token_dt=float(args.token_dt),
        frames_per_token=frames_per_token,
        source="root_refiner_multi_update_segment_0",
        anchor_world_y=initial_anchor[1],
        history_motion_world_5d=None,
        reanchor_to_anchor_xz=False,
    )
    first_plan.anchor_frame_idx = 0
    root_plans = [first_plan]
    stream_conditioner = LdfEvalStreamConditioner(
        sample_batch,
        history_length=int(args.history_length),
        traj_horizon_tokens=int(args.horizon_tokens),
        token_dt=float(args.token_dt),
        frames_per_token=frames_per_token,
        device=device,
    )
    stream_conditioner.timeline = RootTimeline(initial_state)
    stream_conditioner.root_plan = first_plan
    stream_conditioner._anchor_xz = initial_state.world_xz.clone()
    stream_conditioner._anchor_yaw = initial_state.world_yaw.clone()

    text_rollout = StreamTextRolloutController.from_sample_batch(sample_batch)
    stream_recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    condition_traj7 = _compose_multi_rootrefiner_condition_traj7(
        root_plans,
        target_frames=target_total_frames,
        frames_per_token=frames_per_token,
    )
    feedback_batch = apply_updated_traj_to_sample_batch(sample_batch, condition_traj7)

    next_update_idx = 0
    triggered_flags = [False for _ in update_frames]
    switch_frames: list[int] = []
    reanchor_debug_records: list[dict[str, object]] = []
    latent_tokens: list[torch.Tensor] = []
    decoded_chunks: list[torch.Tensor] = []
    chunk_frame_ends: list[int] = []
    first_chunk = True
    generated_frames = 0
    start = time.perf_counter()
    try:
        for commit_index in range(step_count):
            if next_update_idx < len(update_commits):
                update_commit = int(update_commits[next_update_idx])
                if (
                    int(commit_index) >= update_commit
                    and stream_conditioner.timeline.has_exact_state(update_commit)
                ):
                    anchor_state = stream_conditioner.timeline.at_commit(update_commit)
                    history_5d = _generated_history_world_5d(decoded_chunks, device)
                    fallback_frame = token_start_frame(update_commit, frames_per_token)
                    condition_switch_frame = _decoded_anchor_frame_for_update(
                        generated_frames=int(generated_frames),
                        fallback_frame=int(fallback_frame),
                        target_frames=int(target_total_frames),
                    )
                    if history_5d is not None and int(history_5d.shape[0]) > 0:
                        anchor_y = history_5d[-1, 1]
                    else:
                        anchor_y = route_traj7[
                            min(condition_switch_frame, target_total_frames - 1),
                            1,
                        ]
                    segment_end = segment_ends[next_update_idx + 1]
                    segment_source, reanchor_debug = _build_segment_source_from_runtime_anchor(
                        route_traj7,
                        boundary_frame=int(condition_switch_frame),
                        end_frame=int(segment_end),
                        anchor_state=anchor_state,
                        anchor_world_y=anchor_y,
                        reanchor_mode=str(args.root_update_reanchor_mode),
                    )
                    refiner_anchor_state, anchor_yaw_debug = (
                        _resolve_update_anchor_yaw_state(
                            anchor_state,
                            segment_source,
                            mode=str(args.root_refiner_update_anchor_yaw_mode),
                            tangent_frames=int(
                                args.root_refiner_update_anchor_tangent_frames
                            ),
                        )
                    )
                    reanchor_debug = {
                        **reanchor_debug,
                        "rootrefiner_anchor_yaw": anchor_yaw_debug,
                    }
                    reanchor_debug_records.append(_jsonify_debug_record(reanchor_debug))
                    future_frames = (
                        int(args.root_refiner_forced_frames)
                        if args.root_refiner_forced_frames is not None
                        else max(1, int(segment_source.shape[0]) - 1)
                    )
                    future_frames = _clamp_refiner_future_frames(
                        root_refiner,
                        future_frames,
                    )
                    plan = _build_rootrefiner_plan_for_route(
                        stream=stream,
                        text=text_rollout.get_text_for_commit_index(commit_index),
                        route_traj7=segment_source,
                        anchor_state=refiner_anchor_state,
                        forced_future_frames=future_frames,
                        token_dt=float(args.token_dt),
                        frames_per_token=frames_per_token,
                        source=f"root_refiner_multi_update_segment_{next_update_idx + 1}",
                        anchor_world_y=anchor_y,
                        history_motion_world_5d=history_5d,
                        reanchor_to_anchor_xz=True,
                        mask_front_ratio=float(args.root_refiner_update_mask_front_ratio),
                        mask_kind=str(args.root_refiner_update_mask_kind),
                    )
                    plan.anchor_frame_idx = int(condition_switch_frame)
                    plan.debug_rootrefiner_update_anchor_yaw = (
                        _jsonify_debug_record(anchor_yaw_debug)
                    )
                    root_plans.append(plan)
                    stream_conditioner.root_plan = plan
                    condition_traj7 = _compose_multi_rootrefiner_condition_traj7(
                        root_plans,
                        target_frames=target_total_frames,
                        frames_per_token=frames_per_token,
                        prefix_world_5d=history_5d,
                    )
                    feedback_batch = apply_updated_traj_to_sample_batch(
                        sample_batch,
                        condition_traj7,
                    )
                    triggered_flags[next_update_idx] = True
                    switch_frames.append(int(condition_switch_frame))
                    next_update_idx += 1

            current_text = text_rollout.get_text_for_commit_index(commit_index)
            local_commit_index = int(getattr(model, "commit_index", commit_index))
            chunk_size = int(getattr(model, "chunk_size", 1))
            traj_input = stream_conditioner.build_step_payload(
                local_commit_index=local_commit_index,
                absolute_commit_index=int(commit_index),
                chunk_size=chunk_size,
            )
            step_payload = stream.build_step_input(current_text, traj_input=traj_input)
            condition_provider = stream.build_ldf_condition_provider(
                step_payload,
                first_chunk=first_chunk,
                device=device,
            )
            output = model.stream_generate_step(
                step_payload,
                first_chunk=first_chunk,
                condition=condition_provider,
            )
            latent_token = output["generated"][0].detach().cpu()
            if alpha is not None:
                decoded_chunk_raw = _decode_raw_chunk_preserving_feedback_cache(
                    vae,
                    latent_token,
                    first_chunk=first_chunk,
                    device=device,
                )
                decoded_chunk = _replace_chunk_root_from_condition(
                    decoded_chunk_raw,
                    feedback_batch,
                    start_frame=int(generated_frames),
                    previous_decoded_chunks=decoded_chunks,
                    xz_blend_alpha=float(alpha),
                )
                corrected_latent = _encode_corrected_chunk_token(
                    vae,
                    decoded_chunk,
                    first_chunk=first_chunk,
                    device=device,
                )
                _decode_latent_chunk(
                    vae,
                    corrected_latent,
                    first_chunk=first_chunk,
                    device=device,
                )
                _write_committed_latent_to_model(
                    model,
                    corrected_latent,
                    local_commit_index,
                )
                latent_token = corrected_latent
            else:
                decoded_chunk = vae.stream_decode(
                    latent_token.to(device=device).unsqueeze(0),
                    first_chunk=first_chunk,
                )[0].float().detach().cpu()

            first_chunk = False
            latent_tokens.append(latent_token)
            decoded_chunks.append(decoded_chunk)
            generated_frames += int(decoded_chunk.shape[0])
            chunk_frame_ends.append(min(int(generated_frames), target_total_frames))
            stream_conditioner.append_decoded(
                decoded_chunk,
                commit_idx=int(commit_index) + 1,
                recovery=stream_recovery,
            )
            if generated_frames >= target_total_frames:
                break
    finally:
        vae.clear_cache()

    elapsed = time.perf_counter() - start
    decoded_feature = (
        torch.cat(decoded_chunks, dim=0)[:target_total_frames]
        if decoded_chunks
        else torch.zeros((0, 263), dtype=torch.float32)
    )
    latent_stream = (
        torch.cat(latent_tokens, dim=0)
        if latent_tokens
        else torch.zeros((0, model.input_dim), dtype=torch.float32)
    )
    metric_batch = apply_updated_traj_to_sample_batch(sample_batch, condition_traj7)
    condition_metrics = _compute_traj_metrics(
        decoded_feature,
        metric_batch,
        0,
        seg_size=20,
    )
    gt_route_batch = apply_updated_traj_to_sample_batch(
        sample_batch,
        _hold_last_traj7(route_traj7, target_total_frames),
    )
    gt_route_metrics = _compute_traj_metrics(
        decoded_feature,
        gt_route_batch,
        0,
        seg_size=20,
    )
    metrics = dict(condition_metrics)
    metrics.update({f"condition_{key}": value for key, value in condition_metrics.items()})
    metrics.update({f"gt_route_{key}": value for key, value in gt_route_metrics.items()})
    out = {
        "decoded_feature": decoded_feature,
        "decoded_chunks": decoded_chunks,
        "chunk_frame_ends": chunk_frame_ends,
        "latent_stream": latent_stream,
        "original_total_frames": int(original_traj7.shape[0]),
        "target_total_frames": int(target_total_frames),
        "extra_frames": 0,
        "root_replace_feedback": alpha is not None,
        "root_feedback_xz_blend_alpha": 1.0 if alpha is None else float(alpha),
        "condition_traj7": condition_traj7,
        "first_root_plan": root_plans[0],
        "second_root_plan": root_plans[-1] if len(root_plans) > 1 else None,
        "root_plans": root_plans,
        "update_frames": [int(frame) for frame in update_frames],
        "update_commits": [int(commit) for commit in update_commits],
        "switch_frames": [int(frame) for frame in switch_frames],
        "update_triggered": bool(all(triggered_flags)) if triggered_flags else False,
        "update_triggered_flags": [bool(flag) for flag in triggered_flags],
        "update_reanchor_mode": str(args.root_update_reanchor_mode),
        "update_reanchor_debug": reanchor_debug_records,
        "update_mask_front_ratio": float(args.root_refiner_update_mask_front_ratio),
        "update_mask_kind": str(args.root_refiner_update_mask_kind),
        "update_anchor_yaw_mode": str(args.root_refiner_update_anchor_yaw_mode),
        "update_anchor_tangent_frames": int(
            args.root_refiner_update_anchor_tangent_frames
        ),
    }
    return out, metrics, elapsed, seed


def _video_name_for_alpha(alpha: float | None, *, run_idx: int = 0) -> str:
    if alpha is None:
        return f"stream_update_no_feedback_run{int(run_idx)}.mp4"
    suffix = int(round(float(alpha) * 100))
    return f"stream_update_feedback_alpha{suffix:03d}_run{int(run_idx)}.mp4"


def condition_visual_mask(
    num_frames: int,
    display_mode: str,
    update_frames: Iterable[int] | None,
) -> torch.Tensor:
    """Mask red condition overlay for route-update debug videos."""
    count = max(0, int(num_frames))
    if count == 0:
        return torch.zeros(0, dtype=torch.float32)
    if str(display_mode) == "full":
        return torch.ones(count, dtype=torch.float32)
    if str(display_mode) != "future":
        raise ValueError(
            f"condition_traj_display must be 'future' or 'full', got {display_mode!r}"
        )
    starts = []
    for frame in update_frames or []:
        starts.append(max(0, min(int(frame), count)))
    start = min(starts) if starts else 0
    mask = torch.zeros(count, dtype=torch.float32)
    if start < count:
        mask[start:] = 1.0
    return mask


def main() -> int:
    args = _parse_args()
    torch.cuda.set_device(int(args.gpu))
    device = torch.device(f"cuda:{int(args.gpu)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(int(args.seed))
    cfg = load_config(config_path=args.config)
    vae_ckpt = args.vae_ckpt or cfg.get("test_vae_ckpt", None)
    model, vae = load_eval_model_and_vae(
        cfg,
        ckpt_path=args.ckpt,
        vae_ckpt_path=vae_ckpt,
        device=device,
        use_ema=True,
    )
    if str(cfg.get("eval.text_device", "cpu")).lower() == "cpu":
        enable_cpu_text_encoding(model)
    model.cfg_scale_text = float(args.cfg_text)
    model.cfg_scale_traj = float(args.cfg_traj)
    root_refiner = None
    root_text_encoder = None
    root_refiner_cfg = None
    if str(args.condition_source) in {
        "root_refiner",
        "root_refiner_update",
        "root_refiner_multi_update",
    }:
        if not args.root_refiner_ckpt:
            raise ValueError(
                f"--condition_source={args.condition_source} requires --root_refiner_ckpt"
            )
        root_refiner, root_text_encoder, root_refiner_cfg = _load_root_refiner_from_ckpt(
            str(args.root_refiner_ckpt),
            str(device),
        )
        root_refiner = root_refiner.to(device).eval()
        root_text_encoder = root_text_encoder.to(device).eval()

    sample_batch = _load_sample(args, cfg)
    original_traj7 = sample_batch["traj_cond_7d"][0].float().cpu()
    original_frames = int(sample_batch["feature_length"][0].item())
    if args.update_frame is None:
        update_frame = max(
            1,
            original_frames - int(args.update_lead_tokens) * int(args.frames_per_token),
        )
    else:
        update_frame = max(1, min(int(args.update_frame), original_frames - 1))
    suffix_frames = args.suffix_frames if args.suffix_frames is not None else original_frames
    updated_traj7 = compose_updated_route_for_mode(
        original_traj7[:original_frames],
        composition_mode=str(args.composition_mode),
        update_frame=update_frame,
        suffix_frames=suffix_frames,
        source_start_frame=int(args.source_start_frame),
        transition_frames=int(args.transition_frames),
        transition_output_frames=int(args.transition_output_frames),
        anchor_yaw_policy=str(args.anchor_yaw_policy),
        derive_heading_from_path=bool(args.derive_heading_from_path),
        turn_blend_frames=args.turn_blend_frames,
        update_lead_tokens=int(args.update_lead_tokens),
        frames_per_token=int(args.frames_per_token),
        first_straight_frames=int(args.first_straight_frames),
        arc_frames=int(args.arc_frames),
        arc_turn_degrees=float(args.arc_turn_degrees),
        step_lookback_frames=int(args.step_lookback_frames),
        forward_frames=int(args.forward_frames),
        forward_step_length=float(args.forward_step_length),
        four_segment_frames=int(args.four_segment_frames),
        four_segment_forward=float(args.four_segment_forward),
        constant_arc_frames=int(args.constant_arc_frames),
        constant_arc_length=float(args.constant_arc_length),
        constant_arc_turn_degrees=float(args.constant_arc_turn_degrees),
    )
    route_traj7 = updated_traj7
    rootrefiner_plan = None
    if str(args.condition_source) == "root_refiner":
        updated_traj7, rootrefiner_plan = _build_rootrefiner_traj7(
            ldf_model=model,
            root_refiner=root_refiner,
            root_text_encoder=root_text_encoder,
            text=str(sample_batch.get("_caption_text") or sample_batch["text"][0]),
            route_traj7=route_traj7,
            device=device,
            token_dt=float(args.token_dt),
            frames_per_token=int(args.frames_per_token),
            forced_future_frames=args.root_refiner_forced_frames,
        )
        updated_traj7 = _maybe_override_rootrefiner_heading(
            updated_traj7,
            str(args.root_refiner_heading_override),
            path_xyz=route_traj7[:, :3],
        )
    updated_batch = apply_updated_traj_to_sample_batch(sample_batch, updated_traj7)
    updated_xz = updated_traj7[:, [0, 2]].cpu()
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"

    records = []
    decoded_series = {}
    condition_series = {}
    for run_idx in range(max(1, int(args.num_runs))):
        for alpha in [None] + _parse_alphas(args.alphas):
            if str(args.condition_source) == "root_refiner_multi_update":
                run_out, metrics, elapsed, seed = _run_rootrefiner_multi_update_one(
                    model,
                    vae,
                    sample_batch,
                    original_traj7=original_traj7[:original_frames],
                    route_traj7=route_traj7,
                    update_frames=_parse_int_list(args.multi_update_frames),
                    root_refiner=root_refiner,
                    root_text_encoder=root_text_encoder,
                    args=args,
                    device=device,
                    alpha=alpha,
                    run_idx=run_idx,
                )
                condition_traj7 = run_out["condition_traj7"].detach().cpu().float()
                video_traj_xz = condition_traj7[:, [0, 2]]
            elif str(args.condition_source) == "root_refiner_update":
                run_out, metrics, elapsed, seed = _run_rootrefiner_update_one(
                    model,
                    vae,
                    sample_batch,
                    original_traj7=original_traj7[:original_frames],
                    route_traj7=route_traj7,
                    update_frame=update_frame,
                    root_refiner=root_refiner,
                    root_text_encoder=root_text_encoder,
                    args=args,
                    device=device,
                    alpha=alpha,
                    run_idx=run_idx,
                )
                condition_traj7 = run_out["condition_traj7"].detach().cpu().float()
                video_traj_xz = condition_traj7[:, [0, 2]]
            else:
                run_out, metrics, elapsed, seed = _run_one(
                    model,
                    vae,
                    updated_batch,
                    args,
                    device,
                    alpha=alpha,
                    run_idx=run_idx,
                )
                condition_traj7 = updated_traj7.detach().cpu().float()
                video_traj_xz = updated_xz
            decoded = run_out["decoded_feature"]
            video_path = videos_dir / _video_name_for_alpha(alpha, run_idx=run_idx)
            mask_update_frames = run_out.get("switch_frames")
            if mask_update_frames is None:
                mask_update_frames = run_out.get("switch_frame")
            if mask_update_frames is None:
                mask_update_frames = update_frame
            if isinstance(mask_update_frames, (int, float)):
                mask_update_frames = [int(mask_update_frames)]
            condition_mask = condition_visual_mask(
                int(condition_traj7.shape[0]),
                str(args.condition_traj_display),
                mask_update_frames,
            )
            render_motion_video(
                decoded,
                video_path,
                dim=263,
                traj_xz=video_traj_xz,
                traj_mask=torch.ones(
                    int(decoded.shape[0]),
                    dtype=torch.float32,
                    device=decoded.device,
                ),
                cond_traj_mask=condition_mask,
                render_setting={
                    "cond_traj_show_full": (
                        str(args.condition_traj_display) == "full"
                    ),
                    "traj_mask_point_radius": 3,
                    "cond_traj_point_radius": 4,
                },
            )
            label = "no_feedback" if alpha is None else f"alpha_{float(alpha):.2f}"
            series_label = f"{label}_run{run_idx}"
            decoded_series[series_label] = _root_xz(decoded)
            condition_series[f"condition_{series_label}"] = condition_traj7[:, [0, 2]]
            debug_npz = None
            debug_plot = None
            if str(args.condition_source) in {
                "root_refiner_update",
                "root_refiner_multi_update",
            }:
                debug_npz, debug_plot = _save_rootrefiner_update_debug_artifacts(
                    artifacts_dir,
                    label=label,
                    run_idx=run_idx,
                    route_traj7=route_traj7,
                    condition_traj7=condition_traj7,
                    decoded_feature=decoded,
                    run_out=run_out,
                )
            elif str(args.condition_source) == "root_refiner" and rootrefiner_plan is not None:
                debug_npz, debug_plot = _save_rootrefiner_update_debug_artifacts(
                    artifacts_dir,
                    label=label,
                    run_idx=run_idx,
                    route_traj7=route_traj7,
                    condition_traj7=condition_traj7,
                    decoded_feature=decoded,
                    run_out={**run_out, "first_root_plan": rootrefiner_plan},
                )
            first_plan = run_out.get("first_root_plan")
            second_plan = run_out.get("second_root_plan")
            root_plans = run_out.get("root_plans") or []
            records.append(
                {
                    "label": label,
                    "run_idx": int(run_idx),
                    "alpha": None if alpha is None else float(alpha),
                    "seed": int(seed),
                    "elapsed_sec": float(elapsed),
                    "fps": float(run_out["target_total_frames"] / elapsed),
                    "video": str(video_path),
                    "debug_npz": debug_npz,
                    "debug_plot": debug_plot,
                    "condition_frames": int(condition_traj7.shape[0]),
                    "rootrefiner_update_commit": run_out.get("update_commit"),
                    "rootrefiner_update_commits": run_out.get("update_commits"),
                    "rootrefiner_switch_frame": run_out.get("switch_frame"),
                    "rootrefiner_switch_frames": run_out.get("switch_frames"),
                    "rootrefiner_update_triggered": run_out.get("update_triggered"),
                    "rootrefiner_update_triggered_flags": run_out.get(
                        "update_triggered_flags"
                    ),
                    "rootrefiner_update_reanchor_mode": run_out.get(
                        "update_reanchor_mode"
                    ),
                    "rootrefiner_update_reanchor_debug": run_out.get(
                        "update_reanchor_debug"
                    ),
                    "rootrefiner_update_mask_front_ratio": run_out.get(
                        "update_mask_front_ratio"
                    ),
                    "rootrefiner_update_mask_kind": run_out.get("update_mask_kind"),
                    "rootrefiner_update_anchor_yaw_mode": run_out.get(
                        "update_anchor_yaw_mode"
                    ),
                    "rootrefiner_update_anchor_tangent_frames": run_out.get(
                        "update_anchor_tangent_frames"
                    ),
                    "rootrefiner_plan_count": int(len(root_plans)),
                    "rootrefiner_plan_mask_front": [
                        getattr(plan, "debug_rootrefiner_mask_front", None)
                        for plan in root_plans
                    ],
                    "rootrefiner_plan_update_anchor_yaw": [
                        getattr(plan, "debug_rootrefiner_update_anchor_yaw", None)
                        for plan in root_plans
                    ],
                    "rootrefiner_plan_valid_frames": [
                        int(plan.valid_frames) for plan in root_plans
                    ],
                    "rootrefiner_plan_num_tokens_pred": [
                        int(plan.num_tokens_pred) for plan in root_plans
                    ],
                    "first_rootrefiner_valid_frames": (
                        None if first_plan is None else int(first_plan.valid_frames)
                    ),
                    "first_rootrefiner_num_tokens_pred": (
                        None if first_plan is None else int(first_plan.num_tokens_pred)
                    ),
                    "second_rootrefiner_valid_frames": (
                        None if second_plan is None else int(second_plan.valid_frames)
                    ),
                    "second_rootrefiner_num_tokens_pred": (
                        None if second_plan is None else int(second_plan.num_tokens_pred)
                    ),
                    **{key: float(value) if isinstance(value, (int, float)) else value for key, value in metrics.items()},
                }
            )

    plot_path = out_dir / "trajectory_update_turn180.png"
    series = {
        "original_gt": original_traj7[:original_frames, [0, 2]],
        "manual_route": route_traj7[:, [0, 2]].cpu(),
        **condition_series,
        **decoded_series,
    }
    plot_title = (
        "forward_line one-shot root_refiner"
        if str(args.composition_mode) == "forward_line"
        else f"{args.composition_mode} update at frame {update_frame}"
    )
    plot_xz_trajectories(
        plot_path,
        series,
        title=plot_title,
        boundary_frames=[update_frame, original_frames],
    )
    summary = {
        "sample_name": str(sample_batch["name"][0]),
        "caption_index": sample_batch.get("_caption_index"),
        "caption_text": sample_batch.get("_caption_text"),
        "cfg_text": float(args.cfg_text),
        "cfg_traj": float(args.cfg_traj),
        "history_length": int(args.history_length),
        "horizon_tokens": int(args.horizon_tokens),
        "num_runs": max(1, int(args.num_runs)),
        "condition_source": str(args.condition_source),
        "root_refiner_ckpt": (
            None if args.root_refiner_ckpt is None else str(args.root_refiner_ckpt)
        ),
        "root_refiner_forced_frames": (
            None
            if args.root_refiner_forced_frames is None
            else int(args.root_refiner_forced_frames)
        ),
        "root_refiner_valid_frames": (
            None if rootrefiner_plan is None else int(rootrefiner_plan.valid_frames)
        ),
        "root_refiner_num_tokens_pred": (
            None if rootrefiner_plan is None else int(rootrefiner_plan.num_tokens_pred)
        ),
        "root_refiner_source": (
            None if rootrefiner_plan is None else str(rootrefiner_plan.source)
        ),
        "root_refiner_update_mask_front_ratio": float(
            args.root_refiner_update_mask_front_ratio
        ),
        "root_refiner_update_mask_kind": str(args.root_refiner_update_mask_kind),
        "root_refiner_update_anchor_yaw_mode": str(
            args.root_refiner_update_anchor_yaw_mode
        ),
        "root_refiner_update_anchor_tangent_frames": int(
            args.root_refiner_update_anchor_tangent_frames
        ),
        "root_refiner_config_exp_name": (
            None
            if root_refiner_cfg is None
            else str(root_refiner_cfg.get("exp_name", ""))
        ),
        "composition_mode": str(args.composition_mode),
        "condition_traj_display": str(args.condition_traj_display),
        "multi_update_frames": _parse_int_list(args.multi_update_frames),
        "update_frame": int(update_frame),
        "original_frames": int(original_frames),
        "updated_frames": int(updated_traj7.shape[0]),
        "manual_route_frames": int(route_traj7.shape[0]),
        "suffix_frames": int(suffix_frames),
        "source_start_frame": int(args.source_start_frame),
        "transition_frames": int(args.transition_frames),
        "transition_output_frames": int(args.transition_output_frames),
        "first_straight_frames": int(args.first_straight_frames),
        "arc_frames": int(args.arc_frames),
        "arc_turn_degrees": float(args.arc_turn_degrees),
        "constant_arc_frames": int(args.constant_arc_frames),
        "constant_arc_length": float(args.constant_arc_length),
        "constant_arc_turn_degrees": float(args.constant_arc_turn_degrees),
        "step_lookback_frames": int(args.step_lookback_frames),
        "anchor_yaw_policy": str(args.anchor_yaw_policy),
        "derive_heading_from_path": bool(args.derive_heading_from_path),
        "turn_blend_frames": (
            int(args.turn_blend_frames)
            if args.turn_blend_frames is not None
            else int(args.update_lead_tokens) * int(args.frames_per_token)
        ),
        "trajectory_plot": str(plot_path),
        "records": records,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
