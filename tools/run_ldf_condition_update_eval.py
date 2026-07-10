#!/usr/bin/env python3
"""Run LDF-only condition-update experiments from pluggable 7D sources."""

from __future__ import annotations

import argparse
import warnings
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

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

from eval.common.visualization import plot_xz_trajectories, render_motion_video
from eval.ldf.conditioning import LdfEvalStreamConditioner
from eval.ldf.experiments.artifacts import plot_7d_xz_heading
from eval.ldf.experiments.condition_sources import (
    ConditionScenario,
    build_repeat_splice_condition,
    build_sample_condition,
    build_synthetic_condition,
)
from utils.inference.runtime_update.active_condition import compose_active_window_segment
from utils.inference.runtime_update.active_condition import (
    compose_active_window_world_condition,
)
from utils.inference.runtime_update.payload_builder import (
    build_world_condition_stream_payload,
)
from utils.inference.runtime_update.route_tracker import RouteProgressTracker
from utils.inference.runtime_update.root_source import (
    RootSourceProposal,
    condition_scenario_to_proposal,
    proposal_to_world_traj7,
    world_traj7_to_proposal,
)
from eval.ldf.stream_generation import (
    StreamTextRolloutController,
    _decode_latent_chunk,
    _decode_raw_chunk_preserving_feedback_cache,
    _encode_corrected_chunk_token,
    _replace_chunk_root_from_condition,
    _write_committed_latent_to_model,
)
from eval.ldf.stream_setup import (
    _set_seed,
    enable_cpu_text_encoding,
    load_eval_model_and_vae,
)
from metrics.traj import _compute_traj_metrics, _seed_eval_locally, _stable_eval_seed
from tools.run_stream_turn_update_debug import (
    _build_segment_source_from_runtime_anchor,
    _build_rootrefiner_traj7,
    _clamp_refiner_future_frames,
    _compose_multi_rootrefiner_condition_traj7,
    _decoded_anchor_frame_for_update,
    _generated_history_world_5d,
    _hold_last_traj7,
    _load_root_refiner_from_ckpt,
    _maybe_override_rootrefiner_heading,
    _load_sample,
    _root_xz,
    _run_one,
    _video_name_for_alpha,
    apply_updated_traj_to_sample_batch,
    condition_visual_mask,
)
from utils.initialize import load_config
from utils.inference.root_plan import RootPlan
from utils.inference.stream_generator import StreamGenerator
from utils.inference.timeline import RootFrameState
from utils.local_frame import canonicalize_7d
from utils.motion_process import (
    StreamJointRecovery263,
    build_physical_7d_from_5d,
    extract_root_traj_feats_7d_263,
)
from utils.token_frame import frame_idx_to_token_idx, num_tokens_for_frame_len, token_start_frame


def resolve_effective_update_commit(
    raw_update_frame: int,
    first_uncommitted_token: int,
    frames_per_token: int,
) -> int:
    """Resolve when a route update can first affect LDF token generation.

    Route updates may arrive at any frame, but the LDF condition may only
    change at a token boundary that has not already started.  If an update
    arrives inside token k's frame coverage, it is delayed to token k+1.
    """
    raw_commit = frame_idx_to_token_idx(
        int(raw_update_frame),
        int(frames_per_token),
    )
    token_start = token_start_frame(raw_commit, int(frames_per_token))
    if int(raw_update_frame) > int(token_start):
        raw_commit += 1
    return max(int(raw_commit), int(first_uncommitted_token))


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
    parser.add_argument("--probe_tag", default="ldf_condition_update")
    parser.add_argument("--cfg_text", type=float, default=1.2)
    parser.add_argument("--cfg_traj", type=float, default=2.2)
    parser.add_argument("--history_length", type=int, default=30)
    parser.add_argument("--horizon_tokens", type=int, default=10)
    parser.add_argument(
        "--condition_source",
        choices=["sample", "repeat_splice", "synthetic"],
        default="repeat_splice",
    )
    parser.add_argument(
        "--condition_traj_display",
        choices=["future", "full"],
        default="full",
    )
    parser.add_argument(
        "--dynamic_updates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, update_frames are applied during stream_generate_step "
            "by switching the active direct 7D RootPlan instead of exposing the "
            "full condition from frame 0."
        ),
    )
    parser.add_argument(
        "--runtime_update_contract",
        choices=[
            "legacy_reanchor",
            "world_route",
            "relative_route",
            "absolute_route",
            "active_window",
        ],
        default="legacy_reanchor",
        help=(
            "Runtime trajectory update contract. legacy_reanchor keeps the old "
            "segment-source path; relative_route composes a generated-history-"
            "anchored route; world_route keeps the authored world route fixed "
            "so alpha variants share the same future target."
        ),
    )
    parser.add_argument(
        "--runtime_route_lookahead_m",
        type=float,
        default=0.25,
        help="Lookahead distance used by --runtime_update_contract active_window.",
    )
    parser.add_argument(
        "--runtime_bridge_frames",
        type=int,
        default=12,
        help="Bridge frames used by --runtime_update_contract active_window.",
    )
    parser.add_argument(
        "--root_source_refiner_ckpt",
        default=None,
        help=(
            "Optional RootRefiner checkpoint. When set, the external root source "
            "proposal is refined segment-by-segment before the runtime update "
            "contract builds LDF payloads."
        ),
    )
    parser.add_argument(
        "--root_source_refiner_forced_frames",
        type=int,
        default=None,
        help=(
            "Optional forced future frame count for each RootRefiner proposal "
            "segment. Defaults to the segment length, clamped by the refiner."
        ),
    )
    parser.add_argument(
        "--root_source_refiner_heading_override",
        choices=["none", "path_tangent"],
        default="none",
        help="Optional heading postprocess for RootRefiner root-source proposals.",
    )
    parser.add_argument(
        "--dynamic_reanchor_mode",
        choices=["route_boundary", "current_root_translate", "current_root_pose"],
        default="route_boundary",
        help=(
            "Legacy-only: how dynamic updates attach the next condition segment "
            "to the current generated root. route_boundary preserves the old "
            "translation-only behavior; current_root_pose also rotates future "
            "deltas into the current actor yaw frame."
        ),
    )
    parser.add_argument(
        "--dynamic_reanchor_yaw_source",
        choices=["runtime", "tangent"],
        default="runtime",
        help=(
            "Yaw used by current_root_pose re-anchoring. runtime uses the "
            "stream timeline actor yaw; tangent uses the generated root XZ "
            "motion over the recent tangent window."
        ),
    )
    parser.add_argument(
        "--dynamic_reanchor_tangent_frames",
        type=int,
        default=10,
        help="Recent generated-root frames used when dynamic_reanchor_yaw_source=tangent.",
    )
    parser.add_argument("--update_frame", type=int, default=None)
    parser.add_argument("--update_lead_tokens", type=int, default=6)
    parser.add_argument("--suffix_frames", type=int, default=None)
    parser.add_argument(
        "--repeat_mode",
        choices=[
            "center_symmetric",
            "reflected_history",
            "rotated_suffix",
            "rotated_suffix_arc",
            "rotated_suffix_arc_chain",
        ],
        default="center_symmetric",
    )
    parser.add_argument("--source_start_frame", type=int, default=None)
    parser.add_argument(
        "--trim_start_frames",
        type=int,
        default=20,
        help=(
            "Default repeat-splice source start when --source_start_frame is "
            "omitted. The main LDF update probes ignore unstable first frames."
        ),
    )
    parser.add_argument(
        "--trim_end_frames",
        type=int,
        default=20,
        help=(
            "Default repeat-splice tail trim when --update_frame is omitted. "
            "The update anchor becomes original_frames - trim_end_frames - 1, "
            "so the source history is traj7[trim_start_frames:original_frames-trim_end_frames]."
        ),
    )
    parser.add_argument(
        "--derive_heading_from_path",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--transition_frames", type=int, default=0)
    parser.add_argument("--transition_output_frames", type=int, default=0)
    parser.add_argument(
        "--suffix_rotation_deg",
        type=float,
        default=0.0,
        help="Rotation angle for repeat_mode=rotated_suffix; positive turns toward +x.",
    )
    parser.add_argument(
        "--turn_transition_frames",
        type=int,
        default=0,
        help=(
            "Number of suffix delta steps used to smoothly ramp into "
            "--suffix_rotation_deg for repeat_mode=rotated_suffix. For "
            "repeat_mode=rotated_suffix_arc, use -1 to choose this count "
            "from smooth path length / average XZ speed on both sides."
        ),
    )
    parser.add_argument(
        "--arc_speed_window",
        type=int,
        default=8,
        help=(
            "Number of frames on each side used to estimate transition speed "
            "for repeat_mode=rotated_suffix_arc."
        ),
    )
    parser.add_argument(
        "--arc_speed_scale",
        type=float,
        default=1.0,
        help="Multiplier for rotated_suffix_arc transition speed.",
    )
    parser.add_argument(
        "--suffix_blend_frames",
        type=int,
        default=8,
        help=(
            "Number of post-arc suffix deltas blended from the arc terminal "
            "velocity into the reused source shape for repeat_mode=rotated_suffix_arc."
        ),
    )
    parser.add_argument(
        "--suffix_min_speed_factor",
        type=float,
        default=0.25,
        help=(
            "Skip leading source deltas slower than this fraction of the "
            "pre-update speed for repeat_mode=rotated_suffix_arc."
        ),
    )
    parser.add_argument(
        "--suffix_lateral_scale",
        type=float,
        default=0.0,
        help=(
            "How much source lateral drift to keep after projecting suffix "
            "deltas into the target turn direction for repeat_mode=rotated_suffix_arc."
        ),
    )
    parser.add_argument(
        "--arc_y_mode",
        choices=["source", "anchor"],
        default="source",
        help=(
            "How to set root_y for synthesized rotated_suffix_arc updates. "
            "'source' preserves source/smooth Y deltas; 'anchor' keeps the "
            "post-update route at the update anchor height."
        ),
    )
    parser.add_argument(
        "--repeat_heading_blend_frames",
        type=int,
        default=0,
        help=(
            "For rotated_suffix_arc modes, blend the first N visible repeat "
            "heading frames from the smooth seam heading back to the locked "
            "repeat heading. XZ/Y positions are unchanged."
        ),
    )
    parser.add_argument(
        "--arc_smooth_profile",
        choices=["geometric", "heading_residual"],
        default="geometric",
        help=(
            "Smooth bridge profile for rotated_suffix_arc modes. 'geometric' "
            "is the original path-derived bridge. 'heading_residual' uses "
            "stable heading for low-speed updates and preserves recent "
            "heading-minus-tangent residuals."
        ),
    )
    parser.add_argument(
        "--repeat_count",
        type=int,
        default=4,
        help="Number of chained repeats for repeat_mode=rotated_suffix_arc_chain.",
    )
    parser.add_argument(
        "--repeat_angle_choices",
        default="",
        help=(
            "Comma-separated rotation choices sampled per repeat for "
            "repeat_mode=rotated_suffix_arc_chain, e.g. '30,60'. Empty falls "
            "back to --suffix_rotation_deg."
        ),
    )
    parser.add_argument(
        "--repeat_seed",
        type=int,
        default=1234,
        help="Seed for per-repeat angle sampling in rotated_suffix_arc_chain.",
    )
    parser.add_argument(
        "--synthetic_preset",
        choices=["forward_line", "four_segment_curve", "constant_arc"],
        default="four_segment_curve",
    )
    parser.add_argument("--synthetic_frames", type=int, default=240)
    parser.add_argument(
        "--synthetic_update_frames",
        default="",
        help="Comma-separated update frame list for synthetic conditions.",
    )
    parser.add_argument("--synthetic_forward_step_length", type=float, default=0.015)
    parser.add_argument("--synthetic_total_forward", type=float, default=4.2)
    parser.add_argument(
        "--synthetic_arc_turn_deg",
        type=float,
        default=20.0,
        help=(
            "Total heading change for synthetic_preset=constant_arc. Positive "
            "values bend toward +x."
        ),
    )
    parser.add_argument(
        "--alphas",
        default="0.5,1.0",
        help="Comma-separated feedback XZ blend alphas. Empty disables feedback runs.",
    )
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument(
        "--preview_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Only build the external 7D condition and diagnostic plots. "
            "Skips LDF/VAE loading, generation, videos, and metrics."
        ),
    )
    parser.add_argument(
        "--save_debug_npz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-run condition/root arrays for update-boundary debugging.",
    )
    parser.add_argument(
        "--render_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render output videos. Disable for fast numeric diagnostics.",
    )
    parser.add_argument("--num_denoise_steps", type=int, default=None)
    parser.add_argument("--frames_per_token", type=int, default=4)
    parser.add_argument("--token_dt", type=float, default=0.20)
    args = parser.parse_args()
    if args.runtime_update_contract == "absolute_route":
        warnings.warn(
            "absolute_route is deprecated; use world_route",
            DeprecationWarning,
            stacklevel=2,
        )
        args.runtime_update_contract = "world_route"
    elif args.runtime_update_contract == "active_window":
        warnings.warn(
            "active_window was spatially ambiguous and now maps explicitly to "
            "relative_route; use relative_route or world_route",
            DeprecationWarning,
            stacklevel=2,
        )
        args.runtime_update_contract = "relative_route"
    return args


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


def _parse_int_list(raw: str | Iterable[int] | None) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw_text = raw.strip()
        if raw_text.lower() in {"", "none", "null", "false", "no"}:
            return []
        return [int(item.strip()) for item in raw_text.split(",") if item.strip()]
    return [int(item) for item in raw]


def _resolve_update_frame(
    args: argparse.Namespace,
    *,
    original_frames: int,
) -> int:
    if args.update_frame is None:
        trim_end = max(0, int(getattr(args, "trim_end_frames", 20)))
        update = int(original_frames) - trim_end - 1
    else:
        update = int(args.update_frame)
    return max(1, min(update, int(original_frames) - 1))


def _resolve_source_start_frame(args: argparse.Namespace, *, update_frame: int) -> int:
    if args.source_start_frame is None:
        source_start = max(0, int(getattr(args, "trim_start_frames", 20)))
    else:
        source_start = int(args.source_start_frame)
    return max(0, min(source_start, int(update_frame)))


def _build_condition_scenario_from_args(
    args: argparse.Namespace,
    original_traj7: torch.Tensor,
    *,
    original_frames: int,
    sample_name: str,
    caption_index: int | None,
) -> ConditionScenario:
    """Dispatch CLI args to a concrete LDF condition source."""
    source = str(args.condition_source)
    original = original_traj7[: int(original_frames)].detach().cpu().float()
    if source == "sample":
        return build_sample_condition(
            original,
            valid_frames=int(original_frames),
            sample_name=str(sample_name),
            caption_index=caption_index,
        )
    if source == "repeat_splice":
        update_frame = _resolve_update_frame(args, original_frames=int(original_frames))
        source_start_frame = _resolve_source_start_frame(
            args,
            update_frame=int(update_frame),
        )
        suffix_frames = (
            None if args.suffix_frames is None else int(args.suffix_frames)
        )
        scenario = build_repeat_splice_condition(
            original,
            mode=str(args.repeat_mode),
            update_frame=update_frame,
            suffix_frames=suffix_frames,
            source_start_frame=int(source_start_frame),
            derive_heading_from_path=bool(args.derive_heading_from_path),
            transition_frames=int(args.transition_frames),
            transition_output_frames=int(args.transition_output_frames),
            suffix_rotation_deg=float(args.suffix_rotation_deg),
            turn_transition_frames=int(args.turn_transition_frames),
            arc_speed_window=int(args.arc_speed_window),
            arc_speed_scale=float(args.arc_speed_scale),
            suffix_blend_frames=int(args.suffix_blend_frames),
            suffix_min_speed_factor=float(args.suffix_min_speed_factor),
            suffix_lateral_scale=float(args.suffix_lateral_scale),
            arc_y_mode=str(args.arc_y_mode),
            repeat_heading_blend_frames=int(args.repeat_heading_blend_frames),
            arc_smooth_profile=str(args.arc_smooth_profile),
            repeat_count=int(getattr(args, "repeat_count", 4)),
            repeat_angle_choices=str(getattr(args, "repeat_angle_choices", "")),
            repeat_seed=int(getattr(args, "repeat_seed", 1234)),
            sample_name=str(sample_name),
            caption_index=caption_index,
        )
        metadata = {
            **scenario.metadata,
            "trim_start_frames": int(getattr(args, "trim_start_frames", 20)),
            "trim_end_frames": int(getattr(args, "trim_end_frames", 20)),
        }
        return ConditionScenario(
            name=scenario.name,
            condition_traj7=scenario.condition_traj7,
            update_frames=scenario.update_frames,
            visual_mask=scenario.visual_mask,
            base_sample_name=scenario.base_sample_name,
            caption_index=scenario.caption_index,
            metadata=metadata,
        )
    if source == "synthetic":
        return build_synthetic_condition(
            preset=str(args.synthetic_preset),
            base_traj7=original,
            num_frames=int(args.synthetic_frames),
            update_frames=_parse_int_list(args.synthetic_update_frames),
            sample_name=str(sample_name),
            caption_index=caption_index,
            forward_step_length=float(args.synthetic_forward_step_length),
            total_forward=float(args.synthetic_total_forward),
            arc_turn_degrees=float(args.synthetic_arc_turn_deg),
        )
    raise ValueError(f"Unsupported condition_source={source!r}")


def _refine_root_source_proposal_segments(
    root_source: RootSourceProposal,
    *,
    ldf_model,
    root_refiner,
    root_text_encoder,
    text: str,
    device: torch.device,
    token_dt: float,
    frames_per_token: int,
    forced_future_frames: int | None,
    heading_override: str,
) -> RootSourceProposal:
    """Run RootRefiner over each root-source segment, preserving update frames."""
    route = proposal_to_world_traj7(root_source)
    total = int(route.shape[0])
    if total <= 1:
        return root_source
    update_frames = [
        max(1, min(int(frame), total - 1))
        for frame in root_source.metadata.get("update_frames", ())
    ]
    update_frames = sorted(set(update_frames))
    segment_starts = [0] + update_frames
    segment_ends = update_frames + [total - 1]
    refined = route.clone()
    segment_debug: list[dict[str, object]] = []
    for seg_idx, (start, end) in enumerate(zip(segment_starts, segment_ends)):
        start = int(start)
        end = max(start + 1, int(end))
        source_segment = route[start:end + 1]
        requested = (
            None
            if forced_future_frames is None
            else int(forced_future_frames)
        )
        if requested is None:
            requested = max(1, int(source_segment.shape[0]) - 1)
        requested = _clamp_refiner_future_frames(root_refiner, int(requested))
        world_7d, _plan = _build_rootrefiner_traj7(
            ldf_model=ldf_model,
            root_refiner=root_refiner,
            root_text_encoder=root_text_encoder,
            text=str(text),
            route_traj7=source_segment,
            device=device,
            token_dt=float(token_dt),
            frames_per_token=int(frames_per_token),
            forced_future_frames=int(requested),
        )
        world_7d = _maybe_override_rootrefiner_heading(
            world_7d.detach().cpu().float(),
            str(heading_override),
        )
        world_7d = _hold_last_traj7(world_7d, int(source_segment.shape[0]))
        refined[start:end + 1] = world_7d[: end - start + 1].to(dtype=refined.dtype)
        segment_debug.append(
            {
                "segment_index": int(seg_idx),
                "start_frame": int(start),
                "end_frame": int(end),
                "source_frames": int(source_segment.shape[0]),
                "refined_frames": int(world_7d.shape[0]),
                "forced_future_frames": int(requested),
            }
        )
    metadata = {
        **root_source.metadata,
        "root_source_refiner_enabled": True,
        "root_source_refiner_segments": segment_debug,
        "root_source_refiner_heading_override": str(heading_override),
    }
    metadata.update(
        {
            "source_kind": (
                f"{root_source.metadata.get('source_kind', 'unknown')}+root_refiner"
            ),
            "update_frames": tuple(update_frames),
            "anchor_frame_7d": refined[0].detach().cpu(),
        }
    )
    return world_traj7_to_proposal(
        refined,
        source_id=f"root_refiner:{root_source.source_id}",
        version=root_source.version,
        frame_mask=torch.cat(
            [
                torch.ones(1, dtype=torch.bool),
                root_source.future_frame_mask,
            ]
        ),
        strip_anchor=True,
        metadata=metadata,
    )


def _yaw_from_7d(traj7: torch.Tensor) -> torch.Tensor:
    return torch.atan2(traj7[:, 4], traj7[:, 3])


def _build_direct_root_plan_from_world_7d(
    route_traj7: torch.Tensor,
    *,
    anchor_state: RootFrameState,
    anchor_commit_idx: int,
    token_dt: float,
    frames_per_token: int,
    source: str,
) -> RootPlan:
    route = route_traj7.detach().to(
        device=anchor_state.world_xz.device,
        dtype=torch.float32,
    )
    anchor_xz = anchor_state.world_xz.detach().to(
        device=route.device,
        dtype=route.dtype,
    )
    anchor_yaw = anchor_state.world_yaw.detach().to(
        device=route.device,
        dtype=route.dtype,
    )
    waypoints_local = canonicalize_7d(
        route.unsqueeze(0),
        anchor_xz.unsqueeze(0),
        anchor_yaw.reshape(1),
    )[0]
    return RootPlan(
        num_tokens_pred=num_tokens_for_frame_len(
            int(route.shape[0]),
            int(frames_per_token),
        ),
        valid_frames=int(route.shape[0]),
        waypoints_local_7d=waypoints_local,
        frame_dt=float(token_dt) / float(frames_per_token),
        frames_per_token=int(frames_per_token),
        anchor_commit_idx=int(anchor_commit_idx),
        anchor_world_xz=anchor_xz,
        anchor_world_yaw=anchor_yaw,
        source=str(source),
    )


def _anchor_state_with_history_tangent_yaw(
    anchor_state: RootFrameState,
    history_world_5d: torch.Tensor | None,
    *,
    tangent_frames: int,
) -> RootFrameState:
    if history_world_5d is None or int(history_world_5d.shape[0]) < 2:
        return anchor_state
    history = history_world_5d.detach().to(
        device=anchor_state.world_xz.device,
        dtype=anchor_state.world_xz.dtype,
    )
    end = int(history.shape[0]) - 1
    start = max(0, end - max(1, int(tangent_frames)))
    delta = history[end, [0, 2]] - history[start, [0, 2]]
    if float(torch.linalg.norm(delta).detach().cpu().item()) < 1e-6:
        return anchor_state
    yaw = torch.atan2(delta[0], delta[1]).to(
        device=anchor_state.world_yaw.device,
        dtype=anchor_state.world_yaw.dtype,
    )
    return RootFrameState(
        commit_idx=int(anchor_state.commit_idx),
        world_xz=anchor_state.world_xz.clone(),
        world_yaw=yaw,
        source=f"{anchor_state.source}:tangent_yaw",
    )


def _run_ldf_direct_multi_update_one(
    model,
    vae,
    sample_batch: dict,
    *,
    route_traj7: torch.Tensor,
    update_frames: list[int],
    args: argparse.Namespace,
    device: torch.device,
    alpha: float | None,
    run_idx: int = 0,
):
    """Run stream_generate_step while revealing direct 7D condition by segments."""
    seed = _stable_eval_seed(
        int(args.seed),
        f"{args.probe_tag}_direct_multi_update",
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
    update_records = []
    for frame in update_frames:
        raw_commit = frame_idx_to_token_idx(frame, frames_per_token)
        first_uncommitted = raw_commit
        if int(frame) > token_start_frame(raw_commit, frames_per_token):
            first_uncommitted = raw_commit + 1
        effective_commit = resolve_effective_update_commit(
            frame,
            first_uncommitted,
            frames_per_token,
        )
        update_records.append(
            {
                "raw_update_frame": int(frame),
                "raw_update_commit": int(raw_commit),
                "first_uncommitted_token": int(first_uncommitted),
                "effective_update_commit": int(effective_commit),
                "effective_update_frame_start": int(
                    token_start_frame(effective_commit, frames_per_token)
                ),
            }
        )
    update_commits = [
        int(record["effective_update_commit"]) for record in update_records
    ]

    initial_anchor = route_traj7[0].to(device=device, dtype=torch.float32)
    initial_state = RootFrameState(
        commit_idx=0,
        world_xz=initial_anchor[[0, 2]].clone(),
        world_yaw=_yaw_from_7d(route_traj7[:1])[0].to(
            device=device,
            dtype=torch.float32,
        ),
        source="ldf_direct_multi_update_initial_anchor",
    )
    stream = StreamGenerator(
        ldf_model=model,
        device=device,
        history_length=int(args.history_length),
        traj_horizon_tokens=int(args.horizon_tokens),
        token_dt=float(args.token_dt),
    )
    stream.reset(initial_state, text=str(sample_batch.get("_caption_text") or sample_batch["text"][0]))
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
    first_plan = _build_direct_root_plan_from_world_7d(
        first_source,
        anchor_state=initial_state,
        anchor_commit_idx=0,
        token_dt=float(args.token_dt),
        frames_per_token=frames_per_token,
        source="ldf_direct_multi_update_segment_0",
    )
    first_plan.anchor_frame_idx = 0
    root_plans = [first_plan]

    segment_batch = apply_updated_traj_to_sample_batch(sample_batch, first_source)
    stream_conditioner = LdfEvalStreamConditioner(
        segment_batch,
        history_length=int(args.history_length),
        traj_horizon_tokens=int(args.horizon_tokens),
        token_dt=float(args.token_dt),
        frames_per_token=frames_per_token,
        device=device,
    )
    stream_conditioner.timeline = stream.timeline
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
    condition_snapshots = [
        {
            "name": "initial",
            "switch_frame": 0,
            "segment_source": first_source.detach().cpu().float(),
            "condition_traj7": condition_traj7.detach().cpu().float(),
        }
    ]
    payload_snapshots = []
    payload_debug_commits = {
        commit + offset
        for commit in update_commits
        for offset in (-1, 0, 1)
        if commit + offset >= 0
    }

    active_window_tracker = RouteProgressTracker(
        route_traj7.detach().cpu().float(),
        lookahead_m=float(args.runtime_route_lookahead_m),
    )
    next_update_idx = 0
    triggered_flags = [False for _ in update_frames]
    switch_frames: list[int] = []
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
                    reanchor_anchor_state = anchor_state
                    if str(args.dynamic_reanchor_yaw_source) == "tangent":
                        reanchor_anchor_state = _anchor_state_with_history_tangent_yaw(
                            anchor_state,
                            history_5d,
                            tangent_frames=int(args.dynamic_reanchor_tangent_frames),
                        )
                    segment_end = segment_ends[next_update_idx + 1]
                    if str(args.runtime_update_contract) == "relative_route":
                        if history_5d is not None and int(history_5d.shape[0]) > 0:
                            generated_history_traj7 = build_physical_7d_from_5d(
                                history_5d.detach().cpu().float()[:, :5]
                            )
                        else:
                            generated_history_traj7 = route_traj7[
                                : max(1, int(condition_switch_frame) + 1)
                            ].detach().cpu().float()
                        current_frame = min(
                            max(0, int(condition_switch_frame)),
                            int(generated_history_traj7.shape[0]) - 1,
                        )
                        active_anchor_state = RootFrameState(
                            commit_idx=int(update_commit),
                            world_xz=generated_history_traj7[current_frame, [0, 2]].to(
                                device=device, dtype=torch.float32
                            ),
                            world_yaw=reanchor_anchor_state.world_yaw.detach().to(
                                device=device, dtype=torch.float32
                            ),
                            source="active_window_update_anchor",
                        )
                        active_segment = compose_active_window_segment(
                            route_traj7.detach().cpu().float(),
                            generated_history_traj7,
                            current_frame=int(current_frame),
                            current_yaw=active_anchor_state.world_yaw.detach().cpu(),
                            target_end_frame=int(target_total_frames - 1),
                            lookahead_m=float(args.runtime_route_lookahead_m),
                            bridge_frames=int(args.runtime_bridge_frames),
                            min_route_index=int(condition_switch_frame),
                            tracker=active_window_tracker,
                        )
                        segment_source = active_segment.segment_traj7.to(
                            device=device, dtype=torch.float32
                        )
                        reanchor_anchor_state = active_anchor_state
                        reanchor_debug = {
                            "mode": "active_window",
                            "boundary_frame": int(condition_switch_frame),
                            "route_index": int(active_segment.route_index),
                            "future_index": int(active_segment.future_index),
                            "bridge_frames": int(active_segment.bridge_frames),
                        }
                    elif str(args.runtime_update_contract) == "world_route":
                        boundary = max(
                            0,
                            min(int(condition_switch_frame), int(route_traj7.shape[0]) - 1),
                        )
                        end = max(
                            boundary + 1,
                            min(int(segment_end), int(route_traj7.shape[0]) - 1),
                        )
                        segment_source = route_traj7[boundary : end + 1].to(
                            device=device, dtype=torch.float32
                        )
                        reanchor_debug = {
                            "mode": "absolute_route",
                            "boundary_frame": int(boundary),
                            "end_frame": int(end),
                        }
                    else:
                        segment_source, reanchor_debug = _build_segment_source_from_runtime_anchor(
                            route_traj7,
                            boundary_frame=int(condition_switch_frame),
                            end_frame=int(segment_end),
                            anchor_state=reanchor_anchor_state,
                            anchor_world_y=anchor_y,
                            reanchor_mode=str(args.dynamic_reanchor_mode),
                        )
                    if str(args.runtime_update_contract) == "relative_route":
                        condition_traj7 = compose_active_window_world_condition(
                            route_traj7.detach().cpu().float(),
                            generated_history_traj7.detach().cpu().float(),
                            active_segment,
                            current_frame=int(current_frame),
                        ).to(device=condition_traj7.device, dtype=condition_traj7.dtype)
                    elif str(args.runtime_update_contract) == "world_route":
                        condition_traj7 = route_traj7.to(
                            device=condition_traj7.device,
                            dtype=condition_traj7.dtype,
                        )
                    else:
                        plan = _build_direct_root_plan_from_world_7d(
                            segment_source,
                            anchor_state=reanchor_anchor_state,
                            anchor_commit_idx=int(update_commit),
                            token_dt=float(args.token_dt),
                            frames_per_token=frames_per_token,
                            source=(
                                "ldf_direct_multi_update_segment_"
                                f"{next_update_idx + 1}"
                            ),
                        )
                        plan.anchor_frame_idx = int(condition_switch_frame)
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
                    update_record = update_records[next_update_idx]
                    condition_snapshots.append(
                        {
                            "name": f"update_{next_update_idx}",
                            "switch_frame": int(condition_switch_frame),
                            "raw_update_frame": int(update_record["raw_update_frame"]),
                            "raw_update_commit": int(update_record["raw_update_commit"]),
                            "first_uncommitted_token": int(
                                update_record["first_uncommitted_token"]
                            ),
                            "effective_update_commit": int(
                                update_record["effective_update_commit"]
                            ),
                            "effective_update_frame_start": int(
                                update_record["effective_update_frame_start"]
                            ),
                            "segment_source": segment_source.detach().cpu().float(),
                            "condition_traj7": condition_traj7.detach().cpu().float(),
                        }
                    )
                    triggered_flags[next_update_idx] = True
                    switch_frames.append(int(condition_switch_frame))
                    next_update_idx += 1

            current_text = text_rollout.get_text_for_commit_index(commit_index)
            local_commit_index = int(getattr(model, "commit_index", commit_index))
            chunk_size = int(getattr(model, "chunk_size", 1))
            if str(args.runtime_update_contract) in {"relative_route", "world_route"}:
                payload_history_5d = _generated_history_world_5d(decoded_chunks, device)
                payload_history_traj7 = (
                    None
                    if payload_history_5d is None or int(payload_history_5d.shape[0]) <= 0
                    else build_physical_7d_from_5d(
                        payload_history_5d.detach().cpu().float()[:, :5]
                    ).to(device=device, dtype=torch.float32)
                )
                payload_condition_traj7 = condition_traj7.to(device=device, dtype=torch.float32)
                if (
                    str(args.runtime_update_contract) == "relative_route"
                    and
                    payload_history_traj7 is not None
                    and int(payload_history_traj7.shape[0]) > 0
                    and next_update_idx > 0
                ):
                    payload_current_frame = int(payload_history_traj7.shape[0]) - 1
                    payload_current_yaw = torch.atan2(
                        payload_history_traj7[payload_current_frame, 4].detach().cpu(),
                        payload_history_traj7[payload_current_frame, 3].detach().cpu(),
                    )
                    payload_segment = compose_active_window_segment(
                        route_traj7.detach().cpu().float(),
                        payload_history_traj7.detach().cpu().float(),
                        current_frame=payload_current_frame,
                        current_yaw=payload_current_yaw,
                        target_end_frame=int(target_total_frames - 1),
                        lookahead_m=float(args.runtime_route_lookahead_m),
                        bridge_frames=int(args.runtime_bridge_frames),
                        min_route_index=payload_current_frame,
                        tracker=active_window_tracker,
                    )
                    payload_condition_traj7 = route_traj7.detach().clone().to(
                        device=device, dtype=torch.float32
                    )
                    payload_patch_end = min(
                        int(payload_condition_traj7.shape[0]),
                        payload_current_frame + int(payload_segment.segment_traj7.shape[0]),
                    )
                    if payload_patch_end > payload_current_frame:
                        payload_condition_traj7[payload_current_frame:payload_patch_end] = (
                            payload_segment.segment_traj7[: payload_patch_end - payload_current_frame]
                            .to(device=device, dtype=torch.float32)
                        )
                traj_input = build_world_condition_stream_payload(
                    payload_condition_traj7,
                    stream_conditioner.timeline,
                    local_commit_index=local_commit_index,
                    absolute_commit_index=int(commit_index),
                    chunk_size=chunk_size,
                    history_length=int(args.history_length),
                    traj_horizon_tokens=int(args.horizon_tokens),
                    frames_per_token=frames_per_token,
                    generated_history_traj7=payload_history_traj7,
                )
            else:
                traj_input = stream_conditioner.build_step_payload(
                    local_commit_index=local_commit_index,
                    absolute_commit_index=int(commit_index),
                    chunk_size=chunk_size,
                )
            if bool(args.save_debug_npz) and int(commit_index) in payload_debug_commits:
                payload_record = {
                    "commit_index": int(commit_index),
                    "local_commit_index": int(local_commit_index),
                    "has_traj_input": traj_input is not None,
                    "subpayloads": [],
                }
                if traj_input is not None:
                    payload_record.update(
                        {
                            "traj_start_token": int(traj_input.get("traj_start_token", -1)),
                            "traj_abs_start_token": int(traj_input.get("traj_abs_start_token", -1)),
                            "traj_num_tokens": int(traj_input.get("traj_num_tokens", -1)),
                            "body_anchor_abs_token": int(traj_input.get("body_anchor_abs_token", -1)),
                            "traj_cond_7d_frame": traj_input["traj_cond_7d_frame"].detach().cpu().float(),
                            "traj_cond_frame_mask": traj_input["traj_cond_frame_mask"].detach().cpu().float(),
                        }
                    )
                    for subpayload in traj_input.get("traj_substep_payloads", []):
                        payload_record["subpayloads"].append(
                            {
                                "traj_start_token": int(subpayload.get("traj_start_token", -1)),
                                "traj_abs_start_token": int(subpayload.get("traj_abs_start_token", -1)),
                                "traj_num_tokens": int(subpayload.get("traj_num_tokens", -1)),
                                "body_anchor_abs_token": int(subpayload.get("body_anchor_abs_token", -1)),
                                "traj_cond_7d_frame": subpayload["traj_cond_7d_frame"].detach().cpu().float(),
                                "traj_cond_frame_mask": subpayload["traj_cond_frame_mask"].detach().cpu().float(),
                            }
                        )
                payload_snapshots.append(payload_record)
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
            stream.timeline = stream_conditioner.timeline
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
    metric_batch = apply_updated_traj_to_sample_batch(sample_batch, condition_traj7)
    condition_metrics = _compute_traj_metrics(decoded_feature, metric_batch, 0, seg_size=20)
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
        "latent_stream": (
            torch.cat(latent_tokens, dim=0)
            if latent_tokens
            else torch.zeros((0, model.input_dim), dtype=torch.float32)
        ),
        "original_total_frames": int(sample_batch["feature_length"][0].item()),
        "target_total_frames": int(target_total_frames),
        "extra_frames": 0,
        "root_replace_feedback": alpha is not None,
        "root_feedback_xz_blend_alpha": 1.0 if alpha is None else float(alpha),
        "condition_traj7": condition_traj7,
        "condition_snapshots": condition_snapshots,
        "payload_snapshots": payload_snapshots,
        "root_plans": root_plans,
        "update_frames": [int(frame) for frame in update_frames],
        "raw_update_frames": [
            int(record["raw_update_frame"]) for record in update_records
        ],
        "raw_update_commits": [
            int(record["raw_update_commit"]) for record in update_records
        ],
        "first_uncommitted_tokens": [
            int(record["first_uncommitted_token"]) for record in update_records
        ],
        "effective_update_commits": [
            int(record["effective_update_commit"]) for record in update_records
        ],
        "effective_update_frame_starts": [
            int(record["effective_update_frame_start"]) for record in update_records
        ],
        "update_commits": [int(commit) for commit in update_commits],
        "switch_frames": [int(frame) for frame in switch_frames],
        "update_triggered": bool(all(triggered_flags)) if triggered_flags else False,
        "update_triggered_flags": [bool(flag) for flag in triggered_flags],
        "runtime_update_contract": str(args.runtime_update_contract),
        "runtime_route_lookahead_m": float(args.runtime_route_lookahead_m),
        "runtime_bridge_frames": int(args.runtime_bridge_frames),
        "dynamic_reanchor_mode": str(args.dynamic_reanchor_mode),
        "dynamic_reanchor_yaw_source": str(args.dynamic_reanchor_yaw_source),
        "dynamic_reanchor_tangent_frames": int(args.dynamic_reanchor_tangent_frames),
    }
    return out, metrics, elapsed, seed


def main() -> int:
    args = _parse_args()
    if bool(args.preview_only):
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(int(args.seed))
    cfg = load_config(config_path=args.config)
    sample_batch = _load_sample(args, cfg)
    sample_name = str(sample_batch["name"][0])
    caption_index = sample_batch.get("_caption_index")
    original_traj7 = sample_batch["traj_cond_7d"][0].float().cpu()
    original_frames = int(sample_batch["feature_length"][0].item())
    scenario = _build_condition_scenario_from_args(
        args,
        original_traj7,
        original_frames=original_frames,
        sample_name=sample_name,
        caption_index=caption_index,
    )
    root_source = condition_scenario_to_proposal(
        scenario,
        source_kind=str(args.condition_source),
    )

    model = None
    vae = None
    if args.root_source_refiner_ckpt is not None:
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
        root_refiner, root_text_encoder, _root_refiner_cfg = _load_root_refiner_from_ckpt(
            str(args.root_source_refiner_ckpt),
            str(device),
        )
        root_refiner = root_refiner.to(device).eval()
        root_text_encoder = root_text_encoder.to(device).eval()
        root_source = _refine_root_source_proposal_segments(
            root_source,
            ldf_model=model,
            root_refiner=root_refiner,
            root_text_encoder=root_text_encoder,
            text=str(sample_batch.get("_caption_text") or sample_batch["text"][0]),
            device=device,
            token_dt=float(args.token_dt),
            frames_per_token=int(args.frames_per_token),
            forced_future_frames=args.root_source_refiner_forced_frames,
            heading_override=str(args.root_source_refiner_heading_override),
        )

    condition_traj7 = proposal_to_world_traj7(root_source)
    root_source_update_frames = [
        int(frame) for frame in root_source.metadata.get("update_frames", ())
    ]
    root_source_kind = str(root_source.metadata.get("source_kind", "unknown"))
    root_source_name = str(
        root_source.metadata.get("scenario_name", root_source.source_id)
    )
    root_source_visual_mask = root_source.metadata.get("visual_mask")
    condition_batch = apply_updated_traj_to_sample_batch(sample_batch, condition_traj7)
    model_condition_sent = condition_batch["traj_cond_7d"][0].detach().cpu().float()
    artifacts_dir = out_dir / "artifacts"
    condition_source_plot = plot_7d_xz_heading(
        artifacts_dir / "root_source_proposal_xz_heading.png",
        {
            "original_gt": original_traj7[:original_frames],
            "root_source_proposal": condition_traj7,
        },
        update_frames=root_source_update_frames,
        title=f"{scenario.name}: root-source world route proposal",
    )
    model_condition_plot = plot_7d_xz_heading(
        artifacts_dir / "model_condition_sent_xz_heading.png",
        {
            "root_source_proposal": condition_traj7,
            "model_condition_sent": model_condition_sent,
        },
        update_frames=root_source_update_frames,
        title=f"{scenario.name}: root-source proposal after sample batching",
    )
    if bool(args.preview_only):
        plot_path = out_dir / "trajectory_update.png"
        plot_xz_trajectories(
            plot_path,
            {
                "original_gt": original_traj7[:original_frames, [0, 2]],
                "condition": condition_traj7[:, [0, 2]],
            },
            title=f"{scenario.name} condition preview",
            boundary_frames=root_source_update_frames + [original_frames],
        )
        summary = {
            "sample_name": sample_name,
            "caption_index": caption_index,
            "caption_text": sample_batch.get("_caption_text"),
            "cfg_text": float(args.cfg_text),
            "cfg_traj": float(args.cfg_traj),
            "history_length": int(args.history_length),
            "horizon_tokens": int(args.horizon_tokens),
            "num_runs": 0,
            "preview_only": True,
            "condition_source": str(args.condition_source),
            "root_source_kind": root_source_kind,
            "root_source_name": root_source_name,
            "condition_scenario": scenario.name,
            "condition_metadata": scenario.metadata,
            "root_source_metadata": root_source.metadata,
            "dynamic_updates": bool(args.dynamic_updates),
            "condition_traj_display": str(args.condition_traj_display),
            "update_frames": root_source_update_frames,
            "original_frames": int(original_frames),
            "condition_frames": int(condition_traj7.shape[0]),
            "trajectory_plot": str(plot_path),
            "condition_source_plot": str(condition_source_plot),
            "root_source_proposal_plot": str(condition_source_plot),
            "model_condition_sent_plot": str(model_condition_plot),
            "records": [],
        }
        with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if model is None or vae is None:
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

    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    debug_npz_dir = out_dir / "debug_npz"
    if bool(args.save_debug_npz):
        debug_npz_dir.mkdir(parents=True, exist_ok=True)

    records = []
    decoded_series = {}
    dynamic_condition_series = {}
    for run_idx in range(max(1, int(args.num_runs))):
        for alpha in [None] + _parse_alphas(args.alphas):
            if bool(args.dynamic_updates) and root_source_update_frames:
                run_out, metrics, elapsed, seed = _run_ldf_direct_multi_update_one(
                    model,
                    vae,
                    sample_batch,
                    route_traj7=condition_traj7,
                    update_frames=root_source_update_frames,
                    args=args,
                    device=device,
                    alpha=alpha,
                    run_idx=run_idx,
                )
            else:
                run_out, metrics, elapsed, seed = _run_one(
                    model,
                    vae,
                    condition_batch,
                    args,
                    device,
                    alpha=alpha,
                    run_idx=run_idx,
                )
            decoded = run_out["decoded_feature"]
            render_condition_traj7 = (
                run_out.get("condition_traj7", condition_traj7)
                .detach()
                .cpu()
                .float()
            )
            video_path = videos_dir / _video_name_for_alpha(alpha, run_idx=run_idx)
            condition_mask = (
                    root_source_visual_mask
                    if root_source_visual_mask is not None
                    else condition_visual_mask(
                        int(render_condition_traj7.shape[0]),
                        str(args.condition_traj_display),
                        run_out.get("switch_frames", root_source_update_frames),
                    )
                )
            label = "no_feedback" if alpha is None else f"alpha_{float(alpha):.2f}"
            if bool(args.render_video):
                render_motion_video(
                    decoded,
                    video_path,
                    dim=263,
                    traj_xz=render_condition_traj7[:, [0, 2]],
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
            if bool(args.save_debug_npz):
                generated_traj7 = torch.from_numpy(
                    extract_root_traj_feats_7d_263(decoded.numpy().astype(np.float32))
                ).float()
                debug_npz_path = debug_npz_dir / f"{label}_run{run_idx}.npz"
                offline_route_traj7 = condition_traj7.detach().cpu().float()
                model_condition_traj7 = render_condition_traj7.detach().cpu().float()
                debug_payload = {
                    "offline_route_traj7": offline_route_traj7.numpy().astype(np.float32),
                    "root_source_proposal_traj7": offline_route_traj7.numpy().astype(np.float32),
                    "model_condition_traj7": model_condition_traj7.numpy().astype(np.float32),
                    "generated_traj7": generated_traj7.numpy().astype(np.float32),
                    "condition_traj7": model_condition_traj7.numpy().astype(np.float32),
                    "decoded_feature": decoded.numpy().astype(np.float32),
                    "decoded_root_xz": _root_xz(decoded).numpy().astype(np.float32),
                    "route_traj7": offline_route_traj7.numpy().astype(np.float32),
                    "update_frames": np.asarray(run_out.get("update_frames", []), dtype=np.int64),
                    "raw_update_frames": np.asarray(
                        run_out.get("raw_update_frames", run_out.get("update_frames", [])),
                        dtype=np.int64,
                    ),
                    "raw_update_commits": np.asarray(
                        run_out.get("raw_update_commits", []),
                        dtype=np.int64,
                    ),
                    "first_uncommitted_tokens": np.asarray(
                        run_out.get("first_uncommitted_tokens", []),
                        dtype=np.int64,
                    ),
                    "effective_update_commits": np.asarray(
                        run_out.get("effective_update_commits", run_out.get("update_commits", [])),
                        dtype=np.int64,
                    ),
                    "effective_update_frame_starts": np.asarray(
                        run_out.get("effective_update_frame_starts", []),
                        dtype=np.int64,
                    ),
                    "switch_frames": np.asarray(run_out.get("switch_frames", []), dtype=np.int64),
                    "update_commits": np.asarray(run_out.get("update_commits", []), dtype=np.int64),
                }
                snapshot_switch_frames = []
                snapshot_names = []
                for snapshot_idx, snapshot in enumerate(run_out.get("condition_snapshots", [])):
                    snapshot_names.append(str(snapshot.get("name", f"snapshot_{snapshot_idx}")))
                    snapshot_switch_frames.append(int(snapshot.get("switch_frame", 0)))
                    debug_payload[f"snapshot_{snapshot_idx:02d}_condition_traj7"] = (
                        snapshot["condition_traj7"].numpy().astype(np.float32)
                    )
                    debug_payload[f"snapshot_{snapshot_idx:02d}_segment_source_traj7"] = (
                        snapshot["segment_source"].numpy().astype(np.float32)
                    )
                debug_payload["snapshot_names"] = np.asarray(snapshot_names)
                debug_payload["snapshot_switch_frames"] = np.asarray(
                    snapshot_switch_frames,
                    dtype=np.int64,
                )
                payload_names = []
                payload_commits = []
                for payload_idx, payload in enumerate(run_out.get("payload_snapshots", [])):
                    payload_name = f"payload_{payload_idx:02d}"
                    payload_names.append(payload_name)
                    payload_commits.append(int(payload.get("commit_index", -1)))
                    debug_payload[f"{payload_name}_commit_index"] = np.asarray(
                        [int(payload.get("commit_index", -1))],
                        dtype=np.int64,
                    )
                    debug_payload[f"{payload_name}_has_traj_input"] = np.asarray(
                        [bool(payload.get("has_traj_input", False))],
                        dtype=np.bool_,
                    )
                    if bool(payload.get("has_traj_input", False)):
                        debug_payload[f"{payload_name}_traj_abs_start_token"] = np.asarray(
                            [int(payload.get("traj_abs_start_token", -1))],
                            dtype=np.int64,
                        )
                        debug_payload[f"{payload_name}_body_anchor_abs_token"] = np.asarray(
                            [int(payload.get("body_anchor_abs_token", -1))],
                            dtype=np.int64,
                        )
                        debug_payload[f"{payload_name}_traj_num_tokens"] = np.asarray(
                            [int(payload.get("traj_num_tokens", -1))],
                            dtype=np.int64,
                        )
                        debug_payload[f"{payload_name}_traj_cond_7d_frame"] = (
                            payload["traj_cond_7d_frame"].numpy().astype(np.float32)
                        )
                        debug_payload[f"{payload_name}_traj_cond_frame_mask"] = (
                            payload["traj_cond_frame_mask"].numpy().astype(np.float32)
                        )
                        for sub_idx, subpayload in enumerate(payload.get("subpayloads", [])):
                            sub_name = f"{payload_name}_sub{sub_idx:02d}"
                            debug_payload[f"{sub_name}_traj_abs_start_token"] = np.asarray(
                                [int(subpayload.get("traj_abs_start_token", -1))],
                                dtype=np.int64,
                            )
                            debug_payload[f"{sub_name}_body_anchor_abs_token"] = np.asarray(
                                [int(subpayload.get("body_anchor_abs_token", -1))],
                                dtype=np.int64,
                            )
                            debug_payload[f"{sub_name}_traj_num_tokens"] = np.asarray(
                                [int(subpayload.get("traj_num_tokens", -1))],
                                dtype=np.int64,
                            )
                            debug_payload[f"{sub_name}_traj_cond_7d_frame"] = (
                                subpayload["traj_cond_7d_frame"].numpy().astype(np.float32)
                            )
                            debug_payload[f"{sub_name}_traj_cond_frame_mask"] = (
                                subpayload["traj_cond_frame_mask"].numpy().astype(np.float32)
                            )
                debug_payload["payload_names"] = np.asarray(payload_names)
                debug_payload["payload_commits"] = np.asarray(payload_commits, dtype=np.int64)
                np.savez(debug_npz_path, **debug_payload)
                plot_7d_xz_heading(
                    debug_npz_dir / f"{label}_run{run_idx}_route_condition_generated_xz_heading.png",
                    {
                        "offline_route": offline_route_traj7,
                        "model_condition": model_condition_traj7,
                        "generated": generated_traj7,
                    },
                    update_frames=run_out.get("switch_frames", root_source_update_frames),
                    title=f"{scenario.name}: {label} run{run_idx} route/condition/generated",
                )
            decoded_series[f"{label}_run{run_idx}"] = _root_xz(decoded)
            if bool(args.dynamic_updates) and run_idx == 0:
                dynamic_condition_series[f"sent_{label}_run{run_idx}"] = (
                    render_condition_traj7[:, [0, 2]]
                )
            records.append(
                {
                    "label": label,
                    "run_idx": int(run_idx),
                    "alpha": None if alpha is None else float(alpha),
                    "seed": int(seed),
                    "elapsed_sec": float(elapsed),
                    "fps": float(run_out["target_total_frames"] / elapsed),
                    "video": str(video_path) if bool(args.render_video) else None,
                    "condition_frames": int(render_condition_traj7.shape[0]),
                    "dynamic_updates": bool(args.dynamic_updates),
                    "runtime_update_contract": str(run_out.get(
                        "runtime_update_contract",
                        args.runtime_update_contract,
                    )),
                    "dynamic_reanchor_mode": str(run_out.get(
                        "dynamic_reanchor_mode",
                        args.dynamic_reanchor_mode,
                    )),
                    "dynamic_reanchor_yaw_source": str(run_out.get(
                        "dynamic_reanchor_yaw_source",
                        args.dynamic_reanchor_yaw_source,
                    )),
                    "switch_frames": [
                        int(frame) for frame in run_out.get("switch_frames", [])
                    ],
                    "raw_update_frames": [
                        int(frame) for frame in run_out.get("raw_update_frames", [])
                    ],
                    "raw_update_commits": [
                        int(commit) for commit in run_out.get("raw_update_commits", [])
                    ],
                    "first_uncommitted_tokens": [
                        int(commit)
                        for commit in run_out.get("first_uncommitted_tokens", [])
                    ],
                    "effective_update_commits": [
                        int(commit)
                        for commit in run_out.get("effective_update_commits", [])
                    ],
                    "effective_update_frame_starts": [
                        int(frame)
                        for frame in run_out.get("effective_update_frame_starts", [])
                    ],
                    "update_commits": [
                        int(commit) for commit in run_out.get("update_commits", [])
                    ],
                    "update_triggered": bool(
                        run_out.get("update_triggered", False)
                    ),
                    **{
                        key: float(value) if isinstance(value, (int, float)) else value
                        for key, value in metrics.items()
                    },
                }
            )

    plot_path = out_dir / "trajectory_update.png"
    plot_xz_trajectories(
        plot_path,
        {
            "original_gt": original_traj7[:original_frames, [0, 2]],
            "condition": condition_traj7[:, [0, 2]],
            **dynamic_condition_series,
            **decoded_series,
        },
        title=f"{scenario.name} condition update",
        boundary_frames=root_source_update_frames + [original_frames],
    )
    summary = {
        "sample_name": sample_name,
        "caption_index": caption_index,
        "caption_text": sample_batch.get("_caption_text"),
        "cfg_text": float(args.cfg_text),
        "cfg_traj": float(args.cfg_traj),
        "history_length": int(args.history_length),
        "horizon_tokens": int(args.horizon_tokens),
        "num_runs": max(1, int(args.num_runs)),
        "condition_source": str(args.condition_source),
        "root_source_kind": root_source_kind,
        "root_source_name": root_source_name,
        "condition_scenario": scenario.name,
        "condition_metadata": scenario.metadata,
        "root_source_metadata": root_source.metadata,
        "dynamic_updates": bool(args.dynamic_updates),
        "runtime_update_contract": str(args.runtime_update_contract),
        "runtime_route_lookahead_m": float(args.runtime_route_lookahead_m),
        "runtime_bridge_frames": int(args.runtime_bridge_frames),
        "dynamic_reanchor_mode": str(args.dynamic_reanchor_mode),
        "dynamic_reanchor_yaw_source": str(args.dynamic_reanchor_yaw_source),
        "dynamic_reanchor_tangent_frames": int(args.dynamic_reanchor_tangent_frames),
        "condition_traj_display": str(args.condition_traj_display),
        "update_frames": root_source_update_frames,
        "original_frames": int(original_frames),
        "condition_frames": int(condition_traj7.shape[0]),
        "trajectory_plot": str(plot_path),
        "condition_source_plot": str(condition_source_plot),
        "root_source_proposal_plot": str(condition_source_plot),
        "model_condition_sent_plot": str(model_condition_plot),
        "records": records,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
