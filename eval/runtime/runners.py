"""Runtime generation runner data boundaries."""

from __future__ import annotations

import numpy as np
import torch

from dataclasses import dataclass, field
from typing import Any, Mapping
from eval.runtime.transforms import (
    build_eval_root_plan_from_points,
    build_eval_root_plan_from_world_7d,
    compose_turn_root_plan,
    hybridize_root_plan_with_gt_7d,
    recovery_root_state_to_world,
    root_plan_to_world_7d,
    rotate_xz_points,
    rotate_world_7d_about_anchor,
)
from eval.runtime.root_projection import (
    LateDenoiseRootProjectionConfig,
    build_late_denoise_root_projection_callback,
    TimeConsistentRootGuidanceConfig,
    build_time_consistent_root_guidance_callback,
)
from utils.inference.timeline import (
    RootFrameState,
    RootTimeline,
    append_timeline_state_at_token_start_frame,
)
from utils.motion_process import (
    StreamJointRecovery263,
    extract_root_traj_feats_7d_263,
    extract_root_trajectory_263,
)
from utils.inference.root_plan import build_root_plan_stream_payload
from utils.inference.route_condition import (
    RoutePlan,
    sample_route_future,
)
from utils.inference.stream_generator import StreamGenerator
from utils.inference.geometry import (
    assign_uniform_timestamps,
    blend_future_trajs,
    resample_polyline_by_arclength,
    sample_plan_by_time,
    sample_timestamped_trajectory,
    smoothstep01,
)
from utils.token_frame import token_start_frame


@dataclass(frozen=True)
class RuntimeGenerationResult:
    """One web-demo-equivalent runtime generation result."""

    motion_263: Any
    pred_root_world: Any
    target_root_world: Any
    metrics: Mapping[str, Any] = field(default_factory=dict)
    root_plan: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamTextSegment:
    text: str
    token_end: int


class StreamTextRolloutController:
    def __init__(self, segments: list[StreamTextSegment]):
        self.segments = segments or [StreamTextSegment(text="", token_end=0)]

    def get_text_for_commit_index(self, commit_index: int) -> str:
        for segment in self.segments:
            if int(commit_index) < int(segment.token_end):
                return segment.text
        return self.segments[-1].text


def build_rootplan_stream_step_payload(
    model: Any,
    timeline: Any,
    *,
    history_length: int,
    traj_horizon_tokens: int,
    absolute_commit_index: int | None = None,
) -> dict | None:
    """Build the direct 7D payload consumed by ``stream_generate_step``."""

    root_plan = getattr(model, "_active_root_plan", None)
    chunk_size = int(getattr(model, "chunk_size", 1))
    local_commit = int(getattr(model, "commit_index", 0))
    absolute_commit = (
        local_commit if absolute_commit_index is None else int(absolute_commit_index)
    )
    return build_root_plan_stream_payload(
        root_plan,
        timeline,
        local_commit_index=local_commit,
        absolute_commit_index=absolute_commit,
        chunk_size=chunk_size,
        history_length=history_length,
        traj_horizon_tokens=traj_horizon_tokens,
    )


def _stream_generate_step_with_projection(
    model: Any,
    vae: Any,
    step_payload: dict,
    *,
    first_chunk: bool,
    condition_provider,
    root_projection_config: LateDenoiseRootProjectionConfig | None = None,
    time_consistent_guidance_config: TimeConsistentRootGuidanceConfig | None = None,
):
    projection_callback = None
    if time_consistent_guidance_config is not None:
        projection_callback = build_time_consistent_root_guidance_callback(
            vae=vae,
            config=time_consistent_guidance_config,
        )
    if projection_callback is None and root_projection_config is not None:
        projection_callback = build_late_denoise_root_projection_callback(
            vae=vae,
            config=root_projection_config,
        )
    if projection_callback is None:
        return model.stream_generate_step(
            step_payload,
            first_chunk=first_chunk,
            condition=condition_provider,
        )
    return model.stream_generate_step(
        step_payload,
        first_chunk=first_chunk,
        condition=condition_provider,
        projection_callback=projection_callback,
    )


def new_eval_timeline() -> RootTimeline:
    return RootTimeline(
        RootFrameState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    )


def _initial_anchor_state_from_world_7d(traj_7d_world: Any) -> RootFrameState:
    first = torch.as_tensor(traj_7d_world[0], dtype=torch.float32)
    return RootFrameState.initial(
        xz=(float(first[0]), float(first[2])),
        yaw=float(torch.atan2(first[4], first[3])),
        dtype=torch.float32,
    )


def append_eval_timeline_state(
    timeline: RootTimeline,
    *,
    commit_idx: int,
    recovery: Any,
    session_anchor_state: RootFrameState | None = None,
) -> None:
    commit_idx = int(commit_idx)
    if commit_idx <= timeline.head.commit_idx:
        return
    if session_anchor_state is None:
        session_anchor_state = timeline.at_commit(0)
    root, yaw = recovery_root_state_to_world(recovery, session_anchor_state)
    timeline.append(
        RootFrameState(
            commit_idx=commit_idx,
            world_xz=torch.tensor(root[[0, 2]], dtype=torch.float32),
            world_yaw=torch.tensor(yaw, dtype=torch.float32),
            source="stream_eval",
        )
    )


def append_eval_root_history(
    root_5d_history: list,
    frame_idx: int,
    recovery: Any,
    *,
    session_anchor_state: RootFrameState | None = None,
) -> int:
    if session_anchor_state is None:
        root = np.asarray(recovery.r_pos_accum, dtype=np.float32).copy()
        yaw = -2.0 * float(recovery.r_rot_ang_accum)
    else:
        root, yaw = recovery_root_state_to_world(recovery, session_anchor_state)
    root5d = np.asarray(
        [root[0], root[1], root[2], np.cos(yaw), np.sin(yaw)],
        dtype=np.float32,
    )
    root_5d_history.append((int(frame_idx), root5d))
    return int(frame_idx) + 1


def get_eval_root_refiner_history_5d(
    root_5d_history: list,
    anchor_commit: int,
) -> np.ndarray | None:
    if not root_5d_history:
        return None
    anchor_frame = token_start_frame(max(0, int(anchor_commit)))
    frames = [
        np.asarray(root5d, dtype=np.float32)
        for frame_idx, root5d in root_5d_history
        if int(frame_idx) <= anchor_frame
    ]
    if not frames:
        return None
    return np.stack(frames, axis=0).astype(np.float32)


def clear_model_traj_state(model: Any) -> None:
    model._active_root_plan = None


def build_eval_root_plan_for_stream_plan(
    timeline: RootTimeline,
    stream_plan: Any,
    *,
    text: str,
    token_dt: float,
    root_refiner: Any = None,
    root_5d_history: list | None = None,
    frames_per_token: int = 4,
    forced_num_tokens: int | None = None,
    gt_root_7d_world: Any = None,
    hybrid_gt_mode: str | None = None,
    hybrid_source: str | None = None,
) -> Any | None:
    if not timeline.has_exact_state(int(stream_plan.start_commit_index)):
        return None
    anchor_state = timeline.at_commit(int(stream_plan.start_commit_index))
    root_plan_input = stream_plan
    if root_refiner is not None:
        refiner_kwargs = {
            "text": text,
            "route": root_plan_input,
            "anchor_state": anchor_state,
            "history_motion_world_5d": get_eval_root_refiner_history_5d(
                root_5d_history or [],
                int(stream_plan.start_commit_index),
            ),
        }
        if forced_num_tokens is not None:
            refiner_kwargs["forced_num_frames"] = int(1 + 4 * (int(forced_num_tokens) - 1))
        root_plan = root_refiner.build_root_plan(**refiner_kwargs)
        if hybrid_gt_mode is not None:
            if gt_root_7d_world is None:
                raise ValueError("gt_root_7d_world is required for hybrid RootPlan mode")
            root_plan = hybridize_root_plan_with_gt_7d(
                root_plan,
                gt_root_7d_world,
                mode=str(hybrid_gt_mode),
                source=str(hybrid_source or root_plan.source),
            )
    else:
        root_plan = build_eval_root_plan_from_points(
            root_plan_input.points_xyz,
            anchor_state=anchor_state,
            token_dt=token_dt,
            frames_per_token=frames_per_token,
            source=root_plan_input.source or "eval_route",
        )
    return root_plan


def _activate_eval_root_plan(model: Any, root_plan: Any) -> None:
    model._active_root_plan = root_plan
    setattr(model, "_runtime_last_root_plan", root_plan)


def _record_eval_root_plan_event(
    *,
    root_plan: Any,
    stream_plan: Any,
    text: str,
    root_refiner: bool,
    replan_events: list | None = None,
    root_plan_events: list | None = None,
    diagnostic_plan: bool = True,
) -> None:
    if isinstance(root_plan_events, list):
        root_plan_events.append(
            {
                "commit": int(stream_plan.start_commit_index),
                "text": str(text),
                "source": str(stream_plan.source),
                "root_refiner": bool(root_refiner),
                "diagnostic_plan": bool(diagnostic_plan),
                "root_plan": root_plan,
            }
        )
    if isinstance(replan_events, list):
        replan_events.append(
            {
                "commit": int(stream_plan.start_commit_index),
                "text": str(text),
                "source": str(stream_plan.source),
                "root_refiner": bool(root_refiner),
            }
        )


def set_eval_root_plan(
    model: Any,
    timeline: RootTimeline,
    stream_plan: Any,
    *,
    text: str,
    token_dt: float,
    root_refiner: Any = None,
    root_5d_history: list | None = None,
    replan_events: list | None = None,
    root_plan_events: list | None = None,
    frames_per_token: int = 4,
    diagnostic_plan: bool = True,
    forced_num_tokens: int | None = None,
    gt_root_7d_world: Any = None,
    hybrid_gt_mode: str | None = None,
    hybrid_source: str | None = None,
) -> bool:
    root_plan = build_eval_root_plan_for_stream_plan(
        timeline,
        stream_plan,
        text=text,
        token_dt=token_dt,
        root_refiner=root_refiner,
        root_5d_history=root_5d_history,
        frames_per_token=frames_per_token,
        forced_num_tokens=forced_num_tokens,
        gt_root_7d_world=gt_root_7d_world,
        hybrid_gt_mode=hybrid_gt_mode,
        hybrid_source=hybrid_source,
    )
    if root_plan is None:
        return False
    _activate_eval_root_plan(model, root_plan)
    _record_eval_root_plan_event(
        root_plan=root_plan,
        stream_plan=stream_plan,
        text=text,
        root_refiner=root_refiner is not None,
        replan_events=replan_events,
        root_plan_events=root_plan_events,
        diagnostic_plan=diagnostic_plan,
    )
    return True


def set_eval_root_plan_from_world_7d(
    model: Any,
    timeline: RootTimeline,
    traj_7d_world: Any,
    *,
    start_commit_index: int,
    text: str,
    token_dt: float,
    source: str,
    replan_events: list | None = None,
    root_plan_events: list | None = None,
    frames_per_token: int = 4,
    diagnostic_plan: bool = True,
) -> bool:
    if not timeline.has_exact_state(int(start_commit_index)):
        return False
    anchor_state = timeline.at_commit(int(start_commit_index))
    root_plan = build_eval_root_plan_from_world_7d(
        traj_7d_world,
        anchor_state=anchor_state,
        token_dt=token_dt,
        frames_per_token=frames_per_token,
        source=source,
    )
    _activate_eval_root_plan(model, root_plan)
    if isinstance(root_plan_events, list):
        root_plan_events.append(
            {
                "commit": int(start_commit_index),
                "text": str(text),
                "source": str(source),
                "root_refiner": False,
                "diagnostic_plan": bool(diagnostic_plan),
                "root_plan": root_plan,
            }
        )
    if isinstance(replan_events, list):
        replan_events.append(
            {
                "commit": int(start_commit_index),
                "text": str(text),
                "source": str(source),
                "root_refiner": False,
            }
        )
    return True


def slice_stream_plan_from_commit(
    plan_times: np.ndarray,
    plan_points_xyz: np.ndarray,
    *,
    start_commit_index: int,
    token_dt: float,
    waypoint_dt: float,
    version: int,
    source: str,
) -> RoutePlan:
    """Build a future-only stream plan anchored at an absolute commit index."""

    plan_times = np.asarray(plan_times, dtype=np.float32)
    plan_points_xyz = np.asarray(plan_points_xyz, dtype=np.float32)
    start_commit_index = int(start_commit_index)
    elapsed = max(0.0, float(start_commit_index) * float(token_dt))
    if plan_times.size == 0:
        times = np.asarray([0.0, float(waypoint_dt)], dtype=np.float32)
        points = np.zeros((2, 3), dtype=np.float32)
    else:
        end_time = max(elapsed, float(plan_times[-1]))
        npt = max(
            2,
            int(round((end_time - elapsed) / max(float(waypoint_dt), 1e-6))) + 1,
        )
        query_abs = elapsed + np.arange(npt, dtype=np.float32) * np.float32(waypoint_dt)
        points = sample_plan_by_time(plan_times, plan_points_xyz, query_abs)
        times = query_abs - np.float32(elapsed)
    return RoutePlan(
        times=times.astype(np.float32),
        points_xyz=points.astype(np.float32),
        start_commit_index=start_commit_index,
        version=int(version),
        source=str(source),
    )


def _text_segment_start_commit(
    segments: list[StreamTextSegment],
    commit_idx: int,
) -> int:
    start = 0
    commit_idx = int(commit_idx)
    for segment in segments:
        if commit_idx < int(segment.token_end):
            return int(start)
        start = int(segment.token_end)
    return int(start)


def build_turn_metric_target(
    *,
    old_times: np.ndarray,
    old_points_xyz: np.ndarray,
    new_times: np.ndarray,
    new_points_xyz: np.ndarray,
    target_frames: int,
    motion_fps: float,
    edit_commit: int,
    delay_tokens: int,
    blend_tokens: int,
    token_dt: float,
    new_anchor_xz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the time target that matches web delayed-replace RootPlan input."""

    target_t = np.arange(int(target_frames), dtype=np.float32) / np.float32(motion_fps)
    effective_t = (float(edit_commit) + float(delay_tokens)) * float(token_dt)
    activation_t = (
        float(edit_commit) + float(delay_tokens) + float(blend_tokens)
    ) * float(token_dt)
    old_target = sample_plan_by_time(old_times, old_points_xyz, target_t)
    new_target = sample_plan_by_time(new_times, new_points_xyz, target_t)
    if new_anchor_xz is None:
        anchor_point = sample_plan_by_time(
            old_times,
            old_points_xyz,
            np.asarray([effective_t], dtype=np.float32),
        )[0]
        anchor_xz = anchor_point[[0, 2]]
    else:
        anchor_xz = np.asarray(new_anchor_xz, dtype=np.float32).reshape(2)
    new_zero = sample_plan_by_time(
        new_times,
        new_points_xyz,
        np.asarray([effective_t], dtype=np.float32),
    )[0]
    offset = anchor_xz - new_zero[[0, 2]]
    new_target = new_target.copy()
    new_target[:, [0, 2]] += offset[None, :]
    use_new = target_t >= np.float32(activation_t)
    target = old_target.copy()
    target[use_new] = new_target[use_new]
    return target_t.astype(np.float32), target.astype(np.float32)


def run_step_case(
    model: Any,
    vae: Any,
    sample: Mapping[str, Any],
    device: torch.device,
    *,
    hl: int,
    nds: int,
    mode: str,
    **kwargs,
):
    tl = sample["token_length"]
    tfs = 1 + 4 * (tl - 1) if tl > 1 else 1
    hz = int(kwargs.get("hz", 20))
    tdt = float(kwargs.get("tdt", 0.20))
    fps = float(kwargs.get("fps", 20.0))
    condition_path = str(kwargs.get("condition_path", "rootplan_7d"))
    root_refiner = kwargs.get("root_refiner")
    if root_refiner is None:
        root_refiner = kwargs.get("root_refiner_runtime")
    replan_events = kwargs.get("replan_events")
    root_plan_events = kwargs.get("root_plan_events")
    force_no_traj = bool(kwargs.get("force_no_traj", False))
    root_projection_config = kwargs.get("root_projection_config")
    time_consistent_guidance_config = kwargs.get("time_consistent_guidance_config")
    text = sample["text"] if isinstance(sample["text"], str) else sample["text"][0]
    timeline = new_eval_timeline()
    vae.clear_cache()
    stream = StreamGenerator(
        ldf_model=model,
        device=device,
        history_length=hl,
        traj_horizon_tokens=hz,
        token_dt=tdt,
    )
    stream.init_ldf_generation(history_length=hl, batch_size=1, num_denoise_steps=nds)
    model.generated = model.generated.to(device)
    clear_model_traj_state(model)
    root_5d_history, frame_idx = [], 0
    batch_sample = {
        "traj": sample["traj"].unsqueeze(0),
        "token_length": torch.tensor([tl]),
        "traj_length": torch.tensor([sample["traj_length"]]),
        "token_mask": sample["token_mask"].unsqueeze(0),
        "traj_mask": sample["traj_mask"].unsqueeze(0),
    }
    if condition_path == "rootplan_7d" and mode != "step_no_traj" and not force_no_traj:
        plan = RoutePlan(
            times=np.arange(len(sample["traj"]), dtype=np.float32) / fps,
            points_xyz=sample["traj"].numpy().astype(np.float32),
            start_commit_index=0,
            version=0,
            source="bench_step",
        )
        set_eval_root_plan(
            model,
            timeline,
            plan,
            text=text,
            token_dt=tdt,
            root_refiner=root_refiner,
            root_5d_history=root_5d_history,
            replan_events=replan_events,
            root_plan_events=root_plan_events,
        )
    recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    decoded_chunks, roots = [], []
    first_chunk = True
    for commit_idx in range(tl):
        if mode == "step_no_traj" or force_no_traj:
            traj_input = None
        elif condition_path == "rootplan_7d":
            traj_input = build_rootplan_stream_step_payload(
                model,
                timeline,
                history_length=hl,
                traj_horizon_tokens=hz,
                absolute_commit_index=commit_idx,
            )
        elif condition_path == "legacy_xyz":
            raise ValueError("legacy_xyz streaming conditioning has been removed")
        else:
            raise ValueError(f"unknown traj_condition_path {condition_path!r}")
        step_payload = stream.build_step_input(text, traj_input=traj_input)
        condition_provider = stream.build_ldf_condition_provider(
            step_payload,
            first_chunk=first_chunk,
            device=device,
        )
        out = _stream_generate_step_with_projection(
            model,
            vae,
            step_payload,
            first_chunk=first_chunk,
            condition_provider=condition_provider,
            root_projection_config=root_projection_config,
            time_consistent_guidance_config=time_consistent_guidance_config,
        )
        decoded = (
            vae.stream_decode(
                out["generated"][0][None, :].to(device),
                first_chunk=first_chunk,
            )[0]
            .float()
            .cpu()
            .numpy()
        )
        first_chunk = False
        for frame in decoded:
            recovery.process_frame(frame)
            roots.append(recovery.r_pos_accum.copy())
            append_timeline_state_at_token_start_frame(
                timeline,
                frame_idx=frame_idx,
                recovery=recovery,
                source="stream_eval",
            )
            frame_idx = append_eval_root_history(root_5d_history, frame_idx, recovery)
        decoded_chunks.append(decoded)
    vae.clear_cache()
    pred_motion = (
        np.concatenate(decoded_chunks, axis=0)[:tfs]
        if decoded_chunks
        else np.zeros((0, 263))
    )
    pred_root = (
        np.asarray(roots, dtype=np.float32)[:tfs]
        if roots
        else np.zeros((0, 3), dtype=np.float32)
    )
    gt_root = extract_root_trajectory_263(sample["feature"].numpy()[:tfs])
    return pred_motion, pred_root, gt_root


def run_babel_case(
    model: Any,
    vae: Any,
    sample: Mapping[str, Any],
    device: torch.device,
    *,
    hl: int,
    nds: int,
    hz: int,
    tdt: float,
    wpdt: float,
    fps: float,
    mode: str,
    condition_path: str = "rootplan_7d",
    root_refiner: Any = None,
    root_refiner_runtime: Any = None,
    replan_events: list | None = None,
    force_no_traj: bool = False,
    root_plan_events: list | None = None,
    root_projection_config: LateDenoiseRootProjectionConfig | None = None,
    time_consistent_guidance_config: TimeConsistentRootGuidanceConfig | None = None,
):
    if root_refiner is None:
        root_refiner = root_refiner_runtime
    tl = sample["token_length"]
    tfs = 1 + 4 * (tl - 1) if tl > 1 else 1
    gt_route = sample["traj"].numpy()
    timeline = new_eval_timeline()
    duration = (sample["feature_length"] - 1) / fps
    num_points = max(2, int(round(duration / wpdt)) + 1)
    plan_points = resample_polyline_by_arclength(gt_route, num_points)
    plan_times = assign_uniform_timestamps(num_points, wpdt)
    segments = [
        StreamTextSegment(text=text, token_end=token_end)
        for text, token_end in zip(sample["text"], sample["token_text_end"])
    ]
    text_controller = StreamTextRolloutController(segments)
    gt_root = extract_root_trajectory_263(sample["feature"].numpy()[:tfs])
    vae.clear_cache()
    stream = StreamGenerator(
        ldf_model=model,
        device=device,
        history_length=hl,
        traj_horizon_tokens=hz,
        token_dt=tdt,
    )
    stream.init_ldf_generation(history_length=hl, batch_size=1, num_denoise_steps=nds)
    model.generated = model.generated.to(device)
    clear_model_traj_state(model)
    root_5d_history, frame_idx = [], 0
    active_root_refiner_text = None
    if condition_path == "rootplan_7d" and mode != "babel_no_traj" and not force_no_traj:
        initial_text = sample["text"][0] if sample["text"] else ""
        if set_eval_root_plan(
            model,
            timeline,
            slice_stream_plan_from_commit(
                plan_times,
                plan_points,
                start_commit_index=0,
                token_dt=tdt,
                waypoint_dt=wpdt,
                version=0,
                source="bench_babel",
            ),
            text=initial_text,
            token_dt=tdt,
            root_refiner=root_refiner,
            root_5d_history=root_5d_history,
            replan_events=replan_events,
            root_plan_events=root_plan_events,
        ):
            active_root_refiner_text = initial_text
    recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    decoded_chunks, roots = [], []
    first_chunk = True
    for commit_idx in range(tl):
        text = text_controller.get_text_for_commit_index(commit_idx)
        if mode == "babel_no_traj" or force_no_traj:
            traj_input = None
        elif condition_path == "rootplan_7d":
            text_anchor_commit = _text_segment_start_commit(segments, commit_idx)
            if (
                root_refiner is not None
                and text != active_root_refiner_text
                and timeline.has_exact_state(text_anchor_commit)
            ):
                refreshed = set_eval_root_plan(
                    model,
                    timeline,
                    slice_stream_plan_from_commit(
                        plan_times,
                        plan_points,
                        start_commit_index=text_anchor_commit,
                        token_dt=tdt,
                        waypoint_dt=wpdt,
                        version=text_anchor_commit,
                        source="bench_babel_text",
                    ),
                    text=text,
                    token_dt=tdt,
                    root_refiner=root_refiner,
                    root_5d_history=root_5d_history,
                    replan_events=replan_events,
                    root_plan_events=root_plan_events,
                )
                if refreshed:
                    active_root_refiner_text = text
            traj_input = build_rootplan_stream_step_payload(
                model,
                timeline,
                history_length=hl,
                traj_horizon_tokens=hz,
                absolute_commit_index=commit_idx,
            )
        elif condition_path == "legacy_xyz":
            raise ValueError("legacy_xyz streaming conditioning has been removed")
        else:
            raise ValueError(f"unknown traj_condition_path {condition_path!r}")
        step_payload = stream.build_step_input(text, traj_input=traj_input)
        condition_provider = stream.build_ldf_condition_provider(
            step_payload,
            first_chunk=first_chunk,
            device=device,
        )
        out = _stream_generate_step_with_projection(
            model,
            vae,
            step_payload,
            first_chunk=first_chunk,
            condition_provider=condition_provider,
            root_projection_config=root_projection_config,
            time_consistent_guidance_config=time_consistent_guidance_config,
        )
        decoded = (
            vae.stream_decode(
                out["generated"][0][None, :].to(device),
                first_chunk=first_chunk,
            )[0]
            .float()
            .cpu()
            .numpy()
        )
        first_chunk = False
        for frame in decoded:
            recovery.process_frame(frame)
            roots.append(recovery.r_pos_accum.copy())
            append_timeline_state_at_token_start_frame(
                timeline,
                frame_idx=frame_idx,
                recovery=recovery,
                source="stream_eval",
            )
            frame_idx = append_eval_root_history(root_5d_history, frame_idx, recovery)
        decoded_chunks.append(decoded)
    vae.clear_cache()
    pred_motion = (
        np.concatenate(decoded_chunks, axis=0)[:tfs]
        if decoded_chunks
        else np.zeros((0, 263))
    )
    pred_root = (
        np.asarray(roots, dtype=np.float32)[:tfs]
        if roots
        else np.zeros((0, 3), dtype=np.float32)
    )
    return pred_motion, pred_root, gt_root, plan_times, plan_points


def run_real_case(
    model: Any,
    vae: Any,
    sample: Mapping[str, Any],
    device: torch.device,
    *,
    hl: int,
    nds: int,
    hz: int,
    tdt: float,
    wpdt: float,
    fps: float,
    mode: str,
    rotate_plan_deg: float = 0.0,
    condition_path: str = "rootplan_7d",
    root_refiner: Any = None,
    root_refiner_runtime: Any = None,
    replan_events: list | None = None,
    force_no_traj: bool = False,
    gt_motion_7d: bool = False,
    root_plan_events: list | None = None,
    force_root_refiner_num_tokens: bool = False,
    root_refiner_gt_override: str | None = None,
    root_projection_config: LateDenoiseRootProjectionConfig | None = None,
    time_consistent_guidance_config: TimeConsistentRootGuidanceConfig | None = None,
):
    if root_refiner is None:
        root_refiner = root_refiner_runtime
    tl = sample["token_length"]
    tfs = 1 + 4 * (tl - 1) if tl > 1 else 1
    gr_arr = sample["traj"].numpy()
    text = sample["text"] if isinstance(sample["text"], str) else sample["text"][0]
    gr = extract_root_trajectory_263(sample["feature"].numpy()[:tfs])
    gt_traj_7d_world = None
    session_anchor_state = RootFrameState.initial(
        xz=(0.0, 0.0),
        yaw=0.0,
        dtype=torch.float32,
    )
    need_gt_world_7d = (
        gt_motion_7d
        or root_refiner_gt_override is not None
        or (float(rotate_plan_deg) != 0.0 and root_refiner is not None)
    )
    if need_gt_world_7d:
        gt_traj_7d_world = extract_root_traj_feats_7d_263(
            sample["feature"].numpy()[:tfs]
        )
        if rotate_plan_deg:
            gt_traj_7d_world = (
                rotate_world_7d_about_anchor(
                    gt_traj_7d_world,
                    anchor_xyz=gt_traj_7d_world[0, :3],
                    degrees=float(rotate_plan_deg),
                )
                .detach()
                .cpu()
                .numpy()
            )
    if gt_motion_7d:
        session_anchor_state = _initial_anchor_state_from_world_7d(gt_traj_7d_world)
        plan_pts = np.asarray(gt_traj_7d_world[:, :3], dtype=np.float32)
        plan_t = np.arange(len(plan_pts), dtype=np.float32) / float(fps)
    else:
        if (
            float(rotate_plan_deg) != 0.0
            and root_refiner is not None
            and gt_traj_7d_world is not None
        ):
            session_anchor_state = _initial_anchor_state_from_world_7d(
                gt_traj_7d_world
            )
        dur = (sample["feature_length"] - 1) / fps
        npt = max(2, int(round(dur / wpdt)) + 1)
        plan_pts = resample_polyline_by_arclength(gr_arr, npt)
        plan_t = assign_uniform_timestamps(npt, wpdt)
        if rotate_plan_deg:
            plan_pts = rotate_xz_points(plan_pts, plan_pts[0], float(rotate_plan_deg))
    timeline = RootTimeline(session_anchor_state)
    vae.clear_cache()
    stream = StreamGenerator(
        ldf_model=model,
        device=device,
        history_length=hl,
        traj_horizon_tokens=hz,
        token_dt=tdt,
    )
    stream.init_ldf_generation(history_length=hl, batch_size=1, num_denoise_steps=nds)
    model.generated = model.generated.to(device)
    clear_model_traj_state(model)
    root_5d_history, frame_idx = [], 0
    if condition_path == "rootplan_7d" and mode != "real_no_traj" and not force_no_traj:
        if gt_motion_7d:
            set_eval_root_plan_from_world_7d(
                model,
                timeline,
                gt_traj_7d_world,
                start_commit_index=0,
                text=text,
                token_dt=tdt,
                source="bench_real_gt_motion_7d",
                replan_events=replan_events,
                root_plan_events=root_plan_events,
            )
        else:
            set_eval_root_plan(
                model,
                timeline,
                RoutePlan(
                    times=plan_t,
                    points_xyz=plan_pts,
                    start_commit_index=0,
                    version=0,
                    source="bench_real",
                ),
                text=text,
                token_dt=tdt,
                root_refiner=root_refiner,
                root_5d_history=root_5d_history,
                replan_events=replan_events,
                root_plan_events=root_plan_events,
                forced_num_tokens=(
                    tl
                    if force_root_refiner_num_tokens
                    or root_refiner_gt_override is not None
                    else None
                ),
                gt_root_7d_world=gt_traj_7d_world,
                hybrid_gt_mode=root_refiner_gt_override,
                hybrid_source=(
                    f"root_refiner_{root_refiner_gt_override.replace('_', '')}"
                    if root_refiner_gt_override is not None
                    else None
                ),
            )
    recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    decs, roots, first_chunk = [], [], True
    for commit_idx in range(tl):
        if mode == "real_no_traj" or force_no_traj:
            traj_input = None
        elif condition_path == "rootplan_7d":
            traj_input = build_rootplan_stream_step_payload(
                model,
                timeline,
                history_length=hl,
                traj_horizon_tokens=hz,
                absolute_commit_index=commit_idx,
            )
        elif condition_path == "legacy_xyz":
            raise ValueError("legacy_xyz streaming conditioning has been removed")
        else:
            raise ValueError(f"unknown traj_condition_path {condition_path!r}")
        step_payload = stream.build_step_input(text, traj_input=traj_input)
        condition_provider = stream.build_ldf_condition_provider(
            step_payload,
            first_chunk=first_chunk,
            device=device,
        )
        out = _stream_generate_step_with_projection(
            model,
            vae,
            step_payload,
            first_chunk=first_chunk,
            condition_provider=condition_provider,
            root_projection_config=root_projection_config,
            time_consistent_guidance_config=time_consistent_guidance_config,
        )
        dec = (
            vae.stream_decode(
                out["generated"][0][None, :].to(device),
                first_chunk=first_chunk,
            )[0]
            .float()
            .cpu()
            .numpy()
        )
        first_chunk = False
        for frame in dec:
            recovery.process_frame(frame)
            root_world, _ = recovery_root_state_to_world(recovery, session_anchor_state)
            roots.append(root_world)
            append_timeline_state_at_token_start_frame(
                timeline,
                frame_idx=frame_idx,
                recovery=recovery,
                session_anchor_state=session_anchor_state,
                source="stream_eval",
            )
            frame_idx = append_eval_root_history(
                root_5d_history,
                frame_idx,
                recovery,
                session_anchor_state=session_anchor_state,
            )
        decs.append(dec)
    vae.clear_cache()
    pred_motion = np.concatenate(decs, axis=0)[:tfs] if decs else np.zeros((0, 263))
    pred_root = np.array(roots, dtype=np.float32)[:tfs] if roots else np.zeros((0, 3))
    return pred_motion, pred_root, gr, plan_t, plan_pts, float(session_anchor_state.world_yaw.item())


def run_turn_case(
    model: Any,
    vae: Any,
    sample: Mapping[str, Any],
    device: torch.device,
    *,
    hl: int,
    nds: int,
    hz: int,
    tdt: float,
    wpdt: float,
    fps: float,
    mode: str,
    angle: float,
    delay_tokens: int | float = 20,
    blend_tokens: int | float = 4,
    condition_path: str = "rootplan_7d",
    root_refiner: Any = None,
    root_refiner_runtime: Any = None,
    replan_events: list | None = None,
    force_no_traj: bool = False,
    root_plan_events: list | None = None,
    root_projection_config: LateDenoiseRootProjectionConfig | None = None,
    time_consistent_guidance_config: TimeConsistentRootGuidanceConfig | None = None,
):
    if root_refiner is None:
        root_refiner = root_refiner_runtime
    tl = sample["token_length"]
    tfs = 1 + 4 * (tl - 1) if tl > 1 else 1
    gr_arr = sample["traj"].numpy()
    text = sample["text"] if isinstance(sample["text"], str) else sample["text"][0]
    timeline = new_eval_timeline()
    dur = (sample["feature_length"] - 1) / fps
    npt = max(2, int(round(dur / wpdt)) + 1)
    plan_pts = resample_polyline_by_arclength(gr_arr, npt)
    plan_t = assign_uniform_timestamps(npt, wpdt)
    split_tok, split_frame = 15, max(1, 1 + 4 * 14)
    rot_pts = np.concatenate(
        [
            plan_pts[:split_frame],
            rotate_xz_points(plan_pts[split_frame:], plan_pts[split_frame - 1], angle),
        ],
        axis=0,
    )
    rot_t = np.arange(len(rot_pts), dtype=np.float32) * wpdt
    edit_delay = int(delay_tokens) if isinstance(delay_tokens, (int, float)) else 20
    edit_blend = int(blend_tokens) if isinstance(blend_tokens, (int, float)) else 4
    extra = max(0, split_tok + edit_delay + edit_blend - tl)
    total_tl = tl + extra + 8
    total_tfs = 1 + 4 * (total_tl - 1) if total_tl > 1 else 1
    needed_wp = max(len(plan_pts), int((total_tl + hz) * tdt / wpdt) + 2)
    for points, name in [(plan_pts, "plan"), (rot_pts, "rot")]:
        n_points = len(points)
        if needed_wp > n_points:
            start_wp = max(0, n_points - 5)
            velocity = points[-1] - points[start_wp]
            denom = max(1, n_points - 1 - start_wp)
            step = velocity / float(denom)
            n_extra = needed_wp - n_points
            extension = (
                points[-1][None, :]
                + np.arange(1, n_extra + 1, dtype=np.float32)[:, None] * step[None, :]
            )
            if name == "plan":
                plan_pts = np.concatenate([plan_pts, extension.astype(np.float32)], axis=0)
            else:
                rot_pts = np.concatenate([rot_pts, extension.astype(np.float32)], axis=0)
    plan_t = assign_uniform_timestamps(len(plan_pts), wpdt)
    rot_t = np.arange(len(rot_pts), dtype=np.float32) * wpdt
    gr = extract_root_trajectory_263(sample["feature"].numpy()[:tfs])
    vae.clear_cache()
    stream = StreamGenerator(
        ldf_model=model,
        device=device,
        history_length=hl,
        traj_horizon_tokens=hz,
        token_dt=tdt,
    )
    stream.init_ldf_generation(history_length=hl, batch_size=1, num_denoise_steps=nds)
    model.generated = model.generated.to(device)
    clear_model_traj_state(model)
    root_5d_history, frame_idx = [], 0
    edit_commit = split_tok
    effective_commit = edit_commit + edit_delay
    old_plan = RoutePlan(
        times=plan_t,
        points_xyz=plan_pts,
        start_commit_index=0,
        version=0,
        source="bench_old",
    )
    new_plan = slice_stream_plan_from_commit(
        rot_t,
        rot_pts,
        start_commit_index=effective_commit,
        token_dt=tdt,
        waypoint_dt=wpdt,
        version=1,
        source="bench_new",
    )
    if condition_path == "rootplan_7d" and not force_no_traj:
        set_eval_root_plan(
            model,
            timeline,
            old_plan,
            text=text,
            token_dt=tdt,
            root_refiner=root_refiner,
            root_5d_history=root_5d_history,
            replan_events=replan_events,
            root_plan_events=root_plan_events,
            diagnostic_plan=False,
        )
    old_root_plan = getattr(model, "_runtime_last_root_plan", None)
    diagnostic_root_plan = old_root_plan
    new_root_plan_active = False
    recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
    decs, roots, first_chunk = [], [], True
    for commit_idx in range(total_tl):
        offset = commit_idx - edit_commit
        if force_no_traj:
            traj_input = None
        elif condition_path == "rootplan_7d":
            if (
                not new_root_plan_active
                and offset >= edit_delay
                and timeline.has_exact_state(effective_commit)
            ):
                actual_activation_commit = int(commit_idx)
                new_root_plan = build_eval_root_plan_for_stream_plan(
                    timeline,
                    new_plan,
                    text=text,
                    token_dt=tdt,
                    root_refiner=root_refiner,
                    root_5d_history=root_5d_history,
                )
                if old_root_plan is not None and new_root_plan is not None:
                    composed_root_plan = compose_turn_root_plan(
                        old_root_plan,
                        new_root_plan,
                        switch_commit=actual_activation_commit,
                        blend_tokens=edit_blend,
                        source="bench_composed",
                    )
                    _activate_eval_root_plan(model, composed_root_plan)
                    diagnostic_root_plan = composed_root_plan
                    composed_stream_plan = RoutePlan(
                        times=rot_t,
                        points_xyz=rot_pts,
                        start_commit_index=actual_activation_commit,
                        version=2,
                        source="bench_composed",
                    )
                    _record_eval_root_plan_event(
                        root_plan=composed_root_plan,
                        stream_plan=composed_stream_plan,
                        text=text,
                        root_refiner=root_refiner is not None,
                        replan_events=replan_events,
                        root_plan_events=root_plan_events,
                        diagnostic_plan=True,
                    )
                    new_root_plan_active = True
            traj_input = build_rootplan_stream_step_payload(
                model,
                timeline,
                history_length=hl,
                traj_horizon_tokens=hz,
                absolute_commit_index=commit_idx,
            )
        elif condition_path == "legacy_xyz":
            raise ValueError("legacy_xyz streaming conditioning has been removed")
        else:
            raise ValueError(f"unknown traj_condition_path {condition_path!r}")
        step_payload = stream.build_step_input(text, traj_input=traj_input)
        condition_provider = stream.build_ldf_condition_provider(
            step_payload,
            first_chunk=first_chunk,
            device=device,
        )
        out = _stream_generate_step_with_projection(
            model,
            vae,
            step_payload,
            first_chunk=first_chunk,
            condition_provider=condition_provider,
            root_projection_config=root_projection_config,
            time_consistent_guidance_config=time_consistent_guidance_config,
        )
        dec = (
            vae.stream_decode(
                out["generated"][0][None, :].to(device),
                first_chunk=first_chunk,
            )[0]
            .float()
            .cpu()
            .numpy()
        )
        first_chunk = False
        for frame in dec:
            recovery.process_frame(frame)
            roots.append(recovery.r_pos_accum.copy())
            append_timeline_state_at_token_start_frame(
                timeline,
                frame_idx=frame_idx,
                recovery=recovery,
                source="stream_eval",
            )
            frame_idx = append_eval_root_history(root_5d_history, frame_idx, recovery)
        decs.append(dec)
    vae.clear_cache()
    pred_motion = np.concatenate(decs, axis=0)[:total_tfs] if decs else np.zeros((0, 263))
    pred_root = np.array(roots, dtype=np.float32)[:total_tfs] if roots else np.zeros((0, 3))
    target_t = np.arange(int(total_tfs), dtype=np.float32) / np.float32(fps)
    if condition_path == "rootplan_7d" and not force_no_traj and diagnostic_root_plan is not None:
        world_7d = root_plan_to_world_7d(diagnostic_root_plan).detach().cpu().numpy()
        target_pts = world_7d[:, :3].astype(np.float32, copy=False)
        if len(target_pts) < total_tfs and len(target_pts) > 0:
            pad = np.repeat(target_pts[-1:],
                            int(total_tfs) - int(len(target_pts)),
                            axis=0)
            target_pts = np.concatenate([target_pts, pad.astype(np.float32)], axis=0)
        target_pts = target_pts[:total_tfs]
    else:
        target_t, target_pts = build_turn_metric_target(
            old_times=plan_t,
            old_points_xyz=plan_pts,
            new_times=rot_t,
            new_points_xyz=rot_pts,
            target_frames=total_tfs,
            motion_fps=fps,
            edit_commit=edit_commit,
            delay_tokens=edit_delay,
            blend_tokens=edit_blend,
            token_dt=tdt,
        )
    return pred_motion, pred_root, gr, target_t, target_pts, total_tfs


def root_plan_events_to_diagnostic_arrays(
    root_plan_events: list[dict],
) -> tuple[np.ndarray, int]:
    """Return the final diagnostic world 7D root plan and token count."""

    selected_event = None
    for event in root_plan_events:
        root_plan = event.get("root_plan") if isinstance(event, dict) else None
        if root_plan is None:
            continue
        if bool(event.get("diagnostic_plan", False)):
            selected_event = event
        elif selected_event is None:
            selected_event = event
    if selected_event is None:
        return np.zeros((0, 7), dtype=np.float32), 0
    root_plan = selected_event["root_plan"]
    world = root_plan_to_world_7d(root_plan).detach().cpu().numpy()
    if len(world) <= 0:
        return np.zeros((0, 7), dtype=np.float32), int(root_plan.num_tokens_pred)
    return world.astype(np.float32, copy=False), int(root_plan.num_tokens_pred)


__all__ = [
    "RuntimeGenerationResult",
    "append_eval_root_history",
    "append_eval_timeline_state",
    "build_rootplan_stream_step_payload",
    "build_turn_metric_target",
    "clear_model_traj_state",
    "get_eval_root_refiner_history_5d",
    "new_eval_timeline",
    "root_plan_events_to_diagnostic_arrays",
    "run_babel_case",
    "run_real_case",
    "run_step_case",
    "run_turn_case",
    "set_eval_root_plan",
    "set_eval_root_plan_from_world_7d",
    "slice_stream_plan_from_commit",
]
