"""Runtime generation runner data boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from eval.runtime.transforms import (
    build_eval_root_plan_from_points,
    build_eval_root_plan_from_world_7d,
    recovery_root_state_to_world,
    root_plan_to_world_7d,
    rotate_xz_points,
    rotate_world_7d_about_anchor,
)
from utils.inference_glue import InferenceGlueState, InferenceGlueTimeline
from utils.motion_process import (
    StreamJointRecovery263,
    extract_root_traj_feats_7d_263,
    extract_root_trajectory_263,
)
from utils.runtime_rootplan import build_rootplan_stream_payload_from_buffer
from utils.stream_rollout import (
    StreamTextSegment,
    StreamTextRolloutController,
    build_stream_step_model_input,
    build_stream_suffix_conditioning,
)
from utils.stream_traj import (
    StreamTrajectoryPlan,
    assign_uniform_timestamps,
    blend_future_trajs,
    reanchor_stream_plan_to_xz,
    resample_polyline_by_arclength,
    sample_plan_future,
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


def build_rootplan_stream_step_payload(
    model: Any,
    timeline: Any,
    *,
    history_length: int,
    traj_horizon_tokens: int,
    absolute_commit_index: int | None = None,
) -> dict | None:
    """Build the direct 7D payload consumed by ``stream_generate_step``."""

    traj_buf = getattr(model, "_traj_buf", None)
    chunk_size = int(getattr(model, "chunk_size", 1))
    local_commit = int(getattr(model, "commit_index", 0))
    absolute_commit = (
        local_commit if absolute_commit_index is None else int(absolute_commit_index)
    )
    return build_rootplan_stream_payload_from_buffer(
        traj_buf,
        timeline,
        local_commit_index=local_commit,
        absolute_commit_index=absolute_commit,
        chunk_size=chunk_size,
        history_length=history_length,
        traj_horizon_tokens=traj_horizon_tokens,
    )


def new_eval_timeline() -> InferenceGlueTimeline:
    return InferenceGlueTimeline(
        InferenceGlueState.initial(xz=(0.0, 0.0), yaw=0.0, dtype=torch.float32)
    )


def append_eval_timeline_state(
    timeline: InferenceGlueTimeline,
    *,
    commit_idx: int,
    recovery: Any,
    session_anchor_state: InferenceGlueState | None = None,
) -> None:
    commit_idx = int(commit_idx)
    if commit_idx <= timeline.head.commit_idx:
        return
    if session_anchor_state is None:
        session_anchor_state = timeline.at_commit(0)
    root, yaw = recovery_root_state_to_world(recovery, session_anchor_state)
    timeline.append(
        InferenceGlueState(
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
    session_anchor_state: InferenceGlueState | None = None,
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
    traj_buf = getattr(model, "_traj_buf", None)
    if traj_buf is None:
        return
    if hasattr(traj_buf, "reset"):
        traj_buf.reset()
    if hasattr(traj_buf, "clear"):
        traj_buf.clear()


def set_eval_root_plan(
    model: Any,
    timeline: InferenceGlueTimeline,
    stream_plan: Any,
    *,
    text: str,
    token_dt: float,
    root_refiner_runtime: Any = None,
    root_5d_history: list | None = None,
    replan_events: list | None = None,
    root_plan_events: list | None = None,
    frames_per_token: int = 4,
) -> bool:
    if not timeline.has_exact_state(int(stream_plan.start_commit_index)):
        return False
    anchor_state = timeline.at_commit(int(stream_plan.start_commit_index))
    root_plan_input = reanchor_stream_plan_to_xz(
        stream_plan,
        anchor_state.world_xz.detach().cpu().numpy(),
    )
    if root_refiner_runtime is not None:
        root_plan = root_refiner_runtime.build_root_plan(
            text=text,
            plan=root_plan_input,
            anchor_state=anchor_state,
            token_dt=token_dt,
            history_motion_world_5d=get_eval_root_refiner_history_5d(
                root_5d_history or [],
                int(stream_plan.start_commit_index),
            ),
        )
    else:
        root_plan = build_eval_root_plan_from_points(
            root_plan_input.points_xyz,
            anchor_state=anchor_state,
            token_dt=token_dt,
            frames_per_token=frames_per_token,
            source=root_plan_input.source or "eval_route",
        )
    model._traj_buf.set_root_plan(root_plan)
    if isinstance(root_plan_events, list):
        root_plan_events.append(
            {
                "commit": int(stream_plan.start_commit_index),
                "text": str(text),
                "source": str(stream_plan.source),
                "root_refiner": root_refiner_runtime is not None,
                "root_plan": root_plan,
            }
        )
    if isinstance(replan_events, list):
        replan_events.append(
            {
                "commit": int(stream_plan.start_commit_index),
                "text": str(text),
                "source": str(stream_plan.source),
                "root_refiner": root_refiner_runtime is not None,
            }
        )
    return True


def set_eval_root_plan_from_world_7d(
    model: Any,
    timeline: InferenceGlueTimeline,
    traj_7d_world: Any,
    *,
    start_commit_index: int,
    text: str,
    token_dt: float,
    source: str,
    replan_events: list | None = None,
    root_plan_events: list | None = None,
    frames_per_token: int = 4,
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
    model._traj_buf.set_root_plan(root_plan)
    if isinstance(root_plan_events, list):
        root_plan_events.append(
            {
                "commit": int(start_commit_index),
                "text": str(text),
                "source": str(source),
                "root_refiner": False,
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
) -> StreamTrajectoryPlan:
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
    return StreamTrajectoryPlan(
        times=times.astype(np.float32),
        points_xyz=points.astype(np.float32),
        start_commit_index=start_commit_index,
        version=int(version),
        source=str(source),
    )


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
    root_refiner_runtime = kwargs.get("root_refiner_runtime")
    replan_events = kwargs.get("replan_events")
    force_no_traj = bool(kwargs.get("force_no_traj", False))
    text = sample["text"] if isinstance(sample["text"], str) else sample["text"][0]
    timeline = new_eval_timeline()
    vae.clear_cache()
    model.init_generated(hl, batch_size=1, num_denoise_steps=nds)
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
        plan = StreamTrajectoryPlan(
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
            root_refiner_runtime=root_refiner_runtime,
            root_5d_history=root_5d_history,
            replan_events=replan_events,
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
        elif condition_path == "legacy_xyz" and mode == "step_predroot":
            traj_input = build_stream_suffix_conditioning(
                batch_sample, commit_idx, prefer_xyz=True
            )
            if traj_input is not None and len(roots) > 0:
                pred_root = roots[-1]
                gt_root = sample["traj"].numpy()[
                    min(commit_idx * 4, len(sample["traj"]) - 1)
                ].astype(np.float32)
                offset = pred_root.astype(np.float32) - gt_root
                traj = traj_input["traj"]
                if torch.is_tensor(traj):
                    traj_input["traj"] = traj + torch.from_numpy(offset).float().to(traj)
        elif condition_path == "legacy_xyz":
            traj_input = build_stream_suffix_conditioning(
                batch_sample, commit_idx, prefer_xyz=True
            )
        else:
            raise ValueError(f"unknown traj_condition_path {condition_path!r}")
        step_payload = build_stream_step_model_input(text, traj_input=traj_input)
        out = model.stream_generate_step(step_payload, first_chunk=first_chunk)
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
            frame_idx = append_eval_root_history(root_5d_history, frame_idx, recovery)
        append_eval_timeline_state(
            timeline,
            commit_idx=commit_idx + 1,
            recovery=recovery,
        )
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
    root_refiner_runtime: Any = None,
    replan_events: list | None = None,
    force_no_traj: bool = False,
    root_plan_events: list | None = None,
):
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
    model.init_generated(hl, batch_size=1, num_denoise_steps=nds)
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
            root_refiner_runtime=root_refiner_runtime,
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
            if (
                root_refiner_runtime is not None
                and text != active_root_refiner_text
                and timeline.has_exact_state(commit_idx)
            ):
                refreshed = set_eval_root_plan(
                    model,
                    timeline,
                    slice_stream_plan_from_commit(
                        plan_times,
                        plan_points,
                        start_commit_index=commit_idx,
                        token_dt=tdt,
                        waypoint_dt=wpdt,
                        version=commit_idx,
                        source="bench_babel_text",
                    ),
                    text=text,
                    token_dt=tdt,
                    root_refiner_runtime=root_refiner_runtime,
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
        elif condition_path == "legacy_xyz" and mode == "babel_timestamped":
            current_root = np.zeros(3, dtype=np.float32)
            current_root[[0, 2]] = recovery.r_pos_accum[[0, 2]].astype(np.float32)
            query_times = float(commit_idx) * tdt + np.arange(
                hz, dtype=np.float32
            ) * tdt
            gt_times = np.arange(len(gt_route), dtype=np.float32) / 20.0
            future = sample_timestamped_trajectory(gt_times, gt_route, query_times)
            anchor = sample_timestamped_trajectory(
                gt_times,
                gt_route,
                np.asarray([query_times[0]], dtype=np.float32),
            )[0]
            future = current_root + (future - anchor.astype(np.float32))
            traj_input = {
                "traj": torch.from_numpy(future).float().unsqueeze(0),
                "token_mask": torch.ones(1, hz),
            }
        elif condition_path == "legacy_xyz":
            current_root = np.zeros(3, dtype=np.float32)
            current_root[[0, 2]] = recovery.r_pos_accum[[0, 2]].astype(np.float32)
            future = sample_plan_future(
                StreamTrajectoryPlan(
                    times=plan_times,
                    points_xyz=plan_points,
                    start_commit_index=0,
                    version=0,
                    source="bench",
                ),
                current_commit=commit_idx,
                current_root_xyz=current_root,
                horizon_tokens=hz,
                token_dt=tdt,
                reanchor_to_current_root=True,
            )
            traj_input = {
                "traj": torch.from_numpy(future).float().unsqueeze(0),
                "token_mask": torch.ones(1, hz),
            }
        else:
            raise ValueError(f"unknown traj_condition_path {condition_path!r}")
        step_payload = build_stream_step_model_input(text, traj_input=traj_input)
        out = model.stream_generate_step(step_payload, first_chunk=first_chunk)
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
            frame_idx = append_eval_root_history(root_5d_history, frame_idx, recovery)
        append_eval_timeline_state(
            timeline,
            commit_idx=commit_idx + 1,
            recovery=recovery,
        )
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
    root_refiner_runtime: Any = None,
    replan_events: list | None = None,
    force_no_traj: bool = False,
    gt_motion_7d: bool = False,
    root_plan_events: list | None = None,
):
    tl = sample["token_length"]
    tfs = 1 + 4 * (tl - 1) if tl > 1 else 1
    gr_arr = sample["traj"].numpy()
    text = sample["text"] if isinstance(sample["text"], str) else sample["text"][0]
    gr = extract_root_trajectory_263(sample["feature"].numpy()[:tfs])
    gt_traj_7d_world = None
    session_anchor_state = InferenceGlueState.initial(
        xz=(0.0, 0.0),
        yaw=0.0,
        dtype=torch.float32,
    )
    if gt_motion_7d:
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
        first = torch.as_tensor(gt_traj_7d_world[0], dtype=torch.float32)
        session_anchor_state = InferenceGlueState.initial(
            xz=(float(first[0]), float(first[2])),
            yaw=float(torch.atan2(first[4], first[3])),
            dtype=torch.float32,
        )
        plan_pts = np.asarray(gt_traj_7d_world[:, :3], dtype=np.float32)
        plan_t = np.arange(len(plan_pts), dtype=np.float32) / float(fps)
    else:
        dur = (sample["feature_length"] - 1) / fps
        npt = max(2, int(round(dur / wpdt)) + 1)
        plan_pts = resample_polyline_by_arclength(gr_arr, npt)
        plan_t = assign_uniform_timestamps(npt, wpdt)
        if rotate_plan_deg:
            plan_pts = rotate_xz_points(plan_pts, plan_pts[0], float(rotate_plan_deg))
    timeline = InferenceGlueTimeline(session_anchor_state)
    vae.clear_cache()
    model.init_generated(hl, batch_size=1, num_denoise_steps=nds)
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
                StreamTrajectoryPlan(
                    times=plan_t,
                    points_xyz=plan_pts,
                    start_commit_index=0,
                    version=0,
                    source="bench_real",
                ),
                text=text,
                token_dt=tdt,
                root_refiner_runtime=root_refiner_runtime,
                root_5d_history=root_5d_history,
                replan_events=replan_events,
                root_plan_events=root_plan_events,
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
            current_root = np.zeros(3, dtype=np.float32)
            if mode == "real_gtroot":
                current_root = gr_arr[min(commit_idx * 4, len(gr_arr) - 1)].astype(
                    np.float32
                )
            else:
                current_root[[0, 2]] = recovery.r_pos_accum[[0, 2]].astype(np.float32)
            future = sample_plan_future(
                StreamTrajectoryPlan(
                    times=plan_t,
                    points_xyz=plan_pts,
                    start_commit_index=0,
                    version=0,
                    source="bench",
                ),
                current_commit=commit_idx,
                current_root_xyz=current_root,
                horizon_tokens=hz,
                token_dt=tdt,
                reanchor_to_current_root=True,
            )
            traj_input = {
                "traj": torch.from_numpy(future).float().unsqueeze(0),
                "token_mask": torch.ones(1, hz),
            }
        else:
            raise ValueError(f"unknown traj_condition_path {condition_path!r}")
        step_payload = build_stream_step_model_input(text, traj_input=traj_input)
        out = model.stream_generate_step(step_payload, first_chunk=first_chunk)
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
            frame_idx = append_eval_root_history(
                root_5d_history,
                frame_idx,
                recovery,
                session_anchor_state=session_anchor_state,
            )
        append_eval_timeline_state(
            timeline,
            commit_idx=commit_idx + 1,
            recovery=recovery,
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
    root_refiner_runtime: Any = None,
    replan_events: list | None = None,
    force_no_traj: bool = False,
    root_plan_events: list | None = None,
):
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
    model.init_generated(hl, batch_size=1, num_denoise_steps=nds)
    model.generated = model.generated.to(device)
    clear_model_traj_state(model)
    root_5d_history, frame_idx = [], 0
    edit_commit = split_tok
    effective_commit = edit_commit + edit_delay
    old_plan = StreamTrajectoryPlan(
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
            root_refiner_runtime=root_refiner_runtime,
            root_5d_history=root_5d_history,
            replan_events=replan_events,
            root_plan_events=root_plan_events,
        )
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
                and offset >= edit_delay + edit_blend
                and timeline.has_exact_state(effective_commit)
            ):
                new_root_plan_active = set_eval_root_plan(
                    model,
                    timeline,
                    new_plan,
                    text=text,
                    token_dt=tdt,
                    root_refiner_runtime=root_refiner_runtime,
                    root_5d_history=root_5d_history,
                    replan_events=replan_events,
                    root_plan_events=root_plan_events,
                )
            traj_input = build_rootplan_stream_step_payload(
                model,
                timeline,
                history_length=hl,
                traj_horizon_tokens=hz,
                absolute_commit_index=commit_idx,
            )
        elif condition_path == "legacy_xyz":
            current_root = np.zeros(3, dtype=np.float32)
            current_root[[0, 2]] = recovery.r_pos_accum[[0, 2]].astype(np.float32)
            reanchor = dict(
                current_commit=commit_idx,
                current_root_xyz=current_root,
                horizon_tokens=hz,
                token_dt=tdt,
                reanchor_to_current_root=True,
            )
            old_future = sample_plan_future(old_plan, **reanchor)
            if offset < edit_delay:
                future = old_future
            elif offset < edit_delay + edit_blend and edit_blend > 0:
                new_future = sample_plan_future(new_plan, **reanchor)
                weight = smoothstep01(float(offset - edit_delay) / edit_blend)
                future = blend_future_trajs(old_future, new_future, weight)
            else:
                future = sample_plan_future(new_plan, **reanchor)
            traj_input = {
                "traj": torch.from_numpy(future).float().unsqueeze(0),
                "token_mask": torch.ones(1, hz),
            }
        else:
            raise ValueError(f"unknown traj_condition_path {condition_path!r}")
        step_payload = build_stream_step_model_input(text, traj_input=traj_input)
        out = model.stream_generate_step(step_payload, first_chunk=first_chunk)
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
            frame_idx = append_eval_root_history(root_5d_history, frame_idx, recovery)
        append_eval_timeline_state(
            timeline,
            commit_idx=commit_idx + 1,
            recovery=recovery,
        )
        decs.append(dec)
    vae.clear_cache()
    pred_motion = np.concatenate(decs, axis=0)[:total_tfs] if decs else np.zeros((0, 263))
    pred_root = np.array(roots, dtype=np.float32)[:total_tfs] if roots else np.zeros((0, 3))
    effective_frame = token_start_frame(effective_commit)
    if len(pred_root) > 0:
        anchor_frame = int(np.clip(effective_frame, 0, len(pred_root) - 1))
        new_anchor_xz = pred_root[anchor_frame, [0, 2]]
    else:
        new_anchor_xz = None
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
        new_anchor_xz=new_anchor_xz,
    )
    return pred_motion, pred_root, gr, target_t, target_pts, total_tfs


def root_plan_events_to_diagnostic_arrays(
    root_plan_events: list[dict],
) -> tuple[np.ndarray, int]:
    """Return concatenated world 7D root plans and summed token count."""

    world_chunks = []
    num_tokens = 0
    for event in root_plan_events:
        root_plan = event.get("root_plan") if isinstance(event, dict) else None
        if root_plan is None:
            continue
        world = root_plan_to_world_7d(root_plan).detach().cpu().numpy()
        if len(world) > 0:
            world_chunks.append(world.astype(np.float32, copy=False))
        num_tokens += int(root_plan.num_tokens_pred)
    if not world_chunks:
        return np.zeros((0, 7), dtype=np.float32), int(num_tokens)
    return np.concatenate(world_chunks, axis=0).astype(np.float32), int(num_tokens)


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
