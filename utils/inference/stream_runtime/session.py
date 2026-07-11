"""Authoritative prepare/mutate/commit transaction for streaming inference."""

from __future__ import annotations

import copy
import threading
from dataclasses import replace

import numpy as np
import torch

from utils.inference.stream_execution import (
    RootFeedbackConfig,
    decode_token_with_root_feedback,
    restore_ldf_stream_state,
    restore_recovery_state,
    restore_vae_stream_state,
    snapshot_ldf_stream_state,
    snapshot_recovery_state,
    snapshot_vae_stream_state,
)
from utils.inference.timeline import (
    RootFrameState,
    RootTimeline,
    recovery_root_state_to_world,
)
from utils.motion_process import build_physical_7d_from_5d
from utils.token_frame import first_future_frame_abs, token_range_to_frame_slice

from .commands import (
    UNSET,
    RuntimeCommandEnvelope,
    RuntimeCommandQueue,
    SetRuntimeControls,
    reduce_commands,
)
from .composer import ConditionComposer
from .contracts import (
    RouteProgressState,
    RouteStatus,
    RuntimeEvent,
    RuntimeStepConfig,
    SessionResetEvent,
    StreamCommitEvent,
)
from .history import GeneratedRootHistory
from .payload_builder import PayloadBuilder
from .snapshots import restore_rng_state, snapshot_rng_state
from .source_manager import RootSourceManager


class StreamRuntimeSession:
    """The sole owner of stream commit, decode, recovery, and publication."""

    def __init__(
        self,
        *,
        kernel,
        vae,
        recovery,
        timeline: RootTimeline,
        generated_history: GeneratedRootHistory,
        command_queue: RuntimeCommandQueue,
        source_manager: RootSourceManager,
        composer: ConditionComposer,
        payload_builder: PayloadBuilder,
        initial_config: RuntimeStepConfig,
        bridge_frames: int = 8,
    ) -> None:
        self.kernel = kernel
        self.vae = vae
        self.recovery = recovery
        self.timeline = timeline
        self.generated_history = generated_history
        self.command_queue = command_queue
        self.source_manager = source_manager
        self.composer = composer
        self.payload_builder = payload_builder
        self.config = initial_config
        self.initial_config = initial_config
        self.bridge_frames = int(bridge_frames)
        self.first_chunk = generated_history.next_frame_abs == 0
        self.session_anchor_state = copy.deepcopy(timeline.earliest)
        self.session_epoch = 0
        self._step_lock = threading.Lock()
        self._validate_reset_only_config(initial_config)
        self._initialize_model_stream_state(initial_config)
        if hasattr(self.vae, "clear_cache"):
            self.vae.clear_cache()

    @property
    def model(self):
        return self.kernel.ldf_model

    def submit(self, command: RuntimeCommandEnvelope) -> int:
        if isinstance(command, SetRuntimeControls):
            changes_history = (
                command.history_tokens is not UNSET
                and int(command.history_tokens) != int(self.config.history_tokens)
            )
            changes_denoise = (
                command.num_denoise_steps is not UNSET
                and command.num_denoise_steps != self.config.num_denoise_steps
            )
            if changes_history or changes_denoise:
                if not self._step_lock.acquire(blocking=False):
                    raise RuntimeError(
                        "history_tokens and num_denoise_steps require a reset "
                        "while a step is running"
                    )
                try:
                    if int(self.timeline.head.commit_idx) > 0:
                        raise RuntimeError(
                            "history_tokens and num_denoise_steps require a reset"
                        )
                    proposed = self.config
                    if command.history_tokens is not UNSET:
                        proposed = replace(
                            proposed,
                            history_tokens=int(command.history_tokens),
                        )
                    if command.num_denoise_steps is not UNSET:
                        proposed = replace(
                            proposed,
                            num_denoise_steps=command.num_denoise_steps,
                        )
                    self._validate_reset_only_config(proposed)
                    return self.command_queue.submit(command)
                finally:
                    self._step_lock.release()
        return self.command_queue.submit(command)

    def _validate_reset_only_config(self, config: RuntimeStepConfig) -> None:
        steps = config.num_denoise_steps
        if steps is None:
            return
        chunk_size = int(getattr(self.kernel, "chunk_size", 1))
        if int(steps) % chunk_size != 0:
            raise ValueError(
                "num_denoise_steps must be divisible by chunk_size "
                f"({chunk_size}), got {steps}"
            )

    def _initialize_model_stream_state(self, config: RuntimeStepConfig) -> None:
        model = self.model
        num_steps = (
            int(config.num_denoise_steps)
            if config.num_denoise_steps is not None
            else int(getattr(model, "noise_steps", 1))
        )
        if hasattr(model, "init_generated"):
            model.init_generated(
                int(config.history_tokens),
                batch_size=int(getattr(self.kernel, "batch_size", 1)),
                num_denoise_steps=num_steps,
                traj_buffer=None,
            )
            return
        if hasattr(model, "commit_index"):
            model.commit_index = 0
        if hasattr(model, "current_step"):
            model.current_step = 0

    @staticmethod
    def _reset_only_controls_changed(
        before: RuntimeStepConfig,
        after: RuntimeStepConfig,
    ) -> bool:
        return (
            int(before.history_tokens) != int(after.history_tokens)
            or before.num_denoise_steps != after.num_denoise_steps
        )

    def _rng_devices(self):
        device = torch.device(getattr(self.kernel, "device", "cpu"))
        return [device] if device.type == "cuda" else []

    def _snapshot(self) -> dict:
        return {
            "model": snapshot_ldf_stream_state(self.model),
            "vae": snapshot_vae_stream_state(self.vae),
            "recovery": snapshot_recovery_state(self.recovery),
            "rng": snapshot_rng_state(self._rng_devices()),
            "timeline": copy.deepcopy(self.timeline._states),
            "history_base": int(self.generated_history.base_frame_abs),
            "history_frames": self.generated_history.frames_7d.detach().clone(),
            "source": self.source_manager.snapshot_state(),
            "config": self.config,
            "first_chunk": bool(self.first_chunk),
            "anchor": copy.deepcopy(self.session_anchor_state),
            "session_epoch": int(self.session_epoch),
        }

    def _restore(self, state: dict) -> None:
        restore_ldf_stream_state(self.model, state["model"])
        restore_vae_stream_state(self.vae, state["vae"])
        restore_recovery_state(self.recovery, state["recovery"])
        restore_rng_state(state["rng"])
        self.timeline._states = copy.deepcopy(state["timeline"])
        self.generated_history.base_frame_abs = int(state["history_base"])
        self.generated_history.frames_7d = state["history_frames"].detach().clone()
        self.source_manager.restore_state(state["source"])
        self.config = state["config"]
        self.first_chunk = bool(state["first_chunk"])
        self.session_anchor_state = copy.deepcopy(state["anchor"])
        self.session_epoch = int(state["session_epoch"])

    def _reset_owned(
        self,
        *,
        initial_state: RootFrameState | None = None,
        applied_command_version: int,
    ) -> SessionResetEvent:
        previous_epoch = self.session_epoch
        state = initial_state or RootFrameState.initial(
            device=self.timeline.earliest.world_xz.device,
            dtype=self.timeline.earliest.world_xz.dtype,
        )
        self.timeline.reset_to(state)
        self.generated_history.reset_to(0)
        self.session_anchor_state = copy.deepcopy(state)
        self.source_manager.reset()
        self.config = self.initial_config
        self.first_chunk = True
        self.session_epoch += 1
        self._initialize_model_stream_state(self.initial_config)
        if hasattr(self.recovery, "reset"):
            self.recovery.reset()
        if hasattr(self.vae, "clear_cache"):
            self.vae.clear_cache()
        return SessionResetEvent(
            previous_session_epoch=previous_epoch,
            session_epoch=self.session_epoch,
            applied_command_version=int(applied_command_version),
        )

    def reset(
        self,
        initial_state: RootFrameState | None = None,
        *,
        applied_command_version: int = 0,
    ) -> SessionResetEvent:
        if not self._step_lock.acquire(blocking=False):
            raise RuntimeError("concurrent StreamRuntimeSession reset/step is forbidden")
        try:
            return self._reset_owned(
                initial_state=initial_state,
                applied_command_version=applied_command_version,
            )
        finally:
            self._step_lock.release()

    def _compose_payload(self, active, config, commit_abs):
        if active is None:
            return None, None, RouteStatus.INACTIVE, None
        chunk_size = int(getattr(self.kernel, "chunk_size", 1))
        future_slice = token_range_to_frame_slice(
            int(commit_abs),
            chunk_size + int(config.horizon_tokens),
        )
        horizon_frames = int(future_slice.stop - future_slice.start)
        composed = self.composer.compose(
            active,
            self.generated_history,
            self.timeline.head,
            first_future_frame_abs(int(commit_abs)),
            active.progress,
            horizon_frames,
            bridge_frames=self.bridge_frames,
        )
        payload = self.payload_builder.build(
            composed,
            self.timeline,
            local_commit_before=int(getattr(self.model, "commit_index", 0)),
            absolute_commit_before=int(commit_abs),
            chunk_size=chunk_size,
            history_tokens=int(config.history_tokens),
            horizon_tokens=int(config.horizon_tokens),
        )
        return payload, composed.proposed_route_progress, composed.route_status, composed

    def _recover_root_frames(self, decoded_chunk: torch.Tensor, start_frame_abs: int):
        joints = []
        roots_5d = []
        for frame in decoded_chunk:
            joints.append(self.recovery.process_frame(frame.detach().cpu().numpy()))
            world_root, world_yaw = recovery_root_state_to_world(
                self.recovery,
                self.session_anchor_state,
            )
            roots_5d.append(
                [
                    float(world_root[0]),
                    float(frame[3].detach().cpu().item()),
                    float(world_root[2]),
                    float(np.cos(world_yaw)),
                    float(np.sin(world_yaw)),
                ]
            )
        root_5d = torch.as_tensor(roots_5d, dtype=torch.float32)
        if int(self.generated_history.frames_7d.shape[0]):
            joined = torch.cat(
                [self.generated_history.frames_7d[-1:, :5].cpu(), root_5d],
                dim=0,
            )
            root_7d = build_physical_7d_from_5d(joined)[1:]
        else:
            root_7d = build_physical_7d_from_5d(root_5d)
        return torch.as_tensor(np.stack(joints), dtype=torch.float32), root_7d

    def _trim_retained_state(
        self,
        *,
        absolute_commit_after: int,
        config: RuntimeStepConfig,
    ) -> None:
        earliest_anchor_commit = max(
            0,
            int(absolute_commit_after) - int(config.history_tokens),
        )
        self.timeline.trim_before(earliest_anchor_commit)
        earliest_frame = int(
            token_range_to_frame_slice(earliest_anchor_commit, 1).start
        )
        earliest_frame = min(earliest_frame, self.generated_history.next_frame_abs)
        self.generated_history.trim_before(earliest_frame)

    def step(self) -> RuntimeEvent:
        if not self._step_lock.acquire(blocking=False):
            raise RuntimeError("concurrent StreamRuntimeSession.step() is forbidden")
        try:
            return self._step_unlocked()
        finally:
            self._step_lock.release()

    def _step_unlocked(self) -> RuntimeEvent:
        snapshot = self._snapshot()
        batch = None
        try:
            commit_abs = int(self.timeline.head.commit_idx)
            batch = self.command_queue.prepare_due(commit_abs)
            transition = reduce_commands(
                self.config,
                batch,
                self.timeline.head,
                reset_base_config=self.initial_config,
            )
            reset_controls_changed = self._reset_only_controls_changed(
                self.config,
                transition.proposed_config,
            )
            if (
                transition.reset_intent is None
                and reset_controls_changed
                and commit_abs > 0
            ):
                raise RuntimeError(
                    "history_tokens and num_denoise_steps require a reset"
                )
            if transition.reset_intent is not None and transition.diagnostics.get(
                "reset_is_exclusive"
            ):
                event = self._reset_owned(
                    applied_command_version=transition.reset_intent.version
                )
                self.command_queue.ack(
                    batch,
                    discard_through_version=transition.reset_intent.version,
                )
                return event

            reset_epoch = transition.reset_intent is not None
            if reset_epoch:
                self._reset_owned(
                    applied_command_version=transition.reset_intent.version
                )
                commit_abs = 0
            if self._reset_only_controls_changed(
                self.config,
                transition.proposed_config,
            ):
                self._initialize_model_stream_state(transition.proposed_config)
            boundary = self.timeline.head
            prepared_source = self.source_manager.prepare_transition(
                transition.root_source_command,
                boundary,
                commit_abs,
                reset_epoch=reset_epoch,
            )
            config = transition.proposed_config
            self.model.cfg_scale_text = float(config.text_guidance_scale)
            self.model.cfg_scale_traj = float(config.trajectory_guidance_scale)
            payload, progress, route_status, _ = self._compose_payload(
                prepared_source.active,
                config,
                commit_abs,
            )
            kernel_result = self.kernel.generate_token(
                config.text,
                payload,
                first_chunk=self.first_chunk,
                num_denoise_steps=config.num_denoise_steps,
            )
            if int(kernel_result.absolute_commit_before) != commit_abs:
                raise RuntimeError(
                    "LDF/timeline commit mismatch: "
                    f"model={kernel_result.absolute_commit_before}, "
                    f"timeline={commit_abs}"
                )
            feedback = decode_token_with_root_feedback(
                model=self.model,
                vae=self.vae,
                latent_token=kernel_result.raw_latent,
                traj_payload=payload,
                generated_frame_count=self.generated_history.next_frame_abs,
                local_commit_index=kernel_result.local_commit_before,
                first_chunk=self.first_chunk,
                config=RootFeedbackConfig(
                    enabled=config.root_feedback_enabled,
                    xz_blend_alpha=config.root_feedback_xz_blend_alpha,
                ),
                device=getattr(self.kernel, "device", torch.device("cpu")),
            )
            root_start = first_future_frame_abs(commit_abs)
            joint_frames, root_frames = self._recover_root_frames(
                feedback.decoded_motion_chunk,
                root_start,
            )
            history_root_frames = root_frames.to(
                device=self.generated_history.frames_7d.device,
                dtype=self.generated_history.frames_7d.dtype,
            )
            self.generated_history.append(
                history_root_frames,
                start_frame_abs=root_start,
            )
            absolute_after = commit_abs + 1
            last = history_root_frames[-1]
            timeline_state = RootFrameState(
                commit_idx=absolute_after,
                world_xz=last[[0, 2]].to(
                    device=self.session_anchor_state.world_xz.device,
                    dtype=self.session_anchor_state.world_xz.dtype,
                ),
                world_yaw=torch.atan2(last[4], last[3]).to(
                    device=self.session_anchor_state.world_yaw.device,
                    dtype=self.session_anchor_state.world_yaw.dtype,
                ),
                source="stream_runtime_session",
            )
            self.timeline.append(timeline_state)
            lifecycle_events = self.source_manager.commit_transition(
                prepared_source,
                proposed_progress=progress,
                route_status=route_status,
            )
            self.config = config
            self.first_chunk = False
            active = self.source_manager.active
            event = StreamCommitEvent(
                absolute_commit_before=commit_abs,
                absolute_commit_after=absolute_after,
                local_commit_before=kernel_result.local_commit_before,
                local_commit_after=kernel_result.local_commit_after,
                latent_buffer_start_commit_abs=kernel_result.latent_buffer_start_commit_abs,
                latent_buffer_epoch=kernel_result.latent_buffer_epoch,
                committed_latent=feedback.latent_token,
                decoded_chunk=feedback.decoded_motion_chunk,
                joint_frames=joint_frames,
                root_frames_start_abs=root_start,
                root_frames=history_root_frames,
                timeline_state=timeline_state,
                actual_payload=kernel_result.actual_payload,
                source_id=None if active is None else active.proposal.source_id,
                source_version=None if active is None else active.proposal.version,
                actual_activation_commit=(
                    None if active is None else active.actual_activation_commit
                ),
                lifecycle_events=lifecycle_events,
                route_status=self.source_manager.route_status,
                root_feedback_diagnostics=feedback.debug,
            )
            self._trim_retained_state(
                absolute_commit_after=absolute_after,
                config=config,
            )
            self.command_queue.ack(
                batch,
                discard_through_version=(
                    None
                    if transition.reset_intent is None
                    else transition.reset_intent.version
                ),
            )
            return event
        except Exception:
            if batch is not None:
                try:
                    self.command_queue.release(batch)
                except ValueError:
                    pass
            self._restore(snapshot)
            raise


__all__ = ["StreamRuntimeSession"]
