"""
Model Manager for real-time motion generation
Manages model loading, frame buffering, and streaming generation
"""
import sys
import os
import threading
import time
from collections import deque

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from eval.ldf.stream_generation import (
    _decode_latent_chunk,
    _decode_raw_chunk_preserving_feedback_cache,
    _encode_corrected_chunk_token,
    _write_committed_latent_to_model,
)
from utils.motion_process import (
    StreamJointRecovery263,
    append_traj_deltas_5d_to_7d,
    recover_root_rot_pos,
    replace_root_channels_263_window_from_7d,
)
from utils.inference.root_plan import RootPlan
from utils.inference.runtime_update import RootSourceProposal
from utils.inference.route_condition import (
    RoutePlan,
    RouteReferenceMode,
    RouteUpdate,
    reanchor_route_to_xz,
    sample_route_future,
)
from utils.inference.timeline import (
    RootFrameState,
    RootTimeline,
    append_timeline_state_at_token_start_frame,
)
from utils.token_frame import num_frames_for_tokens, token_start_frame
from utils.inference.geometry import (
    assign_uniform_timestamps,
    blend_future_trajs,
    build_remaining_polyline,
    dedupe_polyline,
    ensure_xyz,
    estimate_token_step_distance,
    normalize_manual_waypoints,
    project_point_to_polyline,
    resample_polyline,
    resample_polyline_by_arclength,
    sample_timestamped_trajectory,
    smoothstep01,
    translate_plan_to_current_root,
)
from web_demo.runtime.frame_buffer import FrameBuffer
from web_demo.runtime.contracts import TrajectoryRuntimeControls
from web_demo.runtime.generation_worker import GenerationWorker
from web_demo.runtime.model_loader import (
    build_stream_generator,
    load_ldf_models,
    load_model_bundle,
    reject_normalized_root_refiner_config,
    resolve_repo_path,
)
from web_demo.runtime.rootplan_controller import RootPlanController
from web_demo.runtime.state import GenerationState
from web_demo.runtime.trajectory_controller import TrajectoryController
from web_demo.runtime.web_runtime import WebRuntime


class ModelManager(WebRuntime):
    """Compatibility facade for the staged web runtime refactor."""
    def __init__(self, config_path=None, traj_mask_cfg=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        traj_mask_cfg = traj_mask_cfg or {}
        self.traj_mask_enabled = bool(traj_mask_cfg.get("enabled", False))
        self.traj_mask_keep_ratio_min = float(traj_mask_cfg.get("keep_ratio_min", 0.2))
        self.traj_mask_keep_ratio_max = float(traj_mask_cfg.get("keep_ratio_max", 0.3))
        self.traj_mask_keep_first_last = bool(traj_mask_cfg.get("keep_first_last", True))
        self.traj_mask_rng = np.random.default_rng()
        self._last_traj_mask_keep = None
        self._last_traj_mask_total = None
        
        # Load the complete runtime bundle once so RootRefiner modules are not
        # constructed again through a second StreamGenerator path.
        bundle = self._load_model_bundle(config_path, traj_mask_cfg)
        self.vae = bundle.vae
        self.model = bundle.ldf_model
        self.cfg = bundle.cfg
        self.stream_generator = bundle.stream_generator
        self.rootplan_controller = RootPlanController(self.stream_generator)
        
        # Frame buffer
        self.frame_buffer = FrameBuffer(target_buffer_size=4)
        
        # Stream joint recovery with smoothing
        self.smoothing_alpha = 0.5  # Default: medium smoothing
        self.stream_recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=self.smoothing_alpha)
        self._reset_root_timeline()
        self.root_feedback_enabled = bool(traj_mask_cfg.get("root_feedback_enabled", False))
        self.root_feedback_xz_blend_alpha = self._coerce_root_feedback_alpha(
            traj_mask_cfg.get("root_feedback_xz_blend_alpha", 0.5)
        )
        
        # Generation state
        self.current_text = ""
        self.is_generating = False
        self.generation_worker = GenerationWorker(self._generation_loop)
        self.reset_pending = False  # True while waiting for thread to stop before reset
        self.generation_state = GenerationState.IDLE
        
        # Trajectory control state. This moves into TrajectoryController in stages.
        self.traj_state_lock = threading.Lock()
        self.active_traj_plan: RoutePlan | None = None
        self.pending_update_event: RouteUpdate | None = None
        self._trajectory_state = "none"
        self._model_traj_plan_version = None
        self._plan_version_counter = 0

        self.current_traj_mode = "replace_future"
        self.route_reference_mode = self.stream_generator.condition_manager.route.mode.value
        self.traj_horizon_tokens = int(traj_mask_cfg.get("horizon_tokens", 20))
        self.traj_time_mode = str(traj_mask_cfg.get("time_mode", "timestamped"))
        self.waypoint_dt = float(traj_mask_cfg.get("waypoint_dt", 0.05))
        self.manual_duration_seconds = float(traj_mask_cfg.get("manual_duration_seconds", 5.0))
        self.manual_resample_arclength = bool(traj_mask_cfg.get("manual_resample_arclength", True))
        self.token_dt = float(traj_mask_cfg.get("token_dt", 0.20))
        self.traj_repeat_policy = str(traj_mask_cfg.get("repeat_policy", "translate_from_current_root"))
        self.traj_update_delay_enabled = bool(traj_mask_cfg.get("update_delay_enabled", True))
        self.traj_update_delay_tokens = int(traj_mask_cfg.get("update_delay_tokens", self.traj_horizon_tokens))
        self.traj_update_blend_enabled = bool(traj_mask_cfg.get("update_blend_enabled", True))
        self.traj_update_blend_tokens = int(traj_mask_cfg.get("update_blend_tokens", 4))
        self.trajectory_runtime_controls = TrajectoryRuntimeControls(
            route_mode=self.route_reference_mode,
            horizon_tokens=self.traj_horizon_tokens,
            delay_enabled=self.traj_update_delay_enabled,
            delay_tokens=self.traj_update_delay_tokens,
            blend_enabled=self.traj_update_blend_enabled,
            blend_tokens=self.traj_update_blend_tokens,
        )
        self.trajectory_controller = TrajectoryController(
            self.trajectory_runtime_controls
        )
        self.default_token_step = float(traj_mask_cfg.get("default_token_step", 0.25))
        self.min_token_step = float(traj_mask_cfg.get("min_token_step", 0.05))
        self.max_token_step = float(traj_mask_cfg.get("max_token_step", 1.50))
        self.root_xz_history = deque(maxlen=120)
        self.root_5d_history = deque(maxlen=480)
        self._generated_frame_count = 0
        self._absolute_commit_index = 0
        self.traj_repeat_anchor_root = None
        self.traj_repeat_anchor_cycle = None
        # Compatibility fields for app.py / status endpoints.
        self.current_traj_waypoints = None
        self.current_traj_times = None
        print(
            "Trajectory config: "
            f"time_mode={self.traj_time_mode}, "
            f"waypoint_dt={self.waypoint_dt:.3f}s, "
            f"manual_duration={self.manual_duration_seconds:.2f}s, "
            f"token_dt={self.token_dt:.3f}s, "
            f"horizon_tokens={self.traj_horizon_tokens}, "
            f"repeat_policy={self.traj_repeat_policy}, "
            f"update_delay={self.traj_update_delay_enabled}:{self.traj_update_delay_tokens}, "
            f"update_blend={self.traj_update_blend_enabled}:{self.traj_update_blend_tokens}, "
            f"root_feedback={self.root_feedback_enabled}:{self.root_feedback_xz_blend_alpha:.2f}"
        )
        
        # Model generation state
        self.first_chunk = True
        self.history_length = 30  # Default history window length
        self.denoise_steps = 10  # Default denoising steps

        # Trajectory display: world-space future token positions for frontend viz.
        self._display_traj_lock = threading.Lock()
        self._display_traj = None  # (T, 3) np.ndarray or None

        print("ModelManager initialized successfully")

    @staticmethod
    def _coerce_root_feedback_alpha(value) -> float:
        try:
            alpha = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"root_feedback_xz_blend_alpha must be a number, got {value!r}"
            ) from exc
        return float(np.clip(alpha, 0.0, 1.0))

    def _set_root_feedback_controls(
        self,
        *,
        enabled=None,
        xz_blend_alpha=None,
    ) -> None:
        if not hasattr(self, "root_feedback_enabled"):
            self.root_feedback_enabled = False
        if not hasattr(self, "root_feedback_xz_blend_alpha"):
            self.root_feedback_xz_blend_alpha = 0.5
        if enabled is not None:
            self.root_feedback_enabled = TrajectoryController._coerce_bool(
                enabled,
                default=bool(getattr(self, "root_feedback_enabled", False)),
                name="root_feedback_enabled",
            )
        if xz_blend_alpha is not None:
            self.root_feedback_xz_blend_alpha = self._coerce_root_feedback_alpha(
                xz_blend_alpha
            )

    def _root_feedback_target_from_payload(self, traj_input, decoded_chunk):
        if not bool(getattr(self, "root_feedback_enabled", False)):
            return None
        if not isinstance(traj_input, dict):
            return None
        cond = traj_input.get("traj_cond_7d_frame")
        if cond is None:
            return None
        if torch.is_tensor(cond):
            traj = cond[0] if cond.dim() == 3 else cond
        else:
            traj = torch.as_tensor(cond, dtype=torch.float32)
        if traj.dim() != 2 or traj.shape[-1] < 5 or int(traj.shape[0]) <= 0:
            return None

        abs_start_token = int(
            traj_input.get(
                "traj_abs_start_token",
                traj_input.get("body_anchor_abs_token", 0),
            )
        )
        abs_start_frame = token_start_frame(abs_start_token)
        local_start = max(0, int(self._generated_frame_count) - int(abs_start_frame))
        need = local_start + int(decoded_chunk.shape[0]) + 1
        target = traj.to(device=decoded_chunk.device, dtype=decoded_chunk.dtype)
        if int(target.shape[0]) < need:
            tail = target[-1:].expand(need - int(target.shape[0]), -1)
            target = torch.cat([target, tail], dim=0)
        target = target[local_start:need].clone()
        if int(target.shape[0]) < 2:
            return None

        alpha = float(getattr(self, "root_feedback_xz_blend_alpha", 1.0))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        if alpha < 1.0:
            dummy_tail = decoded_chunk.new_zeros((1, decoded_chunk.shape[-1]))
            generated_prefix = torch.cat([decoded_chunk, dummy_tail], dim=0)
            _, generated_xyz = recover_root_rot_pos(generated_prefix.unsqueeze(0))
            generated_xyz = generated_xyz[0, : int(target.shape[0])]
            target_xz = target[:, [0, 2]].clone()
            target_xz = target_xz - target_xz[:1] + generated_xyz[:1, [0, 2]]
            target[:, [0, 2]] = (
                (1.0 - alpha) * generated_xyz[:, [0, 2]]
                + alpha * target_xz
            )
        return target

    def _apply_root_feedback_to_latent(
        self,
        *,
        generated,
        decoded_raw,
        traj_input,
        local_commit_index: int,
    ):
        device = next(self.model.parameters()).device
        target = self._root_feedback_target_from_payload(traj_input, decoded_raw)
        if target is None:
            latent_token = generated[0].detach()
            decoded = _decode_latent_chunk(
                self.vae,
                latent_token,
                first_chunk=self.first_chunk,
                device=device,
            )
            return latent_token.detach().cpu(), decoded
        corrected = replace_root_channels_263_window_from_7d(
            decoded_raw,
            target,
            start_frame=0,
        )
        corrected_latent = _encode_corrected_chunk_token(
            self.vae,
            corrected,
            first_chunk=self.first_chunk,
            device=device,
        )
        _decode_latent_chunk(
            self.vae,
            corrected_latent,
            first_chunk=self.first_chunk,
            device=device,
        )
        _write_committed_latent_to_model(
            self.model,
            corrected_latent,
            int(local_commit_index),
        )
        return corrected_latent, corrected

    def _sample_waypoint_mask(self, waypoint_len: int) -> np.ndarray:
        """Sample traj_mask over user waypoints (length n), with keep ratio randomly sampled."""
        if not self.traj_mask_enabled:
            mask = np.ones((waypoint_len,), dtype=np.float32)
            self._last_traj_mask_keep = int(mask.sum().item())
            self._last_traj_mask_total = int(mask.shape[0])
            return mask

        if waypoint_len <= 0:
            return np.zeros((0,), dtype=np.float32)
        if waypoint_len == 1:
            return np.ones((1,), dtype=np.float32)

        keep_min = float(np.clip(self.traj_mask_keep_ratio_min, 0.0, 1.0))
        keep_max = float(np.clip(self.traj_mask_keep_ratio_max, 0.0, 1.0))
        if keep_min > keep_max:
            keep_min, keep_max = keep_max, keep_min

        keep_ratio = keep_min if keep_min == keep_max else float(self.traj_mask_rng.uniform(keep_min, keep_max))
        keep_n = int(np.round(waypoint_len * keep_ratio))
        keep_n = int(np.clip(keep_n, 1, waypoint_len))

        mask = np.zeros((waypoint_len,), dtype=np.float32)
        if self.traj_mask_keep_first_last and waypoint_len >= 2:
            # Always keep endpoints.
            keep_n_endpoints = 2
            if keep_n <= keep_n_endpoints:
                mask[0] = 1.0
                mask[waypoint_len - 1] = 1.0
            else:
                remaining = keep_n - keep_n_endpoints
                if remaining > 0 and waypoint_len > 2:
                    mid_idx = np.arange(1, waypoint_len - 1, dtype=np.int64)
                    chosen = self.traj_mask_rng.choice(
                        mid_idx,
                        size=min(remaining, len(mid_idx)),
                        replace=False,
                    )
                    keep_idx = np.sort(
                        np.concatenate(
                            [np.array([0, waypoint_len - 1], dtype=np.int64), chosen.astype(np.int64)]
                        )
                    )
                    mask[keep_idx] = 1.0
                else:
                    mask[0] = 1.0
                    mask[waypoint_len - 1] = 1.0
        else:
            chosen = self.traj_mask_rng.choice(
                np.arange(waypoint_len, dtype=np.int64),
                size=keep_n,
                replace=False,
            )
            mask[chosen] = 1.0

        self._last_traj_mask_keep = int(mask.sum().item())
        self._last_traj_mask_total = int(mask.shape[0])
        return mask
    
    def _load_models(self, config_path):
        return load_ldf_models(config_path, self.device)

    def _load_model_bundle(self, config_path, traj_mask_cfg):
        return load_model_bundle(
            config_path,
            traj_mask_cfg=traj_mask_cfg,
            device=self.device,
        )

    def _resolve_repo_path(self, path):
        return resolve_repo_path(path)

    def _load_stream_generator(self, config_path, traj_mask_cfg):
        return build_stream_generator(
            self.model,
            self.device,
            traj_mask_cfg=traj_mask_cfg,
            history_length=int(getattr(self, "history_length", 30)),
        )

    @staticmethod
    def _reject_normalized_root_refiner_config(cfg) -> None:
        return reject_normalized_root_refiner_config(cfg)
    
    def start_generation(self, text, history_length=None):
        """Start or update generation with new text.

        Clean sessions should call `reset()` first. This method intentionally
        does not clear route/rootplan state so debug presets can install a
        route before starting generation.
        """
        self.current_text = text
        
        if history_length is not None:
            self.history_length = history_length
        
        if not self.is_generating:
            self.generation_state = GenerationState.LOADING
            # Reset state before starting (only once at the beginning)
            self.frame_buffer.clear()
            self.stream_recovery.reset()
            self._reset_root_timeline()
            self.vae.clear_cache()
            self.first_chunk = True
            self.root_xz_history.clear()
            self.root_5d_history.clear()
            self._generated_frame_count = 0
            self._absolute_commit_index = 0
            self.stream_generator.init_ldf_generation(
                history_length=self.history_length,
                batch_size=1,
                num_denoise_steps=self.denoise_steps,
            )
            self.stream_generator.condition_manager.update_text(text, commit_idx=0)
            print(f"Model initialized with history length: {self.history_length}, denoise steps: {self.denoise_steps}")
            
            # Start generation thread
            self._generation_worker().start()
            self.is_generating = True
            self.generation_state = GenerationState.RUNNING
    
    def update_text(self, text):
        """Update text without resetting state (continuous generation with new text)"""
        if text != self.current_text:
            old_text = self.current_text
            self.current_text = text
            self.stream_generator.condition_manager.update_text(
                text,
                commit_idx=self._get_commit_index(),
            )
            # Don't reset first_chunk, stream_recovery, or vae cache
            # This allows continuous generation with text changes
            print(f"Text updated: '{old_text}' -> '{text}' (continuous generation)")

    @staticmethod
    def _path_length_xz(points_xyz: np.ndarray) -> float:
        points = np.asarray(points_xyz, dtype=np.float32)
        if len(points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(points[:, [0, 2]], axis=0), axis=1).sum())

    def _resample_uniform_arclength(self, points_xyz: np.ndarray, num_points: int) -> np.ndarray:
        points = dedupe_polyline(np.asarray(points_xyz, dtype=np.float32))
        if len(points) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if len(points) == 1 or num_points <= 1:
            return points[:1].astype(np.float32)
        total_len = self._path_length_xz(points)
        if total_len <= 1e-6:
            return np.repeat(points[:1].astype(np.float32), num_points, axis=0)
        return resample_polyline(
            points,
            num_tokens=int(num_points),
            token_step=total_len / float(num_points - 1),
        )

    def update_trajectory(
        self, waypoints, mode="replace_future", *, source="manual",
        duration_seconds=None, route_mode=None, horizon_tokens=None,
        delay_enabled=None, delay_tokens=None, blend_enabled=None,
        blend_tokens=None,
    ):
        """Update trajectory control with optional delayed blended replace.

        Does NOT immediately overwrite the active plan.  Instead creates a
        pending ``RouteUpdate`` that takes effect after
        ``update_delay_tokens`` tokens with a smooth blend transition.
        ``waypoints is None`` clears trajectory.
        """
        mode = mode or "replace_future"
        if mode != "replace_future":
            raise ValueError(f"Unsupported trajectory mode: {mode}")
        route_reference_mode = self._set_route_reference_mode(route_mode)
        (
            active_horizon_tokens,
            active_delay_enabled,
            active_delay_tokens,
            active_blend_enabled,
            active_blend_tokens,
        ) = self._apply_trajectory_runtime_controls(
            horizon_tokens=horizon_tokens,
            delay_enabled=delay_enabled,
            delay_tokens=delay_tokens,
            blend_enabled=blend_enabled,
            blend_tokens=blend_tokens,
        )

        # ── Clear ────────────────────────────────────────────────────
        if waypoints is None or len(waypoints) == 0:
            self._clear_runtime_route_state()
            print("Trajectory control cleared")
            return None

        raw = np.asarray(waypoints, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        # ── Timestamped (N,4) or spatial (N,2/3) ────────────────────
        explicit_times = None
        if raw.shape[1] == 4:
            explicit_times = raw[:, 0].astype(np.float32)
            raw = raw[:, 1:4]
        points = ensure_xyz(raw)
        current_root = self._get_current_root_xyz()
        edit_commit = self._get_commit_index()
        controller = self._trajectory_controller()
        _event, _prev_plan = controller.snapshot()
        delay = (
            active_delay_tokens
            if _prev_plan is not None and active_delay_enabled
            else 0
        )
        blend = (
            active_blend_tokens
            if _prev_plan is not None and active_blend_enabled
            else 0
        )
        effective_commit = edit_commit + delay

        if explicit_times is not None:
            times, points = self._prepare_timestamped_route_points(
                explicit_times,
                points,
                current_root=current_root,
                route_mode=route_reference_mode,
            )
        else:
            times, points = self._prepare_manual_route_points(
                points,
                current_root=current_root,
                duration_seconds=duration_seconds,
                route_mode=route_reference_mode,
            )

        new_plan = RoutePlan(
            times=times.astype(np.float32),
            points_xyz=points.astype(np.float32),
            start_commit_index=effective_commit,
            version=self._next_plan_version(),
            source=str(source),
        )
        update_event = self.stream_generator.condition_manager.route.update_route(
            new_plan,
            edit_commit_idx=edit_commit,
            delay_tokens=delay,
            blend_tokens=blend,
        )

        if _prev_plan is None:
            controller.set_active_route(
                new_plan,
                waypoints=points,
                times=times,
                mode=mode,
            )
            self._activate_root_plan_from_stream_plan(new_plan)
            self._trajectory_state = "active_7d"
            preview = sample_route_future(
                new_plan,
                current_commit=edit_commit,
                current_root_xyz=current_root,
                horizon_tokens=active_horizon_tokens,
                token_dt=self.token_dt,
                reanchor_to_current_root=(
                    route_reference_mode == RouteReferenceMode.RELATIVE_TO_ACTOR.value
                ),
            )
            controller.set_display(preview)
            print(
                f"Trajectory updated: {len(points)} points, source={source}, "
                f"horizon={active_horizon_tokens}, "
                f"edit_commit={edit_commit}, effective_commit={effective_commit}, "
                f"delay={delay}, blend=0"
            )
            return self.get_display_traj()

        controller.set_pending_update(
            update_event,
            waypoints=points,
            times=times,
            mode=mode,
        )

        print(
            f"Trajectory updated: {len(points)} points, source={source}, "
            f"horizon={active_horizon_tokens}, "
            f"edit_commit={edit_commit}, effective_commit={effective_commit}, "
            f"delay={delay}, blend={blend}"
        )
        return self.get_display_traj()

    def _apply_trajectory_runtime_controls(
        self,
        *,
        horizon_tokens=None,
        delay_enabled=None,
        delay_tokens=None,
        blend_enabled=None,
        blend_tokens=None,
    ) -> tuple[int, bool, int, bool, int]:
        controls = self._trajectory_controller().update_controls(
            route_mode=self.route_reference_mode,
            horizon_tokens=horizon_tokens,
            delay_enabled=delay_enabled,
            delay_tokens=delay_tokens,
            blend_enabled=blend_enabled,
            blend_tokens=blend_tokens,
        )
        self.trajectory_runtime_controls = controls
        self.traj_horizon_tokens = controls.horizon_tokens
        self.traj_update_delay_enabled = controls.delay_enabled
        self.traj_update_delay_tokens = controls.delay_tokens
        self.traj_update_blend_enabled = controls.blend_enabled
        self.traj_update_blend_tokens = controls.blend_tokens
        stream_generator = getattr(self, "stream_generator", None)
        if stream_generator is not None:
            stream_generator.traj_horizon_tokens = controls.horizon_tokens
        return (
            controls.horizon_tokens,
            controls.delay_enabled,
            controls.delay_tokens,
            controls.blend_enabled,
            controls.blend_tokens,
        )

    def _trajectory_controller(self) -> TrajectoryController:
        controller = getattr(self, "trajectory_controller", None)
        if controller is None:
            controls = getattr(self, "trajectory_runtime_controls", None)
            if controls is None:
                controls = TrajectoryRuntimeControls(
                    route_mode=getattr(self, "route_reference_mode", "relative_to_actor"),
                    horizon_tokens=getattr(self, "traj_horizon_tokens", 20),
                    delay_enabled=getattr(self, "traj_update_delay_enabled", True),
                    delay_tokens=getattr(self, "traj_update_delay_tokens", 20),
                    blend_enabled=getattr(self, "traj_update_blend_enabled", True),
                    blend_tokens=getattr(self, "traj_update_blend_tokens", 4),
                )
            controller = TrajectoryController(controls)
            self.trajectory_controller = controller
        return controller

    @property
    def traj_state_lock(self):
        return self._trajectory_controller().lock

    @traj_state_lock.setter
    def traj_state_lock(self, value):
        self._trajectory_controller().lock = value

    @property
    def active_traj_plan(self):
        return self._trajectory_controller().active_route

    @active_traj_plan.setter
    def active_traj_plan(self, value):
        self._trajectory_controller().active_route = value

    @property
    def pending_update_event(self):
        return self._trajectory_controller().pending_update

    @pending_update_event.setter
    def pending_update_event(self, value):
        self._trajectory_controller().pending_update = value

    @property
    def current_traj_waypoints(self):
        return self._trajectory_controller().current_waypoints

    @current_traj_waypoints.setter
    def current_traj_waypoints(self, value):
        self._trajectory_controller().current_waypoints = value

    @property
    def current_traj_times(self):
        return self._trajectory_controller().current_times

    @current_traj_times.setter
    def current_traj_times(self, value):
        self._trajectory_controller().current_times = value

    @property
    def current_traj_mode(self):
        return self._trajectory_controller().current_mode

    @current_traj_mode.setter
    def current_traj_mode(self, value):
        self._trajectory_controller().current_mode = value

    @property
    def _trajectory_state(self):
        return self._trajectory_controller().state

    @_trajectory_state.setter
    def _trajectory_state(self, value):
        self._trajectory_controller().state = value

    @property
    def _plan_version_counter(self):
        return self._trajectory_controller().plan_version_counter

    @_plan_version_counter.setter
    def _plan_version_counter(self, value):
        self._trajectory_controller().plan_version_counter = int(value)

    @property
    def _display_traj_lock(self):
        return self._trajectory_controller().display_lock

    @_display_traj_lock.setter
    def _display_traj_lock(self, value):
        self._trajectory_controller().display_lock = value

    @property
    def _display_traj(self):
        return self._trajectory_controller()._display_traj

    @_display_traj.setter
    def _display_traj(self, value):
        self._trajectory_controller().set_display(value)

    @property
    def _model_traj_plan_version(self):
        return self._rootplan_controller().model_plan_version

    @_model_traj_plan_version.setter
    def _model_traj_plan_version(self, value):
        self._rootplan_controller().model_plan_version = value

    def _get_current_root_xyz(self) -> np.ndarray:
        root_xyz = np.zeros(3, dtype=np.float32)
        timeline = getattr(self, "_root_timeline", None)
        if timeline is not None:
            root_xyz[[0, 2]] = (
                timeline.head.world_xz.detach().cpu().numpy().astype(np.float32)
            )
            recovery_root = getattr(
                getattr(self, "stream_recovery", None),
                "r_pos_accum",
                None,
            )
            if recovery_root is not None and len(recovery_root) > 1:
                root_xyz[1] = float(np.asarray(recovery_root, dtype=np.float32)[1])
            return root_xyz

        recovery_root = getattr(self.stream_recovery, "r_pos_accum", root_xyz)
        root_xyz[[0, 2]] = np.asarray(recovery_root, dtype=np.float32)[[0, 2]]
        return root_xyz

    def _set_route_reference_mode(self, route_mode=None) -> str:
        mode = RouteReferenceMode(
            route_mode or getattr(self, "route_reference_mode", "relative_to_actor")
        ).value
        self.route_reference_mode = mode
        self.stream_generator.condition_manager.set_route_mode(mode)
        return mode

    def _rootplan_controller(self) -> RootPlanController:
        controller = getattr(self, "rootplan_controller", None)
        if controller is None:
            controller = RootPlanController(self.stream_generator)
            self.rootplan_controller = controller
        return controller

    def _generation_worker(self) -> GenerationWorker:
        worker = getattr(self, "generation_worker", None)
        if worker is None:
            worker = GenerationWorker(self._generation_loop)
            self.generation_worker = worker
        return worker

    def _clear_runtime_route_state(self) -> None:
        self._trajectory_controller().clear()
        if getattr(self, "stream_generator", None) is not None:
            self._rootplan_controller().clear()
            self.stream_generator.condition_manager.route.clear()

    def _prepare_manual_route_points(
        self,
        points: np.ndarray,
        *,
        current_root: np.ndarray,
        duration_seconds,
        route_mode: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if route_mode == RouteReferenceMode.RELATIVE_TO_ACTOR.value:
            duration = (
                float(duration_seconds)
                if duration_seconds is not None
                else self.manual_duration_seconds
            )
            return normalize_manual_waypoints(
                points,
                current_root_xyz=current_root,
                waypoint_dt=self.waypoint_dt,
                manual_duration_seconds=duration,
                resample_arclength=self.manual_resample_arclength,
            )

        out_points = np.asarray(points, dtype=np.float32)
        if self.manual_resample_arclength and len(out_points) >= 2:
            duration = (
                float(duration_seconds)
                if duration_seconds is not None
                else self.manual_duration_seconds
            )
            num_points = max(2, int(duration / float(self.waypoint_dt)) + 1)
            out_points = resample_polyline_by_arclength(out_points, num_points)
        times = assign_uniform_timestamps(len(out_points), self.waypoint_dt)
        return times.astype(np.float32), out_points.astype(np.float32)

    @staticmethod
    def _prepare_timestamped_route_points(
        times: np.ndarray,
        points: np.ndarray,
        *,
        current_root: np.ndarray,
        route_mode: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        out_times = np.asarray(times, dtype=np.float32)
        out_times = out_times - out_times[0] if len(out_times) > 0 else out_times
        out_points = np.asarray(points, dtype=np.float32)
        if route_mode == RouteReferenceMode.RELATIVE_TO_ACTOR.value:
            out_points = translate_plan_to_current_root(out_points, current_root)
        return out_times.astype(np.float32), out_points.astype(np.float32)

    def _estimate_token_step_distance(self) -> float:
        """Thin wrapper around runtime geometry distance estimation."""
        return estimate_token_step_distance(
            list(self.root_xz_history),
            default=self.default_token_step,
            min_step=self.min_token_step,
            max_step=self.max_token_step,
        )

    @staticmethod
    def _project_point_to_polyline(point_xyz: np.ndarray, waypoints_xyz: np.ndarray):
        """Thin wrapper around runtime geometry projection."""
        return project_point_to_polyline(point_xyz, waypoints_xyz)

    @staticmethod
    def _dedupe_polyline(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Thin wrapper around runtime geometry dedupe."""
        return dedupe_polyline(points, eps)

    def _build_remaining_polyline(self, root_xyz: np.ndarray, waypoints_xyz: np.ndarray) -> np.ndarray:
        """Thin wrapper around runtime geometry path trimming."""
        return build_remaining_polyline(root_xyz, waypoints_xyz)

    @staticmethod
    def _resample_polyline(points_xyz: np.ndarray, num_tokens: int, token_step: float) -> np.ndarray:
        """Thin wrapper around runtime geometry path resampling."""
        return resample_polyline(points_xyz, num_tokens, token_step)

    def _sample_timestamped_with_repeat(
        self,
        traj_times: np.ndarray,
        waypoints: np.ndarray,
        query_times: np.ndarray,
    ) -> np.ndarray:
        """Sample timestamped waypoints, optionally as a rolling local template.

        `translate_from_current_root` treats the user/debug trajectory as a
        timed local motion template.  At every streaming step, the first queried
        template phase is aligned to the current generated root, and the future
        horizon is expressed as relative displacement from that phase.

        This keeps repeated plans under the character instead of leaving them in
        the original world location. The returned trajectory remains world-space;
        RootPlan conversion handles body-window anchoring later.
        """
        times = np.asarray(traj_times, dtype=np.float32).reshape(-1)
        points = np.asarray(waypoints, dtype=np.float32)
        queries = np.asarray(query_times, dtype=np.float32).reshape(-1)
        if len(times) < 2 or len(points) < 2 or len(queries) == 0:
            return sample_timestamped_trajectory(times, points, queries)

        start_t = float(times[0])
        end_t = float(times[-1])
        duration = end_t - start_t

        def sample_unwrapped(query_values: np.ndarray) -> np.ndarray:
            query_values = np.asarray(query_values, dtype=np.float32).reshape(-1)
            cycle = np.floor((query_values - start_t) / duration).astype(np.int64)
            cycle = np.maximum(cycle, 0)
            local_t = ((query_values - start_t) % duration) + start_t
            local = sample_timestamped_trajectory(times, points, local_t)
            return local + cycle[:, None].astype(np.float32) * (points[-1] - points[0])

        current_root = self._get_current_root_xyz().astype(np.float32)

        if self.traj_repeat_policy != "translate_from_current_root":
            # Align first queried position to current root but WITHOUT cycle
            # unwrapping — the plan ends at its natural endpoint.
            result = sample_timestamped_trajectory(times, points, queries)
            anchor = sample_timestamped_trajectory(
                times, points,
                np.asarray([queries[0]], dtype=np.float32),
            )[0]
            return (current_root + (result - anchor)).astype(np.float32)

        # translate_from_current_root: same root alignment plus cycle repeat.
        unwrapped = sample_unwrapped(queries)
        anchor = sample_unwrapped(np.asarray([queries[0]], dtype=np.float32))[0]
        self.traj_repeat_anchor_root = current_root.copy()
        self.traj_repeat_anchor_cycle = int(
            max(0, np.floor((float(queries[0]) - start_t) / duration))
        )
        return (current_root + (unwrapped - anchor)).astype(np.float32)

    def _next_plan_version(self) -> int:
        return self._trajectory_controller().next_plan_version()

    def _get_commit_index(self) -> int:
        timeline = getattr(self, "_root_timeline", None)
        if timeline is not None:
            return int(timeline.head.commit_idx)
        return int(getattr(self, "_absolute_commit_index", getattr(self.model, "commit_index", 0)))

    def _get_root_refiner_history_5d(self, anchor_commit: int):
        history = getattr(self, "root_5d_history", None)
        if not history:
            return None
        anchor_frame = token_start_frame(max(0, int(anchor_commit)))
        frames = [
            np.asarray(root5d, dtype=np.float32)
            for frame_idx, root5d in history
            if int(frame_idx) <= anchor_frame
        ]
        if not frames:
            return None
        return np.stack(frames, axis=0).astype(np.float32)

    def _reset_root_timeline(self):
        self._session_anchor_state = RootFrameState.initial(
            device=self.device,
            dtype=torch.float32,
        )
        self._root_timeline = RootTimeline(self._session_anchor_state)
        if getattr(self, "stream_generator", None) is not None:
            self.stream_generator.timeline = self._root_timeline

    def _append_root_state_from_stream_recovery(self, *, frame_idx: int) -> bool:
        timeline = getattr(self, "_root_timeline", None)
        if timeline is None:
            self._reset_root_timeline()
            timeline = self._root_timeline

        appended = append_timeline_state_at_token_start_frame(
            timeline,
            frame_idx=frame_idx,
            recovery=self.stream_recovery,
            session_anchor_state=getattr(self, "_session_anchor_state", None),
            source="stream_recovery",
        )
        if appended:
            history_length = int(getattr(self, "history_length", 0))
            if history_length > 0:
                keep_from = max(0, timeline.head.commit_idx - history_length * 2)
                timeline.trim_before(keep_from)
        return appended

    def _stream_plan_to_root_plan(
        self,
        plan: RoutePlan,
        anchor_state: RootFrameState,
    ) -> RootPlan:
        """Convert a world-space RoutePlan to plan-anchor-local 7D."""
        from utils.local_frame import canonicalize_7d

        device = torch.device(getattr(self, "device", "cpu"))
        frames_per_token = 4
        token_dt = float(getattr(self, "token_dt", 0.20))
        frame_dt = token_dt / float(frames_per_token)
        min_tokens = (
            int(getattr(self, "history_length", 0))
            + int(getattr(getattr(self, "model", None), "chunk_size", 1))
            + int(getattr(self, "traj_horizon_tokens", 0))
        )
        duration = 0.0
        if len(plan.times) > 0:
            duration = max(0.0, float(np.max(plan.times) - np.min(plan.times)))
        duration_tokens = int(np.ceil(duration / max(token_dt, 1e-6))) + 1
        num_tokens_pred = max(1, min_tokens, duration_tokens)
        valid_frames = num_frames_for_tokens(num_tokens_pred, frames_per_token)

        query_times = np.arange(valid_frames, dtype=np.float32) * np.float32(frame_dt)
        xyz = sample_timestamped_trajectory(plan.times, plan.points_xyz, query_times)
        xyz_t = torch.as_tensor(xyz, device=device, dtype=torch.float32)

        xz = xyz_t[:, [0, 2]]
        delta = torch.zeros_like(xz)
        if xz.shape[0] > 1:
            delta[0] = xz[1] - xz[0]
            delta[1:] = xz[1:] - xz[:-1]
        yaw_values = []
        last_yaw = torch.tensor(0.0, device=device, dtype=torch.float32)
        for d in delta:
            if torch.linalg.norm(d) > 1e-6:
                last_yaw = torch.atan2(d[0], d[1])
            yaw_values.append(last_yaw)
        yaw = torch.stack(yaw_values) if yaw_values else xyz_t.new_zeros(0)
        traj_5d_world = torch.cat(
            [xyz_t, torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)],
            dim=-1,
        )
        traj_7d_world = append_traj_deltas_5d_to_7d(traj_5d_world)

        anchor_xz = anchor_state.world_xz.to(device=device, dtype=torch.float32)
        anchor_yaw = anchor_state.world_yaw.to(device=device, dtype=torch.float32)
        traj_7d_local = canonicalize_7d(traj_7d_world, anchor_xz, anchor_yaw)
        return RootPlan(
            num_tokens_pred=num_tokens_pred,
            valid_frames=valid_frames,
            waypoints_local_7d=traj_7d_local,
            frame_dt=frame_dt,
            frames_per_token=frames_per_token,
            anchor_commit_idx=int(anchor_state.commit_idx),
            anchor_world_xz=anchor_xz,
            anchor_world_yaw=anchor_yaw,
            source=str(plan.source),
        )

    def _build_root_plan_from_stream_plan(
        self,
        plan: RoutePlan,
        anchor_state: RootFrameState,
    ) -> RootPlan:
        if (
            getattr(self, "stream_generator", None) is not None
            and self.stream_generator.root_refiner is not None
        ):
            return self.stream_generator.build_root_plan(
                text=getattr(self, "current_text", ""),
                route=plan,
                anchor_state=anchor_state,
                history_motion_world_5d=self._get_root_refiner_history_5d(
                    anchor_state.commit_idx
                ),
            )

        root_plan_input = plan
        route_mode = getattr(
            self,
            "route_reference_mode",
            RouteReferenceMode.RELATIVE_TO_ACTOR.value,
        )
        if route_mode == RouteReferenceMode.RELATIVE_TO_ACTOR.value:
            root_plan_input = reanchor_route_to_xz(
                plan,
                anchor_state.world_xz.detach().cpu().numpy(),
            )
        return self._stream_plan_to_root_plan(root_plan_input, anchor_state)

    def _activate_root_plan_from_stream_plan(self, plan: RoutePlan) -> bool:
        timeline = getattr(self, "_root_timeline", None)
        if timeline is None:
            return False
        anchor_commit = int(plan.start_commit_index)
        if not timeline.has_exact_state(anchor_commit):
            return False
        anchor_state = timeline.at_commit(anchor_commit)
        root_plan = self._build_root_plan_from_stream_plan(plan, anchor_state)
        root_source = RootSourceProposal.from_root_plan(
            root_plan,
            name=f"{plan.source}:root_source",
            source_kind=str(plan.source),
            metadata={"route_plan_version": int(plan.version)},
        )
        self._rootplan_controller().set_active_source(
            root_source,
            contract="absolute_route",
            model_plan_version=int(plan.version),
        )
        self.stream_generator.timeline = timeline
        return True

    def _build_temporary_rootplan_payload(
        self,
        plan: RoutePlan,
        *,
        model_traj_plan_version,
    ):
        timeline = getattr(self, "_root_timeline", None)
        if timeline is None:
            return None
        anchor_commit = int(plan.start_commit_index)
        if not timeline.has_exact_state(anchor_commit):
            return None
        anchor_state = timeline.at_commit(anchor_commit)
        root_plan = self._build_root_plan_from_stream_plan(plan, anchor_state)
        root_source = RootSourceProposal.from_root_plan(
            root_plan,
            name=f"{plan.source}:temporary_root_source",
            source_kind=str(plan.source),
            metadata={"route_plan_version": int(plan.version)},
        )
        with self._rootplan_controller().temporarily_active_source(
            root_source,
            contract="absolute_route",
            model_plan_version=model_traj_plan_version,
        ):
            return self._build_rootplan_stream_traj_input()

    def _build_rootplan_stream_traj_input(self):
        """Build direct 7D stream payload from active RootPlan, if available.

        This is the RootPlan/7D main-path bridge. It produces a payload that
        covers the largest latent window stream_generate_step may use during the
        current one-token commit. The model-side direct 7D helper then slices it
        per denoise sub-step.
        """
        timeline = getattr(self, "_root_timeline", None)
        if (
            timeline is None
            or (
                self._rootplan_controller().active_plan is None
                and self._rootplan_controller().active_source is None
            )
        ):
            return None

        absolute_commit = self._get_commit_index()
        local_commit = int(getattr(self.model, "commit_index", absolute_commit))
        self.stream_generator.timeline = timeline
        return self.stream_generator.build_root_plan_stream_payload(
            local_commit_index=local_commit,
            absolute_commit_index=absolute_commit,
        )

    def _build_stream_traj_input(self):
        current_commit = self._get_commit_index()
        current_root = self._get_current_root_xyz()

        event, plan = self._trajectory_controller().snapshot()

        if plan is None and event is None:
            rootplan_payload = self._build_rootplan_stream_traj_input()
            if rootplan_payload is not None:
                self._trajectory_state = "active_7d"
                return rootplan_payload
            self._trajectory_state = "none"
            return None

        # ── No pending update: sample from active plan ──────────────────
        if event is None:
            if (
                plan is not None
                and self._rootplan_controller().active_plan is None
                and self._rootplan_controller().active_source is None
            ):
                self._activate_root_plan_from_stream_plan(plan)
            future = sample_route_future(
                plan,
                current_commit=current_commit,
                current_root_xyz=current_root,
                horizon_tokens=self.traj_horizon_tokens,
                token_dt=self.token_dt,
                reanchor_to_current_root=(
                    self.route_reference_mode == RouteReferenceMode.RELATIVE_TO_ACTOR.value
                ),
            )
            self._trajectory_state = "active"
            self._trajectory_controller().set_display(future)
            rootplan_payload = self._build_rootplan_stream_traj_input()
            if rootplan_payload is not None:
                self._trajectory_state = "active_7d"
                rootplan_payload.update({
                    "traj_mode": self.current_traj_mode,
                    "traj_plan_version": plan.version if plan else 0,
                    "model_traj_plan_version": self._model_traj_plan_version,
                })
                return rootplan_payload
            self._trajectory_state = "active_7d_unavailable"
            return None

        # ── Pending update: compute blend weight ────────────────────────
        offset = current_commit - event.edit_commit_index
        delay = event.delay_tokens
        blend = event.blend_tokens
        raw_w = 0.0
        if offset >= delay + blend:
            raw_w = 1.0
        elif offset >= delay and blend > 0:
            raw_w = smoothstep01((offset - delay) / blend)
        w = float(raw_w)

        # Sample old and new futures using plan-local time.
        old_future = None
        if event.old_route is not None:
            old_future = sample_route_future(
                event.old_route,
                current_commit=current_commit,
                current_root_xyz=current_root,
                horizon_tokens=self.traj_horizon_tokens,
                token_dt=self.token_dt,
                reanchor_to_current_root=(
                    self.route_reference_mode == RouteReferenceMode.RELATIVE_TO_ACTOR.value
                ),
            )
        new_future = sample_route_future(
            event.new_route,
            current_commit=current_commit,
            current_root_xyz=current_root,
            horizon_tokens=self.traj_horizon_tokens,
            token_dt=self.token_dt,
            reanchor_to_current_root=(
                self.route_reference_mode == RouteReferenceMode.RELATIVE_TO_ACTOR.value
            ),
        )

        future = blend_future_trajs(
            old_future if old_future is not None else new_future,
            new_future, w,
        )

        # ── Transition management ───────────────────────────────────────
        rootplan_payload = None
        model_plan_version = getattr(self, "_model_traj_plan_version", None)
        if w <= 0.0:
            self._trajectory_state = "delay"
            if (
                event.old_route is not None
                and self._rootplan_controller().active_plan is None
                and self._rootplan_controller().active_source is None
            ):
                self._activate_root_plan_from_stream_plan(event.old_route)
            rootplan_payload = self._build_rootplan_stream_traj_input()
            model_plan_version = getattr(self, "_model_traj_plan_version", None)
        elif w < 1.0:
            self._trajectory_state = "blend"
            blend_route = RoutePlan(
                times=(
                    np.arange(len(future), dtype=np.float32)
                    * np.float32(self.token_dt)
                ),
                points_xyz=future.astype(np.float32),
                start_commit_index=int(current_commit),
                version=int(event.version),
                source="blend",
            )
            old_version = (
                int(event.old_route.version)
                if event.old_route is not None
                else int(event.version)
            )
            model_plan_version = f"blend:{old_version}->{int(event.version)}"
            rootplan_payload = self._build_temporary_rootplan_payload(
                blend_route,
                model_traj_plan_version=model_plan_version,
            )
        else:
            self._trajectory_state = "replaced"
            self._trajectory_controller().replace_with_pending(event)
            self._rootplan_controller().clear()
            self._activate_root_plan_from_stream_plan(event.new_route)
            rootplan_payload = self._build_rootplan_stream_traj_input()
            model_plan_version = getattr(self, "_model_traj_plan_version", None)

        self._trajectory_controller().set_display(future)
        if rootplan_payload is not None:
            rootplan_payload.update({
                "traj_mode": self.current_traj_mode,
                "traj_plan_version": event.version,
                "model_traj_plan_version": model_plan_version,
                "traj_update_blend_weight": w,
                "trajectory_state": self._trajectory_state,
            })
            return rootplan_payload
        self._trajectory_state = f"{self._trajectory_state}_7d_unavailable"
        return None
    
    def pause_generation(self, *, target_state=GenerationState.PAUSED):
        """Pause generation (keeps all state)"""
        worker = self._generation_worker()
        if worker.is_running and not worker.stop(timeout=5.0):
            print("Warning: generation thread did not stop within timeout; model state may be unsafe")
            self.generation_state = GenerationState.ERROR
            return False
        self.is_generating = False
        self.generation_state = target_state
        print("Generation paused (state preserved)")
        return True
    
    def resume_generation(self):
        """Resume generation from paused state"""
        if self.is_generating:
            print("Already generating, ignoring resume")
            return
        
        # Restart generation thread with existing state
        self._generation_worker().start()
        self.is_generating = True
        self.generation_state = GenerationState.RUNNING
        print("Generation resumed")
    
    def reset(
        self,
        history_length=None,
        smoothing_alpha=None,
        denoise_steps=None,
        root_feedback_enabled=None,
        root_feedback_xz_blend_alpha=None,
    ):
        """Reset generation state completely
        
        Args:
            history_length: History window length for the model
            smoothing_alpha: EMA smoothing factor (0.0 to 1.0)
                - 1.0 = no smoothing (default)
                - 0.0 = infinite smoothing
                - Recommended: 0.3-0.7 for visible smoothing
            denoise_steps: Number of denoising steps (1-50, default 10)
            root_feedback_enabled: Whether decoded root is blended toward the
                active 7D condition and re-encoded into streaming history.
            root_feedback_xz_blend_alpha: XZ blend strength in [0, 1].
        """
        self.generation_state = GenerationState.RESETTING
        # Stop if running, then poll until thread truly exits (max 10s total)
        if self.is_generating:
            if not self.pause_generation(target_state=GenerationState.RESETTING):
                self.generation_state = GenerationState.ERROR
                return False
        worker = self._generation_worker()
        if worker.is_running:
            self.reset_pending = True
            print("Reset pending — waiting for generation thread to finish...")
            if not worker.stop(timeout=10.0):
                print("Reset failed: generation thread still running after 15s timeout")
                self.reset_pending = False
                self.generation_state = GenerationState.ERROR
                return False
        self.reset_pending = False

        # Clear everything
        self.frame_buffer.clear()
        self.vae.clear_cache()
        self.first_chunk = True
        self.root_xz_history.clear()
        self.root_5d_history.clear()
        self._generated_frame_count = 0
        self._absolute_commit_index = 0
        self._clear_runtime_route_state()
        
        if history_length is not None:
            self.history_length = history_length
        
        if denoise_steps is not None:
            # Ensure denoise_steps is multiple of chunk_size (5)
            chunk_size = 5
            denoise_steps = np.clip(denoise_steps, chunk_size, 50)
            # Round to nearest multiple of chunk_size
            self.denoise_steps = int(np.round(denoise_steps / chunk_size) * chunk_size)
            print(f"Denoising steps updated to: {self.denoise_steps} (must be multiple of {chunk_size})")
        
        # Update smoothing alpha if provided and recreate stream recovery
        if smoothing_alpha is not None:
            self.smoothing_alpha = np.clip(smoothing_alpha, 0.0, 1.0)
            print(f"Smoothing alpha updated to: {self.smoothing_alpha}")

        self._set_root_feedback_controls(
            enabled=root_feedback_enabled,
            xz_blend_alpha=root_feedback_xz_blend_alpha,
        )
        print(
            "Root feedback controls: "
            f"enabled={self.root_feedback_enabled}, "
            f"xz_alpha={self.root_feedback_xz_blend_alpha:.2f}"
        )
        
        # Recreate stream recovery with new smoothing alpha
        self.stream_recovery = StreamJointRecovery263(
            joints_num=22, 
            smoothing_alpha=self.smoothing_alpha
        )
        self._reset_root_timeline()
        
        # Initialize model with denoise steps
        self.stream_generator.init_ldf_generation(
            history_length=self.history_length,
            batch_size=1,
            num_denoise_steps=self.denoise_steps,
        )
        self.generation_state = GenerationState.IDLE
        print(
            f"Model reset - history: {self.history_length}, "
            f"smoothing: {self.smoothing_alpha}, steps: {self.denoise_steps}, "
            f"root_feedback: {self.root_feedback_enabled}:"
            f"{self.root_feedback_xz_blend_alpha:.2f}"
        )
        return True
    
    def _generation_loop(self, stop_event=None):
        """Background loop: each iteration produces one latent token (→ 4 motion frames).

        When trajectory control is active, each step passes a future token-horizon in
        world coordinates. The model-side streaming path then rewrites only the future
        conditioning slots before trajectory encoding.
        """
        print("Generation loop started")
        
        import time
        step_count = 0
        total_gen_time = 0
        
        with torch.no_grad():
            while not (stop_event is not None and stop_event.is_set()):
                # Check if buffer needs more frames
                if self.frame_buffer.needs_generation():
                    try:
                        step_start = time.time()
                        
                        # Generate one token (produces 4 frames from VAE)
                        traj_input = self._build_stream_traj_input()
                        x = self.stream_generator.build_step_input(
                            self.current_text, traj_input=traj_input
                        )
                        condition_provider = self.stream_generator.build_ldf_condition_provider(
                            x,
                            first_chunk=self.first_chunk,
                            device=next(self.model.parameters()).device,
                        )
                        
                        # Generate from model (1 token)
                        # Note: denoise_steps is set in init_generated, not here
                        local_commit_index = int(
                            getattr(
                                self.model,
                                "commit_index",
                                getattr(self, "_absolute_commit_index", 0),
                            )
                        )
                        output = self.model.stream_generate_step(
                            x,
                            first_chunk=self.first_chunk,
                            condition=condition_provider,
                        )
                        generated = output["generated"]
                        self._absolute_commit_index = (
                            int(getattr(self, "_absolute_commit_index", 0))
                            + int(generated.shape[1])
                        )
                        
                        latent_token = generated[0].detach()
                        if bool(getattr(self, "root_feedback_enabled", False)):
                            decoded_raw = _decode_raw_chunk_preserving_feedback_cache(
                                self.vae,
                                latent_token,
                                first_chunk=self.first_chunk,
                                device=next(self.model.parameters()).device,
                            )
                            latent_token, decoded = self._apply_root_feedback_to_latent(
                                generated=generated,
                                decoded_raw=decoded_raw,
                                traj_input=traj_input,
                                local_commit_index=local_commit_index,
                            )
                        else:
                            # Decode with VAE (1 token -> 4 frames)
                            decoded = self.vae.stream_decode(
                                latent_token.to(next(self.model.parameters()).device)[None, :],
                                first_chunk=self.first_chunk,
                            )[0].float().detach().cpu()
                        
                        self.first_chunk = False
                        
                        # Convert each frame to joints
                        for i in range(decoded.shape[0]):
                            frame_data = decoded[i].cpu().numpy()
                            joints = self.stream_recovery.process_frame(frame_data)
                            self.root_xz_history.append(
                                self.stream_recovery.r_pos_accum[[0, 2]].astype(np.float32).copy()
                            )
                            root = self.stream_recovery.r_pos_accum.astype(np.float32).copy()
                            yaw = -2.0 * float(self.stream_recovery.r_rot_ang_accum)
                            root5d = np.array(
                                [root[0], root[1], root[2], np.cos(yaw), np.sin(yaw)],
                                dtype=np.float32,
                            )
                            self.root_5d_history.append((int(self._generated_frame_count), root5d))
                            self._append_root_state_from_stream_recovery(
                                frame_idx=int(self._generated_frame_count)
                            )
                            self._generated_frame_count += 1
                            self.frame_buffer.add_frame(joints)
                        
                        step_time = time.time() - step_start
                        total_gen_time += step_time
                        step_count += 1
                        
                        # Print performance stats every 10 steps
                        if step_count % 10 == 0:
                            avg_time = total_gen_time / step_count
                            fps = decoded.shape[0] / avg_time
                            print(f"[Generation] Step {step_count}: {step_time*1000:.1f}ms, "
                                  f"Avg: {avg_time*1000:.1f}ms, "
                                  f"FPS: {fps:.1f}, "
                                  f"Buffer: {self.frame_buffer.size()}")
                        
                    except Exception as e:
                        print(f"Error in generation: {e}")
                        import traceback
                        traceback.print_exc()
                        time.sleep(0.1)
                else:
                    # Buffer is full, wait a bit
                    time.sleep(0.01)
        
        print("Generation loop stopped")
    
    def get_display_traj(self):
        """Return a copy of the latest world-space trajectory for frontend viz, or None."""
        return self._trajectory_controller().get_display()

    def get_next_frame(self):
        """Get the next frame from buffer and optional trajectory display data."""
        joints = self.frame_buffer.get_frame()
        traj = self.get_display_traj()
        return joints, traj
    
    def get_buffer_status(self):
        """Get buffer status plus trajectory state and update metadata."""
        ev, plan = self._trajectory_controller().snapshot()
        status = {
            "buffer_size": self.frame_buffer.size(),
            "target_size": self.frame_buffer.target_size,
            "is_generating": self.is_generating,
            "generation_state": self.generation_state.value,
            "current_text": self.current_text,
            "trajectory_state": self._trajectory_state,
            "trajectory_active": plan is not None,
            "trajectory_time_mode": self.traj_time_mode,
            "model_traj_plan_version": self._model_traj_plan_version,
            "smoothing_alpha": self.smoothing_alpha,
            "denoise_steps": self.denoise_steps,
            "root_feedback_enabled": bool(
                getattr(self, "root_feedback_enabled", False)
            ),
            "root_feedback_xz_blend_alpha": float(
                getattr(self, "root_feedback_xz_blend_alpha", 0.0)
            ),
        }
        controls = getattr(self, "trajectory_runtime_controls", None)
        if controls is not None:
            status.update(controls.to_status_dict())
        else:
            status.update({
                "trajectory_route_mode": self.route_reference_mode,
                "trajectory_horizon_tokens": self.traj_horizon_tokens,
                "trajectory_delay_enabled": self.traj_update_delay_enabled,
                "trajectory_delay_tokens": self.traj_update_delay_tokens,
                "trajectory_blend_enabled": self.traj_update_blend_enabled,
                "trajectory_blend_tokens": self.traj_update_blend_tokens,
            })
        if plan is not None:
            status["active_plan_version"] = plan.version
            status["active_plan_source"] = plan.source
        preview = self.get_display_traj()
        if preview is not None:
            status["model_used_traj_preview"] = preview[:min(20, len(preview))].tolist()
        if ev is not None:
            status["pending_plan_version"] = ev.version
            status["edit_commit_index"] = ev.edit_commit_index
            status["effective_commit_index"] = ev.effective_commit_index
            status["update_delay_tokens"] = ev.delay_tokens
            status["update_blend_tokens"] = ev.blend_tokens
            status["update_blend_weight"] = round(
                smoothstep01(
                    (self._get_commit_index() - ev.edit_commit_index - ev.delay_tokens)
                    / max(ev.blend_tokens, 1)
                ), 4,
            ) if self._trajectory_state == "blend" else 0.0
        return status


# Global model manager instance
_model_manager = None
_traj_mask_cfg = None
_model_manager_lock = threading.Lock()


def get_model_manager(config_path=None, traj_mask_cfg=None):
    """Get or create the global model manager instance"""
    global _model_manager, _traj_mask_cfg
    if _model_manager is None:
        with _model_manager_lock:
            if _model_manager is None:
                _traj_mask_cfg = traj_mask_cfg or {}
                _model_manager = ModelManager(config_path, traj_mask_cfg=_traj_mask_cfg)
    return _model_manager
