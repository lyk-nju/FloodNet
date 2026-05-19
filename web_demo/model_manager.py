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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch_ema import ExponentialMovingAverage
from utils.initialize import instantiate, load_config
from utils.motion_process import StreamJointRecovery263
from utils.stream_rollout import build_stream_step_model_input
from utils.stream_traj import (
    build_remaining_polyline,
    dedupe_polyline,
    estimate_token_step_distance,
    project_point_to_polyline,
    resample_polyline,
    sample_timestamped_trajectory,
)


class FrameBuffer:
    """
    Thread-safe frame buffer that maintains a queue of generated frames
    """
    def __init__(self, target_buffer_size=4):
        self.buffer = deque(maxlen=100)  # Max 100 frames in buffer
        self.target_size = target_buffer_size
        self.lock = threading.Lock()
        
    def add_frame(self, joints):
        """Add a frame to the buffer"""
        with self.lock:
            self.buffer.append(joints)
    
    def get_frame(self):
        """Get the next frame from buffer"""
        with self.lock:
            if len(self.buffer) > 0:
                return self.buffer.popleft()
            return None
    
    def size(self):
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
    
    def needs_generation(self):
        """Check if buffer needs more frames"""
        return self.size() < self.target_size


class ModelManager:
    """
    Manages model loading and real-time frame generation.
    Trajectory control is active when the user provides waypoints and the model config enables
    trajectory conditioning via ControlNet branch (enabled when model.freeze_backbone=True).
    """
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
        
        # Load models
        self.vae, self.model = self._load_models(config_path)
        
        # Frame buffer
        self.frame_buffer = FrameBuffer(target_buffer_size=4)
        
        # Stream joint recovery with smoothing
        self.smoothing_alpha = 0.5  # Default: medium smoothing
        self.stream_recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=self.smoothing_alpha)
        
        # Generation state
        self.current_text = ""
        self.is_generating = False
        self.generation_thread = None
        self.should_stop = False
        self.reset_pending = False  # True while waiting for thread to stop before reset
        
        # Trajectory control.
        # Public/demo semantics stay in world-space xz coordinates. For streaming inference,
        # we resample a future token-horizon from the character's current root position and
        # let the model-side streaming path normalize the full visible window back to a
        # clip-local origin before encoding.
        self.traj_state_lock = threading.Lock()
        self.current_traj_waypoints = None
        self.current_traj_array = None         # Latest world-space waypoint polyline, shape (N, 3)
        self.current_traj_mode = "replace_future"
        self.traj_plan_version = 0
        self.traj_horizon_tokens = int(traj_mask_cfg.get("horizon_tokens", 20))
        self.traj_time_mode = str(traj_mask_cfg.get("time_mode", "timestamped"))
        self.waypoint_dt = float(traj_mask_cfg.get("waypoint_dt", 0.05))
        self.manual_duration_seconds = float(traj_mask_cfg.get("manual_duration_seconds", 3.0))
        self.manual_resample_arclength = bool(traj_mask_cfg.get("manual_resample_arclength", True))
        self.token_dt = float(traj_mask_cfg.get("token_dt", 0.20))
        self.traj_repeat_policy = str(traj_mask_cfg.get("repeat_policy", "hold"))
        self.default_token_step = float(traj_mask_cfg.get("default_token_step", 0.25))
        self.min_token_step = float(traj_mask_cfg.get("min_token_step", 0.05))
        self.max_token_step = float(traj_mask_cfg.get("max_token_step", 1.50))
        self.current_traj_times = None
        self.traj_plan_start_commit_index = 0
        self.traj_repeat_anchor_root = None
        self.traj_repeat_anchor_cycle = None
        self.root_xz_history = deque(maxlen=120)
        print(
            "Trajectory config: "
            f"time_mode={self.traj_time_mode}, "
            f"waypoint_dt={self.waypoint_dt:.3f}s, "
            f"manual_duration={self.manual_duration_seconds:.2f}s, "
            f"token_dt={self.token_dt:.3f}s, "
            f"horizon_tokens={self.traj_horizon_tokens}, "
            f"repeat_policy={self.traj_repeat_policy}"
        )
        
        # Model generation state
        self.first_chunk = True
        self.history_length = 30  # Default history window length
        self.denoise_steps = 10  # Default denoising steps

        # Trajectory display: world-space future token positions for frontend viz.
        self._display_traj_lock = threading.Lock()
        self._display_traj = None  # (T, 3) np.ndarray or None

        print("ModelManager initialized successfully")

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
        """Load VAE and diffusion models"""
        torch.set_float32_matmul_precision("high")
        
        # Change to parent directory to load config properly
        original_dir = os.getcwd()
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        os.chdir(parent_dir)
        
        try:
            # Load config (same as generate_ldf.py)
            cfg = load_config(config_path=config_path)
            
            # Load VAE
            print("Loading VAE...")
            vae = instantiate(
                target=cfg.test_vae.target,
                cfg=None,
                hfstyle=False,
                **cfg.test_vae.params,
            )
            vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
            
            if "ema_state" in vae_ckpt:
                vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
                vae_ema = ExponentialMovingAverage(
                    vae.parameters(), decay=cfg.test_vae.ema_decay
                )
                vae_ema.load_state_dict(vae_ckpt["ema_state"])
                vae_ema.copy_to(vae.parameters())
                print(f"Loaded VAE with EMA")
            else:
                vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
                print(f"Loaded VAE without EMA")
            
            vae.to(self.device)
            vae.eval()
            
            # Load diffusion model
            print("Loading diffusion model...")
            model = instantiate(
                target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params
            )
            checkpoint = torch.load(cfg.test_ckpt, map_location="cpu", weights_only=False)
            try:
                model.load_state_dict(checkpoint["state_dict"], strict=True)
            except RuntimeError as exc:
                print(
                    "Strict checkpoint load failed; falling back to strict=False for backward compatibility."
                )
                print(f"Reason: {exc}")
                load_result = model.load_state_dict(checkpoint["state_dict"], strict=False)
                if load_result.missing_keys:
                    print(f"Missing keys (initialized from current model): {load_result.missing_keys}")
                if load_result.unexpected_keys:
                    print(f"Unexpected keys (ignored): {load_result.unexpected_keys}")

            if "ema_state" in checkpoint:
                n_shadow = len(checkpoint["ema_state"]["shadow_params"])
                ema_params = [p for p in model.parameters() if p.requires_grad]
                if len(ema_params) != n_shadow:
                    ema_params = list(model.parameters())
                assert len(ema_params) == n_shadow, (
                    f"EMA shadow_params count ({n_shadow}) does not match "
                    f"trainable params ({len([p for p in model.parameters() if p.requires_grad])}) "
                    f"or total params ({len(list(model.parameters()))}). "
                    "Check freeze settings or EMA checkpoint."
                )
                ema = ExponentialMovingAverage(ema_params, decay=cfg.model.ema_decay)
                ema.load_state_dict(checkpoint["ema_state"])
                ema.copy_to(ema_params)
                print(f"Loaded model with EMA ({n_shadow} params)")
            else:
                print("Loaded model without EMA")
            
            model.to(self.device)
            model.eval()
            
            return vae, model
            
        finally:
            # Restore original directory
            os.chdir(original_dir)
    
    def start_generation(self, text, history_length=None):
        """Start or update generation with new text"""
        self.current_text = text
        
        if history_length is not None:
            self.history_length = history_length
        
        if not self.is_generating:
            # Reset state before starting (only once at the beginning)
            self.frame_buffer.clear()
            self.stream_recovery.reset()
            self.vae.clear_cache()
            self.first_chunk = True
            self.root_xz_history.clear()
            self.model.init_generated(self.history_length, batch_size=1, num_denoise_steps=self.denoise_steps)
            print(f"Model initialized with history length: {self.history_length}, denoise steps: {self.denoise_steps}")
            
            # Start generation thread
            self.should_stop = False
            self.generation_thread = threading.Thread(target=self._generation_loop)
            self.generation_thread.daemon = True
            self.generation_thread.start()
            self.is_generating = True
    
    def update_text(self, text):
        """Update text without resetting state (continuous generation with new text)"""
        if text != self.current_text:
            old_text = self.current_text
            self.current_text = text
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
        self,
        waypoints,
        mode="replace_future",
        *,
        source="manual",
        duration_seconds=None,
    ):
        """Update trajectory control from world-space waypoints.

        V1 semantics:
        - `mode=replace_future`: replace only the future plan used by streaming inference.
        - The latent state is preserved; only future trajectory conditioning is updated.
        - Manual non-timestamped paths are normalized to uniform arclength over
          a fixed duration, so browser event density does not change target speed.
        """
        mode = mode or "replace_future"
        if mode != "replace_future":
            raise ValueError(f"Unsupported trajectory mode: {mode}")

        with self.traj_state_lock:
            self.current_traj_mode = mode
            self.traj_plan_version += 1

            if waypoints is None or len(waypoints) == 0:
                self.current_traj_waypoints = None
                self.current_traj_array = None
                self.current_traj_times = None
                self.traj_repeat_anchor_root = None
                self.traj_repeat_anchor_cycle = None
                with self._display_traj_lock:
                    self._display_traj = None
                print("Trajectory control cleared")
                return None

            waypoints = np.asarray(waypoints, dtype=np.float32)
            if waypoints.ndim == 1:
                waypoints = waypoints.reshape(1, -1)
            explicit_times = None
            if waypoints.shape[1] == 4:
                # Explicit timestamped format: [t, x, y, z].
                explicit_times = waypoints[:, 0].astype(np.float32)
                waypoints = waypoints[:, 1:4]
            elif waypoints.shape[1] == 2:
                waypoints = np.c_[
                    waypoints[:, 0],
                    np.zeros(len(waypoints), dtype=np.float32),
                    waypoints[:, 1],
                ]
            if waypoints.shape[1] != 3:
                raise ValueError(
                    "Trajectory waypoints must have shape (N,2), (N,3), "
                    f"or timestamped (N,4); got {waypoints.shape}"
                )

            if explicit_times is None:
                if source == "manual" and len(waypoints) > 0:
                    current_root = self._get_current_root_xyz()
                    if np.linalg.norm(
                        waypoints[0, [0, 2]] - current_root[[0, 2]]
                    ) > 1e-4:
                        waypoints = np.vstack([current_root[None, :], waypoints])
                if source == "manual" and self.manual_resample_arclength:
                    duration = self.manual_duration_seconds
                    if duration_seconds is not None:
                        duration = float(duration_seconds)
                    duration = max(self.waypoint_dt, float(duration))
                    num_points = max(2, int(round(duration / self.waypoint_dt)) + 1)
                    waypoints = self._resample_uniform_arclength(waypoints, num_points)
                else:
                    waypoints = dedupe_polyline(waypoints)
                self.current_traj_times = (
                    np.arange(len(waypoints), dtype=np.float32) * self.waypoint_dt
                )
            else:
                self.current_traj_times = explicit_times.copy()
                if len(self.current_traj_times):
                    self.current_traj_times = self.current_traj_times - self.current_traj_times[0]
            self.current_traj_waypoints = waypoints.copy()
            self.current_traj_array = waypoints.copy()
            self.traj_plan_start_commit_index = int(
                getattr(self.model, "commit_index", 0)
            )
            self.traj_repeat_anchor_root = None
            self.traj_repeat_anchor_cycle = None

        print(
            f"Trajectory updated: {len(waypoints)} waypoints, "
            f"mode={mode}, source={source}, horizon={self.traj_horizon_tokens} tokens, "
            f"time_mode={self.traj_time_mode}, waypoint_dt={self.waypoint_dt:.3f}s, "
            f"duration={(len(waypoints) - 1) * self.waypoint_dt:.2f}s"
        )
        preview = self._build_stream_traj_input()
        if preview is None:
            return None
        return self.get_display_traj()

    def _get_current_root_xyz(self) -> np.ndarray:
        root_xyz = np.zeros(3, dtype=np.float32)
        root_xyz[[0, 2]] = self.stream_recovery.r_pos_accum[[0, 2]].astype(np.float32)
        return root_xyz

    def _estimate_token_step_distance(self) -> float:
        """Thin wrapper — see ``utils.stream_traj.estimate_token_step_distance``."""
        return estimate_token_step_distance(
            list(self.root_xz_history),
            default=self.default_token_step,
            min_step=self.min_token_step,
            max_step=self.max_token_step,
        )

    @staticmethod
    def _project_point_to_polyline(point_xyz: np.ndarray, waypoints_xyz: np.ndarray):
        """Thin wrapper — see ``utils.stream_traj.project_point_to_polyline``."""
        return project_point_to_polyline(point_xyz, waypoints_xyz)

    @staticmethod
    def _dedupe_polyline(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Thin wrapper — see ``utils.stream_traj.dedupe_polyline``."""
        return dedupe_polyline(points, eps)

    def _build_remaining_polyline(self, root_xyz: np.ndarray, waypoints_xyz: np.ndarray) -> np.ndarray:
        """Thin wrapper — see ``utils.stream_traj.build_remaining_polyline``."""
        return build_remaining_polyline(root_xyz, waypoints_xyz)

    @staticmethod
    def _resample_polyline(points_xyz: np.ndarray, num_tokens: int, token_step: float) -> np.ndarray:
        """Thin wrapper — see ``utils.stream_traj.resample_polyline``."""
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
        the original world location.  The returned trajectory remains
        world-space; model-side TrajStreamBuffer still performs its own
        history-window anchor subtract.
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

    def _build_stream_traj_input(self):
        with self.traj_state_lock:
            if self.current_traj_waypoints is None:
                return None
            waypoints = self.current_traj_waypoints.copy()
            traj_times = None if self.current_traj_times is None else self.current_traj_times.copy()
            plan_version = self.traj_plan_version
            traj_mode = self.current_traj_mode
            plan_start_commit = self.traj_plan_start_commit_index

        if self.traj_time_mode == "timestamped" and traj_times is not None:
            commit_index = int(getattr(self.model, "commit_index", 0))
            elapsed_tokens = max(0, commit_index - int(plan_start_commit))
            query_times = (
                float(elapsed_tokens) * self.token_dt
                + np.arange(self.traj_horizon_tokens, dtype=np.float32) * self.token_dt
            )
            future_traj = self._sample_timestamped_with_repeat(
                traj_times, waypoints, query_times
            )
        else:
            current_root = self._get_current_root_xyz()
            token_step = self._estimate_token_step_distance()
            polyline = self._build_remaining_polyline(current_root, waypoints)
            future_traj = self._resample_polyline(
                polyline,
                num_tokens=self.traj_horizon_tokens,
                token_step=token_step,
            )
        token_mask = np.ones((1, future_traj.shape[0]), dtype=np.float32)

        with self._display_traj_lock:
            self._display_traj = future_traj.copy()

        return {
            "traj": future_traj[None, :, :],
            "token_mask": token_mask,
            "traj_mode": traj_mode,
            "traj_plan_version": plan_version,
            "traj_repeat_policy": self.traj_repeat_policy,
        }
    
    def pause_generation(self):
        """Pause generation (keeps all state)"""
        self.should_stop = True
        if self.generation_thread:
            self.generation_thread.join(timeout=5.0)
            if self.generation_thread.is_alive():
                print("Warning: generation thread did not stop within timeout; model state may be unsafe")
                return False
        self.is_generating = False
        print("Generation paused (state preserved)")
        return True
    
    def resume_generation(self):
        """Resume generation from paused state"""
        if self.is_generating:
            print("Already generating, ignoring resume")
            return
        
        # Restart generation thread with existing state
        self.should_stop = False
        self.generation_thread = threading.Thread(target=self._generation_loop)
        self.generation_thread.daemon = True
        self.generation_thread.start()
        self.is_generating = True
        print("Generation resumed")
    
    def reset(self, history_length=None, smoothing_alpha=None, denoise_steps=None):
        """Reset generation state completely
        
        Args:
            history_length: History window length for the model
            smoothing_alpha: EMA smoothing factor (0.0 to 1.0)
                - 1.0 = no smoothing (default)
                - 0.0 = infinite smoothing
                - Recommended: 0.3-0.7 for visible smoothing
            denoise_steps: Number of denoising steps (1-50, default 10)
        """
        # Stop if running, then poll until thread truly exits (max 10s total)
        if self.is_generating:
            if not self.pause_generation():
                return False
        if self.generation_thread is not None and self.generation_thread.is_alive():
            self.reset_pending = True
            print("Reset pending — waiting for generation thread to finish...")
            for _ in range(20):  # up to 10s more (20 × 0.5s)
                self.generation_thread.join(timeout=0.5)
                if not self.generation_thread.is_alive():
                    break
            if self.generation_thread.is_alive():
                print("Reset failed: generation thread still running after 15s timeout")
                self.reset_pending = False
                return False
        self.reset_pending = False

        # Clear everything
        self.frame_buffer.clear()
        self.vae.clear_cache()
        self.first_chunk = True
        self.root_xz_history.clear()
        with self.traj_state_lock:
            self.current_traj_waypoints = None
            self.current_traj_array = None
            self.current_traj_times = None
            self.current_traj_mode = "replace_future"
            self.traj_repeat_anchor_root = None
            self.traj_repeat_anchor_cycle = None
            self.traj_plan_version += 1
        with self._display_traj_lock:
            self._display_traj = None
        
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
        
        # Recreate stream recovery with new smoothing alpha
        self.stream_recovery = StreamJointRecovery263(
            joints_num=22, 
            smoothing_alpha=self.smoothing_alpha
        )
        
        # Initialize model with denoise steps
        self.model.init_generated(self.history_length, batch_size=1, num_denoise_steps=self.denoise_steps)
        print(f"Model reset - history: {self.history_length}, smoothing: {self.smoothing_alpha}, steps: {self.denoise_steps}")
        return True
    
    def _generation_loop(self):
        """Background loop: each iteration produces one latent token (→ 4 motion frames).

        When trajectory control is active, each step passes a future token-horizon in
        world coordinates. The model-side streaming path then rewrites only the future
        conditioning slots and normalizes the full visible context window back to a
        clip-local origin before trajectory encoding.
        """
        print("Generation loop started")
        
        import time
        step_count = 0
        total_gen_time = 0
        
        with torch.no_grad():
            while not self.should_stop:
                # Check if buffer needs more frames
                if self.frame_buffer.needs_generation():
                    try:
                        step_start = time.time()
                        
                        # Generate one token (produces 4 frames from VAE)
                        traj_input = self._build_stream_traj_input()
                        if traj_input is None and hasattr(self.model, "_traj_buf"):
                            self.model._traj_buf.reset()
                        x = build_stream_step_model_input(
                            self.current_text, traj_input=traj_input
                        )
                        
                        # Generate from model (1 token)
                        # Note: denoise_steps is set in init_generated, not here
                        output = self.model.stream_generate_step(
                            x, first_chunk=self.first_chunk
                        )
                        generated = output["generated"]
                        
                        # Decode with VAE (1 token -> 4 frames)
                        decoded = self.vae.stream_decode(
                            generated[0][None, :], first_chunk=self.first_chunk
                        )[0]
                        
                        self.first_chunk = False
                        
                        # Convert each frame to joints
                        for i in range(decoded.shape[0]):
                            frame_data = decoded[i].cpu().numpy()
                            joints = self.stream_recovery.process_frame(frame_data)
                            self.root_xz_history.append(
                                self.stream_recovery.r_pos_accum[[0, 2]].astype(np.float32).copy()
                            )
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
        with self._display_traj_lock:
            if self._display_traj is None:
                return None
            return self._display_traj.copy()

    def get_next_frame(self):
        """Get the next frame from buffer and optional trajectory display data."""
        joints = self.frame_buffer.get_frame()
        traj = self.get_display_traj()
        return joints, traj
    
    def get_buffer_status(self):
        """Get buffer status"""
        return {
            "buffer_size": self.frame_buffer.size(),
            "target_size": self.frame_buffer.target_size,
            "is_generating": self.is_generating,
            "current_text": self.current_text,
            "trajectory_active": self.current_traj_waypoints is not None,
            "trajectory_time_mode": self.traj_time_mode,
            "trajectory_repeat_policy": self.traj_repeat_policy,
            "trajectory_horizon_tokens": self.traj_horizon_tokens,
            "smoothing_alpha": self.smoothing_alpha,
            "denoise_steps": self.denoise_steps,
        }


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
