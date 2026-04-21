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
from utils.initialize import instantiate, load_config, compare_statedict_and_parameters
from utils.motion_process import StreamJointRecovery263
from utils.traj_batch import root_to_traj_feats


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
        self.traj_mask_enabled = bool(traj_mask_cfg.get("enabled", True))
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
        
        # Trajectory control: waypoints → interpolated path
        # current_traj_features_frame: (N*4, 4) float32 — frame-level [x,z,cos,sin].
        # Each token step consumes 4 consecutive frames, passed through LocalTrajEncoder
        # to match the training distribution exactly.
        self.current_traj_waypoints = None
        self.current_traj_array = None         # (N, 3) float32, N = TRAJ_INTERP_LENGTH (token-level)
        self.current_traj_features_frame = None  # (N*4, 4) float32, frame-level features
        self.current_token_mask = None         # (N,) float32, 1=keep, 0=drop per token step
        self.traj_token_index = 0
        self.TRAJ_INTERP_LENGTH = 2000
        
        # Model generation state
        self.first_chunk = True
        self.history_length = 30  # Default history window length
        self.denoise_steps = 10  # Default denoising steps
        
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
                all_params = list(model.parameters())
                if n_shadow == len(all_params):
                    ema_params = all_params
                elif getattr(model, "freeze_backbone", False) and model.controlnet is not None:
                    ema_params = list(model.controlnet.parameters()) + (
                        list(model.traj_encoder.parameters()) if model.traj_encoder is not None else []
                    )
                else:
                    ema_params = all_params
                assert len(ema_params) == n_shadow, (
                    f"EMA shadow_params count ({n_shadow}) does not match "
                    f"selected param group ({len(ema_params)})."
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
            self.traj_token_index = 0
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

    def update_trajectory(self, waypoints):
        """Update trajectory control from waypoints (list of [x,z] or [x,y,z]).
        Waypoints are interpolated to a fixed-length path for streaming.
        Pass None to clear trajectory control.
        """
        if waypoints is None or len(waypoints) == 0:
            self.current_traj_waypoints = None
            self.current_traj_array = None
            self.current_traj_features_frame = None
            self.current_token_mask = None
            print("Trajectory control cleared")
            return
        waypoints = np.array(waypoints, dtype=np.float64)
        if waypoints.ndim == 1:
            waypoints = waypoints.reshape(1, -1)
        if waypoints.shape[1] == 2:
            waypoints = np.c_[waypoints[:, 0], np.zeros(len(waypoints)), waypoints[:, 1]]
        n = len(waypoints)

        # Token-level interpolation (for mask and reference).
        indices_tok = np.linspace(0, n - 1, self.TRAJ_INTERP_LENGTH, dtype=np.float64)
        self.current_traj_array = np.stack([
            np.interp(indices_tok, np.arange(n), waypoints[:, 0]),
            np.interp(indices_tok, np.arange(n), waypoints[:, 1]),
            np.interp(indices_tok, np.arange(n), waypoints[:, 2]),
        ], axis=1).astype(np.float32)  # (N_tok, 3)

        # Frame-level interpolation at 4× resolution — one xyz per motion frame so that
        # each token step gets 4 consecutive frames for LocalTrajEncoder, matching training.
        n_frames = self.TRAJ_INTERP_LENGTH * 4
        indices_frm = np.linspace(0, n - 1, n_frames, dtype=np.float64)
        traj_frame = np.stack([
            np.interp(indices_frm, np.arange(n), waypoints[:, 0]),
            np.interp(indices_frm, np.arange(n), waypoints[:, 1]),
            np.interp(indices_frm, np.arange(n), waypoints[:, 2]),
        ], axis=1).astype(np.float32)  # (N_tok*4, 3)
        self.current_traj_features_frame = root_to_traj_feats(traj_frame)  # (N_tok*4, 4)

        # Randomly mask user waypoints (sparse trajectory hints, aligned with training).
        waypoint_mask = self._sample_waypoint_mask(n)  # (n,)
        interp_mask_idx = np.clip(np.round(indices_tok).astype(np.int64), 0, n - 1)
        interp_mask = waypoint_mask[interp_mask_idx].astype(np.float32)  # (N_tok,)
        self.current_token_mask = interp_mask

        self.current_traj_waypoints = waypoints
        if self._last_traj_mask_keep is not None and self._last_traj_mask_total is not None:
            masked_keep_steps = int(interp_mask.sum().item())
            masked_total_steps = int(interp_mask.shape[0])
            print(
                f"Trajectory updated: {n} waypoints -> {len(self.current_traj_array)} time steps; "
                f"keep (waypoints): {self._last_traj_mask_keep}/{self._last_traj_mask_total}, "
                f"keep (time steps): {masked_keep_steps}/{masked_total_steps}"
            )
        else:
            print(f"Trajectory updated: {n} waypoints -> {len(self.current_traj_array)} points")
    
    def pause_generation(self):
        """Pause generation (keeps all state)"""
        self.should_stop = True
        if self.generation_thread:
            self.generation_thread.join(timeout=5.0)
            if self.generation_thread.is_alive():
                print("Warning: generation thread did not stop within timeout; model state may be unsafe")
        self.is_generating = False
        print("Generation paused (state preserved)")
    
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
            self.pause_generation()
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
                return
        self.reset_pending = False

        # Clear everything
        self.frame_buffer.clear()
        self.vae.clear_cache()
        self.first_chunk = True
        self.traj_token_index = 0
        
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
    
    def _generation_loop(self):
        """Background loop: each iteration produces one latent token (→ 4 motion frames).
        When trajectory is set, passes one traj point per step into the model's stream buffer.
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
                        x = {"text": [self.current_text]}
                        if self.current_traj_features_frame is not None:
                            tok_idx = min(self.traj_token_index, self.TRAJ_INTERP_LENGTH - 1)
                            frm_start = tok_idx * 4
                            frm_end = min(frm_start + 4, self.current_traj_features_frame.shape[0])
                            frames = self.current_traj_features_frame[frm_start:frm_end]  # (≤4, 4)
                            # Pad to exactly 4 frames if near trajectory end
                            if frames.shape[0] < 4:
                                pad = np.zeros((4 - frames.shape[0], 4), dtype=np.float32)
                                frames = np.concatenate([frames, pad], axis=0)
                            # Apply LocalTrajEncoder to match training path exactly:
                            #   training: frame-level (B,T,4,4) → LocalTrajEncoder → (B,T,4) → TrajEncoder
                            #   here:     (1,1,4,4)              → LocalTrajEncoder → (1,1,4) → TrajEncoder
                            frames_t = torch.from_numpy(frames).float().to(device)  # (4, 4)
                            frames_t = frames_t.unsqueeze(0).unsqueeze(0)           # (1, 1, 4, 4)
                            tok_feat = self.model.local_traj_encoder(frames_t)      # (1, 1, 4)
                            x["traj_features"] = tok_feat  # (1, 1, 4) tensor, matches buffer shape
                            if self.current_token_mask is not None:
                                x["token_mask"] = self.current_token_mask[tok_idx : tok_idx + 1][None, :]  # (1,1)
                        
                        # Generate from model (1 token)
                        # Note: denoise_steps is set in init_generated, not here
                        output = self.model.stream_generate_step(
                            x, first_chunk=self.first_chunk
                        )
                        if self.current_traj_features_frame is not None:
                            self.traj_token_index += 1
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
    
    def get_next_frame(self):
        """Get the next frame from buffer"""
        return self.frame_buffer.get_frame()
    
    def get_buffer_status(self):
        """Get buffer status"""
        return {
            "buffer_size": self.frame_buffer.size(),
            "target_size": self.frame_buffer.target_size,
            "is_generating": self.is_generating,
            "current_text": self.current_text,
            "trajectory_active": self.current_traj_array is not None,
            "smoothing_alpha": self.smoothing_alpha,
            "denoise_steps": self.denoise_steps,
        }


# Global model manager instance
_model_manager = None
_traj_mask_cfg = None


def get_model_manager(config_path=None, traj_mask_cfg=None):
    """Get or create the global model manager instance"""
    global _model_manager, _traj_mask_cfg
    if _model_manager is None:
        _traj_mask_cfg = traj_mask_cfg or {}
        _model_manager = ModelManager(config_path, traj_mask_cfg=_traj_mask_cfg)
    return _model_manager

