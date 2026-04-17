"""
Model Manager for real-time motion generation
Manages model loading, frame buffering, and streaming generation
"""
import sys
import os
import threading
import queue
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
    trajectory conditioning (legacy `use_traj_cond` or ControlNet `use_controlnet_traj`).
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
        
        # Trajectory control: waypoints → interpolated path (one point per token step)
        self.current_traj_waypoints = None
        self.current_traj_array = None  # (N, 3) float32, N = TRAJ_INTERP_LENGTH
        self.current_traj_features = None  # (N, 4) float32, [x,z,cos,sin] per token step
        self.current_token_mask = None  # (N,) float32, 1=keep, 0=drop per token step
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
            use_traj_cond = cfg.model.params.get("use_traj_cond", False)
            use_controlnet_traj = cfg.model.params.get("use_controlnet_traj", False)
            strict_load = not (use_traj_cond or use_controlnet_traj)  # allow missing new cond params
            load_result = model.load_state_dict(
                checkpoint["state_dict"], strict=strict_load
            )
            if (use_traj_cond or use_controlnet_traj) and not strict_load:
                if load_result.missing_keys:
                    print(
                        f"Loaded with strict=False (traj). Missing keys (init from scratch): {load_result.missing_keys}"
                    )
                if load_result.unexpected_keys:
                    print(f"Unexpected keys (ignored): {load_result.unexpected_keys}")

            if "ema_state" in checkpoint:
                ema = ExponentialMovingAverage(
                    model.parameters(), decay=cfg.model.ema_decay
                )
                ema.load_state_dict(checkpoint["ema_state"])
                ema.copy_to(model.parameters())
                print("Loaded model with EMA")
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
            self.current_traj_features = None
            self.current_token_mask = None
            print("Trajectory control cleared")
            return
        waypoints = np.array(waypoints, dtype=np.float64)
        if waypoints.ndim == 1:
            waypoints = waypoints.reshape(1, -1)
        if waypoints.shape[1] == 2:
            waypoints = np.c_[waypoints[:, 0], np.zeros(len(waypoints)), waypoints[:, 1]]
        n = len(waypoints)
        indices = np.linspace(0, n - 1, self.TRAJ_INTERP_LENGTH, dtype=np.float64)
        self.current_traj_array = np.stack([
            np.interp(indices, np.arange(n), waypoints[:, 0]),
            np.interp(indices, np.arange(n), waypoints[:, 1]),
            np.interp(indices, np.arange(n), waypoints[:, 2]),
        ], axis=1).astype(np.float32)

        # Randomly mask user waypoints (sparse trajectory hints, aligned with training).
        # Then map waypoint-level mask onto interpolated time steps via nearest neighbor.
        waypoint_mask = self._sample_waypoint_mask(n)  # (n,)
        interp_mask_idx = np.clip(np.round(indices).astype(np.int64), 0, n - 1)  # (TRAJ_INTERP_LENGTH,)
        interp_mask = waypoint_mask[interp_mask_idx].astype(np.float32)  # (TRAJ_INTERP_LENGTH,)
        self.current_token_mask = interp_mask.astype(np.float32)

        # Build token-step [x,z,cos,sin] features (same semantics as dataset traj_features).
        self.current_traj_features = root_to_traj_feats(self.current_traj_array)  # (N,4)
        self.current_traj_features = self.current_traj_features * self.current_token_mask[:, None]

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
            self.generation_thread.join(timeout=2.0)
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
        # Stop if running
        if self.is_generating:
            self.pause_generation()
        
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
                        if self.current_traj_features is not None:
                            idx = min(self.traj_token_index, self.current_traj_features.shape[0] - 1)
                            x["traj_features"] = self.current_traj_features[idx : idx + 1]  # (1,4)
                            if self.current_token_mask is not None:
                                x["token_mask"] = self.current_token_mask[idx : idx + 1]  # (1,)
                        
                        # Generate from model (1 token)
                        # Note: denoise_steps is set in init_generated, not here
                        output = self.model.stream_generate_step(
                            x, first_chunk=self.first_chunk
                        )
                        if self.current_traj_features is not None:
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

