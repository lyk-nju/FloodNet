"""Streaming trajectory buffer for DiffForcingWanModel.stream_generate_step().

Manages per-step traj features / xyz / token_mask writes and LRU embedding cache.
Decoupled from the diffusion model: only holds references to the two traj encoders.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from FloodNet.utils.traj_batch import (
        frames_to_tokens,
        root_to_traj_feats,
    )
except ImportError:  # pragma: no cover - script entrypoints use top-level imports
    from utils.traj_batch import frames_to_tokens, root_to_traj_feats


class TrajStreamBuffer:
    """Rolling buffer for streaming trajectory conditioning.

    The buffer length is ``buf_len = seq_len * 2 + chunk_size`` tokens.
    ``commit_index`` tracks how many tokens have been committed so far; it is
    owned and incremented by the caller (stream_generate_step).

    Two storage modes (mutually exclusive per update):
      - *features mode*: caller provides ``traj_features`` (B, T, 4) already in
        [x, z, cos, sin] space — stored directly, no heading recomputation.
      - *xyz mode*: caller provides ``traj`` (B, T, 3) world-space positions —
        stored as-is; heading is computed lazily at embedding time after
        token-level linear interpolation back to frame level.
    """

    def __init__(
        self,
        batch_size: int | None = None,
        buf_len: int | None = None,
        local_traj_encoder: torch.nn.Module | None = None,
        traj_encoder: torch.nn.Module | None = None,
        use_emb_cache: bool = True,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.batch_size = batch_size
        self.buf_len = buf_len
        self.local_traj_encoder = local_traj_encoder
        self.traj_encoder = traj_encoder
        self.use_emb_cache = use_emb_cache

        self._feat_buf: torch.Tensor | None = None   # (B, buf_len, 4) traj_features
        self._xyz_buf: torch.Tensor | None = None    # (B, buf_len, 3) traj xyz
        self._mask_buf: torch.Tensor | None = None   # (B, buf_len) token mask
        self._version: int = 0
        self._emb_cache: dict = {}

        # T_C_01: RootPlan-aware 7D condition provider state. device/dtype are
        # injected here (not at set_root_plan) so the no-traj branch works from
        # session start (before any plan) with body-consistent device/dtype.
        self._active_plan = None
        self._device = torch.device(device)
        self._dtype = dtype

    # ------------------------------------------------------------------
    # T_C_01: RootPlan-aware 7D condition provider (Stage 2 main path).
    # Stateless transform RootPlan → body-window-local 7D condition. Does NOT
    # call the Refiner, text encoder, or advance the timeline — the upper layer
    # (ModelManager) owns those and passes head_state / body_anchor_state in.
    # ------------------------------------------------------------------

    def set_root_plan(self, root_plan) -> None:
        """Cache the active plan, moving its tensor fields to this buffer's
        device/dtype (the Refiner may return a CPU RootPlan while body runs on
        CUDA — the active-plan branch would otherwise hit a device mismatch)."""
        import dataclasses

        self._active_plan = dataclasses.replace(
            root_plan,
            waypoints_local_7d=root_plan.waypoints_local_7d.to(
                device=self._device, dtype=self._dtype),
            anchor_world_xz=root_plan.anchor_world_xz.to(
                device=self._device, dtype=self._dtype),
            anchor_world_yaw=root_plan.anchor_world_yaw.to(
                device=self._device, dtype=self._dtype),
        )

    def clear(self) -> None:
        self._active_plan = None

    def has_active_plan(self) -> bool:
        return self._active_plan is not None

    def _return_no_traj(self, expected_horizon_frame_slice, *,
                        device=None, dtype=None):
        """Standard no-traj output: zero 7D condition + all-False mask. Single
        source for both no-plan and inactive-plan branches (consistent
        device/dtype/mask-dtype)."""
        if expected_horizon_frame_slice is None:
            raise ValueError("no-traj fallback requires expected_horizon_frame_slice")
        H = expected_horizon_frame_slice.stop - expected_horizon_frame_slice.start
        device = device or self._device
        dtype = dtype or self._dtype
        return (
            torch.zeros(H, 7, device=device, dtype=dtype),
            torch.zeros(H, device=device, dtype=torch.bool),
        )

    def get_body_traj_cond(self, *, head_state, body_anchor_state,
                           horizon_tokens: int,
                           expected_horizon_frame_slice=None):
        """RootPlan → (traj_cond_7d_frame [H,7], traj_cond_frame_mask [H]).

        UNBATCHED output (caller unsqueezes to [1,H,7]/[1,H]). Dual-anchor:
        current_plan_token is derived from head_state (Refiner anchor); the final
        world→local canonicalize uses body_anchor_state (body-window history0).
        expected_horizon_frame_slice ONLY sets the output frame count; the
        plan-local slice is derived from current_plan_token (decoupled).
        """
        from utils.local_frame import canonicalize_7d, uncanonicalize_7d
        from utils.root_plan import slice_plan_with_mask
        from utils.token_frame import token_range_to_frame_slice, token_start_frame

        # no-traj: no active plan
        if self._active_plan is None:
            return self._return_no_traj(expected_horizon_frame_slice)

        plan = self._active_plan

        # Step 1: plan-local offset from the HEAD state (not body anchor).
        current_plan_token = head_state.commit_idx - plan.anchor_commit_idx
        if current_plan_token < 0:
            # plan not yet active → no-traj
            if expected_horizon_frame_slice is None:
                raise ValueError(
                    "inactive plan / no-traj fallback requires explicit "
                    "expected_horizon_frame_slice (active-plan vs no-traj shape "
                    "consistency)"
                )
            return self._return_no_traj(expected_horizon_frame_slice)

        # Step 2a: output frame count H_frame (body-window space, shape only).
        # ⚠ fallback uses token_range_to_frame_slice length (= 4*H for an
        # arbitrary range), NOT num_frames_for_tokens (= 4N-3, prefix only).
        if expected_horizon_frame_slice is not None:
            H_frame = expected_horizon_frame_slice.stop - expected_horizon_frame_slice.start
        else:
            plan_range = token_range_to_frame_slice(
                current_plan_token, horizon_tokens, plan.frames_per_token)
            H_frame = plan_range.stop - plan_range.start

        # Step 2b: plan-local slice (index into plan.waypoints_local_7d).
        plan_frame_start = token_start_frame(current_plan_token, plan.frames_per_token)
        plan_frame_slice = slice(plan_frame_start, plan_frame_start + H_frame)

        # Step 3: slice + overflow hold-last (plan-local frame space).
        traj_plan_local, traj_mask_frame = slice_plan_with_mask(
            plan, frame_slice=plan_frame_slice, hold_last_on_overflow=True)

        # Step 4: plan-anchor-local → world (plan's own anchor).
        traj_world = uncanonicalize_7d(
            traj_plan_local, plan.anchor_world_xz, plan.anchor_world_yaw)

        # Step 5: world → body-window-local (body_anchor_state, NOT head_state).
        # P1-1: defensively cast the anchor pose to the buffer device/dtype — the
        # InferenceGlueState contract leaves casting to the caller, and a legacy
        # buffer (no device/dtype passed) defaults to CPU → CUDA mismatch.
        anchor_xz = body_anchor_state.world_xz.to(device=self._device, dtype=self._dtype)
        anchor_yaw = body_anchor_state.world_yaw.to(device=self._device, dtype=self._dtype)
        traj_body_local = canonicalize_7d(traj_world, anchor_xz, anchor_yaw)

        return traj_body_local, traj_mask_frame

    # ------------------------------------------------------------------
    # public API (legacy xyz/feature streaming path)
    # ------------------------------------------------------------------

    def reset(self):
        self._feat_buf = None
        self._xyz_buf = None
        self._mask_buf = None
        self._version = 0
        self._emb_cache = {}

    def update(self, x: dict, commit_index: int, device):
        """Write traj data from batch dict x into the buffer at commit_index.

        Write semantics: INCREMENTAL APPEND — only the range
        [commit_index, commit_index + len(tf)) is written; positions beyond
        that are NOT cleared.  Callers that want to replace future trajectory
        data should call reset() first, or re-pass the full remaining
        trajectory at every step.

        If x contains no traj fields, returns immediately (buffer is preserved).
        To stop conditioning, call reset() explicitly before the step.
        Increments the internal version to invalidate the embedding cache.
        """
        if commit_index >= self.buf_len:
            return

        no_traj = (
            ("traj" not in x or x["traj"] is None)
            and ("traj_features" not in x or x["traj_features"] is None)
        )
        if no_traj:
            return

        wrote = False

        if "traj_features" in x and x["traj_features"] is not None:
            tf = x["traj_features"]
            if isinstance(tf, np.ndarray):
                tf = torch.from_numpy(tf).float()
            tf = tf.to(device)
            if tf.dim() == 2:
                tf = tf.unsqueeze(0)
            if tf.dim() == 3 and tf.size(0) == self.batch_size:
                self._xyz_buf = None   # switch to features mode
                if self._feat_buf is None:
                    self._feat_buf = torch.zeros(
                        self.batch_size, self.buf_len, tf.size(-1),
                        device=device, dtype=tf.dtype,
                    )
                else:
                    self._feat_buf = self._feat_buf.to(device)
                k = max(0, min(tf.size(1), self.buf_len - commit_index))
                if k > 0:
                    self._feat_buf[:, commit_index:commit_index + k] = tf[:, :k]
                    wrote = True
                self._mask_buf = self._write_mask(
                    x.get("token_mask"), commit_index, k, device, dtype=tf.dtype
                )

        elif "traj" in x and x["traj"] is not None:
            traj_in = x["traj"]
            if isinstance(traj_in, np.ndarray):
                traj_in = torch.from_numpy(traj_in).float()
            traj_in = traj_in.to(device)
            if traj_in.dim() == 2:
                traj_in = traj_in.unsqueeze(0)
            if traj_in.dim() == 1:
                traj_in = traj_in.unsqueeze(0).unsqueeze(0)
            if traj_in.dim() == 3 and traj_in.size(0) == self.batch_size:
                self._feat_buf = None  # switch to xyz mode
                if self._xyz_buf is None:
                    self._xyz_buf = torch.zeros(
                        self.batch_size, self.buf_len, 3,
                        device=device, dtype=torch.float32,
                    )
                else:
                    self._xyz_buf = self._xyz_buf.to(device)
                k = max(0, min(traj_in.size(1), self.buf_len - commit_index))
                if k > 0:
                    self._xyz_buf[:, commit_index:commit_index + k] = traj_in[:, :k]
                    wrote = True
                self._mask_buf = self._write_mask(
                    x.get("token_mask"), commit_index, k, device, dtype=torch.float32,
                    fill_ones_if_absent=(k > 0),
                )

        if wrote:
            self._version += 1
            self._emb_cache = {}

    def build_traj_emb(
        self, end_index: int, seq_len: int, device
    ) -> torch.Tensor | None:
        """Return trajectory embedding (B, ctx_len, traj_out_dim) for the window
        ``[end_index - seq_len, end_index)``.

        Returns None if no traj data is buffered.
        """
        ctx_len = min(end_index, seq_len)
        start_t = max(0, end_index - seq_len)

        if self._feat_buf is not None:
            return self._build_from_features(start_t, end_index, ctx_len, device)
        if self._xyz_buf is not None:
            return self._build_from_xyz(start_t, end_index, ctx_len, device)
        return None

    def get_traj_valid_lens(
        self, end_index: int, seq_len: int, device
    ) -> torch.Tensor | None:
        """Return the last-valid-token position + 1 for each batch item.

        Assumes a contiguous-prefix mask ([1,1,...,1,0,...,0]).  Sparse masks
        (e.g. waypoints) will get lens = last_valid_pos + 1, which may cause
        FlexTraj attention to attend to zero-filled gaps in between.
        Currently safe because mask_ratio=1.0 always produces full-prefix masks.

        Returns None if no mask is tracked (all tokens assumed valid by caller).
        """
        if self._mask_buf is None:
            return None
        ctx_len = min(end_index, seq_len)
        start_t = max(0, end_index - seq_len)
        mask_slice = self._mask_buf[:, start_t:end_index].to(device)
        mask_slice = self._pad_mask_to_ctx(mask_slice, ctx_len, device)
        valid = mask_slice > 0  # (B, ctx_len)
        has_valid = valid.any(dim=1)  # (B,)
        if not has_valid.any():
            return None
        # Distance from the end to the last valid token
        last_from_end = valid.long().flip(1).argmax(dim=1)  # (B,)
        lens = (ctx_len - last_from_end).clamp(min=0)
        lens = torch.where(has_valid, lens, torch.zeros_like(lens))
        return lens.to(dtype=torch.long)

    def roll(self, seq_len: int, device):
        """Shift the buffer left by seq_len when commit_index reaches 2*seq_len.

        Called by stream_generate_step after the rolling reset.
        """
        if self._feat_buf is not None:
            tail = torch.zeros(
                self.batch_size, seq_len, self._feat_buf.size(-1),
                device=device, dtype=self._feat_buf.dtype,
            )
            self._feat_buf = torch.cat(
                [self._feat_buf[:, seq_len:], tail], dim=1
            )
        if self._xyz_buf is not None:
            tail = torch.zeros(
                self.batch_size, seq_len, 3,
                device=device, dtype=self._xyz_buf.dtype,
            )
            self._xyz_buf = torch.cat(
                [self._xyz_buf[:, seq_len:], tail], dim=1
            )
        if self._mask_buf is not None:
            tail = torch.zeros(
                self.batch_size, seq_len,
                device=device, dtype=self._mask_buf.dtype,
            )
            self._mask_buf = torch.cat(
                [self._mask_buf[:, seq_len:], tail], dim=1
            )
        self._version += 1
        self._emb_cache = {}

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _write_mask(self, token_mask, commit_index, k, device, dtype, fill_ones_if_absent=False):
        if self._mask_buf is None:
            self._mask_buf = torch.zeros(
                self.batch_size, self.buf_len, device=device, dtype=dtype,
            )
        else:
            self._mask_buf = self._mask_buf.to(device)

        if token_mask is not None and k > 0:
            tm = token_mask
            if isinstance(tm, np.ndarray):
                tm = torch.from_numpy(tm).float()
            tm = tm.to(device)
            if tm.dim() == 1:
                tm = tm.unsqueeze(0)
            if tm.dim() == 2 and tm.size(0) == self.batch_size:
                self._mask_buf[:, commit_index:commit_index + k] = tm[:, :k]
        elif fill_ones_if_absent and k > 0:
            self._mask_buf[:, commit_index:commit_index + k] = 1.0

        return self._mask_buf

    def _pad_to_ctx(self, buf_slice: torch.Tensor, ctx_len: int, pad_channels: int, device):
        if buf_slice.size(1) < ctx_len:
            pad = buf_slice.new_zeros(self.batch_size, ctx_len - buf_slice.size(1), pad_channels)
            buf_slice = torch.cat([pad, buf_slice], dim=1)
        return buf_slice

    def _pad_mask_to_ctx(self, mask_slice: torch.Tensor | None, ctx_len: int, device):
        if mask_slice is None:
            return None
        if mask_slice.size(1) < ctx_len:
            pad = mask_slice.new_zeros(self.batch_size, ctx_len - mask_slice.size(1))
            mask_slice = torch.cat([pad, mask_slice], dim=1)
        return mask_slice

    def _build_from_features(self, start_t, end_index, ctx_len, device):
        key = ("feat", start_t, end_index, self._version)
        if self.use_emb_cache and key in self._emb_cache:
            return self._emb_cache[key]

        feats = self._feat_buf[:, start_t:end_index, :].to(device)
        feats = self._pad_to_ctx(feats, ctx_len, feats.size(-1), device)
        mask = self._pad_mask_to_ctx(
            self._mask_buf[:, start_t:end_index].to(device) if self._mask_buf is not None else None,
            ctx_len, device,
        )
        if mask is not None:
            feats = feats * mask.unsqueeze(-1).to(dtype=feats.dtype)

        emb = self.traj_encoder(feats)
        if self.use_emb_cache:
            self._emb_cache[key] = emb
        return emb

    def _build_from_xyz(self, start_t, end_index, ctx_len, device):
        # P1-5: the legacy xyz path is translate-only 4D path-heading, NOT B-full
        # canonical / physical-yaw 7D. If the encoder is 7D, the 7D RootPlan path
        # (get_body_traj_cond) is the intended route — warn so a stale legacy
        # callsite doesn't silently feed 4D-style conditioning to a 7D model.
        if getattr(self.traj_encoder, "in_dim", 4) == 7:
            import warnings
            warnings.warn(
                "TrajStreamBuffer._build_from_xyz (legacy 4D translate-only path) "
                "called with a 7D traj_encoder; use get_body_traj_cond (RootPlan "
                "7D, B-full canonical) instead.",
                RuntimeWarning, stacklevel=2,
            )
        key = ("xyz", start_t, end_index, self._version)
        if self.use_emb_cache and key in self._emb_cache:
            return self._emb_cache[key]

        traj_slice = self._xyz_buf[:, start_t:end_index, :].clone().to(device)
        traj_slice = self._pad_to_ctx(traj_slice, ctx_len, 3, device)
        mask = self._pad_mask_to_ctx(
            self._mask_buf[:, start_t:end_index].to(device) if self._mask_buf is not None else None,
            ctx_len, device,
        )

        # Anchor-subtract: shift so the first visible token is the origin.
        if mask is not None:
            valid = mask > 0
            first_valid = valid.to(dtype=torch.long).argmax(dim=1)
            anchor = traj_slice[torch.arange(self.batch_size, device=device), first_valid]
        else:
            anchor = traj_slice[:, 0, :]
        traj_slice = traj_slice - anchor.unsqueeze(1)
        if mask is not None:
            traj_slice = traj_slice * mask.unsqueeze(-1).to(dtype=traj_slice.dtype)

        # token-level xyz → frame-level via linear interpolation → heading features
        traj_frames = _expand_tokens_to_causal_frames(traj_slice)   # (B, 1+4*(N-1), 3)
        feats_frame = root_to_traj_feats(traj_frames)               # (B, T_frames, 4)
        feats_4 = frames_to_tokens(feats_frame, ctx_len)  # (B, ctx_len, 4, 4)
        feats_tok = self.local_traj_encoder(feats_4)                 # (B, ctx_len, 4)

        if mask is not None:
            feats_tok = feats_tok * mask.unsqueeze(-1).to(dtype=feats_tok.dtype)

        emb = self.traj_encoder(feats_tok)
        if self.use_emb_cache:
            self._emb_cache[key] = emb
        return emb


def _expand_tokens_to_causal_frames(traj_tokens: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate token-level xyz to causal-VAE frame-level xyz.

    (B, N, 3) → (B, 1 + 4*(N-1), 3)
    token 0 stays as frame 0; each subsequent token k is split into 4 frames
    by lerp between token k-1 and token k at alpha = [0.25, 0.5, 0.75, 1.0].
    """
    if traj_tokens.size(1) <= 1:
        return traj_tokens[:, :1, :]
    alpha = traj_tokens.new_tensor([0.25, 0.5, 0.75, 1.0]).view(1, 1, 4, 1)
    prev_tok = traj_tokens[:, :-1, :]   # (B, N-1, 3)
    next_tok = traj_tokens[:, 1:, :]    # (B, N-1, 3)
    interp = prev_tok.unsqueeze(2) + (next_tok - prev_tok).unsqueeze(2) * alpha
    interp = interp.reshape(traj_tokens.shape[0], -1, traj_tokens.shape[-1])
    return torch.cat([traj_tokens[:, :1, :], interp], dim=1)
