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
        batch_size: int,
        buf_len: int,
        local_traj_encoder: torch.nn.Module,
        traj_encoder: torch.nn.Module,
        use_emb_cache: bool = True,
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

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self):
        self._feat_buf = None
        self._xyz_buf = None
        self._mask_buf = None
        self._version = 0
        self._emb_cache = {}

    def update(self, x: dict, commit_index: int, device):
        """Write traj data from batch dict x into the buffer at commit_index.

        If x contains no traj fields, the buffer is cleared (stops conditioning).
        Increments the internal version to invalidate the embedding cache.
        """
        if commit_index >= self.buf_len:
            return

        no_traj = (
            ("traj" not in x or x["traj"] is None)
            and ("traj_features" not in x or x["traj_features"] is None)
        )
        if no_traj:
            if self._feat_buf is not None or self._xyz_buf is not None:
                self.reset()
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
                self._feat_buf[:, commit_index:] = 0
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
                self._xyz_buf[:, commit_index:] = 0
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
        self._mask_buf[:, commit_index:] = 0

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
