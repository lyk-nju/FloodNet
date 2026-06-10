"""Trajectory batch utilities: path-heading features [x, z, cos, sin] for DiffForcing/WanModel."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_PATH_HEADING_EPS = 1e-8


def smooth_root_xz(root_xz: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian-smooth the xz root trajectory to reduce high-frequency jitter.

    Args:
        root_xz: (T, 2) array of [x, z] root positions.
        sigma:   Gaussian sigma in frames. 0.0 disables smoothing (returns copy).

    Returns:
        Smoothed (T, 2) float32 array.  The original array is not modified.
    """
    if sigma <= 0.0:
        return root_xz.astype(np.float32)
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(root_xz.astype(np.float64), sigma=sigma, axis=0).astype(np.float32)


def root_to_traj_feats(traj_xyz, eps: float = _PATH_HEADING_EPS):
    """Convert root trajectory xyz to path-heading features [x, z, cos(psi), sin(psi)].

    Accepts either:
      - numpy ndarray (T, 3)  → returns numpy (T, 4)      [dataset / web demo]
      - torch Tensor (B, T, 3) → returns torch (B, T, 4)  [model forward / streaming]

    psi is the xz path heading angle derived from frame-to-frame displacement.
    """
    if isinstance(traj_xyz, np.ndarray):
        arr = np.asarray(traj_xyz, dtype=np.float64)
        t_len = arr.shape[0]
        x, z = arr[:, 0:1], arr[:, 2:3]
        if t_len == 1:
            return np.concatenate(
                [x, z, np.ones((1, 1), dtype=np.float64), np.zeros((1, 1), dtype=np.float64)],
                axis=-1,
            ).astype(np.float32)
        dx = np.empty_like(x)
        dz = np.empty_like(z)
        dx[0:1] = x[1:2] - x[0:1]
        dz[0:1] = z[1:2] - z[0:1]
        dx[1:] = x[1:] - x[:-1]
        dz[1:] = z[1:] - z[:-1]
        sq = dx * dx + dz * dz
        short = sq < eps * eps
        norm = np.sqrt(np.maximum(sq, eps * eps))
        cos_yaw = np.where(short, 1.0, dx / norm)
        sin_yaw = np.where(short, 0.0, dz / norm)
        return np.concatenate([x, z, cos_yaw, sin_yaw], axis=-1).astype(np.float32)
    else:
        # torch Tensor path — (B, T, 3)
        x_coord = traj_xyz[..., 0:1]
        z_coord = traj_xyz[..., 2:3]
        t = x_coord.shape[-2]
        if t == 1:
            return torch.cat(
                [x_coord, z_coord, torch.ones_like(x_coord), torch.zeros_like(z_coord)],
                dim=-1,
            )
        dx = torch.empty_like(x_coord)
        dz = torch.empty_like(z_coord)
        dx[..., 0:1, :] = x_coord[..., 1:2, :] - x_coord[..., 0:1, :]
        dz[..., 0:1, :] = z_coord[..., 1:2, :] - z_coord[..., 0:1, :]
        dx[..., 1:, :] = x_coord[..., 1:, :] - x_coord[..., :-1, :]
        dz[..., 1:, :] = z_coord[..., 1:, :] - z_coord[..., :-1, :]
        sq = dx * dx + dz * dz
        short = sq < eps * eps
        norm = sq.sqrt().clamp(min=eps)
        cos_yaw = torch.where(short, torch.ones_like(dx), dx / norm)
        sin_yaw = torch.where(short, torch.zeros_like(dz), dz / norm)
        return torch.cat([x_coord, z_coord, cos_yaw, sin_yaw], dim=-1)


def frames_to_tokens(feats_frame: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Group frame-level features into per-token 4-frame windows (causal VAE convention).

    Causal VAE: N tokens ↔ 4*(N-1)+1 frames.
      token 0 → frame 0 (padded to 4 copies)
      token k (k≥1) → frames [4k-3, 4k]

    Args:
        feats_frame: (B, T_frame, C) — padded/truncated internally to exact causal length.
        seq_len:     number of output tokens N.
    Returns:
        (B, N, 4, C)
    """
    B, T_frame, C = feats_frame.shape
    total_causal = 4 * (seq_len - 1) + 1 if seq_len > 1 else 1
    if T_frame < total_causal:
        pad = feats_frame.new_zeros(B, total_causal - T_frame, C)
        feats_frame = torch.cat([feats_frame, pad], dim=1)
    feats_frame = feats_frame[:, :total_causal, :]
    tok0 = feats_frame[:, 0:1, :].unsqueeze(2).expand(-1, -1, 4, -1)  # (B, 1, 4, C)
    if seq_len > 1:
        rest = feats_frame[:, 1:, :].reshape(B, seq_len - 1, 4, C)    # (B, N-1, 4, C)
        return torch.cat([tok0, rest], dim=1)                           # (B, N, 4, C)
    return tok0                                                         # (B, 1, 4, C)


def _per_sample_start_tokens(start_token_idx, batch_size: int) -> list[int] | None:
    if torch.is_tensor(start_token_idx):
        if start_token_idx.numel() == 1:
            return None
        values = start_token_idx.detach().cpu().view(-1).tolist()
    elif isinstance(start_token_idx, (list, tuple)):
        if len(start_token_idx) == 1:
            return None
        values = list(start_token_idx)
    else:
        return None
    if len(values) != batch_size:
        raise ValueError(
            "per-sample traj_start_token must have one value per batch item; "
            f"got {len(values)} values for batch size {batch_size}"
        )
    out = [int(v) for v in values]
    if any(v < 0 for v in out):
        raise ValueError(f"traj_start_token must be >= 0, got {out}")
    return out


def frames_to_tokens_range(
    feats_frame: torch.Tensor,
    start_token_idx: int | torch.Tensor,
    num_tokens: int,
    *,
    frames_per_token: int = 4,
) -> torch.Tensor:
    """Group frame-level features for an arbitrary token range.

    ``frames_to_tokens`` is the causal-prefix helper for ``[0, N)``. For a
    rolling range ``[S, S + N)`` where ``S > 0``, every local token covers four
    effective frames. ``feats_frame`` is expected to be window-relative: local
    frame 0 corresponds to ``token_start_frame(S)``.
    """
    from utils.token_frame import token_end_frame, token_start_frame

    per_sample_starts = _per_sample_start_tokens(start_token_idx, feats_frame.shape[0])
    if per_sample_starts is not None:
        return torch.cat(
            [
                frames_to_tokens_range(
                    feats_frame[b : b + 1],
                    start,
                    num_tokens,
                    frames_per_token=frames_per_token,
                )
                for b, start in enumerate(per_sample_starts)
            ],
            dim=0,
        )

    if torch.is_tensor(start_token_idx):
        start_token_idx = int(start_token_idx.item())
    elif isinstance(start_token_idx, (list, tuple)):
        start_token_idx = int(start_token_idx[0])

    if start_token_idx < 0:
        raise ValueError(f"start_token_idx must be >= 0, got {start_token_idx}")
    if num_tokens <= 0:
        B, _, C = feats_frame.shape
        return feats_frame.new_zeros(B, 0, frames_per_token, C)
    if start_token_idx == 0:
        return frames_to_tokens(feats_frame, num_tokens)

    B, T_frame, C = feats_frame.shape
    origin_frame = token_start_frame(start_token_idx, frames_per_token)
    groups = []
    for i in range(num_tokens):
        token_idx = start_token_idx + i
        rel_s = token_start_frame(token_idx, frames_per_token) - origin_frame
        rel_e = token_end_frame(token_idx, frames_per_token) + 1 - origin_frame
        if rel_s >= T_frame:
            chunk = feats_frame.new_zeros(B, 0, C)
        else:
            chunk = feats_frame[:, rel_s:min(rel_e, T_frame), :]
        if chunk.shape[1] < frames_per_token:
            pad = feats_frame.new_zeros(B, frames_per_token - chunk.shape[1], C)
            chunk = torch.cat([chunk, pad], dim=1)
        groups.append(chunk[:, :frames_per_token, :].unsqueeze(1))
    return torch.cat(groups, dim=1)


def frames_to_token_mask_range(
    mask_frame: torch.Tensor,
    num_tokens: int,
    *,
    start_token_idx: int | torch.Tensor = 0,
    frames_per_token: int = 4,
) -> torch.Tensor:
    """Aggregate a window-relative frame mask to token mask by OR."""
    grouped = frames_to_tokens_range(
        mask_frame.unsqueeze(-1),
        start_token_idx,
        num_tokens,
        frames_per_token=frames_per_token,
    ).squeeze(-1)
    return (grouped > 0).any(dim=-1).to(mask_frame.dtype)


def _traj_source(x: dict):
    """The traj tensor encode_traj_batch would use (source priority)."""
    for key in ("traj_features", "traj_cond", "traj"):
        if key in x and x[key] is not None:
            return x[key]
    return None


def _resolve_traj_start_token(x: dict, explicit: int | torch.Tensor | None = None):
    value = x.get("traj_start_token", 0) if explicit is None else explicit
    if torch.is_tensor(value):
        if value.numel() == 1:
            value = int(value.item())
        else:
            value = value.to(dtype=torch.long).view(-1)
            if bool((value < 0).any()):
                raise ValueError(f"traj_start_token must be >= 0, got {value.tolist()}")
            return value
    elif isinstance(value, (list, tuple)):
        if len(value) == 1:
            value = int(value[0])
        else:
            out = torch.tensor([int(v) for v in value], dtype=torch.long)
            if bool((out < 0).any()):
                raise ValueError(f"traj_start_token must be >= 0, got {out.tolist()}")
            return out
    value = int(value)
    if value < 0:
        raise ValueError(f"traj_start_token must be >= 0, got {value}")
    return value


def build_traj_frame_mask(
    x: dict,
    tf: int,
    device,
    *,
    frames_per_token: int = 4,
    traj_start_token: int | None = None,
):
    """Frame-level traj mask [B, tf] (1=valid) from traj_cond_mask/traj_mask, or a
    token_mask expanded to frame level. Padded/truncated to `tf`. None if neither
    is present. Single source for the frame-mask derivation (no second copy)."""
    from utils.token_frame import token_end_frame, token_start_frame

    _cond_mask = x.get("traj_cond_mask", x.get("traj_mask"))
    if _cond_mask is not None:
        mf = _cond_mask.to(device=device, dtype=torch.float32)
    elif x.get("token_mask") is not None:
        tm = x["token_mask"].to(device=device, dtype=torch.float32)
        B_tm, N_tm = tm.shape
        mf = tm.new_zeros(B_tm, tf)
        start_token_idx = _resolve_traj_start_token(x, traj_start_token)
        per_sample_starts = _per_sample_start_tokens(start_token_idx, B_tm)
        if per_sample_starts is None:
            if torch.is_tensor(start_token_idx):
                start_token_idx = int(start_token_idx.item())
            origin_frame = token_start_frame(start_token_idx, frames_per_token)
            for local_k in range(N_tm):
                global_k = start_token_idx + local_k
                sf = token_start_frame(global_k, frames_per_token) - origin_frame
                ef = min(token_end_frame(global_k, frames_per_token) + 1 - origin_frame, tf)
                if sf < tf:
                    sf = max(0, sf)
                    if sf < ef:
                        mf[:, sf:ef] = tm[:, local_k:local_k + 1].expand(-1, ef - sf)
        else:
            for b, start in enumerate(per_sample_starts):
                origin_frame = token_start_frame(start, frames_per_token)
                for local_k in range(N_tm):
                    global_k = start + local_k
                    sf = token_start_frame(global_k, frames_per_token) - origin_frame
                    ef = min(token_end_frame(global_k, frames_per_token) + 1 - origin_frame, tf)
                    if sf < tf:
                        sf = max(0, sf)
                        if sf < ef:
                            mf[b, sf:ef] = tm[b, local_k]
    else:
        return None
    if mf.shape[1] < tf:
        mf = torch.cat([mf, mf.new_zeros(mf.shape[0], tf - mf.shape[1])], dim=1)
    return mf[:, :tf]


def build_traj_token_mask(x: dict, seq_len: int, device, *,
                          horizon_tokens: int | None = None,
                          horizon_active_end_token=0,
                          frames_per_token: int = 4,
                          traj_start_token: int | None = None):
    """[B, seq_len] token mask (1=valid) = build_traj_frame_mask + optional horizon
    truncation → frames_to_token_mask. SINGLE source reused by encode_traj_batch
    (token-embedding zeroing) and _get_traj_seq_lens (attention truncation), so the
    two never derive the mask differently. Returns None when there is neither a
    traj mask nor a horizon."""
    src = _traj_source(x)
    if src is None:
        return None
    tf = src.shape[1]
    start_token_idx = _resolve_traj_start_token(x, traj_start_token)
    mask_frame = build_traj_frame_mask(
        x, tf, device,
        frames_per_token=frames_per_token,
        traj_start_token=start_token_idx,
    )
    if mask_frame is None and horizon_tokens is None:
        return None
    if mask_frame is None:
        mask_frame = torch.ones(src.shape[0], tf, device=device, dtype=torch.float32)
    if horizon_tokens is not None:
        _apply_horizon_mask_tokens_range(
            mask_frame,
            horizon_active_end_token,
            horizon_tokens,
            start_token_idx=start_token_idx,
            frames_per_token=frames_per_token,
        )
    return frames_to_token_mask_range(
        mask_frame,
        seq_len,
        start_token_idx=start_token_idx,
        frames_per_token=frames_per_token,
    )


def _apply_horizon_mask_tokens_range(
    mask_frame: torch.Tensor,
    active_end_token,
    horizon_tokens: int,
    *,
    start_token_idx: int | torch.Tensor,
    frames_per_token: int,
) -> torch.Tensor:
    """Zero a window-relative frame mask at global cutoff ``E + H``."""
    from utils.token_frame import token_start_frame

    per_sample_starts = _per_sample_start_tokens(start_token_idx, mask_frame.shape[0])
    active_is_batch = torch.is_tensor(active_end_token) and active_end_token.dim() > 0
    if per_sample_starts is not None or active_is_batch:
        if per_sample_starts is None:
            if torch.is_tensor(start_token_idx):
                start = int(start_token_idx.item())
            else:
                start = int(start_token_idx)
            per_sample_starts = [start] * mask_frame.shape[0]
        for b, start in enumerate(per_sample_starts):
            active_end = (
                int(active_end_token[b])
                if active_is_batch
                else int(active_end_token)
            )
            origin_frame = token_start_frame(start, frames_per_token)
            cutoff = (
                token_start_frame(active_end + horizon_tokens, frames_per_token)
                - origin_frame
            )
            if cutoff <= 0:
                mask_frame[b, :] = 0
            elif cutoff < mask_frame.shape[-1]:
                mask_frame[b, cutoff:] = 0
        return mask_frame
    if torch.is_tensor(start_token_idx):
        start_token_idx = int(start_token_idx.item())
    origin_frame = token_start_frame(start_token_idx, frames_per_token)
    cutoff = (
        token_start_frame(int(active_end_token) + horizon_tokens, frames_per_token)
        - origin_frame
    )
    if cutoff <= 0:
        mask_frame[...] = 0
    elif cutoff < mask_frame.shape[-1]:
        mask_frame[..., cutoff:] = 0
    return mask_frame


def encode_traj_batch(
    x: dict,
    seq_len: int,
    device,
    local_traj_encoder: torch.nn.Module,
    traj_encoder: torch.nn.Module,
    *,
    horizon_tokens: int | None = None,
    horizon_active_end_token: int = 0,
    frames_per_token: int = 4,
    return_token_mask: bool = False,
    traj_start_token: int | None = None,
):
    """Build trajectory embedding (B, seq_len, traj_out_dim) from a training batch dict.

    Pipeline:
      traj_features (B,T,7) or traj xyz (B,T,3)
        → frame-level mask gate (+ optional T_B_04 horizon truncation)
        → frames_to_tokens
        → LocalTrajEncoder(masked-mean over 4 frames)
        → token-level mask gate
        → TrajEncoder

    Returns:
      - traj_emb (B, seq_len, traj_out_dim) by default, or None if x has no
        trajectory fields (or the entire batch's traj mask is zero).
      - When `return_token_mask=True`, returns (traj_emb, token_mask) where
        `token_mask` is (B, seq_len) in {0, 1} and is None when there is no
        traj/horizon mask. The tuple is also returned (with traj_emb=None,
        token_mask=None) on the no-traj path so callers can unpack uniformly.

    Horizon simulation (T_B_04): when `horizon_tokens` is not None, the
    frame-level traj mask is additionally zeroed at/after the cutoff frame
    `token_start_frame(horizon_active_end_token + horizon_tokens)`, so the model
    only sees `horizon_tokens` of future plan (default active_end=0 = horizon
    measured from the clip start). `horizon_tokens=None` (the default) preserves
    the original behavior exactly.
    """
    # --- source: prioritize traj_cond paths ---
    if "traj_features" in x and x["traj_features"] is not None:
        feats_frame = x["traj_features"].to(device)
    elif "traj_cond" in x and x["traj_cond"] is not None:
        feats_frame = root_to_traj_feats(x["traj_cond"].to(device))
    elif "traj" in x and x["traj"] is not None:
        feats_frame = root_to_traj_feats(x["traj"].to(device))
    else:
        return (None, None) if return_token_mask else None

    # --- frame-level mask (single source: build_traj_frame_mask) ---
    start_token_idx = _resolve_traj_start_token(x, traj_start_token)
    tf = feats_frame.shape[1]
    mask_frame = build_traj_frame_mask(
        x, tf, device,
        frames_per_token=frames_per_token,
        traj_start_token=start_token_idx,
    )

    token_mask_from_frame = None
    if mask_frame is not None or horizon_tokens is not None:
        if mask_frame is None:
            # No base traj mask, but horizon truncation requested → start from
            # all-visible and let the horizon cutoff zero the tail.
            mask_frame = feats_frame.new_ones(feats_frame.shape[0], tf)
        if horizon_tokens is not None:
            _apply_horizon_mask_tokens_range(
                mask_frame, horizon_active_end_token, horizon_tokens,
                start_token_idx=start_token_idx,
                frames_per_token=frames_per_token,
            )
        # If the WHOLE batch's traj mask is zero (clear / no-traj, or a fully-
        # truncated horizon), there is no valid trajectory → return None so the
        # model's no-control path runs.
        if not bool(mask_frame[:, :tf].any()):
            return (None, None) if return_token_mask else None
        feats_frame = feats_frame * mask_frame[:, :tf].unsqueeze(-1).to(dtype=feats_frame.dtype)
        token_mask_from_frame = frames_to_token_mask_range(
            mask_frame[:, :tf],
            seq_len,
            start_token_idx=start_token_idx,
            frames_per_token=frames_per_token,
        )

    # --- frame → token grouping ---
    if feats_frame.shape[1] == seq_len and seq_len > 1:
        raise ValueError(
            "encode_traj_batch expects frame-level traj input [B, T_frame, C] "
            f"(T_frame ~= 4*seq_len), got shape[1]={feats_frame.shape[1]} == "
            f"seq_len={seq_len}; the token-level parallel path is disabled."
        )
    feats_4 = frames_to_tokens_range(
        feats_frame,
        start_token_idx,
        seq_len,
        frames_per_token=frames_per_token,
    )                                                           # (B, seq_len, 4, C)
    # Build a 4-frame mask (B, seq_len, 4) so LocalTrajEncoder can do masked-mean
    # pool — otherwise zero-padded frames in a partial token dilute the mean by
    # 1/valid_count.
    if mask_frame is not None:
        mf_grouped = frames_to_tokens_range(
            mask_frame[:, :tf].unsqueeze(-1).to(feats_frame.dtype),
            start_token_idx,
            seq_len,
            frames_per_token=frames_per_token,
        )
        frame_mask_4 = mf_grouped.squeeze(-1)                    # (B, seq_len, 4)
    else:
        frame_mask_4 = None
    feats_tok = local_traj_encoder(feats_4, frame_mask=frame_mask_4)

    # --- token-level mask gate ---
    # `token_mask` (when present) gates fully-invalid tokens; combine with the
    # frame-derived token mask so downstream callers see a single token-level
    # validity signal.
    combined_token_mask = token_mask_from_frame
    if "token_mask" in x and x["token_mask"] is not None:
        tm = x["token_mask"].to(device=device, dtype=torch.float32)
        if tm.shape[1] < seq_len:
            pad = tm.new_zeros(tm.shape[0], seq_len - tm.shape[1])
            tm = torch.cat([tm, pad], dim=1)
        tm = tm[:, :seq_len]
        feats_tok = feats_tok * tm.unsqueeze(-1).to(dtype=feats_tok.dtype)
        combined_token_mask = (
            tm if combined_token_mask is None else (combined_token_mask * tm)
        )

    traj_emb = traj_encoder(feats_tok)
    # Re-zero by the COMBINED token mask (frame-derived ∧ x["token_mask"]), not
    # just token_mask_from_frame: TrajEncoder's LayerNorm+bias maps a zeroed-input
    # token to a NON-zero embedding, so a token invalid only via x["token_mask"]
    # (when a separate frame cond_mask marks it valid) would otherwise leak.
    if combined_token_mask is not None:
        traj_emb = traj_emb * combined_token_mask[..., None].to(traj_emb.dtype)

    if return_token_mask:
        return traj_emb, combined_token_mask
    return traj_emb


def build_traj_emb(
    x: dict,
    seq_len: int,
    device: torch.device,
    traj_encoder: torch.nn.Module | None,
    use_traj_cond: bool,
    traj_drop_out: float,
    training_dropout: bool,
) -> torch.Tensor | None:
    """Build trajectory embedding (B, T, traj_enc_dim) for WanModel.forward(traj_emb=...).

    Returns None when trajectory is absent or dropped out.
    Prefers traj_features (precomputed 4D); falls back to traj xyz converted on the fly.
    """
    if not use_traj_cond or traj_encoder is None:
        return None
    if training_dropout and np.random.rand() <= traj_drop_out:
        return None

    def align_temporal(feats: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if feats.shape[1] != seq_len:
            t_frames = feats.shape[1]
            # Causal VAE convention: token 0 → frame 0; token k (k≥1) → frames [4k-3, 4k],
            # representative frame = 4k (last frame of the chunk).
            # Use exact frame indices instead of uniform interpolation to avoid ~1.5-frame bias.
            indices = torch.zeros(seq_len, dtype=torch.long, device=feats.device)
            indices[0] = 0
            for k in range(1, seq_len):
                indices[k] = min(4 * k, t_frames - 1)
            feats = feats[:, indices, :]
        if mask is not None:
            m = mask.to(device=device, dtype=torch.float32)
            if m.shape[1] != seq_len:
                m = F.interpolate(
                    m.unsqueeze(1), size=seq_len, mode="nearest"
                ).squeeze(1)
            feats = feats * m.unsqueeze(-1).to(dtype=feats.dtype)
        return feats

    if "traj_features" in x and x["traj_features"] is not None:
        feats = x["traj_features"].to(device)
        mask = x.get("token_mask")
        feats = align_temporal(feats, mask)
    elif "traj" in x:
        traj = x["traj"].to(device)
        mask = x.get("traj_mask")
        # Compute heading from original xyz FIRST, then interpolate the 4D features.
        # Interpolating xyz before heading distorts displacement direction and turn angles.
        feats = root_to_traj_feats(traj)       # (B, T_orig, 4)
        feats = align_temporal(feats, mask)    # → (B, seq_len, 4)
    else:
        return None

    return traj_encoder(feats)
