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
) -> torch.Tensor | None:
    """Build trajectory embedding (B, seq_len, traj_out_dim) from a training batch dict.

    Pipeline:
      traj_features (B,T,4) or traj xyz (B,T,3)
        → frame-level mask gate (+ optional T_B_04 horizon truncation)
        → frames_to_tokens  [skipped if already token-level]
        → local_traj_encoder
        → token-level mask gate
        → traj_encoder

    Returns None if x contains no trajectory fields.

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
        return None

    # --- frame-level mask (traj_cond_mask preferred) ---
    mask_frame = None
    _cond_mask = x.get("traj_cond_mask", x.get("traj_mask"))
    if _cond_mask is not None:
        mask_frame = _cond_mask.to(device=device, dtype=torch.float32)
    elif "token_mask" in x and x["token_mask"] is not None:
        tm = x["token_mask"].to(device=device, dtype=torch.float32)
        B_tm, N_tm = tm.shape
        tf = feats_frame.shape[1]
        mask_frame = tm.new_zeros(B_tm, tf)
        mask_frame[:, 0] = tm[:, 0]
        for k in range(1, N_tm):
            sf = 4 * k - 3
            ef = min(4 * k + 1, tf)
            if sf < tf:
                mask_frame[:, sf:ef] = tm[:, k:k + 1].expand(-1, ef - sf)

    if mask_frame is not None or horizon_tokens is not None:
        tf = feats_frame.shape[1]
        if mask_frame is None:
            # No base traj mask, but horizon truncation requested → start from
            # all-visible and let the horizon cutoff zero the tail.
            mask_frame = feats_frame.new_ones(feats_frame.shape[0], tf)
        elif mask_frame.shape[1] < tf:
            pad = mask_frame.new_zeros(mask_frame.shape[0], tf - mask_frame.shape[1])
            mask_frame = torch.cat([mask_frame, pad], dim=1)
        if horizon_tokens is not None:
            # T_B_04: frame-level horizon truncation (in place on mask_frame).
            from utils.training.horizon_sched import apply_horizon_mask_tokens
            apply_horizon_mask_tokens(
                mask_frame, horizon_active_end_token, horizon_tokens, frames_per_token,
            )
        # B-P0-1: if the WHOLE batch's traj mask is zero (clear / no-traj, or a
        # fully-truncated horizon), there is no valid trajectory → return None so
        # the model's no-control path runs. Otherwise the encoder bias would emit
        # a nonzero embedding (~0.2) that ControlNet treats as a constant control
        # signal — mask=0 must equal no-control, not "constant control". (Mixed
        # batches keep the per-token mask; per-sample tail truncation = Part B,
        # see _get_traj_seq_lens TODO.)
        if not bool(mask_frame[:, :tf].any()):
            return None
        feats_frame = feats_frame * mask_frame[:, :tf].unsqueeze(-1).to(dtype=feats_frame.dtype)

    # --- frame → token grouping ---
    # T_B_07: frame-level is the only supported external entry. The old
    # token-level passthrough (feeding pre-tokenized [B, seq_len, C] features and
    # skipping the local encoder) is the disabled "parallel path": a genuine
    # frame input has ~4x more frames than tokens, so shape[1] == seq_len with
    # seq_len > 1 means token-level data was mis-fed → reject.
    if feats_frame.shape[1] == seq_len and seq_len > 1:
        raise ValueError(
            "encode_traj_batch expects frame-level traj input [B, T_frame, C] "
            f"(T_frame ~= 4*seq_len), got shape[1]={feats_frame.shape[1]} == "
            f"seq_len={seq_len}; the token-level parallel path is disabled."
        )
    feats_4 = frames_to_tokens(feats_frame, seq_len)  # (B, seq_len, 4, C)
    feats_tok = local_traj_encoder(feats_4)           # (B, seq_len, C)

    # --- token-level mask gate ---
    if "token_mask" in x and x["token_mask"] is not None:
        tm = x["token_mask"].to(device=device, dtype=torch.float32)
        if tm.shape[1] < seq_len:
            pad = tm.new_zeros(tm.shape[0], seq_len - tm.shape[1])
            tm = torch.cat([tm, pad], dim=1)
        feats_tok = feats_tok * tm[:, :seq_len].unsqueeze(-1).to(dtype=feats_tok.dtype)

    return traj_encoder(feats_tok)


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
