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
            feats = F.interpolate(
                feats.permute(0, 2, 1),
                size=seq_len,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
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
