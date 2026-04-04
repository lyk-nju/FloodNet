"""Trajectory batch utilities: path-heading features [x, z, cos, sin] for DiffForcing/WanModel."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_PATH_HEADING_EPS = 1e-8


def path_heading_features_from_root_xyz(
    traj_xyz: np.ndarray, eps: float = _PATH_HEADING_EPS
) -> np.ndarray:
    """Convert root trajectory (T, 3) xyz to (T, 4) path-heading features [x, z, cos(psi), sin(psi)].

    psi is the xz path heading (normalised displacement), consistent with xyz_traj_to_features_4d.
    Used for dataset traj_features and inference trajectory conditioning.
    """
    traj_xyz = np.asarray(traj_xyz, dtype=np.float64)
    t_len = traj_xyz.shape[0]
    x = traj_xyz[:, 0:1]
    z = traj_xyz[:, 2:3]
    if t_len == 1:
        cos = np.ones((1, 1), dtype=np.float64)
        sin_p = np.zeros((1, 1), dtype=np.float64)
        return np.concatenate([x, z, cos, sin_p], axis=-1).astype(np.float32)

    dx = np.zeros((t_len, 1), dtype=np.float64)
    dz = np.zeros((t_len, 1), dtype=np.float64)
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


def xyz_traj_to_features_4d(
    traj_xyz: torch.Tensor, eps: float = _PATH_HEADING_EPS
) -> torch.Tensor:
    """Convert (B, T, 3) xyz trajectory to (B, T, 4) path-heading features [x, z, cos(psi), sin(psi)].

    Semantically identical to path_heading_features_from_root_xyz (numpy counterpart).
    """
    x_coord = traj_xyz[..., 0:1]
    z_coord = traj_xyz[..., 2:3]
    _, t, _ = x_coord.shape
    if t == 1:
        cos_yaw = torch.ones_like(x_coord)
        sin_yaw = torch.zeros_like(z_coord)
        return torch.cat([x_coord, z_coord, cos_yaw, sin_yaw], dim=-1)

    dx = torch.zeros_like(x_coord)
    dz = torch.zeros_like(z_coord)
    dx[:, 0:1] = x_coord[:, 1:2] - x_coord[:, 0:1]
    dz[:, 0:1] = z_coord[:, 1:2] - z_coord[:, 0:1]
    dx[:, 1:] = x_coord[:, 1:] - x_coord[:, :-1]
    dz[:, 1:] = z_coord[:, 1:] - z_coord[:, :-1]

    sq = dx * dx + dz * dz
    short = sq < eps * eps
    norm = sq.sqrt().clamp(min=eps)
    cos_yaw = torch.where(short, torch.ones_like(dx), dx / norm)
    sin_yaw = torch.where(short, torch.zeros_like(dz), dz / norm)
    return torch.cat([x_coord, z_coord, cos_yaw, sin_yaw], dim=-1)


def build_traj_emb_from_batch(
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
        traj = align_temporal(traj, mask)
        feats = xyz_traj_to_features_4d(traj)
    else:
        return None

    return traj_encoder(feats)
