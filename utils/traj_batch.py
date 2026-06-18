"""Trajectory conditioning batch helpers.

This module converts frame-level root trajectory features into causal-VAE token
windows and masks. The legacy 4D path-heading path is kept for older WanModel
callers; physical 7D root features live in `utils.motion_process`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from utils.token_frame import (
    FRAMES_PER_TOKEN_DEFAULT,
    num_frames_for_tokens,
    prefix_len_from_tail_invalid,
    token_end_frame,
    token_start_frame,
)

_PATH_HEADING_EPS = 1e-8


def smooth_root_xz(root_xz: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian-smooth root XZ positions; `sigma <= 0` returns a float32 copy."""
    if sigma <= 0.0:
        return root_xz.astype(np.float32)
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(root_xz.astype(np.float64), sigma=sigma, axis=0).astype(
        np.float32
    )


def root_to_traj_feats(traj_xyz, eps: float = _PATH_HEADING_EPS):
    """Convert root xyz to legacy path-heading features `[x, z, cos, sin]`."""
    if isinstance(traj_xyz, np.ndarray):
        arr = np.asarray(traj_xyz, dtype=np.float64)
        num_frames = arr.shape[0]
        x, z = arr[:, 0:1], arr[:, 2:3]
        if num_frames == 1:
            heading = [
                np.ones((1, 1), dtype=np.float64),
                np.zeros((1, 1), dtype=np.float64),
            ]
            return np.concatenate(
                [x, z, *heading],
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
    x_coord = traj_xyz[..., 0:1]
    z_coord = traj_xyz[..., 2:3]
    num_frames = x_coord.shape[-2]
    if num_frames == 1:
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
    """Group frame-level features into causal-VAE token windows `[B, N, 4, C]`."""
    batch_size, num_input_frames, channels = feats_frame.shape
    total_causal = num_frames_for_tokens(seq_len)
    if num_input_frames < total_causal:
        pad = feats_frame.new_zeros(
            batch_size,
            total_causal - num_input_frames,
            channels,
        )
        feats_frame = torch.cat([feats_frame, pad], dim=1)
    feats_frame = feats_frame[:, :total_causal, :]
    token0 = (
        feats_frame[:, 0:1, :]
        .unsqueeze(2)
        .expand(-1, -1, FRAMES_PER_TOKEN_DEFAULT, -1)
    )
    if seq_len > 1:
        rest = feats_frame[:, 1:, :].reshape(
            batch_size,
            seq_len - 1,
            FRAMES_PER_TOKEN_DEFAULT,
            channels,
        )
        return torch.cat([token0, rest], dim=1)
    return token0


def _single_start_token(start_token_idx) -> int:
    if torch.is_tensor(start_token_idx):
        start_token_idx = int(start_token_idx.item())
    elif isinstance(start_token_idx, (list, tuple)):
        start_token_idx = int(start_token_idx[0])
    else:
        start_token_idx = int(start_token_idx)
    if start_token_idx < 0:
        raise ValueError(f"start_token_idx must be >= 0, got {start_token_idx}")
    return start_token_idx


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
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> torch.Tensor:
    """Group window-relative frame features for token range `[start, start + N)`."""
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

    start_token_idx = _single_start_token(start_token_idx)
    if num_tokens <= 0:
        batch_size, _, channels = feats_frame.shape
        return feats_frame.new_zeros(batch_size, 0, frames_per_token, channels)
    if start_token_idx == 0:
        return frames_to_tokens(feats_frame, num_tokens)

    batch_size, num_input_frames, channels = feats_frame.shape
    origin_frame = token_start_frame(start_token_idx, frames_per_token)
    groups = []
    for i in range(num_tokens):
        token_idx = start_token_idx + i
        rel_start = token_start_frame(token_idx, frames_per_token) - origin_frame
        rel_stop = token_end_frame(token_idx, frames_per_token) + 1 - origin_frame
        if rel_start >= num_input_frames:
            chunk = feats_frame.new_zeros(batch_size, 0, channels)
        else:
            chunk = feats_frame[:, rel_start:min(rel_stop, num_input_frames), :]
        if chunk.shape[1] < frames_per_token:
            pad = feats_frame.new_zeros(
                batch_size,
                frames_per_token - chunk.shape[1],
                channels,
            )
            chunk = torch.cat([chunk, pad], dim=1)
        groups.append(chunk[:, :frames_per_token, :].unsqueeze(1))
    return torch.cat(groups, dim=1)


def frames_to_token_mask_range(
    mask_frame: torch.Tensor,
    num_tokens: int,
    *,
    start_token_idx: int | torch.Tensor = 0,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
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
    """Return the frame-level trajectory source, following encoder priority."""
    for key in ("traj_features", "traj_cond", "traj"):
        if key in x and x[key] is not None:
            return x[key]
    return None


def _resolve_traj_start_token(
    x: dict,
    explicit: int | torch.Tensor | None = None,
) -> int | torch.Tensor:
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


def _copy_token_mask_to_frame_mask(
    frame_mask: torch.Tensor,
    token_mask: torch.Tensor,
    start_token_idx: int,
    frames_per_token: int,
    *,
    batch_index: int | None = None,
) -> None:
    """Copy token validity into the frame spans covered by each token."""
    origin_frame = token_start_frame(start_token_idx, frames_per_token)
    _, num_tokens = token_mask.shape
    total_frames = frame_mask.shape[-1]
    for local_token_idx in range(num_tokens):
        global_token_idx = start_token_idx + local_token_idx
        start = token_start_frame(global_token_idx, frames_per_token) - origin_frame
        stop = min(
            token_end_frame(global_token_idx, frames_per_token) + 1 - origin_frame,
            total_frames,
        )
        if start >= total_frames:
            continue
        start = max(0, start)
        if start >= stop:
            continue
        if batch_index is None:
            value = token_mask[:, local_token_idx:local_token_idx + 1].expand(
                -1,
                stop - start,
            )
            frame_mask[:, start:stop] = value
        else:
            frame_mask[batch_index, start:stop] = token_mask[
                batch_index,
                local_token_idx,
            ]


def build_traj_frame_mask(
    x: dict,
    num_frames: int,
    device,
    *,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
    traj_start_token: int | None = None,
):
    """Build frame-level trajectory mask `[B, num_frames]`.

    Prefer explicit frame masks (`traj_cond_mask` / `traj_mask`). If unavailable,
    expand `token_mask` through the causal token/frame layout.
    """
    cond_mask = x.get("traj_cond_mask", x.get("traj_mask"))
    if cond_mask is not None:
        frame_mask = cond_mask.to(device=device, dtype=torch.float32)
    elif x.get("token_mask") is not None:
        token_mask = x["token_mask"].to(device=device, dtype=torch.float32)
        batch_size, _ = token_mask.shape
        frame_mask = token_mask.new_zeros(batch_size, num_frames)
        start_token_idx = _resolve_traj_start_token(x, traj_start_token)
        per_sample_starts = _per_sample_start_tokens(start_token_idx, batch_size)
        if per_sample_starts is None:
            start_token_idx = _single_start_token(start_token_idx)
            _copy_token_mask_to_frame_mask(
                frame_mask,
                token_mask,
                start_token_idx,
                frames_per_token,
            )
        else:
            for batch_idx, start in enumerate(per_sample_starts):
                _copy_token_mask_to_frame_mask(
                    frame_mask,
                    token_mask,
                    start,
                    frames_per_token,
                    batch_index=batch_idx,
                )
    else:
        return None
    if frame_mask.shape[1] < num_frames:
        frame_mask = torch.cat(
            [
                frame_mask,
                frame_mask.new_zeros(
                    frame_mask.shape[0],
                    num_frames - frame_mask.shape[1],
                ),
            ],
            dim=1,
        )
    return frame_mask[:, :num_frames]


def build_traj_token_mask(
    x: dict,
    seq_len: int,
    device,
    *,
    horizon_tokens: int | torch.Tensor | None = None,
    horizon_active_end_token=0,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
    traj_start_token: int | None = None,
):
    """Build [B, seq_len] token mask from frame masks and optional horizon."""
    source = _traj_source(x)
    if source is None:
        return None
    num_source_frames = source.shape[1]
    start_token_idx = _resolve_traj_start_token(x, traj_start_token)
    mask_frame = build_traj_frame_mask(
        x,
        num_source_frames,
        device,
        frames_per_token=frames_per_token,
        traj_start_token=start_token_idx,
    )
    if mask_frame is None and horizon_tokens is None:
        return None
    if mask_frame is None:
        mask_frame = torch.ones(
            source.shape[0],
            num_source_frames,
            device=device,
            dtype=torch.float32,
        )
    if horizon_tokens is not None:
        _truncate_frame_mask_to_token_horizon(
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


def get_traj_seq_lens(
    batch: dict,
    seq_len: int,
    device,
    *,
    horizon_tokens=None,
    horizon_active_end=0,
):
    """Infer valid trajectory token lengths from explicit lengths and masks."""
    batch_size = _infer_batch_size(batch)
    base = _length_tensor(batch.get("traj_num_tokens"), batch_size, seq_len, device)
    if base is None:
        base = _length_tensor(
            batch.get("traj_features_length"), batch_size, seq_len, device
        )
    if base is None and batch.get("traj_length") is not None:
        traj_len = batch["traj_length"].to(device=device, dtype=torch.long)
        tokens = torch.where(
            traj_len <= 1,
            traj_len.clamp(min=0, max=1),
            (traj_len - 2) // FRAMES_PER_TOKEN_DEFAULT + 2,
        )
        base = tokens.clamp(min=0, max=seq_len)
    if base is None:
        base = _length_tensor(batch.get("feature_length"), batch_size, seq_len, device)
    if base is None:
        return None

    token_mask = build_traj_token_mask(
        batch,
        seq_len,
        device,
        horizon_tokens=horizon_tokens,
        horizon_active_end_token=horizon_active_end,
    )
    if token_mask is None:
        return base
    prefix = prefix_len_from_tail_invalid(token_mask).to(device=device)
    return torch.minimum(base, prefix)


def _infer_batch_size(batch: dict) -> int | None:
    for key in ("traj_features", "traj_cond_7d_frame", "traj_cond", "traj", "feature"):
        value = batch.get(key)
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    value = batch.get("feature_length")
    if torch.is_tensor(value) and value.ndim > 0:
        return int(value.numel())
    return None


def _length_tensor(value, batch_size, seq_len: int, device):
    if value is None:
        return None
    if torch.is_tensor(value):
        out = value.to(device=device, dtype=torch.long)
    else:
        out = torch.tensor([int(value)], device=device, dtype=torch.long)
    if out.ndim == 0:
        out = out.view(1)
    if batch_size is not None and out.numel() == 1 and batch_size > 1:
        out = out.expand(batch_size)
    return out.clamp(min=0, max=seq_len)


def _truncate_frame_mask_to_token_horizon(
    mask_frame: torch.Tensor,
    active_end_token,
    horizon_tokens: int | torch.Tensor,
    *,
    start_token_idx: int | torch.Tensor,
    frames_per_token: int,
) -> torch.Tensor:
    """Zero frame-mask entries at and after the token horizon cutoff."""
    per_sample_starts = _per_sample_start_tokens(start_token_idx, mask_frame.shape[0])
    active_is_batch = torch.is_tensor(active_end_token) and active_end_token.dim() > 0
    horizon_is_batch = torch.is_tensor(horizon_tokens) and horizon_tokens.dim() > 0
    if per_sample_starts is not None or active_is_batch or horizon_is_batch:
        if per_sample_starts is None:
            if torch.is_tensor(start_token_idx):
                start = int(start_token_idx.item())
            else:
                start = int(start_token_idx)
            per_sample_starts = [start] * mask_frame.shape[0]
        for batch_idx, start in enumerate(per_sample_starts):
            active_end = (
                int(active_end_token[batch_idx])
                if active_is_batch
                else int(active_end_token)
            )
            horizon = (
                int(horizon_tokens[batch_idx])
                if horizon_is_batch
                else int(horizon_tokens)
            )
            origin_frame = token_start_frame(start, frames_per_token)
            cutoff = (
                token_start_frame(active_end + horizon, frames_per_token)
                - origin_frame
            )
            if cutoff <= 0:
                mask_frame[batch_idx, :] = 0
            elif cutoff < mask_frame.shape[-1]:
                mask_frame[batch_idx, cutoff:] = 0
        return mask_frame
    start_token_idx = _single_start_token(start_token_idx)
    origin_frame = token_start_frame(start_token_idx, frames_per_token)
    cutoff = (
        token_start_frame(int(active_end_token) + int(horizon_tokens), frames_per_token)
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
    traj_encoder: torch.nn.Module,
    *,
    horizon_tokens: int | torch.Tensor | None = None,
    horizon_active_end_token: int = 0,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
    return_token_mask: bool = False,
    traj_start_token: int | None = None,
):
    """Encode frame-level trajectory conditioning to token embeddings."""
    if "traj_features" in x and x["traj_features"] is not None:
        feats_frame = x["traj_features"].to(device)
    elif "traj_cond" in x and x["traj_cond"] is not None:
        feats_frame = root_to_traj_feats(x["traj_cond"].to(device))
    elif "traj" in x and x["traj"] is not None:
        feats_frame = root_to_traj_feats(x["traj"].to(device))
    else:
        return (None, None) if return_token_mask else None

    start_token_idx = _resolve_traj_start_token(x, traj_start_token)
    num_source_frames = feats_frame.shape[1]
    mask_frame = build_traj_frame_mask(
        x,
        num_source_frames,
        device,
        frames_per_token=frames_per_token,
        traj_start_token=start_token_idx,
    )

    token_mask_from_frame = None
    if mask_frame is not None or horizon_tokens is not None:
        if mask_frame is None:
            mask_frame = feats_frame.new_ones(
                feats_frame.shape[0],
                num_source_frames,
            )
        if horizon_tokens is not None:
            _truncate_frame_mask_to_token_horizon(
                mask_frame,
                horizon_active_end_token,
                horizon_tokens,
                start_token_idx=start_token_idx,
                frames_per_token=frames_per_token,
            )
        if not bool(mask_frame[:, :num_source_frames].any()):
            return (None, None) if return_token_mask else None
        visible_mask = mask_frame[:, :num_source_frames].unsqueeze(-1)
        feats_frame = feats_frame * visible_mask.to(dtype=feats_frame.dtype)
        token_mask_from_frame = frames_to_token_mask_range(
            mask_frame[:, :num_source_frames],
            seq_len,
            start_token_idx=start_token_idx,
            frames_per_token=frames_per_token,
        )

    if feats_frame.shape[1] == seq_len and seq_len > 1:
        raise ValueError(
            "encode_traj_batch expects frame-level traj input [B, num_frames, C] "
            f"(not token-level input), got shape[1]={feats_frame.shape[1]} == "
            f"seq_len={seq_len}; the token-level parallel path is disabled."
        )
    token_frames = frames_to_tokens_range(
        feats_frame,
        start_token_idx,
        seq_len,
        frames_per_token=frames_per_token,
    )
    if mask_frame is not None:
        grouped_frame_mask = frames_to_tokens_range(
            mask_frame[:, :num_source_frames].unsqueeze(-1).to(feats_frame.dtype),
            start_token_idx,
            seq_len,
            frames_per_token=frames_per_token,
        )
        frame_mask_4 = grouped_frame_mask.squeeze(-1)
    else:
        frame_mask_4 = None
    traj_emb = traj_encoder(token_frames, frame_mask=frame_mask_4)

    combined_token_mask = token_mask_from_frame
    if "token_mask" in x and x["token_mask"] is not None:
        explicit_token_mask = x["token_mask"].to(device=device, dtype=torch.float32)
        if explicit_token_mask.shape[1] < seq_len:
            pad = explicit_token_mask.new_zeros(
                explicit_token_mask.shape[0],
                seq_len - explicit_token_mask.shape[1],
            )
            explicit_token_mask = torch.cat([explicit_token_mask, pad], dim=1)
        explicit_token_mask = explicit_token_mask[:, :seq_len]
        combined_token_mask = (
            explicit_token_mask
            if combined_token_mask is None
            else (combined_token_mask * explicit_token_mask)
        )

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
    """Build legacy trajectory embedding for WanModel.forward(traj_emb=...)."""
    if not use_traj_cond or traj_encoder is None:
        return None
    if training_dropout and np.random.rand() <= traj_drop_out:
        return None

    def align_temporal(feats: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if feats.shape[1] != seq_len:
            num_input_frames = feats.shape[1]
            indices = torch.zeros(seq_len, dtype=torch.long, device=feats.device)
            indices[0] = 0
            for token_idx in range(1, seq_len):
                indices[token_idx] = min(
                    token_end_frame(token_idx),
                    num_input_frames - 1,
                )
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
        feats = root_to_traj_feats(traj)
        feats = align_temporal(feats, mask)
    else:
        return None

    return traj_encoder(feats)
