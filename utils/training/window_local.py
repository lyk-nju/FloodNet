"""Window-local LDF training helpers.

These helpers build the limited-history training batch. ``precomputed_slice``
uses the existing latent token prefix and anchors full raw 263D trajectory
slices to the sampled window origin. ``online_encode`` clips raw 263D motion,
treats that clip as a new local prefix, and encodes it with the VAE so the
local latent count and local 7D target share the same time contract.
"""

from __future__ import annotations

import torch

from utils.local_frame import canonicalize_7d
from utils.motion_process import recover_root_rot_pos, root_to_traj_feats_7d
from utils.token_frame import (
    num_frames_for_tokens,
    token_range_to_frame_slice,
    token_start_frame,
)
from utils.training.sample_creator import SampleCreator

RAW_HUMANML3D_MOTION_DIM = 263


def _as_long_1d(value, *, batch_size: int, device, name: str) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.to(device=device, dtype=torch.long).view(-1)
    else:
        out = torch.as_tensor(value, device=device, dtype=torch.long).view(-1)
    if out.numel() == 1 and batch_size > 1:
        out = out.expand(batch_size)
    if out.numel() != batch_size:
        raise ValueError(
            f"{name} must be scalar or length {batch_size}; got shape {tuple(out.shape)}"
        )
    return out


def build_window_local_traj_batch(
    *,
    raw_feature_263: torch.Tensor,
    raw_feature_length,
    start_tokens,
    num_tokens,
    frames_per_token: int = 4,
    local_prefix: bool = False,
) -> dict:
    """Build window-local 7D trajectory conditioning from raw 263D motion.

    Args:
        raw_feature_263: ``[B, T, 263]`` raw HumanML3D motion features.
        raw_feature_length: valid raw frame length per sample.
        start_tokens: global token-space window anchor ``S``; scalar or ``[B]``.
        num_tokens: number of trajectory tokens to cover from ``S``; scalar or
            ``[B]``. For window-local training this is usually
            ``latent_valid_len + horizon_tokens``.
        frames_per_token: causal VAE token/frame ratio.

    Returns:
        A model-batch fragment containing frame-level window-local 7D trajectory,
        frame mask, valid frame length, explicit token length, and global start
        token. Tensors are padded to the largest expected frame length in the
        batch; unavailable future tail frames are masked false.
    """
    if (
        raw_feature_263.ndim != 3
        or raw_feature_263.shape[-1] != RAW_HUMANML3D_MOTION_DIM
    ):
        raise ValueError(
            "window-local trajectory requires raw 263D HumanML3D motion; got "
            f"{tuple(raw_feature_263.shape)}"
        )
    device = raw_feature_263.device
    batch_size = int(raw_feature_263.shape[0])
    raw_lengths = _as_long_1d(
        raw_feature_length, batch_size=batch_size, device=device, name="raw_feature_length"
    )
    starts = _as_long_1d(
        start_tokens, batch_size=batch_size, device=device, name="start_tokens"
    )
    counts = _as_long_1d(
        num_tokens, batch_size=batch_size, device=device, name="num_tokens"
    )
    max_raw_frames = int(raw_feature_263.shape[1])
    if bool((raw_lengths < 0).any()) or bool((raw_lengths > max_raw_frames).any()):
        raise ValueError(
            "raw_feature_length must be within the raw_feature_263 tensor frame "
            f"range [0, {max_raw_frames}]; got {raw_lengths.tolist()}"
        )
    if bool((starts < 0).any()):
        raise ValueError(f"start_tokens must be >= 0, got {starts.tolist()}")
    if bool((counts <= 0).any()):
        raise ValueError(f"num_tokens must be > 0, got {counts.tolist()}")

    expected_lengths: list[int] = []
    available_lengths: list[int] = []
    traj_windows: list[torch.Tensor] = []
    for b in range(batch_size):
        start = int(starts[b].item())
        count = int(counts[b].item())
        raw_len = int(raw_lengths[b].item())
        origin_frame = token_start_frame(start, frames_per_token)
        if origin_frame >= raw_len:
            raise ValueError(
                "window-local trajectory requires a valid origin; "
                f"sample={b}, start_token={start}, origin_frame={origin_frame}, "
                f"raw_feature_length={raw_len}"
            )
        if local_prefix:
            expected_len = num_frames_for_tokens(count, frames_per_token)
            available_stop = min(origin_frame + expected_len, raw_len)
            available_len = max(0, available_stop - origin_frame)
            raw_window = raw_feature_263[b : b + 1, origin_frame:available_stop, :]
            if raw_window.shape[1] <= 0:
                raise ValueError(
                    "window-local trajectory produced an empty raw window; "
                    f"sample={b}, start_token={start}, expected_len={expected_len}, "
                    f"raw_feature_length={raw_len}"
                )
            root_quat, root_xyz = recover_root_rot_pos(raw_window)
            traj7 = root_to_traj_feats_7d(root_quat, root_xyz).squeeze(0)
        else:
            frame_slice = token_range_to_frame_slice(start, count, frames_per_token)
            expected_len = int(frame_slice.stop - frame_slice.start)
            available_stop = min(int(frame_slice.stop), raw_len)
            available_len = max(0, available_stop - int(frame_slice.start))
            raw_full = raw_feature_263[b : b + 1, :raw_len, :]
            if raw_full.shape[1] <= 0:
                raise ValueError(
                    "window-local trajectory produced an empty raw window; "
                    f"sample={b}, frame_slice={frame_slice}, raw_feature_length={raw_len}"
                )
            root_quat, root_xyz = recover_root_rot_pos(raw_full)
            full_traj7 = root_to_traj_feats_7d(root_quat, root_xyz).squeeze(0)
            traj7_world = full_traj7[frame_slice.start:available_stop]
            anchor = full_traj7[origin_frame:origin_frame + 1]
            anchor_xz = anchor[..., [0, 2]]
            anchor_yaw = torch.atan2(anchor[..., 4], anchor[..., 3])
            traj7 = canonicalize_7d(
                traj7_world.unsqueeze(0),
                anchor_xz,
                anchor_yaw,
            ).squeeze(0)
        expected_lengths.append(expected_len)
        available_lengths.append(available_len)
        traj_windows.append(traj7)

    max_expected_len = max(expected_lengths) if expected_lengths else 0
    traj_features = raw_feature_263.new_zeros(batch_size, max_expected_len, 7)
    traj_mask = raw_feature_263.new_zeros(batch_size, max_expected_len)
    for b, traj7 in enumerate(traj_windows):
        valid = int(available_lengths[b])
        traj_features[b, :valid, :] = traj7[:valid]
        traj_mask[b, :valid] = 1.0

    return {
        "traj_features": traj_features,
        "traj_cond_mask": traj_mask,
        "traj_length": torch.as_tensor(
            available_lengths, device=device, dtype=torch.long
        ),
        "traj_start_token": torch.zeros_like(starts) if local_prefix else starts,
        "traj_num_tokens": counts,
        "traj_features_length": counts,
    }


def build_window_local_model_batch(
    batch: dict,
    *,
    context_tokens: int,
    horizon_tokens: int,
    sample_policy: str = "variable_history",
    min_history_tokens: int = 1,
    start_tokens=None,
    end_tokens=None,
    frames_per_token: int = 4,
    window_sampling: dict | None = None,
    chunk_size: int | None = None,
    rollout_span: int = 0,
    active_left_tokens=None,
    history_tokens=None,
    sampled_horizon_tokens=None,
    force_start_token_zero: bool = False,
    latent_source: str = "precomputed_slice",
    vae=None,
) -> dict:
    """Build a model batch for window-local limited-history training.

    ``precomputed_slice`` uses ``batch["token"]`` as the latent source.
    ``online_encode`` clips ``batch["feature"]`` in motion space and encodes the
    clip with ``vae.encode``. The returned ``feature`` is padded to the attention
    length ``latent_valid_len + horizon_tokens``, while ``feature_length`` records
    only the valid latent prefix. Future latent slots are zero padding for shape
    only.
    """
    token = batch["token"]
    if token.ndim != 3:
        raise ValueError(f"batch['token'] must be [B,T,D], got {tuple(token.shape)}")
    device = token.device
    batch_size, _, latent_dim = token.shape
    token_length = _as_long_1d(
        batch["token_length"], batch_size=batch_size, device=device, name="token_length"
    )
    context_tokens = int(context_tokens)
    if context_tokens <= 0:
        raise ValueError(f"context_tokens must be > 0, got {context_tokens}")

    latent_source = str(latent_source)
    if latent_source not in {"precomputed_slice", "online_encode"}:
        raise ValueError(
            "latent_source must be 'precomputed_slice' or 'online_encode'; "
            f"got {latent_source!r}"
        )
    if latent_source == "online_encode" and vae is None:
        raise ValueError("latent_source='online_encode' requires a VAE instance")

    sample = SampleCreator(
        context_tokens=context_tokens,
        horizon_tokens=horizon_tokens,
        sample_policy=sample_policy,
        min_history_tokens=min_history_tokens,
        start_tokens=start_tokens,
        end_tokens=end_tokens,
        window_sampling=window_sampling,
        chunk_size=chunk_size,
        rollout_span=rollout_span,
        active_left_tokens=active_left_tokens,
        history_tokens=history_tokens,
        sampled_horizon_tokens=sampled_horizon_tokens,
        force_start_token_zero=force_start_token_zero,
    ).create(token_length)
    starts = sample.global_start_tokens.to(device=device)
    local_starts = (
        sample.local_start_tokens.to(device=device)
        if latent_source == "online_encode"
        else starts
    )
    latent_lengths = sample.latent_tokens.to(device=device)
    traj_token_lengths = sample.traj_tokens.to(device=device)
    sample_policy = sample.sample_policy
    stream_sample = sample.stream_sample
    max_attn_len = int(traj_token_lengths.max().item())

    feature = token.new_zeros(batch_size, max_attn_len, latent_dim)
    token_mask_out = None
    if batch.get("token_mask") is not None:
        token_mask_src = batch["token_mask"].to(device=device, dtype=torch.float32)
        token_mask_out = token_mask_src.new_zeros(batch_size, max_attn_len)

    if "feature" not in batch:
        raise ValueError("window-local training requires raw batch['feature'] 263D motion")
    if "feature_length" not in batch:
        raise ValueError("window-local training requires raw batch['feature_length']")
    raw_feature = batch["feature"].to(device)
    if (
        raw_feature.ndim != 3
        or raw_feature.shape[-1] != RAW_HUMANML3D_MOTION_DIM
    ):
        raise ValueError(
            "window-local training requires raw 263D HumanML3D motion; got "
            f"{tuple(raw_feature.shape)}"
        )
    raw_lengths = _as_long_1d(
        batch["feature_length"],
        batch_size=batch_size,
        device=device,
        name="feature_length",
    )
    if latent_source == "precomputed_slice":
        for b in range(batch_size):
            start = int(starts[b].item())
            valid = int(latent_lengths[b].item())
            feature[b, :valid, :] = token[b, start:start + valid, :]
            if token_mask_out is not None:
                token_mask_out[b, :valid] = token_mask_src[b, start:start + valid]
    else:
        start_frames = sample.global_start_frames.to(device=device)
        frame_lengths = sample.latent_frame_lengths.to(device=device)
        for b in range(batch_size):
            start_frame = int(start_frames[b].item())
            frame_len = int(frame_lengths[b].item())
            stop_frame = start_frame + frame_len
            raw_len = int(raw_lengths[b].item())
            if stop_frame > raw_len:
                raise ValueError(
                    "online_encode motion window exceeds raw feature length; "
                    f"sample={b}, start_frame={start_frame}, frame_len={frame_len}, "
                    f"raw_feature_length={raw_len}"
                )
            raw_window = raw_feature[b : b + 1, start_frame:stop_frame, :]
            with torch.no_grad():
                encoded = vae.encode(raw_window)
            if encoded.ndim != 3 or encoded.shape[0] != 1:
                raise ValueError(
                    "online_encode VAE must return [1,T,D] for each motion window; "
                    f"got {tuple(encoded.shape)}"
                )
            valid = int(latent_lengths[b].item())
            if int(encoded.shape[1]) != valid:
                raise ValueError(
                    "online_encode token count mismatch: sampled token count does "
                    "not match VAE encode output; "
                    f"sample={b}, sampled_tokens={valid}, "
                    f"encoded_tokens={int(encoded.shape[1])}, frame_len={frame_len}"
                )
            if int(encoded.shape[2]) != latent_dim:
                raise ValueError(
                    "online_encode latent dim mismatch with precomputed token dim; "
                    f"sample={b}, encoded_dim={int(encoded.shape[2])}, "
                    f"token_dim={latent_dim}"
                )
            feature[b, :valid, :] = encoded[0].to(device=device, dtype=token.dtype)
            if token_mask_out is not None:
                start = int(starts[b].item())
                token_mask_out[b, :valid] = token_mask_src[b, start:start + valid]

    traj_part = build_window_local_traj_batch(
        raw_feature_263=raw_feature,
        raw_feature_length=raw_lengths,
        start_tokens=starts,
        num_tokens=traj_token_lengths,
        frames_per_token=frames_per_token,
        local_prefix=(latent_source == "online_encode"),
    )

    out = batch.copy()
    for key in (
        "traj", "traj_cond", "traj_cond_7d", "traj_mask", "traj_cond_mask",
        "traj_loss_mask", "traj_features", "traj_length", "traj_features_length",
        "traj_loss_gt", "token_mask",
    ):
        out.pop(key, None)
    out.update(traj_part)
    out["feature"] = feature
    out["feature_length"] = latent_lengths
    out["token"] = feature
    out["token_length"] = latent_lengths
    if token_mask_out is not None:
        out["latent_token_mask"] = token_mask_out
    _crop_segmented_text_fields(out, starts, latent_lengths)
    out["_window_local_traj"] = True
    out["_window_global_start_token"] = starts
    out["_window_local_latent_start_token"] = local_starts
    out["_window_local_latent_valid_len"] = latent_lengths
    out["_window_local_sample_policy"] = sample_policy
    out["_window_local_latent_source"] = latent_source
    out["_window_local_body_aux_mode"] = (
        "local_decode" if latent_source == "online_encode" else "full_prefix_splice"
    )
    if latent_source == "online_encode":
        out["traj_cond_7d"] = out["traj_features"]
    if stream_sample is not None:
        out["_window_sampling_active_left_token"] = stream_sample["active_left_tokens"]
        out["_window_sampling_history_tokens"] = stream_sample["history_tokens"]
        out["_window_sampling_horizon_tokens"] = stream_sample["horizon_tokens"]
        out["_window_sampling_horizon_cap_clip"] = stream_sample["horizon_cap_clip"]
        out["_window_sampling_horizon_short_fallback"] = stream_sample[
            "horizon_short_fallback"
        ]
        out["_window_sampling_rollout_span"] = stream_sample["rollout_span"]
        out["_window_sampling_history_tokens_max_effective"] = (
            stream_sample["history_tokens_max_effective"]
        )
    return out


def _crop_segmented_text_fields(
    batch: dict,
    starts: torch.Tensor,
    latent_lengths: torch.Tensor,
) -> None:
    text = batch.get("text")
    if not text or not isinstance(text[0], list):
        return
    token_text_end = batch.get("token_text_end", batch.get("feature_text_end"))
    if token_text_end is None:
        return
    cropped_text: list[list[str]] = []
    cropped_end: list[list[int]] = []
    for b, segments in enumerate(text):
        ends = token_text_end[b]
        if torch.is_tensor(ends):
            ends = [int(v) for v in ends.view(-1).tolist()]
        else:
            ends = [int(v) for v in ends]
        if len(segments) != len(ends):
            raise ValueError(
                "text/end schedule mismatch for segmented text: "
                f"sample={b}, text_segments={len(segments)}, "
                f"endpoints={len(ends)}"
            )
        prev = 0
        for end in ends:
            if int(end) < prev:
                raise ValueError(
                    "segmented text token endpoints must be monotonic; "
                    f"sample={b}, endpoints={ends}"
                )
            prev = int(end)
        window_start = int(starts[b].item())
        window_end = window_start + int(latent_lengths[b].item())
        prev_end = 0
        sample_text: list[str] = []
        sample_end: list[int] = []
        for segment_text, end in zip(segments, ends):
            seg_start = prev_end
            seg_end = int(end)
            prev_end = seg_end
            inter_start = max(seg_start, window_start)
            inter_end = min(seg_end, window_end)
            if inter_end <= inter_start:
                continue
            sample_text.append(segment_text)
            sample_end.append(inter_end - window_start)
        if not sample_text:
            sample_text = [""]
            sample_end = [int(latent_lengths[b].item())]
        elif sample_end[-1] < int(latent_lengths[b].item()):
            sample_text.append("")
            sample_end.append(int(latent_lengths[b].item()))
        cropped_text.append(sample_text)
        cropped_end.append(sample_end)
    batch["text"] = cropped_text
    batch["token_text_end"] = cropped_end
    # DiffForcingWanModel._prepare_text_context historically reads this field as
    # token endpoints, so keep it synchronized with token_text_end.
    batch["feature_text_end"] = cropped_end


__all__ = ["build_window_local_model_batch", "build_window_local_traj_batch"]
