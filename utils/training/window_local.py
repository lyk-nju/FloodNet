"""Window-local LDF training helpers.

These helpers build the trajectory side of the limited-history training batch.
They intentionally reconstruct 7D root trajectory from raw 263D motion for the
sampled token window instead of cropping an existing full-clip 7D tensor.
"""

from __future__ import annotations

import torch

from utils.motion_process import recover_root_rot_pos, root_to_traj_feats_7d
from utils.token_frame import token_range_to_frame_slice, token_start_frame
from utils.training.window_sampling import sample_stream_window_indices

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
        frame_slice = token_range_to_frame_slice(start, count, frames_per_token)
        expected_len = int(frame_slice.stop - frame_slice.start)
        available_stop = min(int(frame_slice.stop), raw_len)
        available_len = max(0, available_stop - int(frame_slice.start))
        raw_window = raw_feature_263[b : b + 1, frame_slice.start:available_stop, :]
        if raw_window.shape[1] <= 0:
            raise ValueError(
                "window-local trajectory produced an empty raw window; "
                f"sample={b}, frame_slice={frame_slice}, raw_feature_length={raw_len}"
            )
        root_quat, root_xyz = recover_root_rot_pos(raw_window)
        traj7 = root_to_traj_feats_7d(root_quat, root_xyz).squeeze(0)
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
        "traj_start_token": starts,
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
) -> dict:
    """Build a model batch for window-local limited-history training.

    ``batch["token"]`` remains the source of VAE latents; no VAE re-encoding is
    performed. The returned ``feature`` is padded to the attention length
    ``latent_valid_len + horizon_tokens``, but ``feature_length`` records only the
    valid latent prefix. Future latent slots are zero padding for shape only.
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

    ws_cfg = window_sampling or {}
    window_sampling_enabled = bool(ws_cfg.get("enabled", False))
    stream_sample: dict | None = None
    if window_sampling_enabled:
        if chunk_size is None:
            raise ValueError("chunk_size is required when window_sampling is enabled")
        stream_sample = sample_stream_window_indices(
            token_length,
            context_tokens=context_tokens,
            chunk_size=int(chunk_size),
            rollout_span=int(rollout_span),
            history_tokens_min=int(ws_cfg.get("history_tokens_min", 0)),
            history_tokens_max=ws_cfg.get("history_tokens_max", "auto"),
            horizon_tokens_min=int(ws_cfg.get("horizon_tokens_min", 0)),
            horizon_tokens_max=int(ws_cfg.get("horizon_tokens_max", 0)),
            active_left_tokens=active_left_tokens,
            history_tokens=history_tokens,
            horizon_tokens=sampled_horizon_tokens,
        )
        starts = stream_sample["window_left_tokens"]
        latent_lengths = stream_sample["latent_num_tokens"]
        traj_token_lengths = stream_sample["traj_num_tokens"]
        sample_policy = "active_left"
    else:
        sample_policy = str(sample_policy)
        if sample_policy not in {"variable_history", "fixed_window"}:
            raise ValueError(
                "sample_policy must be 'variable_history' or 'fixed_window', "
                f"got {sample_policy!r}"
            )
        horizon_tokens = int(horizon_tokens)
        if horizon_tokens < 0:
            raise ValueError(f"horizon_tokens must be >= 0, got {horizon_tokens}")
        min_history_tokens = int(min_history_tokens)
        if min_history_tokens <= 0:
            raise ValueError(f"min_history_tokens must be > 0, got {min_history_tokens}")
        if context_tokens < min_history_tokens:
            raise ValueError(
                "context_tokens must be >= min_history_tokens; "
                f"got context_tokens={context_tokens}, "
                f"min_history_tokens={min_history_tokens}"
            )

        if sample_policy == "fixed_window":
            if end_tokens is None:
                if bool((token_length < min_history_tokens).any()):
                    raise ValueError(
                        "fixed_window sampling requires token_length >= "
                        f"min_history_tokens={min_history_tokens}; "
                        f"token_length={token_length.tolist()}"
                    )
                ends = torch.stack(
                    [
                        torch.randint(
                            min_history_tokens,
                            int(token_length[b].item()) + 1,
                            (1,),
                            device=device,
                        )[0]
                        for b in range(batch_size)
                    ]
                ).to(dtype=torch.long)
            else:
                ends = _as_long_1d(
                    end_tokens, batch_size=batch_size, device=device, name="end_tokens"
                )
            if bool((ends <= 0).any()):
                raise ValueError(f"end_tokens must be > 0, got {ends.tolist()}")
            if bool((ends > token_length).any()):
                raise ValueError(
                    "end_tokens must be <= token_length; "
                    f"end_tokens={ends.tolist()}, token_length={token_length.tolist()}"
                )
            starts = (ends - context_tokens).clamp(min=0)
        elif start_tokens is None:
            max_start = (token_length - context_tokens).clamp(min=0)
            starts = torch.stack(
                [
                    torch.randint(0, int(max_start[b].item()) + 1, (1,), device=device)[0]
                    for b in range(batch_size)
                ]
            ).to(dtype=torch.long)
        else:
            starts = _as_long_1d(
                start_tokens, batch_size=batch_size, device=device, name="start_tokens"
            )
        if bool((starts < 0).any()):
            raise ValueError(f"start_tokens must be >= 0, got {starts.tolist()}")

        latent_lengths = torch.minimum(
            torch.full_like(token_length, context_tokens),
            token_length - starts,
        )
        if sample_policy == "fixed_window":
            latent_lengths = ends - starts
        if bool((latent_lengths <= 0).any()):
            raise ValueError(
                "window-local model batch requires at least one latent token after "
                f"start; starts={starts.tolist()}, token_length={token_length.tolist()}"
            )
        if bool((latent_lengths < min_history_tokens).any()):
            raise ValueError(
                "window-local model batch produced a window shorter than "
                "min_history_tokens; "
                f"latent_lengths={latent_lengths.tolist()}, "
                f"min_history_tokens={min_history_tokens}"
            )
        traj_token_lengths = latent_lengths + horizon_tokens
    max_attn_len = int(traj_token_lengths.max().item())

    feature = token.new_zeros(batch_size, max_attn_len, latent_dim)
    token_mask_out = None
    if batch.get("token_mask") is not None:
        token_mask_src = batch["token_mask"].to(device=device, dtype=torch.float32)
        token_mask_out = token_mask_src.new_zeros(batch_size, max_attn_len)
    for b in range(batch_size):
        start = int(starts[b].item())
        valid = int(latent_lengths[b].item())
        feature[b, :valid, :] = token[b, start:start + valid, :]
        if token_mask_out is not None:
            token_mask_out[b, :valid] = token_mask_src[b, start:start + valid]

    if "feature" not in batch:
        raise ValueError("window-local training requires raw batch['feature'] 263D motion")
    if "feature_length" not in batch:
        raise ValueError("window-local training requires raw batch['feature_length']")
    traj_part = build_window_local_traj_batch(
        raw_feature_263=batch["feature"].to(device),
        raw_feature_length=batch["feature_length"],
        start_tokens=starts,
        num_tokens=traj_token_lengths,
        frames_per_token=frames_per_token,
    )

    out = batch.copy()
    for key in (
        "traj", "traj_cond", "traj_cond_7d", "traj_mask", "traj_cond_mask",
        "traj_loss_mask", "traj_features", "traj_length", "traj_features_length",
        "traj_loss_gt",
    ):
        out.pop(key, None)
    out.update(traj_part)
    out["feature"] = feature
    out["feature_length"] = latent_lengths
    out["token"] = feature
    out["token_length"] = latent_lengths
    if token_mask_out is not None:
        out["token_mask"] = token_mask_out
    _crop_segmented_text_fields(out, starts, latent_lengths)
    out["_window_local_traj"] = True
    out["_window_local_latent_start_token"] = starts
    out["_window_local_latent_valid_len"] = latent_lengths
    out["_window_local_sample_policy"] = sample_policy
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
