"""Create LDF model batches from HumanML3D dataset batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.local_frame import canonicalize_7d
from utils.motion_process import recover_root_rot_pos, root_to_traj_feats_7d
from utils.token_frame import (
    num_frames_for_tokens,
    num_tokens_for_frame_len,
    token_range_to_frame_slice,
    token_start_frame,
)

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
            f"{name} must be scalar or length {batch_size}; "
            f"got shape {tuple(out.shape)}"
        )
    return out


def _frames_for_tokens_tensor(tokens: torch.Tensor) -> torch.Tensor:
    values = [num_frames_for_tokens(int(v.item())) for v in tokens.view(-1)]
    return torch.as_tensor(
        values,
        device=tokens.device,
        dtype=torch.long,
    ).view_as(tokens)


def _start_frames_tensor(tokens: torch.Tensor) -> torch.Tensor:
    values = [token_start_frame(int(v.item())) for v in tokens.view(-1)]
    return torch.as_tensor(
        values,
        device=tokens.device,
        dtype=torch.long,
    ).view_as(tokens)


def _tokens_for_frames_tensor(frames: torch.Tensor) -> torch.Tensor:
    values = [num_tokens_for_frame_len(int(v.item())) for v in frames.view(-1)]
    return torch.as_tensor(
        values,
        device=frames.device,
        dtype=torch.long,
    ).view_as(frames)


def resolve_history_tokens_max(
    value,
    *,
    context_tokens: int,
    chunk_size: int,
    rollout_span: int,
) -> int:
    context_tokens = int(context_tokens)
    chunk_size = int(chunk_size)
    rollout_span = int(rollout_span)
    if context_tokens <= 0:
        raise ValueError(f"context_tokens must be > 0, got {context_tokens}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if rollout_span < 0:
        raise ValueError(f"rollout_span must be >= 0, got {rollout_span}")
    auto_max = context_tokens - chunk_size - rollout_span
    if auto_max < 0:
        raise ValueError(
            "context_tokens is too small for chunk_size + rollout_span; "
            f"context_tokens={context_tokens}, chunk_size={chunk_size}, "
            f"rollout_span={rollout_span}"
        )
    if value is None or str(value).lower() == "auto":
        return auto_max
    configured = int(value)
    if configured < 0:
        raise ValueError(f"history_tokens_max must be >= 0 or 'auto', got {value!r}")
    return min(configured, auto_max)


def sample_stream_window_indices(
    token_length,
    *,
    context_tokens: int,
    chunk_size: int,
    rollout_span: int,
    history_tokens_min: int,
    history_tokens_max,
    horizon_tokens_min: int,
    horizon_tokens_max: int,
    active_left_tokens=None,
    history_tokens=None,
    horizon_tokens=None,
) -> dict[str, torch.Tensor | int]:
    """Sample active stream windows on the token timeline."""
    if torch.is_tensor(token_length):
        device = token_length.device
        lengths = token_length.to(device=device, dtype=torch.long).view(-1)
    else:
        lengths = torch.as_tensor(token_length, dtype=torch.long).view(-1)
        device = lengths.device
    batch_size = int(lengths.numel())
    if batch_size <= 0:
        raise ValueError("token_length must contain at least one sample")
    if bool((lengths <= 0).any()):
        raise ValueError(f"token_length must be > 0, got {lengths.tolist()}")

    chunk_size = int(chunk_size)
    rollout_span = int(rollout_span)
    history_tokens_min = int(history_tokens_min)
    horizon_tokens_min = int(horizon_tokens_min)
    horizon_tokens_max = int(horizon_tokens_max)
    if history_tokens_min < 0:
        raise ValueError(
            f"history_tokens_min must be >= 0, got {history_tokens_min}"
        )
    if horizon_tokens_min < 0:
        raise ValueError(
            f"horizon_tokens_min must be >= 0, got {horizon_tokens_min}"
        )
    if horizon_tokens_max < horizon_tokens_min:
        raise ValueError(
            "horizon_tokens_max must be >= horizon_tokens_min; "
            f"got min={horizon_tokens_min}, max={horizon_tokens_max}"
        )
    horizon_abs_min = 1
    horizon_pref_min = max(horizon_tokens_min, horizon_abs_min)
    if horizon_tokens_max < horizon_abs_min:
        raise ValueError(
            f"horizon_tokens_max must be >= {horizon_abs_min}, got {horizon_tokens_max}"
        )

    history_max_eff = resolve_history_tokens_max(
        history_tokens_max,
        context_tokens=context_tokens,
        chunk_size=chunk_size,
        rollout_span=rollout_span,
    )
    if history_max_eff < history_tokens_min:
        raise ValueError(
            "history_tokens_max effective value is below history_tokens_min; "
            f"history_tokens_max_effective={history_max_eff}, "
            f"history_tokens_min={history_tokens_min}"
        )

    active_override = (
        None
        if active_left_tokens is None
        else _as_long_1d(
            active_left_tokens,
            batch_size=batch_size,
            device=device,
            name="active_left_tokens",
        )
    )
    history_override = (
        None
        if history_tokens is None
        else _as_long_1d(
            history_tokens,
            batch_size=batch_size,
            device=device,
            name="history_tokens",
        )
    )
    horizon_override = (
        None
        if horizon_tokens is None
        else _as_long_1d(
            horizon_tokens,
            batch_size=batch_size,
            device=device,
            name="horizon_tokens",
        )
    )

    active_values: list[torch.Tensor] = []
    history_values: list[torch.Tensor] = []
    horizon_values: list[torch.Tensor] = []
    horizon_cap_values: list[torch.Tensor] = []
    horizon_short_fallback_values: list[torch.Tensor] = []
    for batch_idx in range(batch_size):
        token_count = int(lengths[batch_idx].item())
        horizon_cap_clip = token_count - chunk_size - rollout_span
        if horizon_cap_clip < horizon_abs_min:
            raise ValueError(
                "stream window sampling requires a full horizon inside the clip; "
                f"sample={batch_idx}, token_length={token_count}, "
                f"horizon_cap_clip={horizon_cap_clip}, "
                f"horizon_abs_min={horizon_abs_min}"
            )
        horizon_cap_history = horizon_cap_clip - history_tokens_min
        if horizon_cap_history < horizon_abs_min:
            raise ValueError(
                "stream window sampling requires room for active chunk, rollout, "
                "minimum history, and at least one horizon token; "
                f"sample={batch_idx}, token_length={token_count}, "
                f"horizon_cap_history={horizon_cap_history}, "
                f"history_tokens_min={history_tokens_min}, "
                f"horizon_abs_min={horizon_abs_min}"
            )
        horizon_cap = min(horizon_cap_clip, horizon_cap_history)
        short_fallback = horizon_cap < horizon_pref_min
        horizon_low = horizon_pref_min if not short_fallback else horizon_abs_min
        horizon_high = min(horizon_tokens_max, horizon_cap)
        if horizon_high < horizon_low:
            raise ValueError(
                "stream window sampling found no valid horizon range; "
                f"sample={batch_idx}, horizon_low={horizon_low}, "
                f"horizon_high={horizon_high}, "
                f"horizon_cap_clip={horizon_cap_clip}, "
                f"horizon_cap_history={horizon_cap_history}"
            )
        if horizon_override is None:
            horizon_value = torch.randint(
                horizon_low,
                horizon_high + 1,
                (1,),
                device=device,
            )[0]
        else:
            horizon_value = horizon_override[batch_idx]
            horizon_int = int(horizon_value.item())
            if horizon_int < horizon_low or horizon_int > horizon_high:
                raise ValueError(
                    "horizon_tokens must be within the complete per-clip range; "
                    f"sample={batch_idx}, horizon_tokens={horizon_int}, "
                    f"valid_range=[{horizon_low}, {horizon_high}]"
                )

        active_low = history_tokens_min
        active_high = (
            token_count - chunk_size - rollout_span - int(horizon_value.item())
        )
        if active_high < active_low:
            raise ValueError(
                "stream window sampling requires room for active chunk, rollout, "
                "chosen horizon, and minimum history; "
                f"sample={batch_idx}, token_length={token_count}, "
                f"low_active_left={active_low}, high_active_left={active_high}, "
                f"chosen_horizon={int(horizon_value.item())}"
            )

        if active_override is None:
            active_value = torch.randint(
                active_low,
                active_high + 1,
                (1,),
                device=device,
            )[0]
        else:
            active_value = active_override[batch_idx]
            active_int = int(active_value.item())
            if active_int < active_low or active_int > active_high:
                raise ValueError(
                    "active_left_tokens must allow full horizon and minimum history; "
                    f"sample={batch_idx}, active_left={active_int}, "
                    f"valid_range=[{active_low}, {active_high}]"
                )

        history_high = min(history_max_eff, int(active_value.item()))
        if history_high < history_tokens_min:
            raise ValueError(
                "stream window sampling found no valid history length; "
                f"sample={batch_idx}, active_left={int(active_value.item())}, "
                f"history_tokens_min={history_tokens_min}, "
                f"history_tokens_max_effective={history_max_eff}"
            )
        if history_override is None:
            history_value = torch.randint(
                history_tokens_min,
                history_high + 1,
                (1,),
                device=device,
            )[0]
        else:
            history_value = history_override[batch_idx]
            history_int = int(history_value.item())
            if history_int < history_tokens_min or history_int > history_high:
                raise ValueError(
                    "history_tokens must fit before active_left and inside context; "
                    f"sample={batch_idx}, history_tokens={history_int}, "
                    f"valid_range=[{history_tokens_min}, {history_high}]"
                )

        active_values.append(active_value.to(dtype=torch.long))
        history_values.append(history_value.to(dtype=torch.long))
        horizon_values.append(horizon_value.to(dtype=torch.long))
        horizon_cap_values.append(
            torch.as_tensor(horizon_cap_clip, device=device, dtype=torch.long)
        )
        horizon_short_fallback_values.append(
            torch.as_tensor(short_fallback, device=device, dtype=torch.bool)
        )

    active = torch.stack(active_values).to(device=device, dtype=torch.long)
    history = torch.stack(history_values).to(device=device, dtype=torch.long)
    horizon = torch.stack(horizon_values).to(device=device, dtype=torch.long)
    horizon_cap = torch.stack(horizon_cap_values).to(device=device, dtype=torch.long)
    horizon_short_fallback = torch.stack(horizon_short_fallback_values).to(
        device=device, dtype=torch.bool
    )
    window_left = active - history
    latent_num_tokens = history + int(chunk_size) + int(rollout_span)
    traj_num_tokens = latent_num_tokens + horizon

    return {
        "window_left_tokens": window_left,
        "active_left_tokens": active,
        "history_tokens": history,
        "horizon_tokens": horizon,
        "latent_num_tokens": latent_num_tokens,
        "traj_num_tokens": traj_num_tokens,
        "horizon_cap_clip": horizon_cap,
        "horizon_short_fallback": horizon_short_fallback,
        "history_tokens_max_effective": int(history_max_eff),
        "rollout_span": int(rollout_span),
    }


@dataclass(frozen=True)
class StreamSample:
    global_start_tokens: torch.Tensor
    local_start_tokens: torch.Tensor
    latent_tokens: torch.Tensor
    traj_tokens: torch.Tensor
    global_start_frames: torch.Tensor
    latent_frame_lengths: torch.Tensor
    traj_frame_lengths: torch.Tensor
    sample_policy: str
    stream_sample: dict | None = None


class SampleCreator:
    """Create full or stream-local LDF batches for model forward."""

    def __init__(
        self,
        *,
        stream_enabled: bool = False,
        context_tokens: int | None = None,
        horizon_tokens: int = 0,
        sample_policy: str = "variable_history",
        window_policy: str = "prefix",
        min_history_tokens: int = 1,
        window_sampling: dict | None = None,
        chunk_size: int | None = None,
        rollout_span: int = 0,
        start_tokens=None,
        end_tokens=None,
        min_prefix_tokens: int | None = None,
        active_left_tokens=None,
        history_tokens=None,
        sampled_horizon_tokens=None,
        force_start_token_zero: bool = False,
        frames_per_token: int = 4,
    ):
        self.stream_enabled = bool(stream_enabled)
        self.context_tokens = None if context_tokens is None else int(context_tokens)
        self.horizon_tokens = int(horizon_tokens)
        self.sample_policy = str(sample_policy)
        self.window_policy = str(window_policy)
        self.min_history_tokens = int(min_history_tokens)
        self.window_sampling = window_sampling or {}
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.rollout_span = int(rollout_span)
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens
        self.min_prefix_tokens = (
            None if min_prefix_tokens is None else int(min_prefix_tokens)
        )
        self.active_left_tokens = active_left_tokens
        self.history_tokens = history_tokens
        self.sampled_horizon_tokens = sampled_horizon_tokens
        self.force_start_token_zero = bool(force_start_token_zero)
        self.frames_per_token = int(frames_per_token)

    def create(self, batch: dict, *, vae=None) -> dict:
        if self.stream_enabled:
            return self._create_online_batch(batch, vae=vae)
        return self._create_batch(batch)

    def _create_batch(self, batch: dict) -> dict:
        token = batch["token"]
        if token.ndim != 3:
            raise ValueError(
                f"batch['token'] must be [B,T,D], got {tuple(token.shape)}"
            )
        batch_size = int(token.shape[0])
        device = token.device
        token_length = _as_long_1d(
            batch["token_length"],
            batch_size=batch_size,
            device=device,
            name="token_length",
        )

        if self.window_policy == "full":
            return self._create_full_batch(batch, token, token_length)
        if self.window_policy != "prefix":
            raise ValueError(
                "non-stream LDF batch creation only supports "
                "window_policy='prefix' or 'full'; "
                f"got {self.window_policy!r}"
            )
        sample = self._sample_prefix_window(token_length, batch_size, device)
        starts = sample.global_start_tokens.to(device=device)
        latent_lengths = sample.latent_tokens.to(device=device)
        traj_token_lengths = sample.traj_tokens.to(device=device)

        max_latent_len = int(latent_lengths.max().item())
        feature = token.new_zeros(batch_size, max_latent_len, int(token.shape[-1]))
        for batch_idx in range(batch_size):
            start_token = int(starts[batch_idx].item())
            valid_tokens = int(latent_lengths[batch_idx].item())
            feature[batch_idx, :valid_tokens, :] = token[
                batch_idx,
                start_token:start_token + valid_tokens,
                :,
            ]

        model_batch = batch.copy()
        model_batch.pop("token_mask", None)
        model_batch["feature"] = feature
        model_batch["feature_length"] = latent_lengths
        model_batch["token"] = feature
        model_batch["token_length"] = latent_lengths
        if "token_mask" in batch:
            token_mask = batch["token_mask"].to(device=device, dtype=torch.float32)
            token_mask_out = token_mask.new_zeros(batch_size, max_latent_len)
            for batch_idx in range(batch_size):
                start_token = int(starts[batch_idx].item())
                valid_tokens = int(latent_lengths[batch_idx].item())
                token_mask_out[batch_idx, :valid_tokens] = token_mask[
                    batch_idx,
                    start_token:start_token + valid_tokens,
                ]
            model_batch["latent_token_mask"] = token_mask_out
        if "token_text_end" in batch:
            model_batch["feature_text_end"] = batch["token_text_end"]
        self._copy_prefix_trajectory_fields(batch, model_batch, traj_token_lengths)
        self._crop_segmented_text_fields(model_batch, starts, latent_lengths)
        model_batch["_window_global_start_token"] = starts
        model_batch["_window_local_latent_start_token"] = torch.zeros_like(starts)
        model_batch["_window_local_latent_valid_len"] = latent_lengths
        model_batch["_window_local_sample_policy"] = sample.sample_policy
        return model_batch

    def _create_full_batch(
        self,
        batch: dict,
        token: torch.Tensor,
        token_length: torch.Tensor,
    ) -> dict:
        """Use the dataset's precomputed latent sequence without window cropping."""
        model_batch = batch.copy()
        model_batch["feature"] = token
        model_batch["feature_length"] = token_length
        model_batch["token"] = token
        model_batch["token_length"] = token_length
        if "token_text_end" in batch:
            model_batch["feature_text_end"] = batch["token_text_end"]
        self._copy_trajectory_fields(batch, model_batch)
        return model_batch

    def _create_online_batch(self, batch: dict, *, vae) -> dict:
        if vae is None:
            raise ValueError("stream SampleCreator requires a VAE for online encode")
        if "feature" not in batch:
            raise ValueError(
                "stream SampleCreator requires raw batch['feature'] motion"
            )
        if "feature_length" not in batch:
            raise ValueError(
                "stream SampleCreator requires raw batch['feature_length']"
            )
        raw_feature = batch["feature"]
        if (
            raw_feature.ndim != 3
            or raw_feature.shape[-1] != RAW_HUMANML3D_MOTION_DIM
        ):
            raise ValueError(
                "stream SampleCreator requires raw 263D HumanML3D motion; got "
                f"{tuple(raw_feature.shape)}"
            )

        device = raw_feature.device
        batch_size = int(raw_feature.shape[0])
        raw_lengths = _as_long_1d(
            batch["feature_length"],
            batch_size=batch_size,
            device=device,
            name="feature_length",
        )
        max_raw_frames = int(raw_feature.shape[1])
        if bool((raw_lengths < 0).any()) or bool((raw_lengths > max_raw_frames).any()):
            raise ValueError(
                "raw_feature_length/feature_length must be within the raw feature "
                "tensor frame range "
                f"[0, {max_raw_frames}]; got {raw_lengths.tolist()}"
            )

        if "token_length" in batch:
            token_length = _as_long_1d(
                batch["token_length"],
                batch_size=batch_size,
                device=device,
                name="token_length",
            )
        else:
            token_length = _tokens_for_frames_tensor(raw_lengths)

        sample = self._sample_window(token_length)
        starts = sample.global_start_tokens.to(device=device)
        latent_lengths = sample.latent_tokens.to(device=device)
        traj_token_lengths = sample.traj_tokens.to(device=device)
        max_latent_len = int(latent_lengths.max().item())
        stream_sample = sample.stream_sample

        encoded_windows: list[torch.Tensor] = []
        start_frames = sample.global_start_frames.to(device=device)
        frame_lengths = sample.latent_frame_lengths.to(device=device)
        for batch_idx in range(batch_size):
            start_frame = int(start_frames[batch_idx].item())
            frame_count = int(frame_lengths[batch_idx].item())
            stop_frame = start_frame + frame_count
            raw_frame_count = int(raw_lengths[batch_idx].item())
            if stop_frame > raw_frame_count:
                raise ValueError(
                    "online VAE encode window exceeds raw feature length; "
                    f"sample={batch_idx}, start_frame={start_frame}, "
                    f"frame_len={frame_count}, raw_feature_length={raw_frame_count}"
                )
            raw_window = raw_feature[
                batch_idx:batch_idx + 1,
                start_frame:stop_frame,
                :,
            ]
            with torch.no_grad():
                encoded = vae.encode(raw_window)
            if encoded.ndim != 3 or encoded.shape[0] != 1:
                raise ValueError(
                    "online VAE encode must return [1,T,D] for each motion window; "
                    f"got {tuple(encoded.shape)}"
                )
            valid_tokens = int(latent_lengths[batch_idx].item())
            if int(encoded.shape[1]) != valid_tokens:
                raise ValueError(
                    "online VAE token count mismatch: sampled token count does not "
                    "match VAE encode output; "
                    f"sample={batch_idx}, sampled_tokens={valid_tokens}, "
                    f"encoded_tokens={int(encoded.shape[1])}, frame_len={frame_count}"
                )
            encoded_windows.append(encoded[0])

        latent_dim = int(encoded_windows[0].shape[-1])
        if "token" in batch and torch.is_tensor(batch["token"]):
            token = batch["token"]
            if token.ndim != 3:
                raise ValueError(
                    f"batch['token'] must be [B,T,D], got {tuple(token.shape)}"
                )
            if int(token.shape[-1]) != latent_dim:
                raise ValueError(
                    "online VAE latent dim mismatch with dataset token dim; "
                    f"encoded_dim={latent_dim}, token_dim={int(token.shape[-1])}"
                )
        feature = encoded_windows[0].new_zeros(batch_size, max_latent_len, latent_dim)
        for batch_idx, encoded in enumerate(encoded_windows):
            valid_tokens = int(latent_lengths[batch_idx].item())
            feature[batch_idx, :valid_tokens, :] = encoded.to(
                device=device,
                dtype=feature.dtype,
            )

        token_mask_out = None
        if batch.get("token_mask") is not None:
            token_mask_src = batch["token_mask"].to(device=device, dtype=torch.float32)
            token_mask_out = token_mask_src.new_zeros(batch_size, max_latent_len)
            for batch_idx in range(batch_size):
                start_token = int(starts[batch_idx].item())
                valid_tokens = int(latent_lengths[batch_idx].item())
                token_mask_out[batch_idx, :valid_tokens] = token_mask_src[
                    batch_idx,
                    start_token:start_token + valid_tokens,
                ]

        traj_part = self._create_local_traj_batch(
            raw_feature_263=raw_feature,
            raw_feature_length=raw_lengths,
            start_tokens=starts,
            num_tokens=traj_token_lengths,
            local_prefix=True,
        )

        out = batch.copy()
        for key in (
            "traj",
            "traj_cond",
            "traj_cond_7d",
            "traj_mask",
            "traj_cond_mask",
            "traj_loss_mask",
            "traj_features",
            "traj_length",
            "traj_features_length",
            "traj_loss_gt",
            "token_mask",
        ):
            out.pop(key, None)
        out.update(traj_part)
        out["feature"] = feature
        out["feature_length"] = latent_lengths
        out["token"] = feature
        out["token_length"] = latent_lengths
        out["traj_cond_7d"] = out["traj_features"]
        if token_mask_out is not None:
            out["latent_token_mask"] = token_mask_out
        self._crop_segmented_text_fields(out, starts, latent_lengths)
        out["_window_local_traj"] = True
        out["_window_global_start_token"] = starts
        out["_window_local_latent_start_token"] = sample.local_start_tokens.to(
            device=device
        )
        out["_window_local_latent_valid_len"] = latent_lengths
        out["_window_local_sample_policy"] = sample.sample_policy
        if stream_sample is not None:
            out["_window_sampling_active_left_token"] = stream_sample[
                "active_left_tokens"
            ]
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

    def _sample_window(self, token_length) -> StreamSample:
        if torch.is_tensor(token_length):
            device = token_length.device
            lengths = token_length.to(device=device, dtype=torch.long).view(-1)
        else:
            lengths = torch.as_tensor(token_length, dtype=torch.long).view(-1)
            device = lengths.device
        batch_size = int(lengths.numel())
        if batch_size <= 0:
            raise ValueError("token_length must contain at least one sample")
        if self.context_tokens is None or self.context_tokens <= 0:
            raise ValueError(f"context_tokens must be > 0, got {self.context_tokens}")

        if bool(self.window_sampling.get("enabled", False)):
            return self._sample_active_window(lengths, batch_size, device)
        return self._sample_v1_window(lengths, batch_size, device)

    def _sample_prefix_window(
        self,
        lengths: torch.Tensor,
        batch_size: int,
        device,
    ) -> StreamSample:
        max_latent = lengths
        if bool((max_latent <= 0).any()):
            raise ValueError(
                "prefix-window batch requires positive token_length; "
                f"token_length={lengths.tolist()}"
            )
        if self.end_tokens is not None:
            latent_tokens = _as_long_1d(
                self.end_tokens,
                batch_size=batch_size,
                device=device,
                name="end_tokens",
            )
        else:
            low = max(
                1,
                int(self.min_history_tokens),
                1 if self.min_prefix_tokens is None else int(self.min_prefix_tokens),
            )
            if bool((max_latent < low).any()):
                raise ValueError(
                    "prefix-window batch found no valid latent window; "
                    f"max_latent={max_latent.tolist()}, "
                    f"min_history_tokens={int(self.min_history_tokens)}, "
                    f"min_prefix_tokens={low}"
                )
            latent_tokens = torch.stack(
                [
                    torch.randint(
                        low,
                        int(max_latent[batch_idx].item()) + 1,
                        (1,),
                        device=device,
                    )[0]
                    for batch_idx in range(batch_size)
                ]
            ).to(dtype=torch.long)
        if bool((latent_tokens <= 0).any()):
            raise ValueError(f"end_tokens must be > 0, got {latent_tokens.tolist()}")
        if self.min_prefix_tokens is not None and bool(
            (latent_tokens < int(self.min_prefix_tokens)).any()
        ):
            raise ValueError(
                "prefix-window batch produced a prefix shorter than "
                "min_prefix_tokens; "
                f"latent_tokens={latent_tokens.tolist()}, "
                f"min_prefix_tokens={int(self.min_prefix_tokens)}"
            )
        if bool((latent_tokens > max_latent).any()):
            raise ValueError(
                "prefix-window latent length must fit token_length; "
                f"latent_tokens={latent_tokens.tolist()}, "
                f"max_latent={max_latent.tolist()}"
            )
        starts = torch.zeros_like(lengths)
        return self._make_sample(
            starts=starts,
            latent_tokens=latent_tokens,
            traj_tokens=lengths,
            sample_policy="prefix",
            stream_sample=None,
        )

    def _sample_active_window(
        self,
        lengths: torch.Tensor,
        batch_size: int,
        device,
    ) -> StreamSample:
        if self.chunk_size is None:
            raise ValueError("chunk_size is required when window_sampling is enabled")
        if self.force_start_token_zero and (
            self.active_left_tokens is not None or self.history_tokens is not None
        ):
            raise ValueError(
                "force_start_token_zero cannot be combined with explicit "
                "active_left_tokens/history_tokens overrides"
            )
        ws_cfg = self.window_sampling
        stream_sample = sample_stream_window_indices(
            lengths,
            context_tokens=self.context_tokens,
            chunk_size=self.chunk_size,
            rollout_span=self.rollout_span,
            history_tokens_min=int(ws_cfg.get("history_tokens_min", 0)),
            history_tokens_max=ws_cfg.get("history_tokens_max", "auto"),
            horizon_tokens_min=int(ws_cfg.get("horizon_tokens_min", 0)),
            horizon_tokens_max=int(ws_cfg.get("horizon_tokens_max", 0)),
            active_left_tokens=self.active_left_tokens,
            history_tokens=self.history_tokens,
            horizon_tokens=self.sampled_horizon_tokens,
        )
        if self.force_start_token_zero:
            history_min = int(ws_cfg.get("history_tokens_min", 0))
            history_max_eff = int(stream_sample["history_tokens_max_effective"])
            horizons = stream_sample["horizon_tokens"]
            high_active = lengths - self.chunk_size - self.rollout_span - horizons
            history_high = torch.minimum(
                torch.full_like(lengths, history_max_eff),
                high_active,
            )
            if bool((history_high < history_min).any()):
                raise ValueError(
                    "force_start_token_zero found no valid zero-start history "
                    "length; "
                    f"history_high={history_high.tolist()}, "
                    f"history_tokens_min={history_min}"
                )
            forced_history = torch.stack(
                [
                    torch.randint(
                        history_min,
                        int(history_high[batch_idx].item()) + 1,
                        (1,),
                        device=device,
                    )[0]
                    for batch_idx in range(batch_size)
                ]
            ).to(dtype=torch.long)
            stream_sample = sample_stream_window_indices(
                lengths,
                context_tokens=self.context_tokens,
                chunk_size=self.chunk_size,
                rollout_span=self.rollout_span,
                history_tokens_min=history_min,
                history_tokens_max=ws_cfg.get("history_tokens_max", "auto"),
                horizon_tokens_min=int(ws_cfg.get("horizon_tokens_min", 0)),
                horizon_tokens_max=int(ws_cfg.get("horizon_tokens_max", 0)),
                active_left_tokens=forced_history,
                history_tokens=forced_history,
                horizon_tokens=horizons,
            )
        return self._make_sample(
            starts=stream_sample["window_left_tokens"],
            latent_tokens=stream_sample["latent_num_tokens"],
            traj_tokens=stream_sample["traj_num_tokens"],
            sample_policy="active_left",
            stream_sample=stream_sample,
        )

    def _sample_v1_window(
        self,
        lengths: torch.Tensor,
        batch_size: int,
        device,
    ) -> StreamSample:
        if self.sample_policy not in {"variable_history", "fixed_window"}:
            raise ValueError(
                "sample_policy must be 'variable_history' or 'fixed_window', "
                f"got {self.sample_policy!r}"
            )
        if self.horizon_tokens < 0:
            raise ValueError(f"horizon_tokens must be >= 0, got {self.horizon_tokens}")
        if self.min_history_tokens <= 0:
            raise ValueError(
                f"min_history_tokens must be > 0, got {self.min_history_tokens}"
            )
        if self.context_tokens < self.min_history_tokens:
            raise ValueError(
                "context_tokens must be >= min_history_tokens; "
                f"got context_tokens={self.context_tokens}, "
                f"min_history_tokens={self.min_history_tokens}"
            )

        ends = None
        if self.force_start_token_zero:
            if self.start_tokens is not None or self.end_tokens is not None:
                raise ValueError(
                    "force_start_token_zero cannot be combined with explicit "
                    "start_tokens/end_tokens overrides"
                )
            starts = torch.zeros_like(lengths)
        elif self.sample_policy == "fixed_window":
            if self.end_tokens is None:
                if bool((lengths < self.min_history_tokens).any()):
                    raise ValueError(
                        "fixed_window sampling requires token_length >= "
                        f"min_history_tokens={self.min_history_tokens}; "
                        f"token_length={lengths.tolist()}"
                    )
                ends = torch.stack(
                    [
                        torch.randint(
                            self.min_history_tokens,
                            int(lengths[batch_idx].item()) + 1,
                            (1,),
                            device=device,
                        )[0]
                        for batch_idx in range(batch_size)
                    ]
                ).to(dtype=torch.long)
            else:
                ends = _as_long_1d(
                    self.end_tokens,
                    batch_size=batch_size,
                    device=device,
                    name="end_tokens",
                )
            if bool((ends <= 0).any()):
                raise ValueError(f"end_tokens must be > 0, got {ends.tolist()}")
            if bool((ends > lengths).any()):
                raise ValueError(
                    "end_tokens must be <= token_length; "
                    f"end_tokens={ends.tolist()}, token_length={lengths.tolist()}"
                )
            starts = (ends - self.context_tokens).clamp(min=0)
        elif self.start_tokens is None:
            max_start = (lengths - self.context_tokens).clamp(min=0)
            starts = torch.stack(
                [
                    torch.randint(
                        0,
                        int(max_start[batch_idx].item()) + 1,
                        (1,),
                        device=device,
                    )[0]
                    for batch_idx in range(batch_size)
                ]
            ).to(dtype=torch.long)
        else:
            starts = _as_long_1d(
                self.start_tokens,
                batch_size=batch_size,
                device=device,
                name="start_tokens",
            )
        if bool((starts < 0).any()):
            raise ValueError(f"start_tokens must be >= 0, got {starts.tolist()}")

        latent_tokens = torch.minimum(
            torch.full_like(lengths, self.context_tokens),
            lengths - starts,
        )
        if self.sample_policy == "fixed_window" and not self.force_start_token_zero:
            latent_tokens = ends - starts
        if bool((latent_tokens <= 0).any()):
            raise ValueError(
                "stream model batch requires at least one latent token after "
                f"start; starts={starts.tolist()}, token_length={lengths.tolist()}"
            )
        if bool((latent_tokens < self.min_history_tokens).any()):
            raise ValueError(
                "stream model batch produced a window shorter than "
                "min_history_tokens; "
                f"latent_lengths={latent_tokens.tolist()}, "
                f"min_history_tokens={self.min_history_tokens}"
            )
        return self._make_sample(
            starts=starts,
            latent_tokens=latent_tokens,
            traj_tokens=latent_tokens + self.horizon_tokens,
            sample_policy=self.sample_policy,
            stream_sample=None,
        )

    def _make_sample(
        self,
        *,
        starts: torch.Tensor,
        latent_tokens: torch.Tensor,
        traj_tokens: torch.Tensor,
        sample_policy: str,
        stream_sample: dict | None,
    ) -> StreamSample:
        local_starts = torch.zeros_like(starts)
        return StreamSample(
            global_start_tokens=starts,
            local_start_tokens=local_starts,
            latent_tokens=latent_tokens,
            traj_tokens=traj_tokens,
            global_start_frames=_start_frames_tensor(starts),
            latent_frame_lengths=_frames_for_tokens_tensor(latent_tokens),
            traj_frame_lengths=_frames_for_tokens_tensor(traj_tokens),
            sample_policy=sample_policy,
            stream_sample=stream_sample,
        )

    def _create_local_traj_batch(
        self,
        *,
        raw_feature_263: torch.Tensor,
        raw_feature_length,
        start_tokens,
        num_tokens,
        local_prefix: bool,
    ) -> dict:
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
            raw_feature_length,
            batch_size=batch_size,
            device=device,
            name="raw_feature_length",
        )
        starts = _as_long_1d(
            start_tokens,
            batch_size=batch_size,
            device=device,
            name="start_tokens",
        )
        counts = _as_long_1d(
            num_tokens,
            batch_size=batch_size,
            device=device,
            name="num_tokens",
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
        for batch_idx in range(batch_size):
            start_token = int(starts[batch_idx].item())
            token_count = int(counts[batch_idx].item())
            raw_frame_count = int(raw_lengths[batch_idx].item())
            origin_frame = token_start_frame(start_token, self.frames_per_token)
            if origin_frame >= raw_frame_count:
                raise ValueError(
                    "window-local trajectory requires a valid origin; "
                    f"sample={batch_idx}, start_token={start_token}, "
                    f"origin_frame={origin_frame}, "
                    f"raw_feature_length={raw_frame_count}"
                )
            if local_prefix:
                expected_len = num_frames_for_tokens(
                    token_count,
                    self.frames_per_token,
                )
                available_stop = min(origin_frame + expected_len, raw_frame_count)
                available_len = max(0, available_stop - origin_frame)
                raw_window = raw_feature_263[
                    batch_idx:batch_idx + 1,
                    origin_frame:available_stop,
                    :,
                ]
                if raw_window.shape[1] <= 0:
                    raise ValueError(
                        "window-local trajectory produced an empty raw window; "
                        f"sample={batch_idx}, start_token={start_token}, "
                        f"expected_len={expected_len}, "
                        f"raw_feature_length={raw_frame_count}"
                    )
                root_quat, root_xyz = recover_root_rot_pos(raw_window)
                traj7 = root_to_traj_feats_7d(root_quat, root_xyz).squeeze(0)
            else:
                frame_slice = token_range_to_frame_slice(
                    start_token,
                    token_count,
                    self.frames_per_token,
                )
                expected_len = int(frame_slice.stop - frame_slice.start)
                available_stop = min(int(frame_slice.stop), raw_frame_count)
                available_len = max(0, available_stop - int(frame_slice.start))
                raw_full = raw_feature_263[
                    batch_idx:batch_idx + 1,
                    :raw_frame_count,
                    :,
                ]
                if raw_full.shape[1] <= 0:
                    raise ValueError(
                        "window-local trajectory produced an empty raw window; "
                        f"sample={batch_idx}, frame_slice={frame_slice}, "
                        f"raw_feature_length={raw_frame_count}"
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
        for batch_idx, traj7 in enumerate(traj_windows):
            valid_frames = int(available_lengths[batch_idx])
            traj_features[batch_idx, :valid_frames, :] = traj7[:valid_frames]
            traj_mask[batch_idx, :valid_frames] = 1.0

        return {
            "traj_features": traj_features,
            "traj_cond_mask": traj_mask,
            "traj_length": torch.as_tensor(
                available_lengths,
                device=device,
                dtype=torch.long,
            ),
            "traj_start_token": torch.zeros_like(starts) if local_prefix else starts,
            "traj_num_tokens": counts,
            "traj_features_length": counts,
        }

    def _copy_prefix_trajectory_fields(
        self,
        batch: dict,
        model_batch: dict,
        traj_token_lengths: torch.Tensor,
    ) -> None:
        device = traj_token_lengths.device
        batch_size = int(traj_token_lengths.numel())
        max_traj_tokens = int(traj_token_lengths.max().item())
        max_traj_frames = num_frames_for_tokens(max_traj_tokens, self.frames_per_token)
        source_lengths = None
        if "traj_length" in batch:
            source_lengths = _as_long_1d(
                batch["traj_length"],
                batch_size=batch_size,
                device=device,
                name="traj_length",
            )

        if "traj_cond_7d" in batch:
            src7 = batch["traj_cond_7d"]
            src_traj = batch.get("traj_cond", batch.get("traj"))
            src_mask = batch.get(
                "traj_cond_mask",
                batch.get("traj_mask", batch.get("traj_loss_mask")),
            )
            traj_features = src7.new_zeros(batch_size, max_traj_frames, src7.shape[-1])
            traj_mask = src7.new_zeros(batch_size, max_traj_frames)
            traj_lengths: list[int] = []
            for batch_idx in range(batch_size):
                frames = num_frames_for_tokens(
                    int(traj_token_lengths[batch_idx].item()),
                    self.frames_per_token,
                )
                valid_src_frames = (
                    int(source_lengths[batch_idx].item())
                    if source_lengths is not None
                    else int(src7.shape[1])
                )
                valid_src_frames = max(0, min(valid_src_frames, int(src7.shape[1])))
                available = min(frames, valid_src_frames)
                traj_lengths.append(available)
                traj_features[batch_idx, :available, :] = src7[
                    batch_idx,
                    :available,
                    :,
                ]
                if src_mask is not None:
                    mask_available = min(available, int(src_mask.shape[1]))
                    traj_mask[batch_idx, :mask_available] = src_mask[
                        batch_idx,
                        :mask_available,
                    ].to(
                        device=device, dtype=traj_mask.dtype,
                    )
                else:
                    traj_mask[batch_idx, :available] = 1.0
            model_batch["traj_features"] = traj_features
            model_batch["traj_cond_7d"] = traj_features
            if src_traj is not None:
                traj = src_traj.new_zeros(
                    batch_size,
                    max_traj_frames,
                    src_traj.shape[-1],
                )
                for batch_idx in range(batch_size):
                    available = min(
                        int(traj_lengths[batch_idx]),
                        int(src_traj.shape[1]),
                    )
                    traj[batch_idx, :available, :] = src_traj[
                        batch_idx,
                        :available,
                        :,
                    ]
                model_batch["traj"] = traj
            model_batch["traj_mask"] = traj_mask
            model_batch["traj_cond_mask"] = traj_mask
            model_batch["traj_length"] = torch.as_tensor(
                traj_lengths,
                device=device,
                dtype=torch.long,
            )
            model_batch["traj_start_token"] = torch.zeros_like(traj_token_lengths)
            model_batch["traj_num_tokens"] = traj_token_lengths
            model_batch["traj_features_length"] = traj_token_lengths
            return

        if "traj_cond" in batch or "traj" in batch:
            src_traj = batch.get("traj_cond", batch.get("traj"))
            src_mask = batch.get(
                "traj_cond_mask",
                batch.get("traj_mask", batch.get("traj_loss_mask")),
            )
            traj = src_traj.new_zeros(batch_size, max_traj_frames, src_traj.shape[-1])
            traj_mask = src_traj.new_zeros(batch_size, max_traj_frames)
            traj_lengths: list[int] = []
            for batch_idx in range(batch_size):
                frames = num_frames_for_tokens(
                    int(traj_token_lengths[batch_idx].item()),
                    self.frames_per_token,
                )
                valid_src_frames = (
                    int(source_lengths[batch_idx].item())
                    if source_lengths is not None
                    else int(src_traj.shape[1])
                )
                valid_src_frames = max(0, min(valid_src_frames, int(src_traj.shape[1])))
                available = min(frames, valid_src_frames)
                traj_lengths.append(available)
                traj[batch_idx, :available, :] = src_traj[
                    batch_idx,
                    :available,
                    :,
                ]
                if src_mask is not None:
                    mask_available = min(available, int(src_mask.shape[1]))
                    traj_mask[batch_idx, :mask_available] = src_mask[
                        batch_idx,
                        :mask_available,
                    ].to(
                        device=device, dtype=traj_mask.dtype,
                    )
                else:
                    traj_mask[batch_idx, :available] = 1.0
            model_batch["traj"] = traj
            model_batch["traj_mask"] = traj_mask
            model_batch["traj_length"] = torch.as_tensor(
                traj_lengths,
                device=device,
                dtype=torch.long,
            )
            model_batch["traj_start_token"] = torch.zeros_like(traj_token_lengths)
            model_batch["traj_num_tokens"] = traj_token_lengths
            model_batch.pop("traj_features", None)
            model_batch.pop("traj_features_length", None)
            return

        self._copy_trajectory_fields(batch, model_batch)

    @staticmethod
    def _copy_trajectory_fields(batch, model_batch) -> None:
        if "traj_cond_7d" in batch:
            model_batch["traj_features"] = batch["traj_cond_7d"]
            model_batch["traj"] = batch.get("traj_cond", batch.get("traj"))
            model_batch["traj_length"] = batch["traj_length"]
            model_batch["traj_mask"] = batch.get(
                "traj_cond_mask",
                batch.get("traj_mask", batch.get("traj_loss_mask")),
            )
            if "token_mask" in batch:
                model_batch["token_mask"] = batch["token_mask"]
            return
        if "traj_cond" in batch:
            model_batch["traj"] = batch["traj_cond"]
            model_batch["traj_length"] = batch["traj_length"]
            model_batch["traj_mask"] = batch.get(
                "traj_cond_mask",
                batch.get("traj_mask", batch.get("traj_loss_mask")),
            )
            model_batch.pop("traj_features", None)
        elif "traj" in batch:
            model_batch["traj"] = batch["traj"]
            model_batch["traj_length"] = batch["traj_length"]
            model_batch["traj_mask"] = batch["traj_mask"]
            if "traj_features" in batch:
                model_batch["traj_features"] = batch["traj_features"]
        if "token_mask" in batch:
            model_batch["token_mask"] = batch["token_mask"]

    @staticmethod
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
        for batch_idx, segments in enumerate(text):
            ends = token_text_end[batch_idx]
            if torch.is_tensor(ends):
                ends = [int(v) for v in ends.view(-1).tolist()]
            else:
                ends = [int(v) for v in ends]
            if len(segments) != len(ends):
                raise ValueError(
                    "text/end schedule mismatch for segmented text: "
                    f"sample={batch_idx}, text_segments={len(segments)}, "
                    f"endpoints={len(ends)}"
                )
            prev = 0
            for end in ends:
                if int(end) < prev:
                    raise ValueError(
                        "segmented text token endpoints must be monotonic; "
                        f"sample={batch_idx}, endpoints={ends}"
                    )
                prev = int(end)
            window_start = int(starts[batch_idx].item())
            window_end = window_start + int(latent_lengths[batch_idx].item())
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
                sample_end = [int(latent_lengths[batch_idx].item())]
            elif sample_end[-1] < int(latent_lengths[batch_idx].item()):
                sample_text.append("")
                sample_end.append(int(latent_lengths[batch_idx].item()))
            cropped_text.append(sample_text)
            cropped_end.append(sample_end)
        batch["text"] = cropped_text
        batch["token_text_end"] = cropped_end
        batch["feature_text_end"] = cropped_end


__all__ = [
    "SampleCreator",
    "StreamSample",
    "resolve_history_tokens_max",
    "sample_stream_window_indices",
]
