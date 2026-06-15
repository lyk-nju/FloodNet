"""Stream training sample creation.

SampleCreator owns token-space stream sampling and the derived motion-space
window lengths. Latent construction stays outside this file so
``precomputed_slice`` and ``online_encode`` use the same sample contract.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from utils.token_frame import num_frames_for_tokens, token_start_frame
from .window_sampling import sample_stream_window_indices


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


def _frames_for_tokens_tensor(tokens: torch.Tensor) -> torch.Tensor:
    values = [num_frames_for_tokens(int(v.item())) for v in tokens.view(-1)]
    return torch.as_tensor(values, device=tokens.device, dtype=torch.long).view_as(tokens)


def _start_frames_tensor(tokens: torch.Tensor) -> torch.Tensor:
    values = [token_start_frame(int(v.item())) for v in tokens.view(-1)]
    return torch.as_tensor(values, device=tokens.device, dtype=torch.long).view_as(tokens)


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
    """Create one stream-training sample plan from token lengths."""

    def __init__(
        self,
        *,
        context_tokens: int,
        horizon_tokens: int,
        sample_policy: str = "variable_history",
        min_history_tokens: int = 1,
        window_sampling: dict | None = None,
        chunk_size: int | None = None,
        rollout_span: int = 0,
        start_tokens=None,
        end_tokens=None,
        active_left_tokens=None,
        history_tokens=None,
        sampled_horizon_tokens=None,
        force_start_token_zero: bool = False,
    ):
        self.context_tokens = int(context_tokens)
        self.horizon_tokens = int(horizon_tokens)
        self.sample_policy = str(sample_policy)
        self.min_history_tokens = int(min_history_tokens)
        self.window_sampling = window_sampling or {}
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.rollout_span = int(rollout_span)
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens
        self.active_left_tokens = active_left_tokens
        self.history_tokens = history_tokens
        self.sampled_horizon_tokens = sampled_horizon_tokens
        self.force_start_token_zero = bool(force_start_token_zero)

    def create(self, token_length) -> StreamSample:
        if torch.is_tensor(token_length):
            device = token_length.device
            lengths = token_length.to(device=device, dtype=torch.long).view(-1)
        else:
            lengths = torch.as_tensor(token_length, dtype=torch.long).view(-1)
            device = lengths.device
        batch_size = int(lengths.numel())
        if batch_size <= 0:
            raise ValueError("token_length must contain at least one sample")
        if self.context_tokens <= 0:
            raise ValueError(f"context_tokens must be > 0, got {self.context_tokens}")

        if bool(self.window_sampling.get("enabled", False)):
            return self._create_window_sampling(lengths, batch_size, device)
        return self._create_v1(lengths, batch_size, device)

    def _create_window_sampling(self, lengths, batch_size: int, device) -> StreamSample:
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
                        int(history_high[b].item()) + 1,
                        (1,),
                        device=device,
                    )[0]
                    for b in range(batch_size)
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

    def _create_v1(self, lengths, batch_size: int, device) -> StreamSample:
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
            end_tokens = self.end_tokens
            if end_tokens is None:
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
                            int(lengths[b].item()) + 1,
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
                    torch.randint(0, int(max_start[b].item()) + 1, (1,), device=device)[0]
                    for b in range(batch_size)
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
                "window-local model batch requires at least one latent token after "
                f"start; starts={starts.tolist()}, token_length={lengths.tolist()}"
            )
        if bool((latent_tokens < self.min_history_tokens).any()):
            raise ValueError(
                "window-local model batch produced a window shorter than "
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


__all__ = ["SampleCreator", "StreamSample"]
