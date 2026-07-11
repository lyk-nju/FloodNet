"""Streaming executor that connects ConditionManager, RootRefiner, and LDF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from utils.conditions.ldf import LDFCondition
from utils.conditions.root_refiner import RootRefinerPathCondition
from utils.inference.condition_manager import ConditionManager
from utils.inference.root_plan import RootPlan
from utils.inference.route_condition import RoutePlan
from utils.inference.timeline import RootFrameState
from utils.inference.stream_runtime import KernelStepResult
from utils.local_frame import canonicalize_5d
from utils.motion_process import build_physical_7d_from_5d
from utils.token_frame import (
    frame_idx_to_token_idx,
    num_tokens_for_frame_len,
    prefix_len_from_tail_invalid,
    token_range_to_frame_slice,
    token_start_frame,
)
from utils.traj_batch import encode_traj_batch


@dataclass(frozen=True)
class StreamStepInput:
    """Inputs passed into one LDF streaming step."""

    text: str
    traj_input: dict | None = None

    def as_dict(self) -> dict:
        payload = {"text": [str(self.text)]}
        if self.traj_input:
            payload.update(self.traj_input)
        return payload


class StreamGenerator:
    """Owns streaming model execution state.

    Model/config/checkpoint construction stays outside this class. The runtime
    receives already-instantiated modules and only coordinates their execution.
    """

    def __init__(
        self,
        *,
        ldf_model,
        condition_manager: ConditionManager | None = None,
        root_refiner=None,
        root_text_encoder=None,
        device=None,
        token_dt: float = 0.20,
        history_length: int = 30,
        traj_horizon_tokens: int = 20,
        default_anchor_y: float = 1.0,
        vae=None,
    ):
        self.ldf_model = ldf_model
        self.condition_manager = condition_manager or ConditionManager()
        self.root_refiner = root_refiner
        self.root_text_encoder = root_text_encoder
        self.device = torch.device(
            device if device is not None else next(ldf_model.parameters()).device
        )
        self.token_dt = float(token_dt)
        self.history_length = int(history_length)
        self.traj_horizon_tokens = int(traj_horizon_tokens)
        self.default_anchor_y = float(default_anchor_y)
        # ``vae`` remains accepted for one compatibility release so old
        # constructors do not break; execution ownership belongs to the session.
        del vae
        self._runtime_session = None

    @property
    def batch_size(self) -> int:
        return int(getattr(self.ldf_model, "batch_size", 1))

    @property
    def chunk_size(self) -> int:
        return int(getattr(self.ldf_model, "chunk_size", 1))

    def reset(self, initial_state: RootFrameState | None = None, *, text: str = "") -> None:
        if self._runtime_session is not None:
            raise RuntimeError(
                "reset is owned by the authoritative runtime session; "
                "submit ResetSession instead"
            )
        del initial_state
        self.condition_manager.reset(text=text)

    def init_ldf_generation(
        self,
        *,
        history_length: int | None = None,
        batch_size: int = 1,
        num_denoise_steps: int | None = None,
    ) -> None:
        if self._runtime_session is not None:
            raise RuntimeError(
                "LDF initialization is owned by the authoritative runtime session"
            )
        if history_length is not None:
            self.history_length = int(history_length)
        self.ldf_model.init_generated(
            self.history_length,
            batch_size=int(batch_size),
            num_denoise_steps=num_denoise_steps,
            traj_buffer=None,
        )

    def build_step_input(self, text: str | None = None, traj_input: dict | None = None) -> dict:
        if text is None:
            text = self.condition_manager.text.text_at(
                int(getattr(self.ldf_model, "commit_index", 0))
            )
        return StreamStepInput(text=str(text), traj_input=traj_input).as_dict()

    @torch.no_grad()
    def build_root_plan(
        self,
        *,
        text: str,
        route: RoutePlan,
        anchor_state: RootFrameState,
        history_motion_world_5d=None,
        forced_num_frames: int | None = None,
        anchor_world_y=None,
        path_condition: RootRefinerPathCondition | None = None,
    ) -> RootPlan:
        if self.root_refiner is None or self.root_text_encoder is None:
            raise RuntimeError("RootRefiner modules are not configured.")
        refiner = self.root_refiner.to(self.device).eval()
        text_encoder = self.root_text_encoder.to(self.device).eval()

        anchor_xz = anchor_state.world_xz.to(device=self.device, dtype=torch.float32)
        anchor_yaw = anchor_state.world_yaw.to(device=self.device, dtype=torch.float32)
        anchor_y = self._resolve_anchor_y(
            anchor_state=anchor_state,
            history_motion_world_5d=history_motion_world_5d,
            anchor_world_y=anchor_world_y,
        )
        max_frames = int(refiner.max_frames)
        condition = path_condition
        if condition is None:
            condition = self.condition_manager.route.build_root_refiner_path_condition_for_route(
                route,
                anchor_state=anchor_state,
                n_path=int(refiner.n_path),
                valid_frame_count=max_frames,
                max_frames=max_frames,
            )
        if condition is None:
            raise ValueError("route produced no valid RootRefiner path condition")

        history_motion, history_mask = self._history_from_world_5d(
            history_motion_world_5d,
            anchor_xz,
            anchor_yaw,
            anchor_y,
        )
        text_emb = text_encoder.encode([str(text)], device=self.device)
        forced_num_frames_t = None
        if forced_num_frames is not None:
            forced_num_frames_t = torch.as_tensor(
                [int(forced_num_frames)],
                dtype=torch.long,
                device=self.device,
            )

        output = refiner(
            text_emb=text_emb,
            path=condition.path.to(self.device).unsqueeze(0),
            path_valid_mask=condition.path_valid_mask.to(self.device).unsqueeze(0),
            path_control_mask=condition.path_control_mask.to(self.device).unsqueeze(0),
            path_features=condition.path_features.to(self.device).unsqueeze(0),
            path_features_raw=condition.path_features_raw.to(self.device).unsqueeze(0),
            history_motion=history_motion,
            history_mask=history_mask,
            anchor_frame=torch.zeros(1, dtype=torch.long, device=self.device),
            num_frames=forced_num_frames_t,
        )

        used_future_frames = int(output["used_frames"][0].detach().cpu().item())
        valid_frames = min(used_future_frames + 1, int(output["waypoints"].shape[1]) + 1)
        frames_per_token = 4
        num_tokens_pred = num_tokens_for_frame_len(valid_frames, frames_per_token)
        waypoints_7d = build_physical_7d_from_5d(output["waypoints"][0])
        anchor_7d = waypoints_7d.new_zeros(1, 7)
        anchor_7d[:, 1] = anchor_y.to(device=waypoints_7d.device, dtype=waypoints_7d.dtype)
        anchor_7d[:, 3] = 1.0
        waypoints_7d = torch.cat([anchor_7d, waypoints_7d], dim=0)
        return RootPlan(
            num_tokens_pred=num_tokens_pred,
            valid_frames=valid_frames,
            waypoints_local_7d=waypoints_7d[:valid_frames],
            frame_dt=self.token_dt / float(frames_per_token),
            frames_per_token=frames_per_token,
            anchor_commit_idx=int(anchor_state.commit_idx),
            anchor_world_xz=anchor_xz,
            anchor_world_yaw=anchor_yaw,
            source="root_refiner_forced" if forced_num_frames is not None else "root_refiner",
        )

    def build_ldf_condition_provider(
        self,
        step_input: dict,
        *,
        first_chunk: bool,
        device=None,
    ):
        device = self.device if device is None else torch.device(device)
        model = self.ldf_model
        batch_size = int(getattr(model, "batch_size", 1))
        use_text_cond = bool(getattr(model, "use_text_cond", True))
        if use_text_cond and "text" in step_input:
            new_text_context = self._encode_stream_text(step_input["text"], device)
        else:
            new_text_context = self._encode_stream_text([""] * batch_size, device)

        if not hasattr(model, "batch_size"):
            model.batch_size = batch_size
        if not hasattr(model, "text_condition_list"):
            model.text_condition_list = [[] for _ in range(batch_size)]
        param_dtype = getattr(model, "param_dtype", torch.float32)
        new_text_context = [item.to(param_dtype) for item in new_text_context]
        text_null_context = [
            item.to(param_dtype)
            for item in self._encode_stream_text([""] * batch_size, device)
        ]

        for batch_idx in range(batch_size):
            if first_chunk:
                model.text_condition_list[batch_idx].extend(
                    [new_text_context[batch_idx]] * int(model.chunk_size)
                )
            else:
                model.text_condition_list[batch_idx].append(new_text_context[batch_idx])

        def provider(*, end_index, model_sl, window_start_token, time_steps, device):
            del time_steps
            text_context = []
            for batch_idx in range(batch_size):
                text_context.extend(
                    model.text_condition_list[batch_idx][:end_index][-model.seq_len :]
                )

            attn_len = int(model_sl)
            traj_emb = None
            traj_seq_lens = None
            traj_token_mask = None
            if _has_direct_traj_payload(step_input):
                traj_emb, traj_seq_lens, traj_token_mask = (
                    build_stream_direct_traj_condition(
                        step_input,
                        model_sl,
                        window_start_token,
                        device,
                        batch_size=batch_size,
                        traj_encoder=model.traj_encoder,
                    )
                )
                if traj_emb is not None:
                    attn_len = max(attn_len, int(traj_emb.shape[1]))

            text_context = extend_stream_text_context(
                text_context,
                batch_size,
                model_sl,
                model_sl,
            )
            return LDFCondition(
                text_context=text_context,
                text_null_context=text_null_context,
                traj_emb=traj_emb,
                traj_seq_lens=traj_seq_lens,
                traj_token_mask=traj_token_mask,
                seq_len=model_sl,
                attn_len=attn_len,
            )

        return provider

    def step(self, num_denoise_steps: int | None = None) -> dict:
        if num_denoise_steps is not None:
            raise ValueError(
                "num_denoise_steps is a runtime command; submit it before step()"
            )
        return self.execute_step()

    def generate_token(
        self,
        text: str,
        payload: dict | None,
        *,
        first_chunk: bool,
        num_denoise_steps: int | None = None,
    ) -> KernelStepResult:
        """Generate one latent token from an already-built model payload.

        Route selection, absolute slicing, VAE decoding, recovery, and timeline
        ownership deliberately live outside this kernel boundary.
        """
        model = self.ldf_model
        if num_denoise_steps is not None:
            model.num_denoise_steps = int(num_denoise_steps)

        pre_metadata = (
            model.stream_buffer_metadata()
            if hasattr(model, "stream_buffer_metadata")
            else None
        )
        pre_start_abs = int(
            getattr(
                pre_metadata,
                "start_commit_abs",
                getattr(model, "latent_buffer_start_commit_abs", 0),
            )
        )
        model_local_before = int(getattr(model, "commit_index", 0))
        absolute_commit_before = pre_start_abs + model_local_before

        step_input = self.build_step_input(text=str(text), traj_input=payload)
        provider = self.build_ldf_condition_provider(
            step_input,
            first_chunk=bool(first_chunk),
            device=self.device,
        )
        output = model.stream_generate_step(
            step_input,
            first_chunk=bool(first_chunk),
            condition=provider,
        )
        generated = output.get("generated") if isinstance(output, dict) else None
        if not torch.is_tensor(generated) or generated.dim() != 3:
            raise ValueError(
                "stream_generate_step must return generated [B,T,C], got "
                f"{None if generated is None else tuple(generated.shape)}"
            )
        if int(generated.shape[1]) != 1:
            raise ValueError(
                "generate_token requires exactly one committed token; "
                f"got {int(generated.shape[1])}"
            )

        post_metadata = (
            model.stream_buffer_metadata()
            if hasattr(model, "stream_buffer_metadata")
            else None
        )
        post_start_abs = int(
            getattr(
                post_metadata,
                "start_commit_abs",
                getattr(model, "latent_buffer_start_commit_abs", 0),
            )
        )
        post_epoch = int(
            getattr(
                post_metadata,
                "epoch",
                getattr(model, "latent_buffer_epoch", 0),
            )
        )
        local_after = int(getattr(model, "commit_index", model_local_before + 1))
        # Express both indices in the post-step buffer coordinate system. This
        # preserves the one-step relation even when this token triggers a roll.
        local_before = absolute_commit_before - post_start_abs
        return KernelStepResult(
            raw_latent=generated[0],
            actual_payload=payload,
            local_commit_before=local_before,
            local_commit_after=local_after,
            latent_buffer_start_commit_abs=post_start_abs,
            latent_buffer_epoch=post_epoch,
        )

    def attach_runtime_session(self, session) -> None:
        """Attach the sole execution owner used by compatibility calls."""
        if getattr(session, "kernel", None) is not self:
            raise ValueError("runtime session kernel must be this StreamGenerator")
        self._runtime_session = session

    def execute_step(
        self,
        *,
        text: str | None = None,
        traj_input: dict | None = None,
        num_denoise_steps: int | None = None,
    ):
        """Compatibility delegate to the authoritative runtime session."""
        if self._runtime_session is None:
            raise RuntimeError(
                "StreamGenerator.execute_step requires an attached "
                "StreamRuntimeSession; direct legacy execution was removed"
            )
        if text is not None or traj_input is not None or num_denoise_steps is not None:
            raise ValueError(
                "per-step overrides are runtime commands; submit them to the "
                "attached StreamRuntimeSession before execute_step()"
            )
        return self._runtime_session.step()

    def _resolve_anchor_y(
        self,
        *,
        anchor_state,
        history_motion_world_5d=None,
        anchor_world_y=None,
    ) -> torch.Tensor:
        value = anchor_world_y
        if value is None:
            value = getattr(anchor_state, "world_y", None)
        if value is None and history_motion_world_5d is not None:
            hist = torch.as_tensor(
                history_motion_world_5d,
                device=self.device,
                dtype=torch.float32,
            )
            if hist.ndim == 2 and hist.shape[0] > 0 and hist.shape[-1] == 5:
                value = hist[-1, 1]
        if value is None:
            value = self.default_anchor_y
        return torch.as_tensor(value, device=self.device, dtype=torch.float32).reshape(())

    def _history_anchor_only(self, anchor_y: torch.Tensor):
        n_hist = int(self.root_refiner.n_hist)
        history = torch.zeros(n_hist, 5, device=self.device, dtype=torch.float32)
        history[-1, 1] = anchor_y
        history[-1, 3] = 1.0
        mask = torch.zeros(n_hist, device=self.device, dtype=torch.bool)
        mask[-1] = True
        return history.unsqueeze(0), mask.unsqueeze(0)

    def _history_from_world_5d(
        self,
        history_motion_world_5d,
        anchor_xz,
        anchor_yaw,
        anchor_y,
    ):
        if history_motion_world_5d is None:
            return self._history_anchor_only(anchor_y)
        history_world = torch.as_tensor(
            history_motion_world_5d,
            device=self.device,
            dtype=torch.float32,
        )
        if history_world.ndim != 2 or history_world.shape[-1] != 5:
            raise ValueError(
                "history_motion_world_5d must be [T,5], got "
                f"{tuple(history_world.shape)}"
            )
        if history_world.shape[0] <= 0:
            return self._history_anchor_only(anchor_y)

        n_hist = int(self.root_refiner.n_hist)
        history_world = history_world[-n_hist:]
        history_local = canonicalize_5d(history_world, anchor_xz, anchor_yaw)
        valid = int(history_local.shape[0])
        if valid < n_hist:
            pad = history_local.new_zeros(n_hist - valid, 5)
            history_local = torch.cat([pad, history_local], dim=0)
        mask = torch.zeros(n_hist, device=self.device, dtype=torch.bool)
        mask[n_hist - valid :] = True
        return history_local.unsqueeze(0), mask.unsqueeze(0)

    def _encode_stream_text(self, text_list, device) -> list[torch.Tensor]:
        encode = getattr(self.ldf_model, "encode_text_with_cache", None)
        if encode is None:
            return [torch.zeros(1, 1, device=device) for _ in text_list]
        return encode(text_list, device)


def build_stream_direct_traj_condition(
    batch,
    model_sl: int,
    window_start_token: int,
    device,
    *,
    batch_size: int,
    traj_encoder,
    traj_sl: int | None = None,
):
    """Encode explicit frame-level 7D stream trajectory payload."""
    subpayloads = batch.get("traj_substep_payloads")
    if subpayloads:
        selected = None
        for subpayload in subpayloads:
            if int(subpayload.get("traj_start_token", -1)) == int(window_start_token):
                selected = subpayload
                break
        if selected is None:
            starts = [
                int(subpayload.get("traj_start_token", -1))
                for subpayload in subpayloads
            ]
            raise ValueError(
                "stream 7D payload has no substep payload for "
                f"window_start_token={window_start_token}; starts={starts}"
            )
        batch = selected

    traj_frame = batch["traj_cond_7d_frame"]
    if isinstance(traj_frame, np.ndarray):
        traj_frame = torch.from_numpy(traj_frame).float()
    traj_frame = traj_frame.to(device=device)
    if traj_frame.dim() == 2:
        traj_frame = traj_frame.unsqueeze(0)
    if traj_frame.dim() != 3 or traj_frame.shape[-1] != 7:
        raise ValueError(
            "traj_cond_7d_frame must be [B,T_frame,7] or [T_frame,7], "
            f"got {tuple(traj_frame.shape)}"
        )
    if traj_frame.shape[0] != batch_size:
        raise ValueError(
            f"traj_cond_7d_frame batch size {traj_frame.shape[0]} does not "
            f"match stream batch_size {batch_size}"
        )

    payload_local_start = int(batch.get("traj_start_token", window_start_token))
    payload_abs_start = int(batch.get("traj_abs_start_token", payload_local_start))
    if payload_local_start > window_start_token:
        raise ValueError(
            "stream 7D payload starts after current latent window start; got "
            f"traj_start_token={payload_local_start}, "
            f"window_start_token={window_start_token}."
        )
    payload_num_tokens = batch.get("traj_num_tokens", None)
    if payload_num_tokens is not None:
        payload_num_tokens = int(payload_num_tokens)
        if payload_num_tokens < model_sl:
            raise ValueError(
                "stream traj_num_tokens must be >= model_sl; got "
                f"traj_num_tokens={payload_num_tokens}, model_sl={model_sl}."
            )
    if traj_sl is None:
        traj_sl = _infer_stream_traj_len(
            traj_frame,
            payload_num_tokens,
            payload_abs_start,
            payload_local_start,
            window_start_token,
            model_sl,
        )
    traj_sl = int(traj_sl)
    if traj_sl < model_sl:
        raise ValueError(
            f"stream traj_sl must be >= model_sl; got traj_sl={traj_sl}, "
            f"model_sl={model_sl}."
        )

    window_abs_start = payload_abs_start + (window_start_token - payload_local_start)
    if payload_local_start < window_start_token:
        traj_frame, traj_sl, rel_start, rel_stop = _crop_stream_traj_frame(
            traj_frame,
            payload_num_tokens,
            payload_abs_start,
            payload_local_start,
            window_start_token,
            model_sl,
        )
    else:
        rel_start = rel_stop = None

    traj_payload = {
        "traj_features": traj_frame,
        "traj_start_token": window_abs_start,
    }
    traj_mask = batch.get("traj_cond_frame_mask", batch.get("traj_cond_mask"))
    if traj_mask is not None:
        traj_mask = _prepare_stream_traj_mask(
            traj_mask,
            batch_size,
            device,
            payload_local_start,
            window_start_token,
            rel_start,
            rel_stop,
        )
        traj_payload["traj_cond_mask"] = traj_mask

    traj_emb, traj_token_mask = encode_traj_batch(
        traj_payload,
        traj_sl,
        device,
        traj_encoder,
        return_token_mask=True,
    )
    if traj_emb is None:
        return None, None, None
    if traj_token_mask is not None:
        traj_seq_lens = prefix_len_from_tail_invalid(traj_token_mask).to(device=device)
    else:
        traj_seq_lens = torch.full(
            (batch_size,),
            traj_sl,
            device=device,
            dtype=torch.long,
        )
    return traj_emb, traj_seq_lens, traj_token_mask


def extend_stream_text_context(text_condition, batch_size: int, model_sl: int, target_sl: int):
    """Pad frame-aligned stream text context to the latent segment length."""
    if target_sl <= model_sl:
        return text_condition
    if len(text_condition) != batch_size * model_sl:
        if len(text_condition) == batch_size:
            return text_condition
        return text_condition
    out = []
    for batch_idx in range(batch_size):
        segment = list(text_condition[batch_idx * model_sl : (batch_idx + 1) * model_sl])
        if not segment:
            continue
        out.extend(segment)
        out.extend([segment[-1]] * (target_sl - model_sl))
    return out


def _has_direct_traj_payload(step_input: dict) -> bool:
    return (
        step_input.get("traj_cond_7d_frame") is not None
        or step_input.get("traj_substep_payloads") is not None
    )


def _infer_stream_traj_len(
    traj_frame,
    payload_num_tokens,
    payload_abs_start,
    payload_local_start,
    window_start_token,
    model_sl,
):
    if payload_num_tokens is not None:
        return payload_num_tokens
    if traj_frame.shape[1] <= 0:
        return model_sl
    origin_frame = token_start_frame(payload_abs_start)
    payload_last_frame = origin_frame + int(traj_frame.shape[1]) - 1
    payload_end_token = frame_idx_to_token_idx(payload_last_frame) + 1
    window_abs_start = payload_abs_start + (window_start_token - payload_local_start)
    return max(model_sl, payload_end_token - window_abs_start)


def _crop_stream_traj_frame(
    traj_frame,
    payload_num_tokens,
    payload_abs_start,
    payload_local_start,
    window_start_token,
    model_sl,
):
    crop_tokens = window_start_token - payload_local_start
    traj_sl = model_sl
    if payload_num_tokens is not None:
        traj_sl = max(model_sl, payload_num_tokens - crop_tokens)
    window_abs_start = payload_abs_start + crop_tokens
    origin_frame = token_start_frame(payload_abs_start)
    needed = token_range_to_frame_slice(window_abs_start, traj_sl)
    rel_start = needed.start - origin_frame
    rel_stop = needed.stop - origin_frame
    if rel_start >= traj_frame.shape[1]:
        traj_frame = traj_frame[:, :0, :]
    else:
        traj_frame = traj_frame[
            :,
            max(0, rel_start) : min(rel_stop, traj_frame.shape[1]),
            :,
        ]
    return traj_frame, traj_sl, rel_start, rel_stop


def _prepare_stream_traj_mask(
    traj_mask,
    batch_size,
    device,
    payload_local_start,
    window_start_token,
    rel_start,
    rel_stop,
):
    if isinstance(traj_mask, np.ndarray):
        traj_mask = torch.from_numpy(traj_mask).float()
    traj_mask = traj_mask.to(device=device)
    if traj_mask.dim() == 1:
        traj_mask = traj_mask.unsqueeze(0)
    if traj_mask.shape[0] != batch_size:
        raise ValueError(
            f"traj_cond_frame_mask batch size {traj_mask.shape[0]} does not "
            f"match stream batch_size {batch_size}"
        )
    if payload_local_start < window_start_token:
        if rel_start >= traj_mask.shape[1]:
            traj_mask = traj_mask[:, :0]
        else:
            traj_mask = traj_mask[
                :,
                max(0, rel_start) : min(rel_stop, traj_mask.shape[1]),
            ]
    return traj_mask


__all__ = [
    "StreamGenerator",
    "StreamStepInput",
    "build_stream_direct_traj_condition",
    "extend_stream_text_context",
]
