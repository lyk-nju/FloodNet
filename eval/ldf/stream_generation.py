"""Single-sample generation helpers for LDF stream evaluation."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from eval.ldf.conditioning import LdfEvalStreamConditioner
from metrics.stream import decode_stream_chunks
from utils.inference.stream_generator import StreamGenerator
from utils.motion_process import (
    StreamJointRecovery263,
    recover_root_rot_pos,
    replace_root_channels_263_window_from_7d,
)
from utils.token_frame import num_tokens_for_frame_len
from utils.training.ldf.validation_conditioning import prepare_ldf_eval_model_batch


@dataclass(frozen=True)
class StreamBestOfKConfig:
    k: int = 1
    score: str = "xz"
    xz_weight: float = 1.0
    fde_weight: float = 1.0
    cont_weight: float = 0.0
    vel_weight: float = 0.5
    rel_margin: float = 0.10
    abs_margin: float = 0.03
    cont_tol: float = 0.03
    force_candidate0: bool = False
    switch_cooldown_steps: int = 0
    debug: bool = False

    @classmethod
    def from_values(
        cls,
        *,
        k: int = 1,
        score: str = "xz",
        xz_weight: float = 1.0,
        fde_weight: float = 1.0,
        cont_weight: float = 0.0,
        vel_weight: float = 0.5,
        rel_margin: float = 0.10,
        abs_margin: float = 0.03,
        cont_tol: float = 0.03,
        force_candidate0: bool = False,
        switch_cooldown_steps: int = 0,
        debug: bool = False,
    ) -> "StreamBestOfKConfig":
        out = cls(
            k=max(1, int(k)),
            score=str(score),
            xz_weight=float(xz_weight),
            fde_weight=float(fde_weight),
            cont_weight=float(cont_weight),
            vel_weight=float(vel_weight),
            rel_margin=float(rel_margin),
            abs_margin=float(abs_margin),
            cont_tol=float(cont_tol),
            force_candidate0=bool(force_candidate0),
            switch_cooldown_steps=int(switch_cooldown_steps),
            debug=bool(debug),
        )
        if out.score != "xz":
            raise ValueError(
                "Only eval_stream_best_of_k_score='xz' is implemented; "
                f"got {out.score!r}."
            )
        return out

    @property
    def enabled(self) -> bool:
        return int(self.k) > 1


class StreamTextRolloutController:
    def __init__(self, texts, token_ends):
        self.texts = [str(text) for text in texts]
        self.token_ends = [int(end) for end in token_ends]
        if not self.texts:
            self.texts = [""]
            self.token_ends = [0]

    @classmethod
    def from_sample_batch(cls, sample_batch):
        text = sample_batch.get("text", [""])
        if isinstance(text, list) and len(text) == 1 and isinstance(text[0], list):
            text = text[0]
        if not isinstance(text, list):
            text = [str(text)]
        token_end = sample_batch.get("token_text_end", [[0]])
        if isinstance(token_end, list) and len(token_end) == 1:
            token_end = token_end[0]
        return cls(text, token_end)

    def get_text_for_commit_index(self, commit_index: int) -> str:
        for text, token_end in zip(self.texts, self.token_ends):
            if int(commit_index) < int(token_end):
                return text
        return self.texts[-1]


def build_stream_input(sample_batch: Dict, device: torch.device, model=None) -> Dict:
    return prepare_ldf_eval_model_batch(sample_batch, device, model=model)


def _to_python_int(value) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _sample_traj7(sample_batch: Dict) -> torch.Tensor:
    traj7 = sample_batch.get("traj_cond_7d")
    if traj7 is None:
        raise ValueError("root feedback requires sample_batch['traj_cond_7d']")
    value = traj7[0] if torch.is_tensor(traj7) and traj7.ndim == 3 else traj7
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32)
    return value.float()


def _repeat_stream_batch_value(value, repeat: int):
    repeat = int(repeat)
    if torch.is_tensor(value):
        if value.dim() > 0 and int(value.shape[0]) == 1:
            return value.repeat((repeat,) + (1,) * (value.dim() - 1))
        return value
    if isinstance(value, dict):
        return {
            key: _repeat_stream_batch_value(item, repeat)
            for key, item in value.items()
        }
    if isinstance(value, list):
        if len(value) == 1 and not isinstance(value[0], dict):
            return [copy.deepcopy(value[0]) for _ in range(repeat)]
        return [_repeat_stream_batch_value(item, repeat) for item in value]
    return value


def _repeat_stream_step_payload(step_payload: Dict, repeat: int) -> Dict:
    return {
        key: _repeat_stream_batch_value(value, repeat)
        for key, value in step_payload.items()
    }


def _replace_chunk_root_from_condition(
    decoded_chunk: torch.Tensor,
    sample_batch: Dict,
    *,
    start_frame: int,
    previous_decoded_chunks: Optional[List[torch.Tensor]] = None,
    xz_blend_alpha: float = 1.0,
) -> torch.Tensor:
    target_traj = _sample_traj7(sample_batch)
    alpha = float(max(0.0, min(1.0, xz_blend_alpha)))
    if alpha < 1.0:
        needed = int(start_frame) + int(decoded_chunk.shape[0]) + 1
        target_traj = target_traj.clone()
        if target_traj.shape[0] < needed:
            tail = target_traj[-1:].expand(needed - target_traj.shape[0], -1)
            target_traj = torch.cat([target_traj, tail], dim=0)
        dummy_tail = decoded_chunk.new_zeros((1, decoded_chunk.shape[-1]))
        prefix_chunks = list(previous_decoded_chunks or []) + [decoded_chunk, dummy_tail]
        generated_prefix = torch.cat(prefix_chunks, dim=0).to(
            device=decoded_chunk.device,
            dtype=decoded_chunk.dtype,
        )
        _, generated_xyz = recover_root_rot_pos(generated_prefix.unsqueeze(0))
        generated_xyz = generated_xyz[0]
        start = max(0, int(start_frame))
        end = min(start + int(decoded_chunk.shape[0]) + 1, target_traj.shape[0])
        valid = max(0, end - start)
        if valid > 0 and generated_xyz.shape[0] >= start + valid:
            target_traj = target_traj.to(device=decoded_chunk.device, dtype=decoded_chunk.dtype)
            generated_xz = generated_xyz[start:start + valid, [0, 2]].to(
                device=target_traj.device,
                dtype=target_traj.dtype,
            )
            condition_xz = target_traj[start:start + valid, [0, 2]]
            target_traj[start:start + valid, [0, 2]] = (
                (1.0 - alpha) * generated_xz + alpha * condition_xz
            )
    return replace_root_channels_263_window_from_7d(
        decoded_chunk,
        target_traj,
        start_frame=int(start_frame),
    )


def _committed_latent_index_after_step(model, local_commit_index: int) -> int:
    local_commit_index = int(local_commit_index)
    current_commit = int(getattr(model, "commit_index", local_commit_index + 1))
    if current_commit == local_commit_index + 1:
        return local_commit_index
    seq_len = int(getattr(model, "seq_len", 0))
    if seq_len > 0 and current_commit == local_commit_index + 1 - seq_len:
        return local_commit_index - seq_len
    return max(0, current_commit - 1)


def _write_committed_latent_to_model(model, latent_token: torch.Tensor, local_commit_index: int) -> None:
    if not hasattr(model, "generated"):
        return
    generated = model.generated
    if generated is None:
        return
    device = generated.device
    dtype = generated.dtype
    latent = latent_token.detach().to(device=device, dtype=dtype)
    if latent.dim() == 1:
        latent = latent.view(1, 1, -1)
    elif latent.dim() == 2:
        latent = latent.unsqueeze(0)
    if hasattr(model, "preprocess"):
        latent_pre = model.preprocess(latent)
    else:
        latent_pre = latent.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    write_index = _committed_latent_index_after_step(model, local_commit_index)
    if 0 <= write_index < generated.shape[2]:
        generated[
            : latent_pre.shape[0],
            :,
            write_index:write_index + 1,
            ...,
        ] = latent_pre


def _clone_cache_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_cache_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cache_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_cache_value(item) for key, item in value.items()}
    return value


def _snapshot_stream_model_state(model) -> dict:
    state = {}
    for name in (
        "generated",
        "commit_index",
        "current_step",
        "batch_size",
        "seq_len",
        "num_denoise_steps",
        "dt",
        "text_condition_list",
    ):
        if hasattr(model, name):
            state[name] = _clone_cache_value(getattr(model, name))
    if hasattr(model, "_traj_buf"):
        try:
            state["_traj_buf"] = copy.deepcopy(getattr(model, "_traj_buf"))
        except Exception:
            state["_traj_buf"] = getattr(model, "_traj_buf")
    return state


def _restore_stream_model_state(model, state: dict) -> None:
    for name, value in state.items():
        if name == "_traj_buf":
            try:
                value = copy.deepcopy(value)
            except Exception:
                pass
        else:
            value = _clone_cache_value(value)
        setattr(model, name, value)


def _expand_stream_model_state_for_candidates(model, candidate_count: int) -> None:
    candidate_count = int(candidate_count)
    generated = getattr(model, "generated", None)
    if not torch.is_tensor(generated):
        raise RuntimeError("best-of-K stream proposal requires model.generated tensor state")
    if int(generated.shape[0]) == candidate_count:
        expanded = generated.detach().clone()
    elif int(generated.shape[0]) == 1:
        expanded = generated.expand(
            (candidate_count,) + tuple(generated.shape[1:])
        ).detach().clone()
    else:
        raise RuntimeError(
            "best-of-K stream proposal expects current batch size 1; "
            f"got generated batch {generated.shape[0]}."
        )
    model.generated = expanded
    model.batch_size = candidate_count
    text_condition_list = getattr(model, "text_condition_list", [[]])
    if len(text_condition_list) == 1:
        model.text_condition_list = [
            _clone_cache_value(text_condition_list[0])
            for _ in range(candidate_count)
        ]
    elif len(text_condition_list) != candidate_count:
        model.text_condition_list = [
            _clone_cache_value(text_condition_list[0] if text_condition_list else [])
            for _ in range(candidate_count)
        ]


def _select_candidate_stream_model_state(model, candidate_idx: int) -> None:
    candidate_idx = int(candidate_idx)
    generated = getattr(model, "generated", None)
    if torch.is_tensor(generated) and generated.shape[0] > 1:
        model.generated = generated[candidate_idx : candidate_idx + 1].detach().clone()
    text_condition_list = getattr(model, "text_condition_list", None)
    if isinstance(text_condition_list, list) and len(text_condition_list) > 1:
        model.text_condition_list = [
            _clone_cache_value(text_condition_list[candidate_idx])
        ]
    model.batch_size = 1


def _randomize_candidate_future_noise(model, local_commit_index: int) -> None:
    generated = getattr(model, "generated", None)
    if not torch.is_tensor(generated) or generated.shape[0] <= 1:
        return
    start = max(0, int(local_commit_index))
    if start >= int(generated.shape[2]):
        return
    # Keep candidate 0 as the default stream proposal for this state. Best-of-K
    # should add alternatives, not silently replace the K=1 proposal.
    generated[1:, :, start:, ...] = torch.randn_like(generated[1:, :, start:, ...])


def _snapshot_vae_decode_cache(vae):
    model = getattr(vae, "model", None)
    if model is None:
        return None
    cache = {}
    for name in ("_conv_num", "_conv_idx", "_feat_map"):
        if hasattr(model, name):
            cache[name] = _clone_cache_value(getattr(model, name))
    return cache


def _restore_vae_decode_cache(vae, cache) -> None:
    if cache is None:
        return
    model = getattr(vae, "model", None)
    if model is None:
        return
    for name, value in cache.items():
        setattr(model, name, _clone_cache_value(value))


def _encode_corrected_chunk_token(
    vae,
    corrected_chunk: torch.Tensor,
    *,
    first_chunk: bool,
    device: torch.device,
) -> torch.Tensor:
    encoded = vae.stream_encode(
        corrected_chunk.to(device=device).unsqueeze(0),
        first_chunk=first_chunk,
    )[0].detach()
    return encoded[-1:].detach().cpu()


def _decode_latent_chunk(
    vae,
    latent_token: torch.Tensor,
    *,
    first_chunk: bool,
    device: torch.device,
) -> torch.Tensor:
    current = latent_token.to(device=device)
    return vae.stream_decode(
        current.unsqueeze(0),
        first_chunk=first_chunk,
    )[0].float().detach().cpu()


def _decode_raw_chunk_preserving_feedback_cache(
    vae,
    latent_token: torch.Tensor,
    *,
    first_chunk: bool,
    device: torch.device,
) -> torch.Tensor:
    cache = _snapshot_vae_decode_cache(vae)
    decoded = _decode_latent_chunk(
        vae,
        latent_token,
        first_chunk=first_chunk,
        device=device,
    )
    _restore_vae_decode_cache(vae, cache)
    return decoded


def _candidate_latents_from_generated_output(generated, expected_count: int) -> torch.Tensor:
    if torch.is_tensor(generated):
        if generated.dim() == 3:
            latents = generated[:, 0, :]
        elif generated.dim() == 2:
            latents = generated
        elif generated.dim() == 1:
            latents = generated.view(1, -1)
        else:
            raise ValueError(
                "stream_generate_step generated tensor must be [B,1,C], [B,C], or [C]; "
                f"got {tuple(generated.shape)}"
            )
    elif isinstance(generated, list):
        items = []
        for item in generated:
            if item is None:
                continue
            if item.dim() == 2:
                items.append(item[0])
            elif item.dim() == 1:
                items.append(item)
            else:
                raise ValueError(
                    "stream_generate_step generated list entries must be [1,C] or [C]; "
                    f"got {tuple(item.shape)}"
                )
        latents = torch.stack(items, dim=0) if items else torch.zeros(0)
    else:
        raise TypeError(f"Unsupported generated output type: {type(generated)!r}")
    if int(latents.shape[0]) != int(expected_count):
        raise ValueError(
            f"best-of-K expected {expected_count} candidates, got {latents.shape[0]}"
        )
    return latents.detach().cpu()


def _target_xz_slice(sample_batch: Dict, *, start_frame: int, frame_count: int) -> torch.Tensor:
    target = _sample_traj7(sample_batch)
    needed = int(start_frame) + int(frame_count)
    if target.shape[0] < needed:
        pad = target[-1:].expand(needed - target.shape[0], -1)
        target = torch.cat([target, pad], dim=0)
    return target[int(start_frame) : needed, [0, 2]].float().cpu()


def _decoded_chunk_root_xz(
    decoded_chunk: torch.Tensor,
    previous_decoded_chunks: Optional[List[torch.Tensor]],
) -> torch.Tensor:
    chunks = list(previous_decoded_chunks or []) + [decoded_chunk]
    full = torch.cat(chunks, dim=0)
    _, root_xyz = recover_root_rot_pos(full.unsqueeze(0))
    start = int(full.shape[0] - decoded_chunk.shape[0])
    return root_xyz[0, start : start + decoded_chunk.shape[0], [0, 2]].float().cpu()


def _stream_candidate_xz_score(
    decoded_chunk: torch.Tensor,
    sample_batch: Dict,
    *,
    start_frame: int,
    previous_decoded_chunks: Optional[List[torch.Tensor]],
    cfg: StreamBestOfKConfig,
) -> tuple[float, dict]:
    pred_xz = _decoded_chunk_root_xz(decoded_chunk, previous_decoded_chunks)
    target_xz = _target_xz_slice(
        sample_batch,
        start_frame=int(start_frame),
        frame_count=int(pred_xz.shape[0]),
    )
    valid = min(int(pred_xz.shape[0]), int(target_xz.shape[0]))
    if valid <= 0:
        return 0.0, {"xz_mean": 0.0, "xz_fde": 0.0, "continuity": 0.0}
    pred_xz = pred_xz[:valid]
    target_xz = target_xz[:valid]
    per_frame = torch.linalg.norm(pred_xz - target_xz, dim=-1)
    xz_mean = float(per_frame.mean().item())
    xz_fde = float(per_frame[-1].item())
    continuity = 0.0
    if float(cfg.cont_weight) != 0.0 and previous_decoded_chunks:
        previous_full = torch.cat(previous_decoded_chunks, dim=0)
        _, previous_xyz = recover_root_rot_pos(previous_full.unsqueeze(0))
        previous_xz = previous_xyz[0, -1, [0, 2]].float().cpu()
        continuity = float(torch.linalg.norm(pred_xz[0] - previous_xz).item())
    score = (
        float(cfg.xz_weight) * xz_mean
        + float(cfg.fde_weight) * xz_fde
        + float(cfg.cont_weight) * continuity
    )
    return float(score), {
        "xz_mean": xz_mean,
        "xz_fde": xz_fde,
        "continuity": continuity,
    }


def _stream_generate_step_best_of_k(
    *,
    model,
    vae,
    stream: StreamGenerator,
    step_payload: Dict,
    sample_batch: Dict,
    first_chunk: bool,
    device: torch.device,
    local_commit_index: int,
    generated_frames: int,
    previous_decoded_chunks: Optional[List[torch.Tensor]],
    cfg: StreamBestOfKConfig,
) -> dict:
    started = time.perf_counter()
    base_state = _snapshot_stream_model_state(model)
    base_vae_cache = _snapshot_vae_decode_cache(vae)
    try:
        # Propose K candidates by temporarily expanding the streaming state.
        # The model still auto-commits internally, so we slice it back to the
        # selected candidate below before the next stream step can observe it.
        _expand_stream_model_state_for_candidates(model, cfg.k)
        _randomize_candidate_future_noise(model, local_commit_index)
        batched_payload = _repeat_stream_step_payload(step_payload, cfg.k)
        condition_provider = stream.build_ldf_condition_provider(
            batched_payload,
            first_chunk=first_chunk,
            device=device,
        )
        output = model.stream_generate_step(
            batched_payload,
            first_chunk=first_chunk,
            condition=condition_provider,
        )
        candidate_latents = _candidate_latents_from_generated_output(
            output["generated"],
            cfg.k,
        )
        scores: list[float] = []
        score_parts: list[dict] = []
        for candidate_idx in range(cfg.k):
            _restore_vae_decode_cache(vae, base_vae_cache)
            latent_token = candidate_latents[candidate_idx : candidate_idx + 1]
            decoded = _decode_raw_chunk_preserving_feedback_cache(
                vae,
                latent_token,
                first_chunk=first_chunk,
                device=device,
            )
            score, parts = _stream_candidate_xz_score(
                decoded,
                sample_batch,
                start_frame=generated_frames,
                previous_decoded_chunks=previous_decoded_chunks,
                cfg=cfg,
            )
            scores.append(score)
            score_parts.append(parts)

        selected_idx = int(torch.as_tensor(scores).argmin().item())
        # Commit selected latent only: keep the selected batch row of
        # model.generated/text_condition_list and discard all other proposals.
        _select_candidate_stream_model_state(model, selected_idx)
        _restore_vae_decode_cache(vae, base_vae_cache)
        selected_latent = candidate_latents[selected_idx : selected_idx + 1]
        selected_decoded = _decode_latent_chunk(
            vae,
            selected_latent,
            first_chunk=first_chunk,
            device=device,
        )
        diversity = 0.0
        if candidate_latents.shape[0] > 1:
            diversity = float(
                (candidate_latents - candidate_latents[:1]).abs().max().item()
            )
        return {
            "latent_token": selected_latent,
            "decoded_chunk": selected_decoded,
            "record": {
                "selected_idx": selected_idx,
                "scores": [float(value) for value in scores],
                "selected_score": float(scores[selected_idx]),
                "score_parts": score_parts,
                "latent_pairwise_max_abs": diversity,
                "elapsed_sec": float(time.perf_counter() - started),
            },
        }
    except Exception:
        _restore_stream_model_state(model, base_state)
        _restore_vae_decode_cache(vae, base_vae_cache)
        raise


def run_stream_generate_sample(model, vae, sample_batch: Dict, device: torch.device, num_denoise_steps: Optional[int]):
    model_batch = build_stream_input(sample_batch, device, model=model)
    latent_chunks: List[torch.Tensor] = []
    for output in model.stream_generate(model_batch, num_denoise_steps=num_denoise_steps):
        latent_chunk = output["generated"][0]
        if latent_chunk is None or latent_chunk.shape[0] == 0:
            continue
        latent_chunks.append(latent_chunk.detach())

    decoded_feature, decoded_chunks, chunk_frame_ends = decode_stream_chunks(vae, latent_chunks)
    latent_stream = (
        torch.cat([chunk.detach().cpu() for chunk in latent_chunks], dim=0)
        if latent_chunks
        else torch.zeros((0, model.input_dim), dtype=torch.float32)
    )
    return {
        "decoded_feature": decoded_feature,
        "decoded_chunks": decoded_chunks,
        "chunk_frame_ends": chunk_frame_ends,
        "latent_stream": latent_stream,
    }


def run_stream_generate_step_sample(
    model,
    vae,
    sample_batch: Dict,
    device: torch.device,
    history_length: int,
    num_denoise_steps: Optional[int],
    traj_horizon_tokens: Optional[int] = None,
    token_dt: float = 0.20,
    frames_per_token: int = 4,
    extra_frames: int = 0,
    root_replace_feedback: bool = False,
    root_feedback_xz_blend_alpha: float = 1.0,
    best_of_k: int = 1,
    best_of_k_score: str = "xz",
    best_of_k_xz_weight: float = 1.0,
    best_of_k_fde_weight: float = 1.0,
    best_of_k_cont_weight: float = 0.0,
    best_of_k_vel_weight: float = 0.5,
    best_of_k_rel_margin: float = 0.10,
    best_of_k_abs_margin: float = 0.03,
    best_of_k_cont_tol: float = 0.03,
    best_of_k_force_candidate0: bool = False,
    best_of_k_switch_cooldown_steps: int = 0,
    best_of_k_debug: bool = False,
):
    best_of_k_cfg = StreamBestOfKConfig.from_values(
        k=best_of_k,
        score=best_of_k_score,
        xz_weight=best_of_k_xz_weight,
        fde_weight=best_of_k_fde_weight,
        cont_weight=best_of_k_cont_weight,
        vel_weight=best_of_k_vel_weight,
        rel_margin=best_of_k_rel_margin,
        abs_margin=best_of_k_abs_margin,
        cont_tol=best_of_k_cont_tol,
        force_candidate0=best_of_k_force_candidate0,
        switch_cooldown_steps=best_of_k_switch_cooldown_steps,
        debug=best_of_k_debug,
    )
    if best_of_k_cfg.enabled and root_replace_feedback:
        raise ValueError(
            "best-of-K candidate selection commits the selected latent directly "
            "and cannot be combined with root replacement feedback in this "
            "minimal eval path."
        )
    if best_of_k_cfg.enabled and sample_batch.get("traj_cond_7d") is None:
        raise ValueError(
            "best-of-K xz candidate selection requires sample_batch['traj_cond_7d']."
        )
    total_tokens = _to_python_int(sample_batch["token_length"][0])
    total_frames = _to_python_int(sample_batch["feature_length"][0])
    extra_frames = max(0, int(extra_frames))
    target_total_frames = int(total_frames) + extra_frames
    target_tokens = num_tokens_for_frame_len(
        target_total_frames,
        int(frames_per_token),
    )
    step_count = max(total_tokens, target_tokens)

    if num_denoise_steps is None:
        num_denoise_steps = int(getattr(model, "noise_steps"))

    stream = StreamGenerator(
        ldf_model=model,
        device=device,
        history_length=history_length,
        traj_horizon_tokens=int(traj_horizon_tokens or 0),
        token_dt=float(token_dt),
    )
    stream.init_ldf_generation(
        history_length=history_length,
        batch_size=1,
        num_denoise_steps=num_denoise_steps,
    )
    vae.clear_cache()

    text_rollout = StreamTextRolloutController.from_sample_batch(sample_batch)
    stream_conditioner = (
        LdfEvalStreamConditioner(
            sample_batch,
            history_length=history_length,
            traj_horizon_tokens=int(traj_horizon_tokens or 0),
            token_dt=float(token_dt),
            frames_per_token=int(frames_per_token),
            device=device,
            extra_frames=extra_frames,
        )
        if "traj_cond_7d" in sample_batch and sample_batch["traj_cond_7d"] is not None
        else None
    )
    stream_recovery = (
        StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
        if stream_conditioner is not None
        else None
    )
    first_chunk = True
    latent_tokens: List[torch.Tensor] = []
    decoded_chunks: List[torch.Tensor] = []
    chunk_frame_ends: List[int] = []
    generated_frames = 0
    best_of_k_records: list[dict] = []
    best_of_k_total_elapsed_sec = 0.0

    try:
        for commit_index in range(step_count):
            current_text = text_rollout.get_text_for_commit_index(commit_index)
            local_commit_index = int(getattr(model, "commit_index", commit_index))
            if stream_conditioner is not None:
                chunk_size = int(getattr(model, "chunk_size", 1))
                traj_input = stream_conditioner.build_step_payload(
                    local_commit_index=local_commit_index,
                    absolute_commit_index=commit_index,
                    chunk_size=chunk_size,
                )
            else:
                traj_input = None
            step_payload = stream.build_step_input(
                current_text,
                traj_input=traj_input,
            )
            if best_of_k_cfg.enabled:
                proposal = _stream_generate_step_best_of_k(
                    model=model,
                    vae=vae,
                    stream=stream,
                    step_payload=step_payload,
                    sample_batch=sample_batch,
                    first_chunk=first_chunk,
                    device=device,
                    local_commit_index=local_commit_index,
                    generated_frames=generated_frames,
                    previous_decoded_chunks=decoded_chunks,
                    cfg=best_of_k_cfg,
                )
                latent_token = proposal["latent_token"]
                decoded_chunk_raw = proposal["decoded_chunk"]
                record = proposal["record"]
                record["commit_index"] = int(commit_index)
                record["local_commit_index"] = int(local_commit_index)
                best_of_k_records.append(record)
                best_of_k_total_elapsed_sec += float(record["elapsed_sec"])
            else:
                condition_provider = stream.build_ldf_condition_provider(
                    step_payload,
                    first_chunk=first_chunk,
                    device=device,
                )
                output = model.stream_generate_step(
                    step_payload,
                    first_chunk=first_chunk,
                    condition=condition_provider,
                )
                latent_token = output["generated"][0].detach().cpu()
                if root_replace_feedback:
                    decoded_chunk_raw = _decode_raw_chunk_preserving_feedback_cache(
                        vae,
                        latent_token,
                        first_chunk=first_chunk,
                        device=device,
                    )
                else:
                    decoded_chunk_raw = vae.stream_decode(
                        output["generated"][0][None, :], first_chunk=first_chunk
                    )[0].float().detach().cpu()
            chunk_start_frame = int(generated_frames)
            if root_replace_feedback:
                decoded_chunk = _replace_chunk_root_from_condition(
                    decoded_chunk_raw,
                    sample_batch,
                    start_frame=chunk_start_frame,
                    previous_decoded_chunks=decoded_chunks,
                    xz_blend_alpha=float(root_feedback_xz_blend_alpha),
                )
                corrected_latent = _encode_corrected_chunk_token(
                    vae,
                    decoded_chunk,
                    first_chunk=first_chunk,
                    device=device,
                )
                _decode_latent_chunk(
                    vae,
                    corrected_latent,
                    first_chunk=first_chunk,
                    device=device,
                )
                _write_committed_latent_to_model(
                    model,
                    corrected_latent,
                    local_commit_index,
                )
                latent_token = corrected_latent
            else:
                decoded_chunk = decoded_chunk_raw
            first_chunk = False

            latent_tokens.append(latent_token)
            decoded_chunks.append(decoded_chunk)
            generated_frames += decoded_chunk.shape[0]
            chunk_frame_ends.append(min(generated_frames, target_total_frames))
            if stream_conditioner is not None and stream_recovery is not None:
                stream_conditioner.append_decoded(
                    decoded_chunk,
                    commit_idx=commit_index + 1,
                    recovery=stream_recovery,
                )
    finally:
        vae.clear_cache()

    decoded_feature = (
        torch.cat(decoded_chunks, dim=0)[:target_total_frames]
        if decoded_chunks
        else torch.zeros((0, 263), dtype=torch.float32)
    )
    latent_stream = (
        torch.cat(latent_tokens, dim=0)
        if latent_tokens
        else torch.zeros((0, model.input_dim), dtype=torch.float32)
    )
    return {
        "decoded_feature": decoded_feature,
        "decoded_chunks": decoded_chunks,
        "chunk_frame_ends": chunk_frame_ends,
        "latent_stream": latent_stream,
        "original_total_frames": int(total_frames),
        "target_total_frames": int(target_total_frames),
        "extra_frames": int(extra_frames),
        "root_replace_feedback": bool(root_replace_feedback),
        "root_feedback_xz_blend_alpha": float(root_feedback_xz_blend_alpha),
        "stream_best_of_k": {
            "enabled": bool(best_of_k_cfg.enabled),
            "k": int(best_of_k_cfg.k),
            "score": str(best_of_k_cfg.score),
            "xz_weight": float(best_of_k_cfg.xz_weight),
            "fde_weight": float(best_of_k_cfg.fde_weight),
            "cont_weight": float(best_of_k_cfg.cont_weight),
            "vel_weight": float(best_of_k_cfg.vel_weight),
            "rel_margin": float(best_of_k_cfg.rel_margin),
            "abs_margin": float(best_of_k_cfg.abs_margin),
            "cont_tol": float(best_of_k_cfg.cont_tol),
            "force_candidate0": bool(best_of_k_cfg.force_candidate0),
            "switch_cooldown_steps": int(best_of_k_cfg.switch_cooldown_steps),
            "total_elapsed_sec": float(best_of_k_total_elapsed_sec),
            "records": best_of_k_records,
        },
    }


def run_offline_generate_sample(model, vae, sample_batch: Dict, device: torch.device, num_denoise_steps: Optional[int]):
    model_batch = build_stream_input(sample_batch, device, model=model)
    output = model.generate(model_batch, num_denoise_steps=num_denoise_steps)
    latent = output["generated"][0].detach()
    decoded = vae.decode(latent.unsqueeze(0))[0].float().detach().cpu()
    return {
        "decoded_feature": decoded,
        "latent": latent.detach().cpu(),
    }



__all__ = [
    "StreamTextRolloutController",
    "StreamBestOfKConfig",
    "build_stream_input",
    "run_offline_generate_sample",
    "run_stream_generate_sample",
    "run_stream_generate_step_sample",
]
