"""Single-sample generation helpers for LDF stream evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from metrics.stream import decode_stream_chunks
from utils.inference.runtime_update import world_traj7_to_proposal
from utils.inference.stream_generator import StreamGenerator
from utils.inference.stream_runtime import (
    ConditionComposer,
    GeneratedRootHistory,
    PayloadBuilder,
    RootSourceManager,
    RuntimeCommandQueue,
    RuntimeStepConfig,
    SetRootSource,
    SetText,
    SpaceContract,
    StreamRuntimeSession,
)
from utils.inference.timeline import RootFrameState, RootTimeline
from utils.motion_process import (
    StreamJointRecovery263,
    recover_root_rot_pos,
    replace_root_channels_263_window_from_7d,
)
from utils.token_frame import num_tokens_for_frame_len
from utils.training.ldf.validation_conditioning import prepare_ldf_eval_model_batch


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


# Compatibility helpers for explicit legacy/debug runners. The default
# stream_generate_step eval below must execute through StreamRuntimeSession.
def _committed_latent_index_after_step(model, local_commit_index: int) -> int:
    local_commit_index = int(local_commit_index)
    current_commit = int(getattr(model, "commit_index", local_commit_index + 1))
    if current_commit == local_commit_index + 1:
        return local_commit_index
    seq_len = int(getattr(model, "seq_len", 0))
    if seq_len > 0 and current_commit == local_commit_index + 1 - seq_len:
        return local_commit_index - seq_len
    return max(0, current_commit - 1)


def _write_committed_latent_to_model(model, latent_token, local_commit_index):
    generated = getattr(model, "generated", None)
    if generated is None:
        return
    latent = latent_token.detach().to(device=generated.device, dtype=generated.dtype)
    if latent.dim() == 1:
        latent = latent.view(1, 1, -1)
    elif latent.dim() == 2:
        latent = latent.unsqueeze(0)
    latent_pre = (
        model.preprocess(latent)
        if hasattr(model, "preprocess")
        else latent.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    )
    write_index = _committed_latent_index_after_step(model, local_commit_index)
    if 0 <= write_index < generated.shape[2]:
        generated[: latent_pre.shape[0], :, write_index : write_index + 1, ...] = latent_pre


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


def _snapshot_vae_decode_cache(vae):
    model = getattr(vae, "model", None)
    if model is None:
        return None
    return {
        name: _clone_cache_value(getattr(model, name))
        for name in ("_conv_num", "_conv_idx", "_feat_map")
        if hasattr(model, name)
    }


def _restore_vae_decode_cache(vae, cache) -> None:
    model = getattr(vae, "model", None)
    if model is None or cache is None:
        return
    for name, value in cache.items():
        setattr(model, name, _clone_cache_value(value))


def _encode_corrected_chunk_token(vae, corrected_chunk, *, first_chunk, device):
    encoded = vae.stream_encode(
        corrected_chunk.to(device=device).unsqueeze(0),
        first_chunk=first_chunk,
    )[0].detach()
    return encoded[-1:].detach().cpu()


def _decode_latent_chunk(vae, latent_token, *, first_chunk, device):
    return vae.stream_decode(
        latent_token.to(device=device).unsqueeze(0),
        first_chunk=first_chunk,
    )[0].float().detach().cpu()


def _decode_raw_chunk_preserving_feedback_cache(
    vae,
    latent_token,
    *,
    first_chunk,
    device,
):
    cache = _snapshot_vae_decode_cache(vae)
    decoded = _decode_latent_chunk(
        vae,
        latent_token,
        first_chunk=first_chunk,
        device=device,
    )
    _restore_vae_decode_cache(vae, cache)
    return decoded


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
):
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
    text_rollout = StreamTextRolloutController.from_sample_batch(sample_batch)
    initial_text = text_rollout.get_text_for_commit_index(0)
    timeline = RootTimeline(
        RootFrameState.initial(device=device, dtype=torch.float32)
    )
    session = StreamRuntimeSession(
        kernel=stream,
        vae=vae,
        recovery=StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0),
        timeline=timeline,
        generated_history=GeneratedRootHistory.empty(device=device),
        command_queue=RuntimeCommandQueue(),
        source_manager=RootSourceManager(),
        composer=ConditionComposer(),
        payload_builder=PayloadBuilder(frames_per_token=int(frames_per_token)),
        initial_config=RuntimeStepConfig(
            text=initial_text,
            text_guidance_scale=float(getattr(model, "cfg_scale_text", 1.0)),
            trajectory_guidance_scale=float(getattr(model, "cfg_scale_traj", 1.0)),
            root_feedback_enabled=bool(root_replace_feedback),
            root_feedback_xz_blend_alpha=float(root_feedback_xz_blend_alpha),
            history_tokens=int(history_length),
            horizon_tokens=int(traj_horizon_tokens or 0),
            num_denoise_steps=int(num_denoise_steps),
        ),
        bridge_frames=0,
    )
    stream.attach_runtime_session(session)
    command_version = 0
    traj7 = sample_batch.get("traj_cond_7d")
    if traj7 is not None:
        world_traj7 = _sample_traj7(sample_batch)
        frame_mask = sample_batch.get("traj_cond_mask", sample_batch.get("traj_mask"))
        if frame_mask is None:
            mask = torch.ones(world_traj7.shape[0], dtype=torch.bool)
        else:
            mask = torch.as_tensor(frame_mask).reshape(-1).bool()
            mask = mask[: int(world_traj7.shape[0])]
            if int(mask.shape[0]) < int(world_traj7.shape[0]):
                mask = torch.cat(
                    [
                        mask,
                        torch.zeros(
                            int(world_traj7.shape[0]) - int(mask.shape[0]),
                            dtype=torch.bool,
                        ),
                    ]
                )
        traj_length = sample_batch.get("traj_length")
        if traj_length is not None:
            valid = min(
                _to_python_int(torch.as_tensor(traj_length).reshape(-1)[0]),
                int(mask.shape[0]),
            )
            mask[valid:] = False
        command_version += 1
        session.submit(
            SetRootSource(
                version=command_version,
                requested_commit_abs=0,
                proposal=world_traj7_to_proposal(
                    world_traj7,
                    source_id=str(sample_batch.get("name", ["eval"])[0]),
                    version=command_version,
                    metadata={"source_kind": "dataset_eval"},
                    frame_mask=mask,
                    strip_anchor=True,
                ),
                space_contract=SpaceContract.WORLD_ROUTE,
            )
        )

    latent_tokens: List[torch.Tensor] = []
    decoded_chunks: List[torch.Tensor] = []
    chunk_frame_ends: List[int] = []
    generated_frames = 0
    active_text = initial_text

    try:
        for commit_index in range(step_count):
            current_text = text_rollout.get_text_for_commit_index(commit_index)
            if current_text != active_text:
                command_version += 1
                session.submit(
                    SetText(
                        version=command_version,
                        requested_commit_abs=commit_index,
                        text=current_text,
                    )
                )
                active_text = current_text
            event = session.step()
            latent_token = event.committed_latent.detach().cpu()
            decoded_chunk = event.decoded_chunk.detach().cpu()

            latent_tokens.append(latent_token)
            decoded_chunks.append(decoded_chunk)
            generated_frames += decoded_chunk.shape[0]
            chunk_frame_ends.append(min(generated_frames, target_total_frames))
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
    "build_stream_input",
    "run_offline_generate_sample",
    "run_stream_generate_sample",
    "run_stream_generate_step_sample",
]
