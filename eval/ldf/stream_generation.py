"""Single-sample generation helpers for LDF stream evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from eval.ldf.conditioning import LdfEvalStreamConditioner
from metrics.stream import decode_stream_chunks
from utils.inference.stream_generator import StreamGenerator
from utils.motion_process import StreamJointRecovery263
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
):
    total_tokens = _to_python_int(sample_batch["token_length"][0])
    total_frames = _to_python_int(sample_batch["feature_length"][0])
    step_count = total_tokens

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

    try:
        for commit_index in range(step_count):
            current_text = text_rollout.get_text_for_commit_index(commit_index)
            if stream_conditioner is not None:
                local_commit_index = int(getattr(model, "commit_index", commit_index))
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
            decoded_chunk = vae.stream_decode(
                output["generated"][0][None, :], first_chunk=first_chunk
            )[0].float().detach().cpu()
            first_chunk = False

            latent_tokens.append(latent_token)
            decoded_chunks.append(decoded_chunk)
            generated_frames += decoded_chunk.shape[0]
            chunk_frame_ends.append(min(generated_frames, total_frames))
            if stream_conditioner is not None and stream_recovery is not None:
                stream_conditioner.append_decoded(
                    decoded_chunk,
                    commit_idx=commit_index + 1,
                    recovery=stream_recovery,
                )
    finally:
        vae.clear_cache()

    decoded_feature = (
        torch.cat(decoded_chunks, dim=0)[:total_frames]
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
