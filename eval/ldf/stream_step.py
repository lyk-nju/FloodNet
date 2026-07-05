"""Shared one-step runtime helper for LDF stream evaluation."""

from __future__ import annotations

from typing import Any, Dict

import torch

from eval.ldf.stream_state import StepOutput


def run_one_stream_step(
    *,
    model: Any,
    vae: Any,
    stream: Any,
    step_payload: Dict,
    first_chunk: bool,
    device: torch.device,
    local_commit_index: int,
    generated_frames: int,
    frames_per_token: int,
) -> StepOutput:
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
    latent = output["generated"][0].detach().cpu()
    decoded = vae.stream_decode(
        output["generated"][0][None, :], first_chunk=first_chunk
    )[0].float().detach().cpu()

    chunk_frames = int(decoded.shape[0])
    frame_start = int(generated_frames)
    frame_end = frame_start + chunk_frames
    token_start = int(local_commit_index)
    token_end = token_start + 1
    return StepOutput(
        clean_committed_latent=latent,
        decoded_chunk=decoded,
        commit_token_range=(token_start, token_end),
        commit_frame_range=(
            token_start * int(frames_per_token),
            token_end * int(frames_per_token),
        ),
        decoded_chunk_frame_range=(frame_start, frame_end),
        target_xz_frame_range=(frame_start, frame_end),
        debug={"ready_to_commit_token": token_start},
    )
