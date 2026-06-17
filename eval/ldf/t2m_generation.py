"""Generation helpers for full-sequence T2M metric evaluation."""

from __future__ import annotations

import torch

from utils.training.ldf.t2m_generation_modes import (
    T2M_GENERATE,
    T2M_STREAM_GENERATE,
)


def _batch_size_from_model_batch(model_batch: dict) -> int:
    if "feature_length" in model_batch:
        return int(len(model_batch["feature_length"]))
    if "token_length" in model_batch:
        return int(len(model_batch["token_length"]))
    if "feature" in model_batch:
        return int(model_batch["feature"].shape[0])
    text = model_batch.get("text", [])
    return int(len(text))


def _lengths_from_model_batch(model_batch: dict, batch_size: int):
    lengths = model_batch.get("feature_length", model_batch.get("token_length"))
    if lengths is None:
        return [None] * batch_size
    if torch.is_tensor(lengths):
        return [int(v.item()) for v in lengths.reshape(-1)[:batch_size]]
    return [int(v) for v in list(lengths)[:batch_size]]


def _empty_generated_like(model_batch: dict) -> torch.Tensor:
    feature = model_batch.get("feature")
    if torch.is_tensor(feature):
        return feature.new_zeros((0, int(feature.shape[-1])))
    return torch.zeros(0, 0)


def _batch_text(model_batch: dict, batch_size: int):
    text = model_batch.get("output_text", model_batch.get("text", [""] * batch_size))
    if isinstance(text, list) and text and isinstance(text[0], list):
        text = [" ////////// ".join(map(str, item)) for item in text]
    return text


@torch.no_grad()
def run_t2m_generation_mode(
    model,
    model_batch: dict,
    mode: str,
    *,
    num_denoise_steps=None,
) -> dict:
    """Run one full-sequence generation mode for T2M metric evaluation.

    ``generate`` already returns a list of full latent sequences. ``stream_generate``
    yields committed chunks, so this helper concatenates those chunks into the
    same output shape before metric decoding.
    """
    if mode == T2M_GENERATE:
        return model.generate(model_batch, num_denoise_steps=num_denoise_steps)
    if mode != T2M_STREAM_GENERATE:
        raise ValueError(f"unknown T2M generation mode: {mode!r}")

    batch_size = _batch_size_from_model_batch(model_batch)
    chunks: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
    for step_output in model.stream_generate(
        model_batch,
        num_denoise_steps=num_denoise_steps,
    ):
        step_generated = step_output.get("generated", [])
        for idx in range(min(batch_size, len(step_generated))):
            chunk = step_generated[idx]
            if chunk is not None and int(chunk.shape[0]) > 0:
                chunks[idx].append(chunk)

    lengths = _lengths_from_model_batch(model_batch, batch_size)
    empty = _empty_generated_like(model_batch)
    generated = []
    for idx in range(batch_size):
        if chunks[idx]:
            sample = torch.cat(chunks[idx], dim=0)
        else:
            sample = empty
        if lengths[idx] is not None:
            sample = sample[: lengths[idx]]
        generated.append(sample)
    return {
        "generated": generated,
        "text": _batch_text(model_batch, batch_size),
    }


__all__ = ["run_t2m_generation_mode"]
