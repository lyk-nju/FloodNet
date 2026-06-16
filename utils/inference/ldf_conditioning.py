from __future__ import annotations

import torch

from utils.inference.stream_conditioning import (
    build_stream_direct_traj_condition,
    extend_stream_text_context,
)
from utils.ldf_condition import LDFCondition
from utils.traj_batch import encode_traj_batch, get_traj_seq_lens


def prepare_generate_condition(
    model,
    batch: dict,
    device,
    *,
    seq_len: int | None = None,
) -> LDFCondition:
    """Build the prepared condition used by offline LDF generation."""
    feature_length = batch["feature_length"]
    batch_size = len(feature_length)
    if seq_len is None:
        seq_len = _max_length(feature_length, device)
    attn_len = int(seq_len) + int(model.chunk_size)

    text_context = prepare_generate_text_context(model, batch, attn_len, device)
    text_null_context = [
        item.to(model.param_dtype)
        for item in model.encode_text_with_cache([""] * batch_size, device)
    ]
    traj_emb, traj_token_mask = encode_traj_batch(
        batch,
        attn_len,
        device,
        model.local_traj_encoder,
        model.traj_encoder,
        return_token_mask=True,
    )
    traj_seq_lens = get_traj_seq_lens(batch, attn_len, device)
    return LDFCondition(
        text_context=text_context,
        text_null_context=text_null_context,
        traj_emb=traj_emb,
        traj_seq_lens=traj_seq_lens,
        traj_token_mask=traj_token_mask,
        seq_len=attn_len,
        attn_len=attn_len,
    )


def prepare_generate_text_context(model, batch: dict, seq_len: int, device) -> list:
    if model.use_text_cond and "text" in batch:
        text_list = batch["text"]
        if text_list and isinstance(text_list[0], list):
            all_text_context = []
            for single_text_list, single_text_end_list in zip(
                text_list,
                batch["feature_text_end"],
            ):
                single_text_end_list = [0] + [
                    min(int(t), int(seq_len)) for t in single_text_end_list
                ]
                single_text_length_list = [
                    t - b
                    for t, b in zip(
                        single_text_end_list[1:],
                        single_text_end_list[:-1],
                    )
                ]
                single_context = [
                    item.to(model.param_dtype)
                    for item in model.encode_text_with_cache(single_text_list, device)
                ]
                for item, duration in zip(single_context, single_text_length_list):
                    all_text_context.extend([item] * duration)
                all_text_context.extend(
                    [single_context[-1]] * (int(seq_len) - single_text_end_list[-1])
                )
            return all_text_context
        return [
            item.to(model.param_dtype)
            for item in model.encode_text_with_cache(list(text_list), device)
        ]

    return [
        item.to(model.param_dtype)
        for item in model.encode_text_with_cache(
            [""] * int(batch["feature"].shape[0]),
            device,
        )
    ]


def build_stream_step_condition_provider(
    model,
    step_input: dict,
    *,
    first_chunk: bool,
    device,
):
    """Build a stream-step condition provider outside the model layer."""
    traj_buf = getattr(model, "_traj_buf", None)
    if traj_buf is not None:
        traj_buf.update(step_input, model.commit_index, device)

    use_text_cond = bool(getattr(model, "use_text_cond", True))
    if use_text_cond and "text" in step_input:
        new_text_context = _encode_stream_text(model, step_input["text"], device)
    else:
        fallback_batch_size = int(getattr(model, "batch_size", 1))
        new_text_context = _encode_stream_text(model, [""] * fallback_batch_size, device)
    batch_size = int(getattr(model, "batch_size", len(new_text_context)))
    if not hasattr(model, "batch_size"):
        model.batch_size = batch_size
    if not hasattr(model, "text_condition_list"):
        model.text_condition_list = [[] for _ in range(batch_size)]
    param_dtype = getattr(model, "param_dtype", torch.float32)
    new_text_context = [item.to(param_dtype) for item in new_text_context]
    text_null_context = [
        item.to(param_dtype)
        for item in _encode_stream_text(model, [""] * batch_size, device)
    ]

    for batch_idx in range(batch_size):
        if first_chunk:
            model.text_condition_list[batch_idx].extend(
                [new_text_context[batch_idx]] * model.chunk_size
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
                    local_traj_encoder=model.local_traj_encoder,
                    traj_encoder=model.traj_encoder,
                )
            )
            if traj_emb is not None:
                attn_len = max(attn_len, int(traj_emb.shape[1]))
        elif traj_buf is not None:
            traj_emb = traj_buf.build_traj_emb(
                end_index,
                model.seq_len,
                device,
            )
            if traj_emb is not None:
                attn_len = max(attn_len, int(traj_emb.shape[1]))
                traj_seq_lens = traj_buf.get_traj_valid_lens(
                    end_index,
                    model.seq_len,
                    device,
                )
                traj_token_mask = traj_buf.get_traj_token_mask(
                    end_index,
                    model.seq_len,
                    device,
                )

        text_context = extend_stream_text_context(
            text_context,
            batch_size,
            model_sl,
            attn_len,
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


def _max_length(value, device) -> int:
    if torch.is_tensor(value):
        tensor = value.to(device=device, dtype=torch.long)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=torch.long)
    return int(tensor.reshape(-1).max().item())


def _has_direct_traj_payload(step_input: dict) -> bool:
    return (
        step_input.get("traj_cond_7d_frame") is not None
        or step_input.get("traj_substep_payloads") is not None
    )


def _encode_stream_text(model, text_list, device) -> list[torch.Tensor]:
    encode = getattr(model, "encode_text_with_cache", None)
    if encode is None:
        return [torch.zeros(1, 1, device=device) for _ in text_list]
    return encode(text_list, device)
