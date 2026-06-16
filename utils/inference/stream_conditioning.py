from __future__ import annotations

import numpy as np
import torch

from utils.token_frame import (
    frame_idx_to_token_idx,
    prefix_len_from_tail_invalid,
    token_range_to_frame_slice,
    token_start_frame,
)
from utils.traj_batch import encode_traj_batch


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
                "stream_generate_step 7D payload has no substep payload "
                f"for window_start_token={window_start_token}; available "
                f"starts={starts}."
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
            "stream_generate_step 7D payload starts after current latent "
            "window start; got traj_start_token="
            f"{payload_local_start}, window_start_token={window_start_token}. "
            "Build direct 7D payloads from the earliest denoise substep "
            "window start or earlier."
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

    window_abs_start = payload_abs_start + (
        window_start_token - payload_local_start
    )
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
        window_abs_start = payload_abs_start

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
        traj_seq_lens = prefix_len_from_tail_invalid(traj_token_mask).to(
            device=device
        )
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
    for i in range(batch_size):
        segment = list(text_condition[i * model_sl : (i + 1) * model_sl])
        if not segment:
            continue
        out.extend(segment)
        out.extend([segment[-1]] * (target_sl - model_sl))
    return out


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
    window_abs_start = payload_abs_start + (
        window_start_token - payload_local_start
    )
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
