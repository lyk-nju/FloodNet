"""LDF training condition preparation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.conditions.ldf import LDFCondition
from utils.traj_batch import encode_traj_batch, get_traj_seq_lens


@dataclass(frozen=True, init=False)
class PreparedCondition:
    condition: LDFCondition
    text_dropped_flags: list
    traj_dropped: bool

    def __init__(
        self,
        *,
        text_context: list | None = None,
        text_dropped_flags: list,
        traj_emb: torch.Tensor | None = None,
        traj_seq_lens: torch.Tensor | None = None,
        traj_dropped: bool,
        traj_token_mask: torch.Tensor | None = None,
        condition: LDFCondition | None = None,
    ):
        if condition is None:
            if text_context is None:
                text_context = []
            condition = LDFCondition(
                text_context=text_context,
                traj_emb=traj_emb,
                traj_seq_lens=traj_seq_lens,
                traj_token_mask=traj_token_mask,
            )
        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "text_dropped_flags", text_dropped_flags)
        object.__setattr__(self, "traj_dropped", bool(traj_dropped))

    @property
    def text_context(self):
        return self.condition.text_context

    @property
    def traj_emb(self):
        return self.condition.traj_emb

    @property
    def traj_seq_lens(self):
        return self.condition.traj_seq_lens

    @property
    def traj_token_mask(self):
        return self.condition.traj_token_mask

    def with_traj(
        self,
        traj_emb,
        traj_seq_lens,
        traj_dropped: bool,
        traj_token_mask=None,
    ) -> "PreparedCondition":
        attn_len = self.condition.attn_len
        if traj_emb is not None:
            attn_len = int(traj_emb.shape[1])
        return PreparedCondition(
            condition=LDFCondition(
                text_context=self.text_context,
                text_null_context=self.condition.text_null_context,
                traj_emb=traj_emb,
                traj_seq_lens=traj_seq_lens,
                traj_token_mask=traj_token_mask,
                seq_len=self.condition.seq_len,
                attn_len=attn_len,
            ),
            text_dropped_flags=self.text_dropped_flags,
            traj_dropped=bool(traj_dropped),
        )


def prepare_condition(
    model,
    batch,
    seq_len: int,
    device,
    *,
    traj_dropped: bool | None = None,
    horizon_tokens=None,
    horizon_active_end=0,
) -> PreparedCondition:
    text_context, text_dropped_flags = prepare_text_condition(
        model, batch, seq_len, device
    )
    traj_emb, traj_seq_lens, traj_dropped, traj_token_mask = prepare_traj_condition(
        model,
        batch,
        seq_len,
        device,
        traj_dropped=traj_dropped,
        horizon_tokens=horizon_tokens,
        horizon_active_end=horizon_active_end,
    )
    condition = LDFCondition(
        text_context=text_context,
        traj_emb=traj_emb,
        traj_seq_lens=traj_seq_lens,
        traj_token_mask=traj_token_mask,
        seq_len=seq_len,
        attn_len=_resolve_traj_pad_len(batch, seq_len, device),
    )
    return PreparedCondition(
        condition=condition,
        text_dropped_flags=text_dropped_flags,
        traj_dropped=traj_dropped,
    )


def prepare_text_condition(
    model,
    batch,
    seq_len: int,
    device,
    *,
    text_dropped_flags=None,
):
    if text_dropped_flags is None:
        text_dropped_flags = sample_dropout_flags(
            batch["feature"].shape[0],
            device,
            p=model.text_dropout,
            training=model.training,
        )
    text_context = prepare_text_context(
        batch,
        seq_len,
        device,
        use_text_cond=model.use_text_cond,
        training=model.training,
        param_dtype=model.param_dtype,
        text_dropped_flags=text_dropped_flags,
        encode_text=model.encode_text_with_cache,
    )
    return text_context, text_dropped_flags


def prepare_traj_condition(
    model,
    batch,
    seq_len: int,
    device,
    *,
    traj_dropped: bool | None = None,
    horizon_tokens=None,
    horizon_active_end=0,
):
    if traj_dropped is None:
        traj_dropped = sample_traj_dropout(model, device)
    else:
        traj_dropped = bool(traj_dropped)

    traj_emb = None
    traj_seq_lens = None
    traj_token_mask = None
    if not traj_dropped:
        traj_pad_len = _resolve_traj_pad_len(batch, seq_len, device)
        traj_emb, traj_token_mask = encode_traj_batch(
            batch,
            traj_pad_len,
            device,
            model.traj_encoder,
            horizon_tokens=horizon_tokens,
            horizon_active_end_token=horizon_active_end,
            return_token_mask=True,
        )
        traj_seq_lens = get_traj_seq_lens(
            batch,
            traj_pad_len,
            device,
            horizon_tokens=horizon_tokens,
            horizon_active_end=horizon_active_end,
        )

    return traj_emb, traj_seq_lens, traj_dropped, traj_token_mask


def sample_traj_dropout(model, device) -> bool:
    if not model.training:
        return False
    drop = torch.empty(1, device=device).uniform_()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(drop, src=0)
    return drop.item() < model.traj_dropout


def sample_dropout_flags(batch_size: int, device, *, p: float, training: bool) -> list:
    if not training:
        return [False] * batch_size
    drop = torch.empty(batch_size, device=device).uniform_()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(drop, src=0)
    return (drop < p).tolist()


def prepare_text_context(
    batch,
    seq_len,
    device,
    *,
    use_text_cond: bool,
    training: bool,
    param_dtype,
    text_dropped_flags=None,
    encode_text,
):
    if use_text_cond and "text" in batch:
        text_list = batch["text"]
        if isinstance(text_list[0], list):
            text_end_list = batch["feature_text_end"]
            all_text_context = []
            for sample_idx, (single_text_list, single_text_end_list) in enumerate(
                zip(text_list, text_end_list)
            ):
                sample_dropped = (
                    text_dropped_flags[sample_idx]
                    if text_dropped_flags is not None
                    else False
                )
                if (not training) or (not sample_dropped):
                    single_text_end_list = [0] + [
                        min(end_token, seq_len)
                        for end_token in single_text_end_list
                    ]
                else:
                    single_text_list = [""]
                    single_text_end_list = [0, seq_len]
                single_text_length_list = [
                    end_token - begin_token
                    for end_token, begin_token in zip(
                        single_text_end_list[1:], single_text_end_list[:-1]
                    )
                ]
                single_text_context = encode_text(single_text_list, device)
                single_text_context = [
                    item.to(param_dtype) for item in single_text_context
                ]
                for context_item, duration in zip(
                    single_text_context,
                    single_text_length_list,
                ):
                    all_text_context.extend(
                        [context_item for _ in range(duration)]
                    )
                all_text_context.extend(
                    [
                        single_text_context[-1]
                        for _ in range(seq_len - single_text_end_list[-1])
                    ]
                )
        else:
            if training and text_dropped_flags is not None:
                all_text_context = [
                    ("" if text_dropped_flags[sample_idx] else text)
                    for sample_idx, text in enumerate(text_list)
                ]
            else:
                all_text_context = list(text_list)
            all_text_context = encode_text(all_text_context, device)
            all_text_context = [item.to(param_dtype) for item in all_text_context]
    else:
        all_text_context = [""] * batch["feature"].shape[0]
        all_text_context = encode_text(all_text_context, device)
        all_text_context = [item.to(param_dtype) for item in all_text_context]
    return all_text_context


def _resolve_traj_pad_len(batch, seq_len: int, device) -> int:
    pad_len = int(seq_len)
    for key in ("traj_num_tokens", "traj_features_length"):
        value = batch.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            tensor = value.to(device=device, dtype=torch.long).view(-1)
        else:
            tensor = torch.as_tensor(value, device=device, dtype=torch.long).view(-1)
        if tensor.numel() > 0:
            pad_len = max(pad_len, int(tensor.max().item()))
            break
    return pad_len


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
    latent_len = int(seq_len) + int(model.chunk_size)
    traj_len = _resolve_traj_pad_len(batch, latent_len, device)

    text_context = prepare_generate_text_context(model, batch, latent_len, device)
    text_null_context = [
        item.to(model.param_dtype)
        for item in model.encode_text_with_cache([""] * batch_size, device)
    ]
    traj_emb, traj_token_mask = encode_traj_batch(
        batch,
        traj_len,
        device,
        model.traj_encoder,
        return_token_mask=True,
    )
    traj_seq_lens = get_traj_seq_lens(batch, traj_len, device)
    return LDFCondition(
        text_context=text_context,
        text_null_context=text_null_context,
        traj_emb=traj_emb,
        traj_seq_lens=traj_seq_lens,
        traj_token_mask=traj_token_mask,
        seq_len=latent_len,
        attn_len=traj_len,
    )


def prepare_generate_text_context(model, batch: dict, seq_len: int, device) -> list:
    """Build text context for offline generation batches."""
    if model.use_text_cond and "text" in batch:
        text_list = batch["text"]
        if text_list and isinstance(text_list[0], list):
            all_text_context = []
            for single_text_list, single_text_end_list in zip(
                text_list,
                batch["feature_text_end"],
            ):
                single_text_end_list = [0] + [
                    min(int(end), int(seq_len)) for end in single_text_end_list
                ]
                single_text_length_list = [
                    end - begin
                    for end, begin in zip(
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

    batch_size = int(batch["feature"].shape[0]) if "feature" in batch else len(batch["feature_length"])
    return [
        item.to(model.param_dtype)
        for item in model.encode_text_with_cache([""] * batch_size, device)
    ]


def _max_length(value, device) -> int:
    if torch.is_tensor(value):
        tensor = value.to(device=device, dtype=torch.long)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=torch.long)
    return int(tensor.reshape(-1).max().item())
