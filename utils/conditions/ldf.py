from __future__ import annotations

import torch

from dataclasses import dataclass


@dataclass(frozen=True)
class LDFCondition:
    """Prepared condition consumed by DiffForcingWanModel."""

    text_context: list
    text_null_context: list | None = None
    traj_emb: torch.Tensor | None = None
    traj_seq_lens: torch.Tensor | None = None
    traj_token_mask: torch.Tensor | None = None
    seq_len: int | None = None
    attn_len: int | None = None

    def latent_len(self) -> int:
        if self.seq_len is None:
            raise ValueError("LDFCondition requires seq_len for latent length.")
        return int(self.seq_len)

    def attention_len(self) -> int:
        length = self.attn_len if self.attn_len is not None else self.seq_len
        if length is None:
            raise ValueError("LDFCondition requires seq_len or attn_len.")
        return int(length)

    def validate(self, *, batch_size: int | None = None) -> None:
        if self.seq_len is not None and int(self.seq_len) <= 0:
            raise ValueError(f"seq_len must be > 0, got {self.seq_len}")
        if self.attn_len is not None and int(self.attn_len) <= 0:
            raise ValueError(f"attn_len must be > 0, got {self.attn_len}")
        if self.traj_emb is not None:
            if self.traj_emb.dim() != 3:
                raise ValueError(
                    "traj_emb must be [B,T,C], "
                    f"got shape {tuple(self.traj_emb.shape)}"
                )
            if batch_size is not None and int(self.traj_emb.shape[0]) != int(batch_size):
                raise ValueError(
                    f"traj_emb batch size {self.traj_emb.shape[0]} "
                    f"!= batch_size {batch_size}"
                )
        if self.traj_seq_lens is not None:
            if self.traj_seq_lens.dim() != 1:
                raise ValueError(
                    "traj_seq_lens must be [B], "
                    f"got shape {tuple(self.traj_seq_lens.shape)}"
                )
            if batch_size is not None and int(self.traj_seq_lens.numel()) != int(batch_size):
                raise ValueError(
                    f"traj_seq_lens batch size {self.traj_seq_lens.numel()} "
                    f"!= batch_size {batch_size}"
                )
        if self.traj_token_mask is not None:
            if self.traj_token_mask.dim() != 2:
                raise ValueError(
                    "traj_token_mask must be [B,T], "
                    f"got shape {tuple(self.traj_token_mask.shape)}"
                )
            if batch_size is not None and int(self.traj_token_mask.shape[0]) != int(batch_size):
                raise ValueError(
                    f"traj_token_mask batch size {self.traj_token_mask.shape[0]} "
                    f"!= batch_size {batch_size}"
                )


@dataclass(frozen=True)
class LDFTrainingCondition:
    condition: LDFCondition
    text_dropped_flags: list
    traj_dropped: bool
