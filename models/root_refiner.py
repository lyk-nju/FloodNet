"""RootRefiner: transformer that maps (text, user path, current motion) → root plan.

Architecture (docs/design.md §3.3, docs/TODO.md §T_A_05):

  Token sequence (single non-AR Transformer Encoder pass):
    [CLS,
     text,
     path_stats,
     path × n_path,          (n_path = 64)
     history × n_hist,       (n_hist = 20)
     queries × max_frames]   (max_frames = num_frames_for_tokens(max_tokens))

  Total length L = 1 + 1 + 1 + n_path + n_hist + max_frames
                 = 1 + 1 + 1 + 64 + 20 + 193 = 280 (defaults).

  Heads:
    num_token_head : CLS hidden → [max_tokens - min_tokens + 1] logits
                     (classifies plan token count).
    waypoint_head  : last max_frames query hiddens → [max_frames, 7] regress
                     [x, y, z, cos_h, sin_h, fwd_delta, yaw_delta] (plan-anchor-local).
                     Heading channels [3:5] are L2-normalized to lie on the
                     unit circle.

Input mask semantics (PyTorch `src_key_padding_mask`: True = ignore):
  - path positions: ignored where the dataset's `path_mask` is False
    (degenerate-path fallback).
  - history positions: ignored where `history_mask` is False
    (full-plan mode pads 19 of 20 leading slots with zeros).
  - CLS / text / path_stats / waypoint queries are NEVER masked — they always
    participate in attention so the model can route information from the
    structured inputs to the queries.

Loss is implemented in T_A_08 `train_refiner.py`. This module owns only the
forward pass + heads + heading normalization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.token_frame import num_frames_for_tokens


class RootRefiner(nn.Module):
    """Transformer-encoder Refiner. Forward signature is dict-in / dict-out."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_tokens: int = 49,
        min_tokens: int = 4,
        frames_per_token: int = 4,
        n_path: int = 64,
        n_hist: int = 20,
        text_emb_dim: int = 512,
        path_stats_dim: int = 3,
        norm_first: bool = True,
    ):
        super().__init__()
        if max_tokens < min_tokens:
            raise ValueError(f"max_tokens ({max_tokens}) < min_tokens ({min_tokens})")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.frames_per_token = frames_per_token
        self.n_path = n_path
        self.n_hist = n_hist
        self.text_emb_dim = text_emb_dim
        self.path_stats_dim = path_stats_dim
        self.max_frames = num_frames_for_tokens(max_tokens, frames_per_token)

        # Input projections.
        self.text_proj = nn.Linear(text_emb_dim, d_model)
        self.path_proj = nn.Linear(2, d_model)
        self.stats_proj = nn.Linear(path_stats_dim, d_model)
        self.hist_proj = nn.Linear(5, d_model)

        # Learnable / positional embeddings.
        self.cls_token = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.waypoint_queries = nn.Parameter(torch.zeros(self.max_frames, d_model))
        nn.init.trunc_normal_(self.waypoint_queries, std=0.02)

        self.path_pos_emb = nn.Embedding(n_path, d_model)
        self.hist_pos_emb = nn.Embedding(n_hist, d_model)
        self.query_pos_emb = nn.Embedding(self.max_frames, d_model)

        # Backbone.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Heads.
        self.num_token_logits_dim = max_tokens - min_tokens + 1
        self.num_token_head = nn.Linear(d_model, self.num_token_logits_dim)
        self.waypoint_head = nn.Linear(d_model, 7)

        # Token-section sizes / slice indices for assembling sequence + mask.
        self._n_specials = 3   # CLS + text + path_stats
        self._path_start = self._n_specials
        self._path_end = self._n_specials + n_path
        self._hist_start = self._path_end
        self._hist_end = self._hist_start + n_hist
        self._query_start = self._hist_end
        self._query_end = self._query_start + self.max_frames
        self._seq_len = self._query_end

    # ------------------------------------------------------------------

    def forward(
        self,
        text_emb: Tensor,            # [B, text_emb_dim]
        xz_path: Tensor,             # [B, n_path, 2]
        path_mask: Tensor,           # [B, n_path] bool (True = valid)
        path_stats: Tensor,          # [B, path_stats_dim]
        current_motion: Tensor,      # [B, n_hist, 5]
        history_mask: Tensor,        # [B, n_hist] bool (True = valid)
    ) -> dict[str, Tensor]:
        """Forward pass.

        Returns:
            dict with:
              num_token_logits : [B, max_tokens - min_tokens + 1]
              waypoints        : [B, max_frames, 7] (heading channels 3:5 unit-norm)
        """
        B = text_emb.shape[0]
        device = text_emb.device

        # Project each input section.
        text_tok = self.text_proj(text_emb).unsqueeze(1)                        # [B, 1, D]
        stats_tok = self.stats_proj(path_stats).unsqueeze(1)                    # [B, 1, D]
        path_pos = self.path_pos_emb.weight.unsqueeze(0).expand(B, -1, -1)      # [B, 64, D]
        path_toks = self.path_proj(xz_path) + path_pos                          # [B, 64, D]
        hist_pos = self.hist_pos_emb.weight.unsqueeze(0).expand(B, -1, -1)      # [B, 20, D]
        hist_toks = self.hist_proj(current_motion) + hist_pos                   # [B, 20, D]
        query_toks_base = (
            self.waypoint_queries + self.query_pos_emb.weight
        )                                                                          # [max_T, D]
        query_toks = query_toks_base.unsqueeze(0).expand(B, -1, -1)             # [B, max_T, D]
        cls_tok = self.cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)     # [B, 1, D]

        seq = torch.cat(
            [cls_tok, text_tok, stats_tok, path_toks, hist_toks, query_toks],
            dim=1,
        )                                                                          # [B, L, D]
        assert seq.shape[1] == self._seq_len, (
            f"seq length {seq.shape[1]} != expected {self._seq_len}"
        )

        # Build src_key_padding_mask: True = ignore. Special tokens / queries
        # are NEVER masked. path / history positions follow the per-sample bool
        # masks (NOT path_mask means "ignore").
        not_path = ~path_mask.bool()
        not_hist = ~history_mask.bool()

        spec_pad = torch.zeros(B, self._n_specials, dtype=torch.bool, device=device)
        query_pad = torch.zeros(B, self.max_frames, dtype=torch.bool, device=device)
        src_key_padding_mask = torch.cat(
            [spec_pad, not_path, not_hist, query_pad], dim=1,
        )                                                                          # [B, L]

        # Defensive: degenerate all-True padding row would cause NaN attention
        # (no valid keys). We can't mathematically guard upstream — flag here.
        # In normal operation `cls_tok` row is always non-masked so this is OK.
        assert not src_key_padding_mask.all(dim=1).any().item(), (
            "src_key_padding_mask has at least one row fully True; "
            "attention would NaN. CLS/text/stats/queries should always be unmasked."
        )

        hidden = self.transformer(seq, src_key_padding_mask=src_key_padding_mask)

        # num_token classification from CLS hidden.
        num_token_logits = self.num_token_head(hidden[:, 0])                    # [B, K]

        # Waypoint regression from the trailing max_frames query slots.
        waypoint_hidden = hidden[:, self._query_start : self._query_end]        # [B, max_T, D]
        waypoints = self.waypoint_head(waypoint_hidden)                         # [B, max_T, 7]

        # Heading unit-normalization on channels [3:5] (cos, sin).
        head = waypoints[..., 3:5]
        head = F.normalize(head, dim=-1, eps=1e-6)
        waypoints = torch.cat(
            [waypoints[..., :3], head, waypoints[..., 5:7]], dim=-1,
        )

        return {
            "num_token_logits": num_token_logits,
            "waypoints": waypoints,
        }

    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Trainable parameter count (for sanity / model size budget)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["RootRefiner"]


# Re-export math so the loss helpers in T_A_08 can pick it up — keeps this
# module self-contained without an external dependency on the test bookkeeping.
_ = math.pi
