"""RootRefiner: path-conditioned duration-first / trajectory-second root planner.

Architecture (redesigned — duration-first / trajectory-second):

  Stage 1 — condition-only num-token predictor:
      cond_seq = [CLS, text, path_stats, path × n_path, history × n_hist]
      cond_hidden = cond_transformer(cond_seq, cond_pad)
      num_token_logits = num_token_head(cond_hidden[:, 0])
    The horizon (num_tokens) is decided from the conditions ALONE — it never
    attends to the 193 frame-level waypoint queries, so the CLS representation
    is not diluted and waypoint-regression gradients do not dominate it.

  Stage 2 — token-level plan-latent generator (conditioned on the chosen horizon):
      chosen_class = (training & num_tokens given) ? num_tokens - min_tokens
                                                   : argmax(num_token_logits)
      token_seq = [num_token_emb(chosen_class), cond_hidden, plan_queries × max_tokens]
      token_hidden = token_transformer(token_seq, token_pad)
      plan_token_hidden = token_hidden[:, -max_tokens:]
    Plan queries past `chosen_num_tokens` are key-masked (token_query_pad).

  Stage 3 — token→frame decoder:
      dense = frame_decoder(plan_token_hidden)         # [B, max_tokens*fpt, 7]
      waypoints = causal_trim(dense)                   # [B, num_frames_for_tokens(max_tokens), 7]
      heading channels [3:5] L2-normalized.

Causal-VAE frame convention: num_frames_for_tokens(N) = 4N - 3 (fpt=4). The
decoder emits max_tokens*fpt frames; the causal trim keeps token 0 as ONE
effective frame and tokens 1..N-1 as 4 frames each:
    waypoints = cat([dense[:, :1], dense[:, fpt:]], dim=1).

Output keys (downstream-compatible): num_token_logits, pred_num_tokens,
chosen_num_tokens, waypoints. Loss lives in train_refiner.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.token_frame import num_frames_for_tokens


def _make_encoder(d_model: int, n_heads: int, ff_dim: int, dropout: float,
                  n_layers: int, norm_first: bool) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout,
        activation="gelu", batch_first=True, norm_first=norm_first,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class TokenToFrameDecoder(nn.Module):
    """Upsample token-level plan hiddens [B, N, D] → frame-level [B, N*fpt, out_dim].

    Conv1d → GELU → nearest-Upsample(×fpt) → Conv1d → GELU → Conv1d. Simple and
    stable; the caller applies the causal trim + heading normalization.
    """

    def __init__(self, d_model: int, out_dim: int = 7, frames_per_token: int = 4):
        super().__init__()
        self.frames_per_token = frames_per_token
        self.conv_in = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.up = nn.Upsample(scale_factor=frames_per_token, mode="nearest")
        self.conv_mid = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.conv_out = nn.Conv1d(d_model, out_dim, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, token_hidden: Tensor) -> Tensor:   # [B, N, D]
        h = token_hidden.transpose(1, 2)                 # [B, D, N]
        h = self.act(self.conv_in(h))
        h = self.up(h)                                   # [B, D, N*fpt]
        h = self.act(self.conv_mid(h))
        h = self.conv_out(h)                             # [B, out_dim, N*fpt]
        return h.transpose(1, 2)                         # [B, N*fpt, out_dim]


class RootRefiner(nn.Module):
    """Duration-first / trajectory-second Refiner. dict-in / dict-out forward."""

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
        n_layers_cond: int | None = None,
        n_layers_token: int | None = None,
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

        # Split the layer budget into cond / token stages (backward compatible:
        # absent n_layers_cond/token → split n_layers ~evenly, e.g. 6 → 3 + 3).
        if n_layers_cond is None:
            n_layers_cond = max(1, n_layers // 2)
        if n_layers_token is None:
            n_layers_token = max(1, n_layers - n_layers_cond)
        self.n_layers_cond = n_layers_cond
        self.n_layers_token = n_layers_token

        # Input projections (shared by the condition stage).
        self.text_proj = nn.Linear(text_emb_dim, d_model)
        self.path_proj = nn.Linear(2, d_model)
        self.stats_proj = nn.Linear(path_stats_dim, d_model)
        self.hist_proj = nn.Linear(5, d_model)

        self.cls_token = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.path_pos_emb = nn.Embedding(n_path, d_model)
        self.hist_pos_emb = nn.Embedding(n_hist, d_model)

        # Stage 1: condition-only num-token predictor.
        self.cond_transformer = _make_encoder(
            d_model, n_heads, ff_dim, dropout, n_layers_cond, norm_first)
        self.num_token_logits_dim = max_tokens - min_tokens + 1
        self.num_token_head = nn.Linear(d_model, self.num_token_logits_dim)

        # Stage 2: token-level plan generator, conditioned on the chosen horizon.
        self.num_token_emb = nn.Embedding(self.num_token_logits_dim, d_model)
        self.plan_token_queries = nn.Parameter(torch.zeros(max_tokens, d_model))
        nn.init.trunc_normal_(self.plan_token_queries, std=0.02)
        self.plan_token_pos_emb = nn.Embedding(max_tokens, d_model)
        self.token_transformer = _make_encoder(
            d_model, n_heads, ff_dim, dropout, n_layers_token, norm_first)

        # Stage 3: token → frame decoder.
        self.frame_decoder = TokenToFrameDecoder(
            d_model=d_model, out_dim=7, frames_per_token=frames_per_token)

        # CLS + text + path_stats are the 3 never-masked "special" condition tokens.
        self._n_specials = 3

    # ------------------------------------------------------------------

    def forward(
        self,
        text_emb: Tensor,            # [B, text_emb_dim]
        xz_path: Tensor,             # [B, n_path, 2]
        path_mask: Tensor,           # [B, n_path] bool (True = valid)
        path_stats: Tensor,          # [B, path_stats_dim]
        current_motion: Tensor,      # [B, n_hist, 5]
        history_mask: Tensor,        # [B, n_hist] bool (True = valid)
        num_tokens: Tensor | None = None,   # [B] long GT horizon (training teacher-forcing)
    ) -> dict[str, Tensor]:
        B = text_emb.shape[0]
        device = text_emb.device

        # ---- Stage 1: condition-only num-token prediction ----
        cls_tok = self.cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)      # [B,1,D]
        text_tok = self.text_proj(text_emb).unsqueeze(1)                         # [B,1,D]
        stats_tok = self.stats_proj(path_stats).unsqueeze(1)                     # [B,1,D]
        path_toks = self.path_proj(xz_path) + self.path_pos_emb.weight.unsqueeze(0)   # [B,n_path,D]
        hist_toks = self.hist_proj(current_motion) + self.hist_pos_emb.weight.unsqueeze(0)  # [B,n_hist,D]
        cond_seq = torch.cat([cls_tok, text_tok, stats_tok, path_toks, hist_toks], dim=1)

        spec_pad = torch.zeros(B, self._n_specials, dtype=torch.bool, device=device)
        cond_pad = torch.cat([spec_pad, ~path_mask.bool(), ~history_mask.bool()], dim=1)
        # CLS/text/stats never masked → no all-True row → attention can't NaN
        # (structural guarantee; no host-sync assertion needed).
        cond_hidden = self.cond_transformer(cond_seq, src_key_padding_mask=cond_pad)
        num_token_logits = self.num_token_head(cond_hidden[:, 0])                # [B, K]

        pred_class = num_token_logits.argmax(dim=-1)                             # [B] in [0,K-1]
        if self.training and num_tokens is not None:
            chosen_class = (num_tokens.to(device=device, dtype=torch.long) - self.min_tokens)
            chosen_class = chosen_class.clamp(0, self.num_token_logits_dim - 1)  # no host sync
        else:
            chosen_class = pred_class
        pred_num_tokens = pred_class + self.min_tokens
        chosen_num_tokens = chosen_class + self.min_tokens

        # ---- Stage 2: token-level plan generator (chosen-horizon conditioned) ----
        chosen_tok = self.num_token_emb(chosen_class).unsqueeze(1)               # [B,1,D]
        plan_q = (self.plan_token_queries + self.plan_token_pos_emb.weight)      # [max_tokens,D]
        plan_q = plan_q.unsqueeze(0).expand(B, -1, -1)                           # [B,max_tokens,D]
        token_seq = torch.cat([chosen_tok, cond_hidden, plan_q], dim=1)

        ar = torch.arange(self.max_tokens, device=device)
        token_query_pad = ~(ar.unsqueeze(0) < chosen_num_tokens.unsqueeze(1))    # [B,max_tokens]
        chosen_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)
        token_pad = torch.cat([chosen_pad, cond_pad, token_query_pad], dim=1)
        token_hidden = self.token_transformer(token_seq, src_key_padding_mask=token_pad)
        plan_token_hidden = token_hidden[:, -self.max_tokens:]                   # [B,max_tokens,D]

        # ---- Stage 3: token → frame decode + causal trim + heading norm ----
        dense = self.frame_decoder(plan_token_hidden)        # [B, max_tokens*fpt, 7]
        waypoints = torch.cat(
            [dense[:, :1], dense[:, self.frames_per_token:]], dim=1,
        )                                                    # [B, 4N-3, 7]
        assert waypoints.shape[1] == self.max_frames, (      # int compare, no host sync
            f"decoded frames {waypoints.shape[1]} != max_frames {self.max_frames}"
        )
        head = F.normalize(waypoints[..., 3:5], dim=-1, eps=1e-6)
        waypoints = torch.cat([waypoints[..., :3], head, waypoints[..., 5:7]], dim=-1)

        return {
            "num_token_logits": num_token_logits,
            "pred_num_tokens": pred_num_tokens,
            "chosen_num_tokens": chosen_num_tokens,
            "waypoints": waypoints,
        }

    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["RootRefiner", "TokenToFrameDecoder"]
