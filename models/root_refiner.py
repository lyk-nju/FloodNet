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
      chosen_class = (num_tokens given) ? num_tokens - min_tokens   # teacher-force
                                        : argmax(num_token_logits)   # inference
      (gated on num_tokens presence, NOT on train/eval mode → val + oracle eval
       can teacher-force the GT horizon)
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

from utils.token_frame import frame_idx_to_token_idx, num_frames_for_tokens


def _make_encoder(d_model: int, n_heads: int, ff_dim: int, dropout: float,
                  n_layers: int, norm_first: bool) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout,
        activation="gelu", batch_first=True, norm_first=norm_first,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class ResBlock1D(nn.Module):
    """Dilated 1D residual block: Conv(dilated)→GELU→[Dropout]→Conv → + residual."""

    def __init__(self, width: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:              # [B, width, T]
        h = self.act(self.conv1(x))
        h = self.drop(h)
        h = self.conv2(h)
        return x + h


class TokenToFrameDecoder(nn.Module):
    """Simple decoder: Conv→GELU→nearest-Upsample(×fpt)→Conv→GELU→Conv, then the
    causal trim to effective frames. Returns [B, num_frames_for_tokens(N), out_dim]
    (= 4N-3 for fpt=4) so it is interchangeable with PathCondFrameDecoder. Ignores
    the path-conditioning kwargs (kept for a uniform decoder signature).
    """

    def __init__(self, d_model: int, out_dim: int = 7, frames_per_token: int = 4):
        super().__init__()
        self.frames_per_token = frames_per_token
        self.conv_in = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.up = nn.Upsample(scale_factor=frames_per_token, mode="nearest")
        self.conv_mid = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.conv_out = nn.Conv1d(d_model, out_dim, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, token_hidden: Tensor, *, xz_path=None, path_mask=None,
                chosen_num_tokens=None) -> Tensor:       # [B, N, D]
        h = token_hidden.transpose(1, 2)                 # [B, D, N]
        h = self.act(self.conv_in(h))
        h = self.up(h)                                   # [B, D, N*fpt]
        h = self.act(self.conv_mid(h))
        h = self.conv_out(h)                             # [B, out_dim, N*fpt]
        dense = h.transpose(1, 2)                        # [B, N*fpt, out_dim]
        fpt = self.frames_per_token
        # Causal trim: token0=1 effective frame, token k≥1=fpt frames → 4N-3 frames.
        return torch.cat([dense[:, :1], dense[:, fpt:]], dim=1)


class PathCondFrameDecoder(nn.Module):
    """Path-conditioned root-trajectory decoder (replaces the blind upsampler).

    Builds the EFFECTIVE causal frame grid directly (token 0 → 1 frame, token k≥1
    → fpt frames; no upsample-then-trim, so no off-by-one), injects a dense
    frame-level path condition repeatedly (added before each dilated frame
    ResBlock), and outputs [B, num_frames_for_tokens(max_tokens), out_dim].

    path_cond[t] (per frame, dim 6) = [path_x, path_z, tangent_x, tangent_z,
    horizon_progress, grid_progress], built by interpolating the (anchor-local,
    arclength-resampled) xz_path at the per-frame horizon progress. For samples
    whose path is degenerate (path_mask valid points < 2) the whole path_cond is
    zeroed so the decoder falls back to the token-latent plan.
    """

    def __init__(self, d_model: int, max_tokens: int, *, frames_per_token: int = 4,
                 n_path: int = 64, width: int = 512, path_cond_dim: int = 6,
                 token_res_depth: int = 2, frame_res_depth: int = 4,
                 dilation_growth_rate: int = 2, dropout: float = 0.0, out_dim: int = 7):
        super().__init__()
        # _build_path_cond emits a fixed 6-D condition [x, z, tan_x, tan_z,
        # horizon_progress, grid_progress]; the param exists for config symmetry
        # only — anything else would shape-mismatch path_proj. Fail at construction.
        if path_cond_dim != 6:
            raise ValueError(
                f"PathCondFrameDecoder builds a 6-D path condition; "
                f"path_cond_dim must be 6, got {path_cond_dim}"
            )
        self.frames_per_token = frames_per_token
        self.max_tokens = max_tokens
        self.n_path = n_path
        self.max_frames = num_frames_for_tokens(max_tokens, frames_per_token)
        self.path_cond_dim = path_cond_dim

        # Constant causal frame→token expansion index (frame f pulls token f2t[f]).
        f2t = [frame_idx_to_token_idx(f, frames_per_token) for f in range(self.max_frames)]
        self.register_buffer("frame_to_token", torch.tensor(f2t, dtype=torch.long),
                             persistent=False)

        self.in_proj = nn.Conv1d(d_model, width, 3, padding=1)
        self.token_blocks = nn.ModuleList(
            [ResBlock1D(width, dilation=1, dropout=dropout) for _ in range(token_res_depth)]
        )
        self.path_proj = nn.Linear(path_cond_dim, width)
        self.frame_blocks = nn.ModuleList(
            [ResBlock1D(width, dilation=dilation_growth_rate ** i, dropout=dropout)
             for i in range(frame_res_depth)]
        )
        self.out_conv1 = nn.Conv1d(width, width, 3, padding=1)
        self.out_conv2 = nn.Conv1d(width, out_dim, 3, padding=1)
        self.act = nn.GELU()

    def _build_path_cond(self, xz_path: Tensor, path_mask: Tensor,
                         chosen_num_tokens: Tensor) -> Tensor:
        """Dense per-frame path condition [B, max_frames, path_cond_dim]. Fully
        vectorized — no host sync / Python per-sample loop."""
        B = xz_path.shape[0]
        device, dtype = xz_path.device, xz_path.dtype
        T, n_path, fpt = self.max_frames, self.n_path, self.frames_per_token

        t = torch.arange(T, device=device, dtype=dtype)                       # [T]
        # Effective-grid progress: valid frames = num_frames_for_tokens(chosen) =
        # fpt*chosen - (fpt-1); last valid frame index = that - 1.
        valid_eff = (fpt * chosen_num_tokens.to(dtype) - (fpt - 1))            # [B]
        denom = (valid_eff - 1.0).clamp(min=1.0)                              # [B]
        horizon = (t[None, :] / denom[:, None]).clamp(0.0, 1.0)              # [B, T]
        grid = (t / float(max(T - 1, 1)))[None, :].expand(B, T)              # [B, T] absolute pos

        # Linear interp along xz_path at horizon progress (gather-based lerp).
        path_pos = horizon * (n_path - 1)                                     # [B, T]
        idx0 = path_pos.floor().long().clamp(0, n_path - 1)                   # [B, T]
        idx1 = (idx0 + 1).clamp(max=n_path - 1)
        alpha = (path_pos - idx0.to(dtype))[..., None]                        # [B, T, 1]
        idx0e = idx0[..., None].expand(-1, -1, 2)
        idx1e = idx1[..., None].expand(-1, -1, 2)
        p0 = torch.gather(xz_path, 1, idx0e)                                  # [B, T, 2]
        p1 = torch.gather(xz_path, 1, idx1e)
        path_xy = (1.0 - alpha) * p0 + alpha * p1                            # [B, T, 2]

        # Tangent (path direction); pad last with previous; eps-safe normalize.
        diff = path_xy[:, 1:] - path_xy[:, :-1]                              # [B, T-1, 2]
        tangent = torch.cat([diff, diff[:, -1:]], dim=1)                     # [B, T, 2]
        tangent = F.normalize(tangent, dim=-1, eps=1e-6)

        cond = torch.cat(
            [path_xy, tangent, horizon[..., None], grid[..., None]], dim=-1,  # [B, T, 6]
        )
        # Degenerate path (valid points < 2) → zero the whole sample's condition.
        path_valid = (path_mask > 0).sum(dim=1) >= 2                          # [B] bool
        return cond * path_valid[:, None, None].to(dtype)

    def forward(self, token_hidden: Tensor, *, xz_path: Tensor, path_mask: Tensor,
                chosen_num_tokens: Tensor) -> Tensor:
        device, dtype = token_hidden.device, token_hidden.dtype
        fpt = self.frames_per_token
        # Validity masks: tokens / frames past the chosen horizon are zeroed and
        # RE-MASKED after EVERY block. Conv1d/ResBlock have a bias and a kernel that
        # reads neighbors, so an unmasked invalid region re-grows nonzero features
        # each layer and the dilated frame convs leak them back into the boundary
        # VALID frames (incl. the past-horizon path condition). Re-masking keeps a
        # valid frame's conv inputs (its invalid neighbors) at a deterministic 0.
        tok_ar = torch.arange(self.max_tokens, device=device)
        token_valid = (tok_ar[None, :] < chosen_num_tokens[:, None]).to(dtype)        # [B, max_tokens]
        valid_eff = fpt * chosen_num_tokens - (fpt - 1)                               # [B]
        fr_ar = torch.arange(self.max_frames, device=device)
        frame_valid = (fr_ar[None, :] < valid_eff[:, None]).to(dtype)                 # [B, max_frames]
        tok_m = token_valid[:, None, :]                                               # [B, 1, max_tokens]
        fr_m = frame_valid[:, None, :]                                                # [B, 1, max_frames]

        # Token stage (token grid).
        h = self.act(self.in_proj(token_hidden.transpose(1, 2))) * tok_m              # [B, width, max_tokens]
        for blk in self.token_blocks:
            h = blk(h) * tok_m
        # Causal expand to the effective frame grid (no upsample/trim → no offset).
        h = h.index_select(2, self.frame_to_token) * fr_m                            # [B, width, max_frames]
        # Repeated additive path conditioning before each dilated frame block. Mask
        # path_emb on invalid frames too — path_proj has a bias, so path_proj(cond=0)
        # is NOT zero; only post-projection masking kills the past-horizon path leak.
        cond = self._build_path_cond(xz_path, path_mask, chosen_num_tokens)  # [B, max_frames, 6]
        path_emb = (self.path_proj(cond).transpose(1, 2)) * fr_m                      # [B, width, max_frames]
        for blk in self.frame_blocks:
            h = blk(h + path_emb) * fr_m
        h = self.act(self.out_conv1(h)) * fr_m
        h = self.out_conv2(h)                                                # [B, out_dim, max_frames]
        return h.transpose(1, 2)                                             # [B, max_frames, out_dim]


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
        decoder_type: str = "path_cond",
        decoder_width: int | None = None,
        decoder_path_cond_dim: int = 6,
        decoder_token_res_depth: int = 2,
        decoder_frame_res_depth: int = 4,
        decoder_dilation_growth_rate: int = 2,
        decoder_dropout: float = 0.0,
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

        # Stage 3: token → frame decoder (both return effective [B, max_frames, 7]).
        self.decoder_type = decoder_type
        if decoder_type == "path_cond":
            width = decoder_width if decoder_width is not None else d_model
            self.frame_decoder = PathCondFrameDecoder(
                d_model=d_model, max_tokens=max_tokens, frames_per_token=frames_per_token,
                n_path=n_path, width=width, path_cond_dim=decoder_path_cond_dim,
                token_res_depth=decoder_token_res_depth,
                frame_res_depth=decoder_frame_res_depth,
                dilation_growth_rate=decoder_dilation_growth_rate,
                dropout=decoder_dropout, out_dim=7,
            )
        elif decoder_type == "simple":
            self.frame_decoder = TokenToFrameDecoder(
                d_model=d_model, out_dim=7, frames_per_token=frames_per_token)
        else:
            raise ValueError(
                f"decoder_type must be 'path_cond' or 'simple', got {decoder_type!r}"
            )

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
        num_tokens: Tensor | None = None,   # [B] long GT horizon → teacher-force when given (train/val/oracle); argmax when None
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
        # Teacher-force the horizon whenever a num_tokens is PROVIDED — training,
        # validation, AND oracle-duration eval — and fall back to the model's own
        # argmax when it is absent (real inference / normal benchmark). Gated on
        # `num_tokens is not None`, NOT on self.training, so val/oracle can
        # teacher-force in eval() mode (else val waypoint loss would be scored
        # against a possibly-wrong predicted horizon).
        if num_tokens is not None:
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
        token_valid = ar.unsqueeze(0) < chosen_num_tokens.unsqueeze(1)           # [B,max_tokens]
        token_query_pad = ~token_valid
        chosen_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)
        token_pad = torch.cat([chosen_pad, cond_pad, token_query_pad], dim=1)
        token_hidden = self.token_transformer(token_seq, src_key_padding_mask=token_pad)
        plan_token_hidden = token_hidden[:, -self.max_tokens:]                   # [B,max_tokens,D]

        # Zero the hiddens of plan tokens PAST the chosen horizon. Attention only
        # key-masks them (others can't read them) but the transformer still emits a
        # (garbage, input-dependent) hidden for those query positions; the frame
        # decoder's Conv1d(kernel=3) would then mix that tail into the boundary
        # VALID waypoints. Zeroing here makes valid frames a clean function of the
        # valid token hiddens + deterministic zero padding only.
        plan_token_hidden = plan_token_hidden * token_valid.unsqueeze(-1).to(plan_token_hidden.dtype)

        # ---- Stage 3: token → frame decode (decoder returns EFFECTIVE frames
        # [B, num_frames_for_tokens(max_tokens), 7] directly; path_cond decoder also
        # fuses the frame-level path condition) + heading norm. ----
        waypoints = self.frame_decoder(
            plan_token_hidden, xz_path=xz_path, path_mask=path_mask,
            chosen_num_tokens=chosen_num_tokens,
        )                                                    # [B, max_frames, 7]
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


__all__ = ["RootRefiner", "TokenToFrameDecoder", "PathCondFrameDecoder", "ResBlock1D"]
