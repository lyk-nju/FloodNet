from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResBlock1D(nn.Module):
    """Temporal 1D residual block."""

    def __init__(self, width: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.conv1(x))
        h = self.drop(h)
        h = self.conv2(h)
        return x + h


class RootDurationHead(nn.Module):
    """Predict waypoints length for root plan."""

    def __init__(
        self,
        d_model: int,
        *,
        path_features_dim: int,
        min_frames: int,
        max_frames: int,
        dropout: float = 0.1,
        pace_text_dim: int = 32,
    ):
        super().__init__()
        if max_frames < min_frames:
            raise ValueError(f"max_frames ({max_frames}) < min_frames ({min_frames})")
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.path_features_dim = int(path_features_dim)
        self.pace_text_dim = int(pace_text_dim)

        self.duration_feature_proj = nn.Sequential(
            nn.Linear(self.path_features_dim, d_model),
            nn.GELU(),
        )
        self.condition_proj = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.raw_feature_proj = nn.Sequential(
            nn.Linear(self.path_features_dim, d_model),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(d_model, self.pace_text_dim),
            nn.GELU(),
        )
        hidden = max(1, d_model // 2)
        self.pace_head = nn.Sequential(
            nn.Linear(2 * d_model + self.pace_text_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        *,
        cls_summary: Tensor,
        path_summary: Tensor,
        history_summary: Tensor,
        text_summary: Tensor,
        path_features: Tensor,
        path_features_raw: Tensor,
    ) -> dict[str, Tensor]:
        duration_feature = self.duration_feature_proj(path_features)
        condition_summary = self.condition_proj(
            torch.cat(
                [
                    cls_summary,
                    path_summary,
                    history_summary,
                    duration_feature,
                ],
                dim=-1,
            )
        )
        raw_feature_summary = self.raw_feature_proj(path_features_raw)
        text_style_summary = self.text_proj(text_summary)
        pace_summary = torch.cat(
            [condition_summary, raw_feature_summary, text_style_summary],
            dim=-1,
        )
        pred_log_pace = self.pace_head(pace_summary).squeeze(-1)
        effective_length = (
            path_features_raw[:, 0].clamp_min(0.0)
            + path_features_raw[:, 3].clamp_min(0.0)
        )
        pred_frames_float = pred_log_pace.clamp(-8.0, 8.0).exp() * effective_length
        pred_frames_pace = (
            pred_frames_float.round().long().clamp(self.min_frames, self.max_frames)
        )
        return {
            "pred_log_pace": pred_log_pace,
            "pred_frames_float": pred_frames_float,
            "pred_frames_pace": pred_frames_pace,
            "pred_frames": pred_frames_pace,
        }


class RootPlanDecoder(nn.Module):
    """Decode root features into waypoints."""

    def __init__(
        self,
        d_model: int,
        *,
        width: int | None = None,
        depth: int = 4,
        dilation_growth_rate: int = 2,
        dropout: float = 0.0,
        out_dim: int = 5,
    ):
        super().__init__()
        width = int(width if width is not None else d_model)
        self.in_proj = nn.Conv1d(d_model, width, 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                ResBlock1D(
                    width,
                    dilation=dilation_growth_rate ** i,
                    dropout=dropout,
                )
                for i in range(depth)
            ]
        )
        self.out_conv1 = nn.Conv1d(width, width, 3, padding=1)
        self.out_conv2 = nn.Conv1d(width, out_dim, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, root_hidden: Tensor, frame_mask: Tensor) -> Tensor:
        mask = frame_mask[:, None, :].to(root_hidden.dtype)
        h = self.act(self.in_proj(root_hidden.transpose(1, 2))) * mask
        for block in self.blocks:
            h = block(h) * mask
        h = self.act(self.out_conv1(h)) * mask
        h = self.out_conv2(h)
        return h.transpose(1, 2)


class RootRefiner(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_frames: int = 193,
        min_frames: int = 1,
        n_path: int = 64,
        n_hist: int = 20,
        text_emb_dim: int = 512,
        path_features_dim: int = 5,
        norm_first: bool = True,
        n_layers_cond: int | None = None,
        n_layers_root: int | None = None,
        decoder_width: int | None = None,
        decoder_depth: int = 4,
        decoder_dilation_growth_rate: int = 2,
        decoder_dropout: float = 0.0,
        pace_text_dim: int = 32,
    ):
        super().__init__()
        if max_frames < min_frames:
            raise ValueError(f"max_frames ({max_frames}) < min_frames ({min_frames})")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = int(d_model)
        self.max_frames = int(max_frames)
        self.min_frames = int(min_frames)
        self.n_path = int(n_path)
        self.n_hist = int(n_hist)
        self.text_emb_dim = int(text_emb_dim)
        self.path_features_dim = int(path_features_dim)

        if n_layers_cond is None:
            n_layers_cond = max(1, n_layers // 2)
        if n_layers_root is None:
            n_layers_root = max(1, n_layers - n_layers_cond)
        self.n_layers_cond = int(n_layers_cond)
        self.n_layers_root = int(n_layers_root)

        # Condition inputs.
        self.text_proj = nn.Linear(text_emb_dim, d_model)
        self.path_proj = nn.Linear(2, d_model)
        self.path_control_proj = nn.Linear(1, d_model)
        self.stats_proj = nn.Linear(path_features_dim, d_model)
        self.hist_proj = nn.Linear(5, d_model)

        self.cls_token = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.path_pos_emb = nn.Embedding(n_path, d_model)
        self.hist_pos_emb = nn.Embedding(n_hist, d_model)

        self.cond_transformer = self._make_encoder(
            d_model,
            n_heads,
            ff_dim,
            dropout,
            self.n_layers_cond,
            norm_first,
        )

        self.duration_head = RootDurationHead(
            d_model=d_model,
            path_features_dim=self.path_features_dim,
            min_frames=self.min_frames,
            max_frames=self.max_frames,
            dropout=dropout,
            pace_text_dim=pace_text_dim,
        )

        # Root plan queries.
        self.root_queries = nn.Parameter(torch.zeros(self.max_frames, d_model))
        nn.init.trunc_normal_(self.root_queries, std=0.02)
        self.root_pos_emb = nn.Embedding(self.max_frames, d_model)
        self.root_progress_proj = nn.Linear(1, d_model)
        self.frame_count_emb = nn.Embedding(self.max_frames - self.min_frames + 1, d_model)
        self.root_transformer = self._make_encoder(
            d_model,
            n_heads,
            ff_dim,
            dropout,
            self.n_layers_root,
            norm_first,
        )

        self.decoder_pred_dim = 5
        self.root_decoder = RootPlanDecoder(
            d_model=d_model,
            width=decoder_width,
            depth=decoder_depth,
            dilation_growth_rate=decoder_dilation_growth_rate,
            dropout=decoder_dropout,
            out_dim=self.decoder_pred_dim,
        )

        self._num_condition_specials = 3

    def forward(
        self,
        text_emb: Tensor,
        path: Tensor | None = None,
        path_valid_mask: Tensor | None = None,
        path_control_mask: Tensor | None = None,
        path_mode=None,
        path_features: Tensor | None = None,
        path_features_raw: Tensor | None = None,
        sample_mode=None,
        history_motion: Tensor | None = None,
        history_mask: Tensor | None = None,
        anchor_frame: Tensor | None = None,
        num_frames: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if (
            path is None
            or path_valid_mask is None
            or path_features is None
            or history_motion is None
            or history_mask is None
        ):
            raise TypeError(
                "RootRefiner.forward requires path/path_valid_mask/path_features/"
                "history_motion/history_mask"
            )
        batch_size = text_emb.shape[0]
        device = text_emb.device
        if path_control_mask is None:
            path_control_mask = path_valid_mask
        path_control_mask = (
            path_control_mask.to(device=device).bool()
            & path_valid_mask.to(device=device).bool()
        )
        if path_features_raw is None:
            path_features_raw = path_features
        path_features_raw = path_features_raw.to(device=device, dtype=path_features.dtype)

        (
            condition_hidden,
            cls_summary,
            path_summary,
            history_summary,
            text_summary,
        ) = self._encode_condition(
            text_emb=text_emb,
            path=path,
            path_valid_mask=path_valid_mask,
            path_control_mask=path_control_mask,
            path_features=path_features,
            history_motion=history_motion,
            history_mask=history_mask,
        )

        duration = self.duration_head(
            cls_summary=cls_summary,
            path_summary=path_summary,
            history_summary=history_summary,
            text_summary=text_summary,
            path_features=path_features,
            path_features_raw=path_features_raw,
        )
        if num_frames is not None:
            used_frames = num_frames.to(device=device, dtype=torch.long).clamp(
                self.min_frames,
                self.max_frames,
            )
        else:
            used_frames = duration["pred_frames"]

        frame_mask = self._frame_mask(used_frames, device)
        root_hidden = self._build_root_hidden(
            condition_hidden=condition_hidden,
            condition_pad=self._condition_pad(path_valid_mask, history_mask, device),
            used_frames=used_frames,
            frame_mask=frame_mask,
            dtype=path.dtype,
        )

        raw_waypoints = self.root_decoder(root_hidden, frame_mask)
        heading = F.normalize(raw_waypoints[..., 3:5], dim=-1, eps=1e-6)
        future_waypoints = torch.cat([raw_waypoints[..., :3], heading], dim=-1)
        future_waypoints = future_waypoints * frame_mask[..., None].to(future_waypoints.dtype)

        return {
            **duration,
            "used_frames": used_frames,
            "frame_mask": frame_mask,
            "future_waypoints": future_waypoints,
            "waypoints": future_waypoints,
        }

    def _encode_condition(
        self,
        *,
        text_emb: Tensor,
        path: Tensor,
        path_valid_mask: Tensor,
        path_control_mask: Tensor,
        path_features: Tensor,
        history_motion: Tensor,
        history_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size = text_emb.shape[0]
        device = text_emb.device

        cls_token = self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        text_token = self.text_proj(text_emb).unsqueeze(1)
        stats_token = self.stats_proj(path_features).unsqueeze(1)
        path_tokens = (
            self.path_proj(path)
            + self.path_pos_emb.weight.unsqueeze(0)
            + self.path_control_proj(path_control_mask.to(path.dtype).unsqueeze(-1))
        )
        history_tokens = (
            self.hist_proj(history_motion) + self.hist_pos_emb.weight.unsqueeze(0)
        )
        condition_seq = torch.cat(
            [cls_token, text_token, stats_token, path_tokens, history_tokens],
            dim=1,
        )
        condition_hidden = self.cond_transformer(
            condition_seq,
            src_key_padding_mask=self._condition_pad(path_valid_mask, history_mask, device),
        )
        cls_summary = condition_hidden[:, 0]
        text_summary = condition_hidden[:, 1]
        path_hidden = condition_hidden[
            :,
            self._num_condition_specials:self._num_condition_specials + self.n_path,
        ]
        history_hidden = condition_hidden[:, self._num_condition_specials + self.n_path:]
        path_mask = path_valid_mask.bool().to(path_hidden.dtype)
        path_denominator = path_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        path_summary = (path_hidden * path_mask[..., None]).sum(dim=1) / path_denominator
        history_indices = torch.arange(self.n_hist, device=device)
        last_history_index_1d = (
            history_mask.bool().long() * history_indices.unsqueeze(0)
        ).amax(dim=1)
        last_history_index = last_history_index_1d.view(batch_size, 1, 1).expand(
            -1,
            1,
            history_hidden.shape[-1],
        )
        history_summary = history_hidden.gather(1, last_history_index).squeeze(1)
        return condition_hidden, cls_summary, path_summary, history_summary, text_summary

    def _build_root_hidden(
        self,
        *,
        condition_hidden: Tensor,
        condition_pad: Tensor,
        used_frames: Tensor,
        frame_mask: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        batch_size = condition_hidden.shape[0]
        device = condition_hidden.device
        frame_idx = torch.arange(self.max_frames, device=device)
        progress_denominator = (used_frames.to(dtype=dtype) - 1.0).clamp(min=1.0)
        frame_progress = (
            frame_idx[None, :].to(dtype) / progress_denominator[:, None]
        ).clamp(0.0, 1.0)
        frame_class = (used_frames - self.min_frames).clamp(
            0,
            self.max_frames - self.min_frames,
        )
        count_token = self.frame_count_emb(frame_class).unsqueeze(1)
        root_queries = (
            self.root_queries
            + self.root_pos_emb.weight
        ).unsqueeze(0).expand(batch_size, -1, -1)
        root_queries = root_queries + self.root_progress_proj(frame_progress[..., None])
        root_seq = torch.cat([count_token, condition_hidden, root_queries], dim=1)

        count_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        root_query_pad = ~frame_mask
        root_pad = torch.cat([count_pad, condition_pad, root_query_pad], dim=1)
        root_hidden = self.root_transformer(root_seq, src_key_padding_mask=root_pad)
        root_hidden = root_hidden[:, -self.max_frames:]
        return root_hidden * frame_mask.unsqueeze(-1).to(root_hidden.dtype)

    def _condition_pad(
        self,
        path_valid_mask: Tensor,
        history_mask: Tensor,
        device,
    ) -> Tensor:
        batch_size = path_valid_mask.shape[0]
        special_pad = torch.zeros(
            batch_size,
            self._num_condition_specials,
            dtype=torch.bool,
            device=device,
        )
        return torch.cat(
            [special_pad, ~path_valid_mask.bool(), ~history_mask.bool()],
            dim=1,
        )

    def _frame_mask(self, used_frames: Tensor, device) -> Tensor:
        frame_idx = torch.arange(self.max_frames, device=device)
        return frame_idx.unsqueeze(0) < used_frames.unsqueeze(1)

    @staticmethod
    def _make_encoder(
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        n_layers: int,
        norm_first: bool,
    ) -> nn.TransformerEncoder:
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        return nn.TransformerEncoder(layer, num_layers=n_layers)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["RootRefiner", "RootDurationHead", "RootPlanDecoder", "ResBlock1D"]
