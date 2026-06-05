"""Lightning module for RootRefiner training.

The training entrypoint owns run orchestration and dataset construction. This
module owns the RootRefiner forward path, losses, validation modes, and
optimizer/scheduler construction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.root_refiner import RootRefiner
from utils.initialize import instantiate
from utils.motion_process import build_physical_7d_from_normalized_5d
from utils.refiner.config_validate import validate_refiner_config
from utils.refiner.losses import (
    dense_path_control_loss,
    goal_point_control_loss,
    masked_mean,
    ordinal_duration_loss,
    second_order_diff_l2,
    smooth_l1_masked,
    sparse_path_control_loss,
)
from utils.text_encoder_resolver import resolve_text_encoder

log = logging.getLogger(__name__)


class RootRefinerLightningModule(pl.LightningModule):
    """Lightning module for RootRefiner training and validation."""

    def __init__(self, cfg: dict, text_encoder: nn.Module | None = None):
        super().__init__()
        validate_refiner_config(cfg)
        self.cfg = cfg
        model_block = cfg["model"]
        model_cfg = dict(model_block["params"])
        target = model_block.get("target", "models.root_refiner.RootRefiner")
        if target == "models.root_refiner.RootRefiner":
            self.refiner = RootRefiner(**model_cfg)
        else:
            self.refiner = instantiate(target=target, cfg=None, hfstyle=False, **model_cfg)
        self.min_tokens = model_cfg["min_tokens"]
        self.max_tokens = model_cfg["max_tokens"]
        text_emb_dim = model_cfg.get("text_emb_dim", 512)
        self.text_encoder = resolve_text_encoder(cfg, text_encoder, text_emb_dim)
        self.loss_weights = dict(cfg.get("loss_weights", {}))
        self.heading_form = cfg.get("loss", {}).get("heading_form", "cosine")
        self.ordinal_sigma = float(cfg.get("loss", {}).get("ordinal_sigma", 1.0))
        self._register_waypoint_stats(cfg)
        self.save_hyperparameters(ignore=["text_encoder"])

    def forward(self, batch: dict, *, duration_mode: str = "groundtruth_duration") -> dict:
        if duration_mode not in {"groundtruth_duration", "pred_duration"}:
            raise ValueError(
                "duration_mode must be 'groundtruth_duration' or 'pred_duration', "
                f"got {duration_mode!r}."
            )
        text_emb = self.text_encoder.encode(batch["text"], device=self.device)
        num_tokens = batch.get("num_tokens") if duration_mode == "groundtruth_duration" else None
        return self.refiner(
            text_emb=text_emb,
            path=batch["path"],
            path_valid_mask=batch["path_valid_mask"],
            path_control_mask=batch.get("path_control_mask"),
            path_mode=batch.get("path_mode"),
            path_features=batch["path_features"],
            history_motion=batch["history_motion"],
            history_mask=batch["history_mask"],
            offset_start_frames=batch.get("offset_start_frames"),
            num_tokens=num_tokens,
        )

    def _compute_loss(
        self,
        out: dict,
        batch: dict,
        *,
        target_mask: torch.Tensor | None = None,
    ) -> dict:
        target_wp = batch["waypoints"]
        if target_mask is None:
            target_mask = batch["waypoints_mask"]

        target_class = batch["num_tokens"] - self.min_tokens
        dur = ordinal_duration_loss(
            out["num_token_logits"], batch["num_tokens"],
            min_tokens=self.min_tokens, sigma=self.ordinal_sigma,
        )
        L_num = dur["ordinal_ce"]
        L_num_soft = dur["expected"]
        expected_class = dur["expected_num_tokens"] - float(self.min_tokens)

        L_xyz = smooth_l1_masked(out["waypoints"][..., 0:3], target_wp[..., 0:3], target_mask)

        pred_h = F.normalize(out["waypoints"][..., 3:5], dim=-1, eps=1e-6)
        gt_h = target_wp[..., 3:5]
        if self.heading_form == "cosine":
            head_term = 1.0 - (pred_h * gt_h).sum(-1)
            L_head = masked_mean(head_term, target_mask)
        else:
            L_head = smooth_l1_masked(pred_h, gt_h, target_mask)

        pred5 = torch.cat([out["waypoints"][..., :3], pred_h], dim=-1)
        gt5 = torch.cat(
            [target_wp[..., :3], F.normalize(target_wp[..., 3:5], dim=-1, eps=1e-6)],
            dim=-1,
        )
        pred_delta = self._to_physical_7d(pred5)[..., 5:7]
        if "waypoints_physical" in batch:
            gt7_phys = batch["waypoints_physical"].to(device=pred5.device, dtype=pred5.dtype)
        elif "target_waypoints_physical" in batch:
            gt7_phys = batch["target_waypoints_physical"].to(device=pred5.device, dtype=pred5.dtype)
        else:
            gt7_phys = self._to_physical_7d(gt5)
        gt_delta = gt7_phys[..., 5:7]
        delta_mask = target_mask.clone()
        delta_mask[:, 0] = False
        L_fwd_delta = smooth_l1_masked(pred_delta[..., 0:1], gt_delta[..., 0:1], delta_mask)
        L_yaw_delta = smooth_l1_masked(pred_delta[..., 1:2], gt_delta[..., 1:2], delta_mask)
        w = self.loss_weights
        L_smooth = second_order_diff_l2(pred_delta, delta_mask)
        if float(w.get("path_control", 0.0)) == 0.0:
            L_path_control = out["waypoints"].new_zeros(())
        else:
            L_path_control = self._compute_path_control_loss(out, batch, target_mask)
        loss = (
            w.get("num_token", 1.0) * L_num
            + w.get("num_token_soft", 0.1) * L_num_soft
            + w.get("xyz", 5.0) * L_xyz
            + w.get("heading", 1.0) * L_head
            + w.get("fwd_delta", 0.5) * L_fwd_delta
            + w.get("yaw_delta", 0.5) * L_yaw_delta
            + w.get("path_control", 0.0) * L_path_control
            + w.get("smoothness", 0.0) * L_smooth
        )

        with torch.no_grad():
            pred_class = out["num_token_logits"].argmax(dim=-1)
            pred_token_class = out["pred_num_tokens"].to(target_class.device) - self.min_tokens
            err = (pred_token_class - target_class).abs().float()
            argmax_err = (pred_class - target_class).abs().float()
            soft_err = (expected_class - target_class.to(expected_class.dtype)).abs()
        return {
            "loss": loss,
            "num_token": L_num,
            "num_token_soft": L_num_soft,
            "xyz": L_xyz,
            "heading": L_head,
            "fwd_delta": L_fwd_delta,
            "yaw_delta": L_yaw_delta,
            "path_control": L_path_control,
            "smoothness": L_smooth,
            "num_token_acc": (err == 0).float().mean(),
            "num_token_acc_pm1": (err <= 1).float().mean(),
            "num_token_acc_pm2": (err <= 2).float().mean(),
            "num_token_mae": err.mean(),
            "num_token_argmax_mae": argmax_err.mean(),
            "num_token_soft_mae": soft_err.mean(),
        }

    def _compute_path_control_loss(
        self, out: dict, batch: dict, target_mask: torch.Tensor
    ) -> torch.Tensor:
        if "path" not in batch or "path_control_mask" not in batch:
            return out["waypoints"].new_zeros(())

        path_modes = batch.get("path_mode")
        if path_modes is None:
            path_modes = ["dense_path"] * out["waypoints"].shape[0]
        offset_start_frames = batch.get("offset_start_frames")
        if offset_start_frames is None:
            offset_start_frames = torch.zeros(
                out["waypoints"].shape[0],
                dtype=torch.long,
                device=out["waypoints"].device,
            )
        else:
            offset_start_frames = offset_start_frames.to(out["waypoints"].device)

        losses = []
        for mode in ("dense_path", "sparse_path", "goal_point"):
            idx = [i for i, sample_mode in enumerate(path_modes) if sample_mode == mode]
            if not idx:
                continue
            index = torch.as_tensor(idx, dtype=torch.long, device=out["waypoints"].device)
            pred = out["waypoints"].index_select(0, index)
            path = batch["path"].to(out["waypoints"].device).index_select(0, index)
            control = batch["path_control_mask"].to(out["waypoints"].device).index_select(0, index)
            if mode == "dense_path":
                base_supervision = batch.get("path_supervision_mask", target_mask)
                supervision = (
                    base_supervision.to(out["waypoints"].device).bool().index_select(0, index)
                    & target_mask.to(out["waypoints"].device).bool().index_select(0, index)
                )
                losses.append(
                    dense_path_control_loss(
                        pred,
                        path,
                        supervision,
                    )
                )
            elif mode == "sparse_path":
                base_supervision = batch.get("path_supervision_mask", target_mask)
                supervision = (
                    base_supervision.to(out["waypoints"].device).bool().index_select(0, index)
                    & target_mask.to(out["waypoints"].device).bool().index_select(0, index)
                )
                losses.append(
                    sparse_path_control_loss(
                        pred,
                        path,
                        control,
                        supervision,
                        offset_start_frames.index_select(0, index),
                    )
                )
            else:
                losses.append(
                    goal_point_control_loss(
                        pred,
                        target_mask.to(out["waypoints"].device).index_select(0, index),
                        path,
                        control,
                    )
                )
        if not losses:
            return out["waypoints"].new_zeros(())
        return torch.stack(losses).mean()

    METRIC_KEYS = (
        "num_token_acc",
        "num_token_acc_pm1",
        "num_token_acc_pm2",
        "num_token_mae",
        "num_token_argmax_mae",
        "num_token_soft_mae",
    )

    def _register_waypoint_stats(self, cfg: dict) -> None:
        data_cfg = cfg.get("data", {}) or {}
        normalize = bool(data_cfg.get("normalize", False))
        stats_dir = data_cfg.get("stats_dir")
        if not normalize:
            self.register_buffer("_wp_mean", None, persistent=False)
            self.register_buffer("_wp_std", None, persistent=False)
            self.register_buffer("_wp_norm_idx", None, persistent=False)
            return
        if not stats_dir:
            raise FileNotFoundError(
                "data.normalize=true requires data.stats_dir for physical waypoint losses"
            )
        stats_path = Path(stats_dir)
        if not stats_path.is_dir():
            raise FileNotFoundError(
                f"data.normalize=true requires an existing stats_dir, got {stats_path}"
            )
        required = (
            "waypoint_mean.npy",
            "waypoint_std.npy",
            "waypoint_norm_indices.npy",
        )
        missing = [name for name in required if not (stats_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"stats_dir {stats_path} is missing required waypoint stats: {missing}"
            )
        self.register_buffer(
            "_wp_mean",
            torch.as_tensor(np.load(stats_path / "waypoint_mean.npy"), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_wp_std",
            torch.as_tensor(np.load(stats_path / "waypoint_std.npy"), dtype=torch.float32).clamp(min=1e-6),
            persistent=False,
        )
        self.register_buffer(
            "_wp_norm_idx",
            torch.as_tensor(np.load(stats_path / "waypoint_norm_indices.npy"), dtype=torch.long),
            persistent=False,
        )

    def _to_physical_7d(self, waypoints5: torch.Tensor) -> torch.Tensor:
        return build_physical_7d_from_normalized_5d(
            waypoints5,
            self._wp_mean,
            self._wp_std,
            self._wp_norm_idx,
        )

    def _common_prefix_mask(self, batch: dict, out: dict) -> torch.Tensor:
        target_mask = batch["waypoints_mask"]
        used_num_tokens = out["used_num_tokens"].to(target_mask.device, dtype=torch.long)
        valid_eff = (
            self.refiner.frames_per_token * used_num_tokens
            - (self.refiner.frames_per_token - 1)
        )
        frame_idx = torch.arange(target_mask.shape[1], device=target_mask.device)
        return target_mask.bool() & (frame_idx.unsqueeze(0) < valid_eff.unsqueeze(1))

    def training_step(self, batch: dict, batch_idx: int):
        out = self(batch)
        losses = self._compute_loss(out, batch)
        loss = losses["loss"]
        if not torch.isfinite(loss):
            log.warning(
                "non-finite train loss (%s) at global_step=%d batch_idx=%d; "
                "skipping optimizer step for this batch.",
                loss.detach().item(), self.global_step, batch_idx,
            )
            self.log("train/nonfinite_skip", 1.0, prog_bar=False, on_step=True, on_epoch=False)
            return None
        for k, v in losses.items():
            self.log(f"train/{k}", v, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        modes = (self.cfg.get("validation") or {}).get(
            "eval_modes",
            ["groundtruth_duration", "pred_duration"],
        )
        batch_size = int(batch["num_tokens"].shape[0])
        first_loss = None
        for mode in modes:
            out = self(batch, duration_mode=mode)
            metric_mask = (
                self._common_prefix_mask(batch, out)
                if mode == "pred_duration"
                else batch["waypoints_mask"]
            )
            losses = self._compute_loss(out, batch, target_mask=metric_mask)
            if first_loss is None:
                first_loss = losses["loss"]
            for k, v in losses.items():
                self.log(
                    f"val_{mode}/{k}",
                    v,
                    prog_bar=(k == "loss" and mode == "groundtruth_duration"),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
        return first_loss

    def configure_optimizers(self):
        opt_cfg = self.cfg["optimizer"]
        optim_target = opt_cfg["target"]
        if len(optim_target.split(".")) == 1:
            optim_target = "torch.optim." + optim_target
        optimizer = instantiate(
            target=optim_target,
            cfg=None,
            hfstyle=False,
            params=(p for p in self.parameters() if p.requires_grad),
            **dict(opt_cfg.get("params") or {}),
        )

        sched_cfg = self.cfg.get("lr_scheduler") or {}
        sched_target = sched_cfg.get("target")
        if not sched_target:
            return optimizer
        if len(sched_target.split(".")) == 1:
            sched_target = "torch.optim.lr_scheduler." + sched_target
        scheduler = instantiate(
            target=sched_target,
            cfg=None,
            hfstyle=False,
            optimizer=optimizer,
            **dict(sched_cfg.get("params") or {}),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": sched_cfg.get("interval", "step"),
                "frequency": int(sched_cfg.get("frequency", 1)),
            },
        }


RefinerLightningModule = RootRefinerLightningModule


__all__ = ["RootRefinerLightningModule", "RefinerLightningModule"]
