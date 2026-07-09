"""Lightning module for RootRefiner training.

The training entrypoint owns run orchestration and dataset construction. This
module owns the RootRefiner forward path, losses, validation modes, and
optimizer/scheduler construction.
"""

from __future__ import annotations

import logging
import math

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from torch import nn

from models.root_refiner import RootRefiner
from utils.initialize import instantiate
from utils.local_frame import wrap_angle
from utils.motion_process import build_physical_7d_from_5d
from utils.training.root_refiner.config_validate import validate_refiner_config
from utils.training.root_refiner.losses import (
    dense_path_control_loss,
    goal_point_control_loss,
    masked_mean,
    second_order_diff_l2,
    smooth_l1_masked,
    sparse_path_control_loss,
)
from utils.training.root_refiner.sampling_schedule import TrainingSchedule
from utils.training.root_refiner.text_encoder import resolve_text_encoder

log = logging.getLogger(__name__)


class RootRefinerLightningModule(pl.LightningModule):
    """Lightning module for RootRefiner training and validation."""

    _FREEZE_REFINER_PARAMETER_PREFIXES = {
        "condition_encoder": (
            "refiner.cls_token",
            "refiner.text_proj.",
            "refiner.path_proj.",
            "refiner.path_control_proj.",
            "refiner.stats_proj.",
            "refiner.hist_proj.",
            "refiner.path_pos_emb.",
            "refiner.hist_pos_emb.",
            "refiner.cond_transformer.",
        ),
        "duration_head": (
            "refiner.duration_head.",
        ),
        "root_transformer": (
            "refiner.root_queries",
            "refiner.root_pos_emb.",
            "refiner.root_progress_proj.",
            "refiner.frame_count_emb.",
            "refiner.root_transformer.",
        ),
        "root_decoder": (
            "refiner.root_decoder.",
        ),
    }
    _FREEZE_REFINER_PARAMETER_PREFIXES["root_branch"] = (
        *_FREEZE_REFINER_PARAMETER_PREFIXES["root_transformer"],
        *_FREEZE_REFINER_PARAMETER_PREFIXES["root_decoder"],
    )

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
            self.refiner = instantiate(
                target=target,
                cfg=None,
                hfstyle=False,
                **model_cfg,
            )
        self.min_frames = model_cfg["min_frames"]
        self.max_frames = model_cfg["max_frames"]
        text_emb_dim = model_cfg.get("text_emb_dim", 512)
        self.text_encoder = resolve_text_encoder(cfg, text_encoder, text_emb_dim)
        self.loss_weights = dict(cfg.get("loss_weights", {}))
        self.heading_form = cfg.get("loss", {}).get("heading_form", "cosine")
        self.validation_suite_names = self._validation_suite_names(cfg)
        self._permanently_frozen_parameter_names: set[str] = set()
        self.freeze_refiner_modules = self._freeze_refiner_modules_from_cfg(cfg)
        self.schedule_freeze_refiner_modules = (
            self._schedule_freeze_refiner_modules_from_cfg(cfg)
        )
        self._apply_freeze_config()
        self.save_hyperparameters(ignore=["text_encoder"])

    @staticmethod
    def _validation_suite_names(cfg: dict) -> list[str]:
        suites = (cfg.get("validation") or {}).get("suites") or []
        names = [
            str(suite.get("name"))
            for suite in suites
            if isinstance(suite, dict) and suite.get("name")
        ]
        return names or ["default"]

    @staticmethod
    def _freeze_refiner_modules_from_cfg(cfg: dict) -> tuple[str, ...]:
        freeze_cfg = cfg.get("freeze") or {}
        modules = freeze_cfg.get("refiner_modules") or []
        if isinstance(modules, str):
            modules = [modules]
        return tuple(str(module) for module in modules)

    @staticmethod
    def _schedule_freeze_refiner_modules_from_cfg(cfg: dict) -> tuple[str, ...]:
        schedule = TrainingSchedule.from_config(cfg)
        if schedule is None:
            return ()
        return schedule.all_schedule_freeze_modules()

    @classmethod
    def _matches_parameter_prefix(cls, name: str, prefixes: tuple[str, ...]) -> bool:
        return any(name == prefix or name.startswith(prefix) for prefix in prefixes)

    def _apply_freeze_config(self) -> None:
        if not self.freeze_refiner_modules:
            return
        prefixes = tuple(
            prefix
            for module_name in self.freeze_refiner_modules
            for prefix in self._FREEZE_REFINER_PARAMETER_PREFIXES[module_name]
        )
        frozen_parameters = 0
        frozen_tensors = 0
        for name, parameter in self.named_parameters():
            if self._matches_parameter_prefix(name, prefixes):
                parameter.requires_grad_(False)
                self._permanently_frozen_parameter_names.add(name)
                frozen_parameters += int(parameter.numel())
                frozen_tensors += 1
        trainable_parameters = sum(
            int(parameter.numel())
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        log.info(
            "RootRefiner freeze config: modules=%s, frozen_tensors=%d, "
            "frozen_parameters=%d, trainable_parameters=%d",
            list(self.freeze_refiner_modules),
            frozen_tensors,
            frozen_parameters,
            trainable_parameters,
        )

    def apply_schedule_freeze(self, refiner_modules: list[str] | tuple[str, ...]) -> None:
        controlled_modules = self.schedule_freeze_refiner_modules or tuple(
            str(module) for module in refiner_modules
        )
        if not controlled_modules:
            return
        controlled_prefixes = self._prefixes_for_refiner_modules(controlled_modules)
        frozen_prefixes = self._prefixes_for_refiner_modules(refiner_modules)
        for name, parameter in self.named_parameters():
            if name in self._permanently_frozen_parameter_names:
                continue
            if self._matches_parameter_prefix(name, controlled_prefixes):
                parameter.requires_grad_(
                    not self._matches_parameter_prefix(name, frozen_prefixes)
                )

    @classmethod
    def _prefixes_for_refiner_modules(
        cls,
        refiner_modules: list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        return tuple(
            prefix
            for module_name in refiner_modules
            for prefix in cls._FREEZE_REFINER_PARAMETER_PREFIXES[str(module_name)]
        )

    def forward(
        self,
        batch: dict,
        *,
        duration_mode: str = "groundtruth_duration",
    ) -> dict:
        if duration_mode not in {"groundtruth_duration", "pred_duration"}:
            raise ValueError(
                "duration_mode must be 'groundtruth_duration' or 'pred_duration', "
                f"got {duration_mode!r}."
            )
        text_emb = self.text_encoder.encode(batch["text"], device=self.device)
        num_frames = (
            batch.get("num_frames")
            if duration_mode == "groundtruth_duration"
            else None
        )
        return self.refiner(
            text_emb=text_emb,
            path=batch["path"],
            path_valid_mask=batch["path_valid_mask"],
            path_control_mask=batch.get("path_control_mask"),
            path_features=batch["path_features"],
            path_features_raw=batch.get("path_features_raw", batch["path_features"]),
            history_motion=batch["history_motion"],
            history_mask=batch["history_mask"],
            anchor_frame=batch.get("anchor_frame"),
            num_frames=num_frames,
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

        raw_features = batch.get("path_features_raw", batch["path_features"]).to(
            device=out["waypoints"].device,
            dtype=out["waypoints"].dtype,
        )
        effective_length = (
            raw_features[:, 0].clamp_min(0.0)
            + raw_features[:, 3].clamp_min(0.0)
        )
        pace_valid = effective_length >= 0.05
        target_frames = batch["num_frames"].to(
            device=out["waypoints"].device,
            dtype=out["waypoints"].dtype,
        )
        target_log_pace = torch.log(
            target_frames.clamp_min(1.0) / effective_length.clamp_min(0.05)
        )
        pace_terms = F.smooth_l1_loss(
            out["pred_log_pace"],
            target_log_pace,
            reduction="none",
        )
        loss_pace = masked_mean(pace_terms, pace_valid)
        frame_pace_terms = F.smooth_l1_loss(
            out["pred_frames_float"],
            target_frames,
            reduction="none",
        )
        loss_frame_pace = masked_mean(frame_pace_terms, pace_valid)

        loss_xyz = smooth_l1_masked(
            out["waypoints"][..., 0:3],
            target_wp[..., 0:3],
            target_mask,
        )

        pred_h = F.normalize(out["waypoints"][..., 3:5], dim=-1, eps=1e-6)
        gt_h = F.normalize(target_wp[..., 3:5], dim=-1, eps=1e-6)
        heading_dot = (pred_h * gt_h).sum(-1).clamp(-1.0, 1.0)
        if self.heading_form == "cosine":
            head_term = 1.0 - heading_dot
            loss_heading = masked_mean(head_term, target_mask)
        else:
            loss_heading = smooth_l1_masked(pred_h, gt_h, target_mask)
        heading_flip_margin = math.cos(math.radians(60.0))
        loss_heading_flip = masked_mean(
            F.relu(heading_flip_margin - heading_dot).pow(2),
            target_mask,
        )

        pred_waypoints5 = torch.cat([out["waypoints"][..., :3], pred_h], dim=-1)
        target_waypoints5 = torch.cat([target_wp[..., :3], gt_h], dim=-1)
        pred_delta = self._to_physical_7d(
            self._prepend_local_anchor_5d(pred_waypoints5)
        )[:, 1:, 5:7]
        if "waypoints_physical" in batch:
            target_physical = batch["waypoints_physical"].to(
                device=pred_waypoints5.device,
                dtype=pred_waypoints5.dtype,
            )
        elif "target_waypoints_physical" in batch:
            target_physical = batch["target_waypoints_physical"].to(
                device=pred_waypoints5.device,
                dtype=pred_waypoints5.dtype,
            )
        else:
            target_physical = self._to_physical_7d(
                self._prepend_local_anchor_5d(target_waypoints5)
            )[:, 1:]
        target_delta = target_physical[..., 5:7]
        delta_mask = target_mask.clone()
        loss_fwd_delta = smooth_l1_masked(
            pred_delta[..., 0:1],
            target_delta[..., 0:1],
            delta_mask,
        )
        loss_yaw_delta = smooth_l1_masked(
            pred_delta[..., 1:2],
            target_delta[..., 1:2],
            delta_mask,
        )
        if pred_h.shape[1] < 2:
            loss_yaw_jump = pred_h.new_zeros(())
        else:
            target_valid = target_mask.bool()
            valid_heading = target_valid & (pred_h.norm(dim=-1) > 1e-6)
            identity_heading = pred_h.new_tensor([1.0, 0.0]).view(1, 1, 2)
            yaw_jump_heading = torch.where(
                valid_heading.unsqueeze(-1),
                pred_h,
                identity_heading,
            )
            pred_yaw = torch.atan2(yaw_jump_heading[..., 1], yaw_jump_heading[..., 0])
            pred_yaw_delta = wrap_angle(pred_yaw[:, 1:] - pred_yaw[:, :-1])
            yaw_jump_mask = target_valid[:, 1:] & target_valid[:, :-1]
            yaw_jump_threshold = math.radians(45.0)
            loss_yaw_jump = masked_mean(
                F.relu(pred_yaw_delta.abs() - yaw_jump_threshold).pow(2),
                yaw_jump_mask,
            )
        weights = self.loss_weights
        loss_smoothness = second_order_diff_l2(pred_delta, delta_mask)
        if float(weights.get("path_control", 0.0)) == 0.0:
            loss_path_control = out["waypoints"].new_zeros(())
        else:
            loss_path_control = self._compute_path_control_loss(out, batch, target_mask)
        loss = (
            weights.get("pace", 0.0) * loss_pace
            + weights.get("frame_pace", 1.0) * loss_frame_pace
            + weights.get("xyz", 5.0) * loss_xyz
            + weights.get("heading", 1.0) * loss_heading
            + weights.get("heading_flip", 0.0) * loss_heading_flip
            + weights.get("fwd_delta", 0.5) * loss_fwd_delta
            + weights.get("yaw_delta", 0.5) * loss_yaw_delta
            + weights.get("yaw_jump", 0.0) * loss_yaw_jump
            + weights.get("path_control", 0.0) * loss_path_control
            + weights.get("smoothness", 0.0) * loss_smoothness
        )

        with torch.no_grad():
            float_err = (
                out["pred_frames_float"].to(target_frames.device)
                - target_frames
            ).abs()
            float_err_mean = masked_mean(float_err, pace_valid.to(float_err.device))
            physical_metrics = self._compute_physical_xyz_metrics(
                out,
                batch,
                target_mask,
            )
        return {
            "loss": loss,
            "pace": loss_pace,
            "frame_pace": loss_frame_pace,
            "xyz": loss_xyz,
            "heading": loss_heading,
            "heading_flip": loss_heading_flip,
            "fwd_delta": loss_fwd_delta,
            "yaw_delta": loss_yaw_delta,
            "yaw_jump": loss_yaw_jump,
            "path_control": loss_path_control,
            "smoothness": loss_smoothness,
            "frame_pace_mae": float_err_mean,
            "frame_pace_acc_pm1": masked_mean(
                (float_err <= 1.0).to(float_err.dtype),
                pace_valid.to(float_err.device),
            ),
            "frame_pace_acc_pm4": masked_mean(
                (float_err <= 4.0).to(float_err.dtype),
                pace_valid.to(float_err.device),
            ),
            **physical_metrics,
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
            sample_indices = [
                sample_idx
                for sample_idx, sample_mode in enumerate(path_modes)
                if sample_mode == mode
            ]
            if not sample_indices:
                continue
            index_tensor = torch.as_tensor(
                sample_indices,
                dtype=torch.long,
                device=out["waypoints"].device,
            )
            pred_waypoints = out["waypoints"].index_select(0, index_tensor)
            path = batch["path"].to(out["waypoints"].device).index_select(
                0,
                index_tensor,
            )
            control_mask = batch["path_control_mask"].to(
                out["waypoints"].device
            ).index_select(0, index_tensor)
            if mode == "dense_path":
                base_supervision = batch.get("path_supervision_mask", target_mask)
                supervision = (
                    base_supervision.to(out["waypoints"].device)
                    .bool()
                    .index_select(0, index_tensor)
                    & target_mask.to(out["waypoints"].device)
                    .bool()
                    .index_select(0, index_tensor)
                )
                losses.append(
                    dense_path_control_loss(
                        pred_waypoints,
                        path,
                        supervision,
                    )
                )
            elif mode == "sparse_path":
                base_supervision = batch.get("path_supervision_mask", target_mask)
                supervision = (
                    base_supervision.to(out["waypoints"].device)
                    .bool()
                    .index_select(0, index_tensor)
                    & target_mask.to(out["waypoints"].device)
                    .bool()
                    .index_select(0, index_tensor)
                )
                losses.append(
                    sparse_path_control_loss(
                        pred_waypoints,
                        path,
                        control_mask,
                        supervision,
                        offset_start_frames.index_select(0, index_tensor),
                    )
                )
            else:
                losses.append(
                    goal_point_control_loss(
                        pred_waypoints,
                        target_mask.to(out["waypoints"].device).index_select(
                            0,
                            index_tensor,
                        ),
                        path,
                        control_mask,
                    )
                )
        if not losses:
            return out["waypoints"].new_zeros(())
        return torch.stack(losses).mean()

    METRIC_KEYS = (
        "frame_pace_mae",
        "frame_pace_acc_pm1",
        "frame_pace_acc_pm4",
        "xyz_ADE_m",
        "xyz_FDE_m",
    )

    def _to_physical_7d(self, waypoints5: torch.Tensor) -> torch.Tensor:
        return build_physical_7d_from_5d(waypoints5)

    @staticmethod
    def _prepend_local_anchor_5d(waypoints5: torch.Tensor) -> torch.Tensor:
        anchor = waypoints5.new_zeros(waypoints5.shape[0], 1, waypoints5.shape[-1])
        anchor[..., 1] = waypoints5[:, :1, 1]
        anchor[..., 3] = 1.0
        return torch.cat([anchor, waypoints5], dim=1)

    def _common_prefix_mask(self, batch: dict, out: dict) -> torch.Tensor:
        target_mask = batch["waypoints_mask"]
        used_frames = out["used_frames"].to(target_mask.device, dtype=torch.long)
        frame_idx = torch.arange(target_mask.shape[1], device=target_mask.device)
        return target_mask.bool() & (frame_idx.unsqueeze(0) < used_frames.unsqueeze(1))

    def _compute_physical_xyz_metrics(
        self,
        out: dict,
        batch: dict,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred_physical = self._to_physical_7d(out["waypoints"])
        if "waypoints_physical" in batch:
            target_physical = batch["waypoints_physical"].to(
                device=pred_physical.device,
                dtype=pred_physical.dtype,
            )
        else:
            target_waypoints5 = torch.cat(
                [
                    batch["waypoints"][..., :3],
                    F.normalize(batch["waypoints"][..., 3:5], dim=-1, eps=1e-6),
                ],
                dim=-1,
            )
            target_physical = self._to_physical_7d(target_waypoints5)
        valid = mask.to(device=pred_physical.device).bool()
        xyz_err = (pred_physical[..., :3] - target_physical[..., :3]).norm(dim=-1)
        valid_float = valid.to(dtype=xyz_err.dtype)
        ade = (xyz_err * valid_float).sum() / valid_float.sum().clamp_min(1.0)
        valid_counts = valid.long().sum(dim=1)
        has_valid = valid_counts > 0
        last_indices = valid_counts.sub(1).clamp(min=0)
        fde_per_sample = xyz_err.gather(1, last_indices.view(-1, 1)).squeeze(1)
        has_valid_float = has_valid.to(dtype=xyz_err.dtype)
        fde = (
            (fde_per_sample * has_valid_float).sum()
            / has_valid_float.sum().clamp_min(1.0)
        )
        return {"xyz_ADE_m": ade, "xyz_FDE_m": fde}

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
            self.log(
                "train/nonfinite_skip",
                1.0,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
            return None
        logger_cfg = self.cfg.get("logger") or {}
        wandb_cfg = logger_cfg.get("wandb") or {}
        exclude_log_keys = set(
            str(key) for key in (wandb_cfg.get("train_exclude_log_keys") or [])
        )
        for key, value in losses.items():
            if key in exclude_log_keys:
                continue
            self.log(
                f"train/{key}",
                value,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
        return loss

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        modes = (self.cfg.get("validation") or {}).get(
            "eval_modes",
            ["groundtruth_duration", "pred_duration"],
        )
        if dataloader_idx < len(self.validation_suite_names):
            suite_name = self.validation_suite_names[dataloader_idx]
        else:
            suite_name = f"suite_{dataloader_idx}"
        validation_cfg = self.cfg.get("validation") or {}
        log_keys_cfg = validation_cfg.get("log_keys")
        log_keys = set(str(key) for key in log_keys_cfg) if log_keys_cfg else None
        batch_size = int(batch["num_frames"].shape[0])
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
            for key, value in losses.items():
                if log_keys is not None and key not in log_keys:
                    continue
                self.log(
                    f"val_{suite_name}/{mode}/{key}",
                    value,
                    prog_bar=(key == "loss" and mode == "groundtruth_duration"),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                    add_dataloader_idx=False,
                )
        return first_loss

    def configure_optimizers(self):
        opt_cfg = self.cfg["optimizer"]
        optim_target = opt_cfg["target"]
        if len(optim_target.split(".")) == 1:
            optim_target = "torch.optim." + optim_target
        dynamic_prefixes = self._prefixes_for_refiner_modules(
            self.schedule_freeze_refiner_modules
        )
        trainable_parameters = [
            parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
            or (
                dynamic_prefixes
                and name not in self._permanently_frozen_parameter_names
                and self._matches_parameter_prefix(name, dynamic_prefixes)
            )
        ]
        if not trainable_parameters:
            raise ValueError(
                "RootRefiner has no trainable parameters after applying "
                "freeze.refiner_modules."
            )
        optimizer = instantiate(
            target=optim_target,
            cfg=None,
            hfstyle=False,
            params=trainable_parameters,
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
