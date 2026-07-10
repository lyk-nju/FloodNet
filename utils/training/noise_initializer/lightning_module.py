"""Lightning wrapper for online residual NoiseInitializer training."""

from __future__ import annotations

from typing import Callable

import lightning.pytorch as pl
import torch
from torch import nn

from models.noise_initializer import NoiseInitializer
from utils.training.noise_initializer.losses import (
    anchored_root_xz_loss,
    delta_zT_l2_regularization,
)
from utils.training.noise_initializer.shadow_rollout import (
    run_residual_shadow_rollout,
)


def _cfg_get(cfg: dict, *keys, default=None):
    value = cfg
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _build_initializer(cfg: dict) -> NoiseInitializer:
    model_cfg = dict(_cfg_get(cfg, "model", default={}) or {})
    params = dict(model_cfg.get("params", model_cfg))
    return NoiseInitializer(**params)


def _single_sample_xz(value: torch.Tensor) -> torch.Tensor:
    if value.dim() == 3:
        return value[0]
    return value


class NoiseInitializerLightningModule(pl.LightningModule):
    """Train only ``NoiseInitializer`` against frozen LDF/VAE shadow rollouts."""

    def __init__(
        self,
        cfg: dict,
        *,
        initializer: nn.Module | None = None,
        ldf_model: nn.Module | None = None,
        vae: nn.Module | None = None,
        rollout_fn: Callable | None = None,
        decode_latents_fn: Callable | None = None,
    ):
        super().__init__()
        self.cfg = dict(cfg)
        self.initializer = initializer if initializer is not None else _build_initializer(self.cfg)
        self.ldf_model = ldf_model
        self.vae = vae
        self.rollout_fn = rollout_fn
        self.decode_latents_fn = decode_latents_fn or self._default_decode_latents
        self.save_hyperparameters(ignore=["initializer", "ldf_model", "vae", "rollout_fn", "decode_latents_fn"])

    @staticmethod
    def _default_decode_latents(vae, latents: torch.Tensor) -> torch.Tensor:
        if vae is None:
            return latents[..., :2]
        decoded = vae.decode(latents) if hasattr(vae, "decode") else vae(latents)
        return decoded[..., :2]

    def training_step(self, batch: dict, batch_idx: int):
        del batch_idx
        if self.ldf_model is None:
            raise RuntimeError("NoiseInitializerLightningModule requires ldf_model")
        if self.rollout_fn is None:
            raise RuntimeError("NoiseInitializerLightningModule requires rollout_fn")

        rollout_cfg = dict(self.cfg.get("rollout", {}) or {})
        loss_cfg = dict(self.cfg.get("loss", {}) or {})
        result = run_residual_shadow_rollout(
            model=self.ldf_model,
            vae=self.vae,
            initializer=self.initializer,
            context=batch["context"],
            rollout_fn=self.rollout_fn,
            alpha=float(rollout_cfg.get("alpha", 1.0)),
            rollout_tokens=int(rollout_cfg.get("loss_horizon_tokens", 1)),
            first_chunk=bool(batch.get("first_chunk", True)),
            max_delta_norm_ratio=rollout_cfg.get("max_delta_norm_ratio", None),
        )
        pred_xz = self.decode_latents_fn(
            self.vae,
            result.shadow_latents,
            committed_prefix_latents=batch.get("committed_prefix_latents"),
            shadow_start_token=int(batch.get("commit_index", 0)),
            frames_per_token=int(batch.get("frames_per_token", 4)),
        )
        target_xz = batch["target_xz"].to(device=pred_xz.device, dtype=pred_xz.dtype)
        target_mask = batch["target_mask"].to(device=pred_xz.device, dtype=pred_xz.dtype)
        traj_loss, parts = anchored_root_xz_loss(
            _single_sample_xz(pred_xz),
            _single_sample_xz(target_xz),
            target_mask[0] if target_mask.dim() == 2 else target_mask,
            history_frames=int(batch.get("history_frames", 0)),
            lambda_vel=float(loss_cfg.get("lambda_vel", 0.0)),
            anchor_mode=str(loss_cfg.get("anchor_mode", "target_anchor_abs")),
            generated_anchor_xz=batch.get("generated_anchor_xz"),
        )
        delta_reg = delta_zT_l2_regularization(result.raw_delta_zT)
        loss = traj_loss + float(loss_cfg.get("lambda_delta", 0.0)) * delta_reg
        raw_norm = result.raw_delta_zT.detach().float().reshape(
            int(result.raw_delta_zT.shape[0]), -1
        ).norm(dim=1)
        clipped_norm = result.delta_zT.detach().float().reshape(
            int(result.delta_zT.shape[0]), -1
        ).norm(dim=1)
        base_norm = batch["context"].frontier_base_zT.detach().float().reshape(
            int(result.delta_zT.shape[0]), -1
        ).norm(dim=1)
        scale = result.delta_scale.detach().float().reshape(int(result.delta_zT.shape[0]), -1)
        self.last_step_diagnostics = {
            "raw_delta_norm": float(raw_norm.mean().cpu().item()),
            "clipped_delta_norm": float(clipped_norm.mean().cpu().item()),
            "base_zT_norm": float(base_norm.mean().cpu().item()),
            "clipped_to_base_ratio": float(
                (clipped_norm / base_norm.clamp(min=1e-12)).mean().cpu().item()
            ),
            "delta_scale_mean": float(scale.mean().cpu().item()),
            "clip_saturation_ratio": float((scale < 0.999999).float().mean().cpu().item()),
        }
        if getattr(self, "_trainer", None) is not None:
            self.log("train/loss", loss, prog_bar=True)
            self.log("train/traj_loss", torch.as_tensor(parts["traj_loss"], device=loss.device))
            self.log("train/vel_loss", torch.as_tensor(parts["vel_loss"], device=loss.device))
            self.log("train/delta_reg", delta_reg)
        return loss

    def configure_optimizers(self):
        opt_cfg = dict(self.cfg.get("optimizer", {}) or {})
        lr = float(opt_cfg.get("lr", 1e-4))
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))
        parameters = [p for p in self.initializer.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("NoiseInitializer has no trainable parameters")
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


__all__ = ["NoiseInitializerLightningModule"]
