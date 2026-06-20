"""Lightning module for LDF training and validation."""

from __future__ import annotations

import os
import time

import torch
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch_ema import ExponentialMovingAverage

from metrics.t2m import T2MMetrics
from utils.initialize import check_state_dict, instantiate
from utils.training import ckpt_step_info
from utils.training.lightning_module import BasicLightningModule
from utils.training.ldf import (
    SampleCreator,
    SelfForcingTrainer,
    compute_control_loss_xz,
    control_loss_train_mode,
    self_forcing_enabled,
    t2m_metric_enabled,
    validate_self_forcing_runtime_config,
)
from utils.training.ldf.config_validate import (
    validate_7d_requires_self_forcing,
    validate_ldf_training_config,
    validate_traj_dim_consistency,
)
from utils.training.ldf.model_step import run_model_step
from utils.training.ldf.t2m_generation import run_t2m_generation_mode
from utils.training.ldf.t2m_generation_modes import (
    T2M_GENERATE,
    resolve_t2m_generation_modes,
)
from utils.training.ldf.validation_conditioning import (
    build_windowed_metric_ground_truth as _build_windowed_metric_ground_truth,
    prepare_ldf_eval_model_batch,
)
from utils.training.ldf.validation_generation import run_validation_generation_eval
from utils.training.ldf.validation_summary import process_validation_generation_results


class LDFLightningModule(BasicLightningModule):
    """Lightning module wiring diffusion model, VAE, EMA, and eval metrics.

    Routes training between standard (auto opt) and self-forcing (manual opt)
    via ``SelfForcingTrainer``. Handles ControlNet re-init on resume, validation
    generation eval, and T2M metric computation.
    """

    def __init__(self, cfg):
        validate_traj_dim_consistency(cfg)
        validate_7d_requires_self_forcing(cfg)
        validate_ldf_training_config(cfg)
        self._validation_eval_dedup = {}
        self._resume_step_offset = 0
        super().__init__(cfg)
        validate_self_forcing_runtime_config(
            cfg, prediction_type=self.model.prediction_type
        )
        sf_enabled = self_forcing_enabled(cfg)
        self.automatic_optimization = not sf_enabled
        self._sf_trainer = (
            SelfForcingTrainer(self) if sf_enabled else None
        )

        z_stats_dir = (cfg.get("history_corruption", {}) or {}).get("z_stats_dir")
        inner = getattr(getattr(self, "model", None), "model", None)
        if z_stats_dir and hasattr(inner, "load_z_stats"):
            inner.load_z_stats(z_stats_dir)
            rank_zero_info(f"[z_stats] loaded cached-z stats from {z_stats_dir}")

    def build_prefix_sample_creator(
        self,
        *,
        min_prefix_tokens: int | None = None,
    ) -> SampleCreator:
        window_policy = str(
            OmegaConf.select(self.cfg, "ldf_training.window_policy", default="prefix")
        )
        if window_policy == "full":
            return SampleCreator(
                window_policy="full",
                chunk_size=getattr(self.model, "chunk_size", None),
            )
        return SampleCreator(
            window_policy="prefix",
            sample_policy=str(
                OmegaConf.select(
                    self.cfg,
                    "ldf_training.prefix_sample_policy",
                    default="variable_history",
                )
            ),
            min_history_tokens=int(
                OmegaConf.select(self.cfg, "ldf_training.min_history_tokens", default=1)
            ),
            min_prefix_tokens=min_prefix_tokens,
            chunk_size=getattr(self.model, "chunk_size", None),
        )

    def _log_step(
        self, loss_dict, optimizer, net_start_time, extra_metrics=None, lr_value=None
    ):
        net_end_time = time.time()
        data_time = (
            self.batch_ready_time - self.last_batch_end_time
            if self.last_batch_end_time is not None
            else 0.0
        )
        net_time = net_end_time - net_start_time
        batch_size = self.cfg.data.train_bs
        self.log(
            "lr",
            optimizer.param_groups[0]["lr"] if lr_value is None else float(lr_value),
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "data_time", data_time, on_step=True, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "net_time", net_time, on_step=True, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "ckpt_absolute_step",
            float(ckpt_step_info(self).metric_value),
            on_step=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        if extra_metrics:
            for key, value in extra_metrics.items():
                self.log(
                    key,
                    float(value),
                    on_step=True,
                    # Surface the body aux terms (root_xz/root_y/heading/fwd_delta/
                    # yaw_delta) on the tqdm bar too; other extra metrics stay
                    # wandb-only to avoid cluttering the progress bar.
                    prog_bar=key.startswith("body_aux/"),
                    batch_size=batch_size,
                )
        for key, value in loss_dict.items():
            self.log(
                f"train_loss/{key}",
                value.item(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    def _compute_control_loss(self, pred_list, batch):
        if pred_list is None:
            return None
        traj_loss_gt = batch.get("traj_loss_gt", batch.get("traj"))
        if traj_loss_gt is None:
            return None
        traj = traj_loss_gt
        traj_mask = batch.get("traj_loss_mask", batch.get("traj_mask"))
        traj_length = batch["traj_length"]
        train_mode = control_loss_train_mode(self.cfg)
        chunk_size_tokens = getattr(self.model, "chunk_size", None)
        return compute_control_loss_xz(
            pred_list,
            traj,
            traj_mask,
            traj_length,
            self.vae,
            self.device,
            train_mode=train_mode,
            chunk_size_tokens=chunk_size_tokens,
        )

    def initialize_metrics(self):
        ##############################
        # vae (frozen decoder for L_control)
        ##############################
        self.vae = instantiate(
            target=self.cfg.test_vae.target,
            cfg=None,
            hfstyle=False,
            **self.cfg.test_vae.params,
        )
        vae_ckpt = torch.load(
            self.cfg.test_vae_ckpt, map_location="cpu", weights_only=False
        )
        if "ema_state" in vae_ckpt:
            self.vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
            vae_ema = ExponentialMovingAverage(
                self.vae.parameters(), decay=self.cfg.test_vae.ema_decay
            )
            vae_ema.load_state_dict(vae_ckpt["ema_state"])
            vae_ema.copy_to(self.vae.parameters())
            del vae_ema  # EMA weights now in self.vae; no need to keep shadow copy
            rank_zero_info(f"Loaded VAE model from {self.cfg.test_vae_ckpt} with EMA")
        else:
            self.vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
            rank_zero_info(f"Loaded VAE model from {self.cfg.test_vae_ckpt} w/o EMA")

        check_state_dict(
            state_dict=self.vae.state_dict(),
            named_parameters=self.vae.named_parameters(),
            named_buffers=self.vae.named_buffers(),
        )
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        ##############################
        # t2m metrics
        ##############################
        self.recover_dim = self.cfg.metrics.dim
        self.t2m_enabled = t2m_metric_enabled(self.cfg)
        self.t2m_generation_modes = resolve_t2m_generation_modes(self.cfg)
        self.t2m_metrics = (
            torch.nn.ModuleDict({
                mode: T2MMetrics(self.cfg.metrics.t2m)
                for mode in self.t2m_generation_modes
            })
            if self.t2m_enabled
            else None
        )

    def load_checkpoint(self, checkpoint):
        # NOTE: super() not called — state_dict / EMA / optimizer are restored
        # manually so ControlNet re-init and new-param EMA reset work correctly.
        ##############################
        # global_step sync (SF auto→manual)
        ##############################
        self._resume_step_offset = int(checkpoint.get("global_step", 0))
        if getattr(self, "_sf_trainer", None) is not None:
            self._resume_step_offset = self._sf_trainer.on_load_checkpoint(checkpoint)
        else:
            rank_zero_info(
                f"[resume] loaded checkpoint global_step={self._resume_step_offset}"
            )
        ##############################
        # state_dict
        ##############################
        ckpt_keys = set(checkpoint["state_dict"].keys())
        controlnet_missing = not any(k.startswith("controlnet.") for k in ckpt_keys)

        strict = not controlnet_missing
        result = self.model.load_state_dict(checkpoint["state_dict"], strict=strict)
        has_new_cond_params = controlnet_missing and bool(result.missing_keys)
        if not strict and result.missing_keys:
            rank_zero_info(
                "Loaded pretrained LDF with strict=False (base checkpoint without ControlNet). "
                f"Missing keys (new modules init from scratch): {result.missing_keys}"
            )

        if (controlnet_missing
                and any("controlnet." in k for k in result.missing_keys)):
            self.model.controlnet.init_from_backbone(self.model.model)
            rank_zero_info("Re-initialized ControlNet from loaded pretrained backbone weights")
            if result.unexpected_keys:
                rank_zero_info(
                    f"Unexpected keys in checkpoint (ignored): {result.unexpected_keys}"
                )
        ##############################
        # EMA
        ##############################
        if "ema_state" in checkpoint and not has_new_cond_params:
            self.ema.load_state_dict(checkpoint["ema_state"])
            rank_zero_info("init ema from ckpt")
        else:
            self.ema = ExponentialMovingAverage(
                [p for p in self.model.parameters() if p.requires_grad],
                decay=self.cfg.model.ema_decay,
            )
            rank_zero_info("init ema from current model weights")
        ##############################
        # optimizer / scheduler
        ##############################
        # NOTE: Set to empty lists (not pop) so Lightning passes "key exists"
        # check but restores nothing, letting opt/sched follow current config.
        reset_optim_on_resume = bool(self.cfg.get("resume_reset_optimizer", False))

        if has_new_cond_params or reset_optim_on_resume:
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []
            reasons = []
            if has_new_cond_params:
                reasons.append("new cond params")
            if reset_optim_on_resume:
                reasons.append("resume_reset_optimizer")
            rank_zero_info(
                f"Skip restoring optimizer/scheduler ({', '.join(reasons)})"
            )
        check_state_dict(
            state_dict=self.model.state_dict(),
            named_parameters=self.model.named_parameters(),
            named_buffers=self.model.named_buffers(),
        )
        self._skip_next_lightning_load_state_dict = True

        if os.environ.get("FLOODNET_DEBUG", "") == "1":
            ema_n = len(self.ema.shadow_params)
            trainable_n = len([p for p in self.model.parameters() if p.requires_grad])
            total_n = len(list(self.model.parameters()))
            rank_zero_info(
                f"[DEBUG load] ema shadow_params={ema_n} trainable={trainable_n} total={total_n}"
            )
            all_params = list(self.model.parameters())
            ema_shadow_map = {
                id(p): s for p, s in zip(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.ema.shadow_params,
                )
            }
            backbone_with_ema = 0
            backbone_total = 0
            for name, p in self.model.named_parameters():
                if not p.requires_grad and "controlnet" not in name and "traj_encoder" not in name:
                    backbone_total += 1
                    if id(p) in ema_shadow_map:
                        backbone_with_ema += 1
                        s = ema_shadow_map[id(p)]
                        diff = (s - p).abs().mean().item()
                        if backbone_with_ema == 1:
                            rank_zero_info(
                                f"[DEBUG load] backbone param '{name}': raw={p.abs().mean():.6f} "
                                f"ema_shadow={s.abs().mean():.6f} diff={diff:.8f}"
                            )
            rank_zero_info(
                f"[DEBUG load] backbone params with EMA shadow: {backbone_with_ema}/{backbone_total}"
            )

    def on_load_checkpoint(self, checkpoint):
        # Lightning hook; the named method carries the actual resume logic.
        self.load_checkpoint(checkpoint)

    def train(self, mode: bool = True):
        super().train(mode)
        # VAE is a frozen decoder used only for L_control; keep it in eval mode regardless
        # of Lightning's train/eval switches so its parameters never enter train behaviour.
        if hasattr(self, "vae"):
            self.vae.eval()
        return self

    def _step(self, batch, is_training=True, model_batch=None):
        loss_batch = batch
        if model_batch is None:
            model_batch = self.build_prefix_sample_creator().create(batch)
            loss_batch = model_batch
        out = run_model_step(self.model, model_batch)

        ##############################
        # control loss (XZ trajectory alignment via VAE decode)
        ##############################
        if "control_aux" in out and "traj" in loss_batch:
            control_weight = self.cfg.model.params.get("control_loss_weight", 1.0)
            if control_weight > 0:
                control_aux = out["control_aux"]
                # multi-step SF: weighted average over rollout steps
                if "pred_x0_latent_list_steps" in control_aux:
                    step_weights = [
                        float(w)
                        for w in control_aux.get(
                            "step_weights",
                            [1.0] * len(control_aux["pred_x0_latent_list_steps"]),
                        )
                    ]
                    weighted_control = None
                    total_step_weight = 0.0
                    for pred_list, step_weight in zip(
                        control_aux["pred_x0_latent_list_steps"], step_weights
                    ):
                        if pred_list is None:
                            continue
                        step_loss = self._compute_control_loss(
                            pred_list, loss_batch
                        )
                        if step_loss is None:
                            continue
                        if weighted_control is None:
                            weighted_control = step_loss * step_weight
                        else:
                            weighted_control = weighted_control + step_loss * step_weight
                        total_step_weight += step_weight
                    loss_control = (
                        weighted_control / total_step_weight
                        if weighted_control is not None and total_step_weight > 0
                        else None
                    )
                # single-window forward: direct control loss
                else:
                    pred_list = control_aux["pred_x0_latent_list"]
                    loss_control = self._compute_control_loss(
                        pred_list, loss_batch
                    )
                if loss_control is not None:
                    out["total"] = out["total"] + control_weight * loss_control
                    out["control"] = loss_control

        if "control_aux" in out:
            del out["control_aux"]
        return out

    def training_step(self, batch, batch_idx):
        if getattr(self, "_sf_trainer", None) is not None:
            return self._sf_trainer.training_step(batch)
        return super().training_step(batch, batch_idx)

    def update_metrics(self, batch):
        if not self.t2m_enabled or self.t2m_metrics is None:
            return
        # Save/restore CUDA RNG — model.generate() consumes random state via
        # torch.randn() for latent init, which would otherwise shift the noise
        # used in subsequent training steps.
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            with self.ema.average_parameters([p for p in self.model.parameters() if p.requires_grad]):
                model_batch = prepare_ldf_eval_model_batch(batch, self.device, model=self.model)
                outputs = {
                    mode: run_t2m_generation_mode(self.model, model_batch, mode)
                    for mode in self.t2m_generation_modes
                }
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        (
            ground_truth_token,
            gt_token_length,
            ground_truth_feature,
            gt_feature_length,
        ) = _build_windowed_metric_ground_truth(batch, model_batch)

        for mode, output in outputs.items():
            metric = self.t2m_metrics[mode]
            generated = output["generated"]
            for i in range(len(generated)):
                ##############################
                # decode generated motion
                ##############################
                single_generated = generated[i]
                decoded_single_generated = self.vae.decode(
                    single_generated[None, :].to(self.device)
                )[0]
                decoded_single_generated = decoded_single_generated.float().to(self.device)
                ##############################
                # decode ground truth
                ##############################
                single_gt_r = ground_truth_token[i][: int(gt_token_length[i].item())]
                decoded_single_gt_r = self.vae.decode(single_gt_r[None, :].to(self.device))[
                    0
                ]
                decoded_single_gt_r = decoded_single_gt_r.float().to(self.device)
                ##############################
                # original ground truth (VAE vs raw feature for fid_target)
                ##############################
                single_gt_o = ground_truth_feature[i]
                decoded_single_gt_o = single_gt_o[: gt_feature_length[i], :].to(self.device)
                decoded_single_gt_o = decoded_single_gt_o.float().to(self.device)
                text_tokens_single = batch["text_tokens"][i]
                if self.cfg.metrics.t2m.fid_target == "vae":
                    metric.update(
                        feats_rst=decoded_single_generated[None, ...],
                        feats_ref=decoded_single_gt_r[None, ...],
                        lengths_rst=[int(decoded_single_generated.shape[0])],
                        lengths_ref=[int(decoded_single_gt_r.shape[0])],
                        text_tokens=[text_tokens_single],
                    )
                else:
                    metric.update(
                        feats_rst=decoded_single_generated[None, ...],
                        feats_ref=decoded_single_gt_o[None, ...],
                        lengths_rst=[int(decoded_single_generated.shape[0])],
                        lengths_ref=[int(decoded_single_gt_o.shape[0])],
                        text_tokens=[text_tokens_single],
                    )
        return

    def compute_metrics(self):
        if not self.t2m_enabled or self.t2m_metrics is None:
            return
        legacy_single_generate = self.t2m_generation_modes == (T2M_GENERATE,)
        for mode in self.t2m_generation_modes:
            t2m_output = self.t2m_metrics[mode].compute(
                sanity_flag=self.trainer.sanity_checking
            )
            log_prefix = (
                "metrics/t2m_metrics"
                if legacy_single_generate
                else f"metrics/t2m_metrics/{mode}"
            )
            for key, value in t2m_output.items():
                self.log(f"{log_prefix}/{key}", value, sync_dist=True)

    def finish_validation_epoch(self):
        _force = getattr(self, "_eval_on_resume", False)
        if (
            not self.trainer.sanity_checking
            and self.global_step > 0
            and (_force or self.global_step % self.cfg.validation.test_steps == 0)
        ):
            self.on_test_epoch_end()
            self._validation_eval_dedup.clear()
            # Re-save checkpoint: ModelCheckpoint saved earlier (on_validation_end)
            # with stale EMA. Now that validation eval has run, overwrite with the
            # EMA that was actually used.
            _step_val = int(ckpt_step_info(self).metric_value)
            _ckpt_path = os.path.join(
                self.cfg.save_dir, f"step_{_step_val:06.0f}.ckpt"
            )
            rank_zero_info(
                f"[re-save] overwriting {_ckpt_path} with step={_step_val}"
            )
            self.trainer.save_checkpoint(_ckpt_path, weights_only=False)
        self.compute_metrics()

    def on_validation_epoch_end(self):
        # Lightning hook; keep the framework name as a thin adapter.
        self.finish_validation_epoch()

    def update_test(self, batch, batch_idx=None, test_loader_idx=0):
        return run_validation_generation_eval(
            self,
            batch,
            batch_idx=batch_idx,
            test_loader_idx=test_loader_idx,
        )

    def process_test_results(self):
        process_validation_generation_results(self)


__all__ = ["LDFLightningModule"]
