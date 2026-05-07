import os
import time

import torch
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

from metrics.t2m import T2MMetrics
from eval.inline_eval_runner import run_inline_generation_eval
from eval.inline_eval_summary import process_inline_generation_results
from utils.initialize import (
    compare_statedict_and_parameters,
    get_function,
    get_shared_run_time,
    instantiate,
    load_config,
    save_config_and_codes,
)
from utils.lightning_module import BasicLightningModule
from utils.training import (
    build_checkpoint_step_info,
    build_step_semantics,
    compute_control_loss_xz,
    load_resume_step_offset,
    resolve_runtime_scheduler_steps,
    resolve_runtime_max_steps,
)

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomLightningModule(BasicLightningModule):
    def __init__(self, cfg):
        self._inline_eval_seen = {}
        self._self_forcing_runtime_validated = False
        self._resume_step_offset = 0
        super().__init__(cfg)
        # Must set AFTER super().__init__() because LightningModule.__init__
        # forces automatic_optimization=True; setting it before is silently
        # overwritten and crashes self.manual_backward at runtime.
        self_forcing_enabled = bool(
            cfg.model.params.get("self_forcing_enabled", False)
        )
        self.automatic_optimization = not self_forcing_enabled

    def _build_step_semantics(self, phase_step: int | None = None):
        trainer = getattr(self, "trainer", None)
        trainer_max_steps = getattr(trainer, "max_steps", None) if trainer is not None else None
        # `trainer.max_steps` is absolute (Lightning compares it against the
        # absolute `global_step`).  For phase-relative semantics we need the
        # phase length, so subtract the resume offset here.
        sf_enabled = bool(getattr(self.model, "self_forcing_enabled", False))
        if (
            sf_enabled
            and trainer_max_steps is not None
            and int(trainer_max_steps) > 0
        ):
            phase_total = int(trainer_max_steps) - int(self._resume_step_offset)
            trainer_max_steps = max(1, phase_total)
        return build_step_semantics(
            phase_step=self._get_phase_global_step() if phase_step is None else int(phase_step),
            trainer_max_steps=trainer_max_steps,
            resume_step_offset=self._resume_step_offset,
            self_forcing_enabled=sf_enabled,
        )

    def _get_phase_global_step(self) -> int:
        # Lightning restores `global_step` to the ckpt's absolute value on
        # resume (e.g. 240000).  For self-forcing schedule progress we need
        # the *phase-relative* step (steps trained since this phase started),
        # so subtract the resume offset.
        return max(0, int(self.global_step) - int(self._resume_step_offset))

    def _get_effective_global_step(self):
        return self._build_step_semantics().absolute_step

    def _get_checkpoint_step_value(self):
        return build_checkpoint_step_info(
            self._build_step_semantics(),
            include_next_step=False,
        ).metric_value

    def _get_step_tag(self):
        return build_checkpoint_step_info(
            self._build_step_semantics(),
            include_next_step=False,
        ).step_tag

    def _get_test_probe_tags(self) -> list[str]:
        tags = getattr(self, "test_loader_tags", None)
        if tags:
            return list(tags)
        return ["test"]

    def _resolve_test_probe_tag(self, test_loader_idx: int) -> str:
        tags = self._get_test_probe_tags()
        if 0 <= test_loader_idx < len(tags):
            return tags[test_loader_idx]
        return f"test_loader_{test_loader_idx}"

    def _get_generation_eval_cfg(self):
        val_cfg = self.cfg.get("validation", {})
        return {
            "enabled": bool(val_cfg.get("eval_generation_metrics", True)),
            "num_runs": int(val_cfg.get("eval_num_runs", 10)),
            "seg_size": int(val_cfg.get("eval_seg_size", 20)),
            "forward_ctrl_loss": bool(val_cfg.get("eval_forward_control_loss", True)),
            "forward_ctrl_window_mode": str(
                val_cfg.get("eval_forward_control_loss_window_mode", "mean_chunk_windows")
            ),
        }

    def _build_self_forcing_runtime(self):
        semantics = self._build_step_semantics()
        runtime_metrics = {
            "self_forcing/enabled": 1.0,
            "self_forcing/active": 1.0,
            "self_forcing/progress": float(semantics.progress),
            "self_forcing/k": 0.0,
            "self_forcing/phase_step": float(semantics.phase_step),
            "self_forcing/absolute_step": float(semantics.absolute_step),
            "self_forcing/resume_step_offset": float(semantics.resume_step_offset),
            "self_forcing/phase_total_steps": float(semantics.phase_total_steps),
            "self_forcing/absolute_target_step": float(semantics.absolute_target_step),
        }
        return semantics, runtime_metrics

    def _build_model_batch(self, batch, is_training=True):
        model_batch = batch.copy()
        model_batch["feature"] = batch["token"]
        model_batch["feature_length"] = batch["token_length"]
        if "token_text_end" in batch:
            model_batch["feature_text_end"] = batch["token_text_end"]
        self._copy_traj_fields_to_model_batch(batch, model_batch)
        return model_batch

    def _validate_self_forcing_runtime(self):
        if self._self_forcing_runtime_validated:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        accumulate_grad_batches = int(
            getattr(trainer, "accumulate_grad_batches", 1)
        )
        if accumulate_grad_batches != 1:
            raise NotImplementedError(
                "self-forcing manual optimization does not yet support "
                f"accumulate_grad_batches={accumulate_grad_batches}. Set it to 1."
            )
        self._self_forcing_runtime_validated = True

    def _log_training_step_outputs(
        self, loss_dict, optimizer, net_start_time, extra_metrics=None
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
            optimizer.param_groups[0]["lr"],
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
            float(self._get_checkpoint_step_value()),
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
                    prog_bar=False,
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

    def _compute_control_loss_for_pred_list(self, pred_list, batch):
        if pred_list is None or "traj" not in batch:
            return None
        traj = batch["traj"]
        traj_mask = batch["traj_mask"]
        traj_length = batch["traj_length"]
        train_mode = self.cfg.get("control_loss_train_mode", 3)
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
        # vae
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

        compare_statedict_and_parameters(
            state_dict=self.vae.state_dict(),
            named_parameters=self.vae.named_parameters(),
            named_buffers=self.vae.named_buffers(),
        )
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        # metric models
        self.recover_dim = self.cfg.metrics.dim
        self.t2m_enabled = bool(self.cfg.get("t2m_metric", False))
        self.t2m_metrics = T2MMetrics(self.cfg.metrics.t2m) if self.t2m_enabled else None

    def on_load_checkpoint(self, checkpoint):
        # super() not called: we handle state_dict / EMA / optimizer restore manually
        # so that ControlNet re-init and new-param EMA reset work correctly.
        self._resume_step_offset = int(checkpoint.get("global_step", 0))
        rank_zero_info(
            f"[resume] loaded checkpoint global_step={self._resume_step_offset}"
        )
        # When the ckpt was saved under automatic_optimization=True but we now
        # resume under manual_optimization=False (self-forcing), Lightning
        # reads `global_step` from the *manual* counter, which is missing in
        # the ckpt (defaults to 0).  Result: self.global_step = 0 forever
        # until it climbs past 240000, so phase_step / ckpt naming get stuck.
        # Mirror the auto-counter's progress into the manual counter to keep
        # `self.global_step` consistent across the auto→manual switch.
        if not self.automatic_optimization:
            try:
                fit_loop = checkpoint["loops"]["fit_loop"]
                auto_progress = (
                    fit_loop["epoch_loop.automatic_optimization.optim_progress"]
                    ["optimizer"]["step"]
                )
                completed = int(auto_progress["total"]["completed"])
                ready = int(auto_progress["total"]["ready"])
                manual_key = "epoch_loop.manual_optimization.optim_step_progress"
                manual_progress = fit_loop.setdefault(
                    manual_key,
                    {"total": {"ready": 0, "completed": 0},
                     "current": {"ready": 0, "completed": 0}},
                )
                if int(manual_progress["total"]["completed"]) < completed:
                    manual_progress["total"]["ready"] = ready
                    manual_progress["total"]["completed"] = completed
                    rank_zero_info(
                        f"[resume] mirrored auto→manual optim_step_progress "
                        f"completed={completed} (was 0); keeps self.global_step "
                        f"aligned with ckpt"
                    )
            except (KeyError, TypeError) as exc:
                rank_zero_info(
                    f"[resume] could not mirror auto→manual progress ({exc!r}); "
                    f"global_step may start from 0"
                )
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
        # Re-init ControlNet from the *loaded* backbone when starting from a base ckpt.
        # init_from_backbone in __init__ runs before weights are loaded, so it copies random
        # weights — moving the call here ensures it copies the actual pretrained backbone.
        if (controlnet_missing
                and any("controlnet." in k for k in result.missing_keys)):
            self.model.controlnet.init_from_backbone(self.model.model)
            rank_zero_info("Re-initialized ControlNet from loaded pretrained backbone weights")
            if result.unexpected_keys:
                rank_zero_info(
                    f"Unexpected keys in checkpoint (ignored): {result.unexpected_keys}"
                )
        # When loading pretrained with new traj params, ema_state has wrong param count -> reinit EMA
        if "ema_state" in checkpoint and not has_new_cond_params:
            self.ema.load_state_dict(checkpoint["ema_state"])
            rank_zero_info("init ema from ckpt")
        else:
            self.ema = ExponentialMovingAverage(
                [p for p in self.model.parameters() if p.requires_grad],
                decay=self.cfg.model.ema_decay,
            )
            rank_zero_info("init ema from current model weights")
        # When has_new_cond_params, optimizer/scheduler param groups mismatch -> skip restore.
        # When resume_reset_optimizer=True, skip restore so LR/schedule follow current yaml (not ckpt).
        # Set to empty lists (don't pop) so Lightning passes "key exists" check but restores nothing.
        reset_optim_on_resume = bool(self.cfg.get("resume_reset_optimizer", False))
        if has_new_cond_params or reset_optim_on_resume:
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []
            if has_new_cond_params and reset_optim_on_resume:
                rank_zero_info(
                    "Skip restoring optimizer/scheduler (new cond params + resume_reset_optimizer)"
                )
            elif has_new_cond_params:
                rank_zero_info("Skip restoring optimizer/scheduler (new cond params)")
            else:
                rank_zero_info("Skip restoring optimizer/scheduler (resume_reset_optimizer)")
        compare_statedict_and_parameters(
            state_dict=self.model.state_dict(),
            named_parameters=self.model.named_parameters(),
            named_buffers=self.model.named_buffers(),
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)

    def train(self, mode: bool = True):
        super().train(mode)
        # VAE is a frozen decoder used only for L_control; keep it in eval mode regardless
        # of Lightning's train/eval switches so its parameters never enter train behaviour.
        if hasattr(self, "vae"):
            self.vae.eval()
        return self

    @staticmethod
    def _copy_traj_fields_to_model_batch(batch, model_batch):
        if "traj" in batch:
            model_batch["traj"] = batch["traj"]
            model_batch["traj_length"] = batch["traj_length"]
            model_batch["traj_mask"] = batch["traj_mask"]
        if "token_mask" in batch:
            model_batch["token_mask"] = batch["token_mask"]
        if "traj_features" in batch:
            model_batch["traj_features"] = batch["traj_features"]
            # traj_features is frame-level; length is traj_length and mask is traj_mask/token_mask

    def _step(self, batch, is_training=True, model_batch=None):
        if model_batch is None:
            model_batch = self._build_model_batch(batch, is_training=is_training)
        out = self.model(model_batch)

        # MotionLCM-style control loss: explicit trajectory alignment in motion space
        if "control_aux" in out and "traj" in batch:
            control_weight = self.cfg.model.params.get("control_loss_weight", 1.0)
            if control_weight > 0:
                control_aux = out["control_aux"]
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
                        step_loss = self._compute_control_loss_for_pred_list(
                            pred_list, batch
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
                else:
                    pred_list = control_aux["pred_x0_latent_list"]
                    loss_control = self._compute_control_loss_for_pred_list(
                        pred_list, batch
                    )
                if loss_control is not None:
                    out["total"] = out["total"] + control_weight * loss_control
                    out["control"] = loss_control

        if "control_aux" in out:
            del out["control_aux"]
        return out

    def _run_standard_training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def _log_self_forcing_metrics(self, runtime_metrics):
        log_every_n_steps = max(
            1, int(getattr(getattr(self, "trainer", None), "log_every_n_steps", 100))
        )
        if int(runtime_metrics["self_forcing/phase_step"]) % log_every_n_steps == 0:
            rank_zero_info(
                "[self_forcing] "
                f"phase_step={int(runtime_metrics['self_forcing/phase_step'])} "
                f"absolute_step={int(runtime_metrics['self_forcing/absolute_step'])} "
                f"resume_step_offset={int(runtime_metrics['self_forcing/resume_step_offset'])} "
                f"phase_total_steps={int(runtime_metrics['self_forcing/phase_total_steps'])} "
                f"absolute_target_step={int(runtime_metrics['self_forcing/absolute_target_step'])} "
                f"active={int(runtime_metrics['self_forcing/active'])} "
                f"progress={runtime_metrics['self_forcing/progress']:.6f}"
            )

    def _resolve_self_forcing_k(self, progress):
        k = int(self.model.self_forcing_k_schedule[0][1])
        for threshold, candidate_k in self.model.self_forcing_k_schedule:
            if progress >= threshold:
                k = int(candidate_k)
            else:
                break
        return max(1, k)

    def _prepare_self_forcing_plan(self, feature_length, device, progress):
        target_k = self._resolve_self_forcing_k(progress)
        # Cross-rank consensus on effective_k so DDP ranks run the same number
        # of rollout steps (shortest valid sequence wins).
        min_k_local = int(feature_length.min().item()) - self.model.chunk_size + 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tmp = torch.tensor([min_k_local], device=device, dtype=torch.long)
            torch.distributed.all_reduce(tmp, op=torch.distributed.ReduceOp.MIN)
            min_k_supported = int(tmp.item())
        else:
            min_k_supported = min_k_local
        if min_k_supported < 1:
            raise ValueError(
                "self-forcing requires feature_length >= chunk_size for every sample"
            )
        effective_k = min(target_k, min_k_supported)

        max_start = feature_length.to(device=device, dtype=torch.long) - effective_k + 1
        # Mix self-forcing with standard supervision so the t < 1 regime
        # (very short prefixes) keeps getting gradient updates.  Otherwise
        # ControlNet drifts on the early-time regime that inference still
        # exercises.  ``low = 1`` matches the standard `forward` end_index
        # range; samples with start < chunk_size simply fall back to a
        # standard supervised step (rollout has no effect when replace_idx
        # would land before position 0).
        start_end_indices = []
        for b in range(feature_length.shape[0]):
            low = 1
            high = int(max_start[b].item())
            if high < low:
                raise ValueError(
                    f"Invalid self-forcing start range for sample {b}: "
                    f"low={low}, high={high}, valid_len={int(feature_length[b].item())}, "
                    f"effective_k={effective_k}"
                )
            if high == low:
                start_end_indices.append(low)
            else:
                start_end_indices.append(
                    int(torch.randint(low, high + 1, (1,), device=device).item())
                )
        start_end_indices = torch.tensor(
            start_end_indices, device=device, dtype=torch.long
        )
        # Per-sample fractional time offset (matches standard `forward`).
        # Sharing a single scalar across the batch correlates noise schedules
        # for every sample and collapses effective batch diversity.
        batch_size = int(feature_length.shape[0])
        phase_offset = torch.empty(
            batch_size, device=device, dtype=torch.float32
        ).uniform_(0.0, 1.0 / self.model.chunk_size)
        return effective_k, start_end_indices, phase_offset

    def _run_self_forcing_rollout(self, model_batch, progress):
        feature = model_batch["feature"]
        feature_length = model_batch["feature_length"]
        _, seq_len, _ = feature.shape
        device = feature.device
        all_text_context = self.model._prepare_text_context(model_batch, seq_len, device)
        traj_dropped = self.model._sample_traj_dropout_decision(device)
        effective_k, start_end_indices, phase_offset = self._prepare_self_forcing_plan(
            feature_length, device, progress
        )
        traj_emb, traj_seq_lens, _ = self.model._prepare_traj_condition(
            model_batch, seq_len, device, traj_dropped_override=traj_dropped
        )

        current_feature = feature.clone()
        final_step_result = None
        for step_idx in range(effective_k):
            end_indices = start_end_indices + step_idx
            time_steps = (
                (end_indices.to(dtype=torch.float32) - 1.0) / self.model.chunk_size
                + phase_offset
            )
            is_final_step = step_idx == effective_k - 1
            if is_final_step:
                final_step_result = self.model._run_single_window_forward(
                    model_batch,
                    current_feature,
                    time_steps,
                    all_text_context,
                    traj_emb,
                    traj_seq_lens,
                    traj_dropped,
                    enable_scheduled_sampling=False,
                )
                break

            with torch.no_grad():
                rollout_result = self.model._run_single_window_forward(
                    model_batch,
                    current_feature,
                    time_steps,
                    all_text_context,
                    traj_emb,
                    traj_seq_lens,
                    traj_dropped,
                    enable_scheduled_sampling=False,
                )

            # Diagnostic: when self_forcing_disable_replace is set, we still
            # run the rollout forward (so any side-effect of the extra forward
            # pass remains), but skip the substitution entirely.  K=2 with this
            # flag should behave identically to K=1.  If FID still degrades
            # with this flag on, the bug is in the dual-forward path itself
            # (e.g. RNG / autograd state).  If FID is restored, the bug is in
            # the substituted value.
            disable_replace = bool(
                self.cfg.get("self_forcing_disable_replace", False)
            )
            next_feature = current_feature.clone()
            replace_diffs = []
            if not disable_replace:
                for b in range(feature.shape[0]):
                    # When start < chunk_size (early-time short-prefix samples),
                    # replace_idx < 0 means there is no past token to roll out;
                    # this sample degenerates to a plain supervised step.
                    replace_idx = int(end_indices[b].item()) - self.model.chunk_size
                    if replace_idx < 0:
                        continue
                    pred_seq = rollout_result["x0_latent_list"][b]
                    if replace_idx >= pred_seq.shape[0]:
                        continue
                    replacement = pred_seq[replace_idx].detach().to(
                        device=current_feature.device, dtype=current_feature.dtype
                    )
                    gt_token = current_feature[b, replace_idx, :]
                    replace_diffs.append(
                        (replacement - gt_token).abs().mean().item()
                    )
                    next_feature[b, replace_idx, :] = replacement
            current_feature = next_feature
            if replace_diffs:
                self._self_forcing_last_replace_diff = float(
                    sum(replace_diffs) / len(replace_diffs)
                )

        if final_step_result is None:
            raise RuntimeError(
                f"self-forcing expected at least one supervised step, got effective_k={effective_k}"
            )
        return final_step_result, effective_k

    def _finalize_self_forcing_loss(self, final_step_result, batch):
        step_diff_loss = final_step_result["loss"]
        total_loss = step_diff_loss
        step_control_loss = None
        control_weight = float(self.cfg.model.params.get("control_loss_weight", 1.0))
        if control_weight > 0.0 and "traj" in batch:
            step_control_loss = self._compute_control_loss_for_pred_list(
                final_step_result["pred_x0_latent_list"], batch
            )
            if step_control_loss is not None:
                total_loss = total_loss + control_weight * step_control_loss
        return total_loss, step_diff_loss, step_control_loss

    def _run_self_forcing_training_step(self, batch):
        self._validate_self_forcing_runtime()
        net_start_time = time.time()
        semantics, runtime_metrics = self._build_self_forcing_runtime()
        model_batch = self._build_model_batch(batch, is_training=True)
        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()
        self._log_self_forcing_metrics(runtime_metrics)

        optimizer.zero_grad(set_to_none=True)
        final_step_result, effective_k = self._run_self_forcing_rollout(
            model_batch, semantics.progress
        )
        runtime_metrics["self_forcing/k"] = float(effective_k)
        replace_diff = getattr(self, "_self_forcing_last_replace_diff", None)
        if replace_diff is not None:
            runtime_metrics["self_forcing/replace_abs_diff"] = float(replace_diff)
            self._self_forcing_last_replace_diff = None
        total_loss, step_diff_loss, step_control_loss = self._finalize_self_forcing_loss(
            final_step_result, batch
        )

        self.manual_backward(total_loss)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        # Honour trainer.gradient_clip_val if user sets it; otherwise fall back to
        # a conservative default (manual_optimization disables Lightning's auto
        # clipping, so without this self-forcing has zero clipping).
        clip_val = getattr(getattr(self, "trainer", None), "gradient_clip_val", None)
        if clip_val is None or float(clip_val) <= 0:
            clip_val = float(self.cfg.get("self_forcing_grad_clip", 1.0))
        else:
            clip_val = float(clip_val)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, clip_val)
        runtime_metrics["self_forcing/grad_norm"] = float(grad_norm)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_dict = {"total": total_loss.detach(), "mse": step_diff_loss.detach()}
        if step_control_loss is not None:
            loss_dict["control"] = step_control_loss.detach()
        self._log_training_step_outputs(
            loss_dict,
            optimizer,
            net_start_time,
            extra_metrics=runtime_metrics,
        )
        return total_loss

    def training_step(self, batch, batch_idx):
        if not getattr(self.model, "self_forcing_enabled", False):
            return self._run_standard_training_step(batch, batch_idx)
        return self._run_self_forcing_training_step(batch)

    def update_metrics(self, batch):
        if not self.t2m_enabled or self.t2m_metrics is None:
            return
        with self.ema.average_parameters([p for p in self.model.parameters() if p.requires_grad]):
            model_batch = batch.copy()
            model_batch["feature"] = batch["token"]
            model_batch["feature_length"] = batch["token_length"]
            if "token_text_end" in batch:
                model_batch["feature_text_end"] = batch["token_text_end"]
            self._copy_traj_fields_to_model_batch(batch, model_batch)
            output = self.model.generate(model_batch)
        generated = output["generated"]
        ground_truth_token = batch["token"]
        gt_token_length = batch["token_length"]
        ground_truth_feature = batch["feature"]
        gt_feature_length = batch["feature_length"]

        for i in range(len(generated)):
            # Decode motion
            single_generated = generated[i]
            decoded_single_generated = self.vae.decode(
                single_generated[None, :].to(self.device)
            )[0]
            decoded_single_generated = decoded_single_generated.float().to(self.device)
            # Decode ground truth
            single_gt_r = ground_truth_token[i][: gt_token_length[i]]
            decoded_single_gt_r = self.vae.decode(single_gt_r[None, :].to(self.device))[
                0
            ]
            decoded_single_gt_r = decoded_single_gt_r.float().to(self.device)
            # Original ground truth
            single_gt_o = ground_truth_feature[i]
            decoded_single_gt_o = single_gt_o[: gt_feature_length[i], :].to(self.device)
            decoded_single_gt_o = decoded_single_gt_o.float().to(self.device)
            text_tokens_single = batch["text_tokens"][i]
            if self.cfg.metrics.t2m.fid_target == "vae":
                self.t2m_metrics.update(
                    feats_rst=decoded_single_generated[None, ...],
                    feats_ref=decoded_single_gt_r[None, ...],
                    lengths_rst=[int(decoded_single_generated.shape[0])],
                    lengths_ref=[int(decoded_single_gt_r.shape[0])],
                    text_tokens=[text_tokens_single],
                )
            else:
                self.t2m_metrics.update(
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
        t2m_output = self.t2m_metrics.compute(sanity_flag=self.trainer.sanity_checking)
        for key, value in t2m_output.items():
            self.log(f"metrics/t2m_metrics/{key}", value, sync_dist=True)

    def on_validation_epoch_end(self):
        if (
            not self.trainer.sanity_checking
            and self.global_step > 0
            and self.global_step % self.cfg.validation.test_steps == 0
        ):
            self.on_test_epoch_end()
            self._inline_eval_seen.clear()
        self.compute_metrics()

    def update_test(self, batch, batch_idx=None, test_loader_idx=0):
        return run_inline_generation_eval(
            self,
            batch,
            batch_idx=batch_idx,
            test_loader_idx=test_loader_idx,
        )

    def process_test_results(self):
        process_inline_generation_results(self)


def _build_test_probe_loaders(cfg, collate_fn):
    probe_cfg = cfg.data.get("test_probe_meta_paths", None)
    probe_specs = []
    if probe_cfg:
        for probe_tag, meta_paths in probe_cfg.items():
            probe_specs.append((str(probe_tag), list(meta_paths)))
    else:
        test_meta_paths = cfg.data.get("test_meta_paths", None)
        if test_meta_paths is not None:
            probe_specs.append(("test", list(test_meta_paths)))
        else:
            # No meta paths (e.g. GenerateDataset): use dataset directly with split="test"
            probe_specs.append(("test", None))

    loaders, tags = [], []
    total_probe_samples = 0
    test_target = cfg.data.get("test_target", cfg.data.target)
    for probe_tag, meta_paths in probe_specs:
        probe_cfg_obj = OmegaConf.create(OmegaConf.to_container(cfg.config, resolve=False))
        if meta_paths is not None:
            OmegaConf.update(probe_cfg_obj, "data.test_meta_paths", meta_paths)
        probe_dataset = instantiate(test_target, cfg=probe_cfg_obj, split="test")
        probe_loader = DataLoader(
            probe_dataset,
            batch_size=cfg.data.test_bs,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.data.num_workers,
            persistent_workers=False,
            prefetch_factor=8,
            collate_fn=collate_fn,
        )
        loaders.append(probe_loader)
        tags.append(probe_tag)
        total_probe_samples += len(probe_dataset)
        rank_zero_info(f"Test probe[{probe_tag}]: {len(probe_dataset)} samples")
    return loaders, tags, total_probe_samples


def main():
    # init
    torch.set_float32_matmul_precision("high")
    cfg = load_config()
    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    run_time = get_shared_run_time(cfg.save_dir)
    save_dir = os.path.join(cfg.save_dir, f"{run_time}_{cfg.exp_name}")
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.update(cfg.config, "save_dir", save_dir)
    rank_zero_info(
        f"Save dir: {save_dir}, current working dir: {os.getcwd()}, exp_name: {cfg.exp_name}"
    )
    save_config_and_codes(cfg, cfg.save_dir)

    logger = None
    if not cfg.debug:
        wandb_key = cfg.logger.wandb.wandb_key
        if wandb_key and wandb_key.strip():
            os.environ["WANDB_API_KEY"] = wandb_key
            logger = WandbLogger(
                project=cfg.logger.wandb.project,
                name=f"{cfg.exp_name}_{run_time}",
                entity=cfg.logger.wandb.entity,
                config=OmegaConf.to_container(cfg.config, resolve=True),
                save_dir=cfg.save_dir,
            )
            rank_zero_info("WandB logging enabled")
        else:
            rank_zero_info("WandB API key not provided, skipping WandB logging")

    # dataloader
    collate_fn = (
        get_function(cfg.data.collate_fn) if cfg.data.get("collate_fn", None) else None
    )

    train_dataset = (
        instantiate(cfg.data.target, cfg=cfg.config, split="train")
        if cfg.train
        else None
    )
    val_dataset = instantiate(
        cfg.data.get("val_target", cfg.data.target), cfg=cfg.config, split="val"
    )

    train_dataloader = (
        DataLoader(
            train_dataset,
            batch_size=cfg.data.train_bs,
            shuffle=True,
            drop_last=False,
            num_workers=cfg.data.num_workers,
            persistent_workers=True,
            prefetch_factor=8,
            collate_fn=collate_fn,
        )
        if cfg.train
        else None
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.data.val_bs,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=False,
        prefetch_factor=8,
        collate_fn=collate_fn,
    )

    test_probe_loaders, test_loader_tags, total_probe_samples = _build_test_probe_loaders(
        cfg, collate_fn
    )

    rank_zero_info(
        f"Train dataset: {len(train_dataset) if train_dataset is not None else 0}, "
        f"Val dataset: {len(val_dataset) if val_dataset is not None else 0}, "
        f"Test probe samples: {total_probe_samples}"
    )
    
    trainer_absolute_max_steps = int(cfg.trainer.max_steps)
    # Lightning restores `global_step` to the ckpt's absolute value on resume
    # and compares it against `trainer.max_steps` (also absolute).  We must
    # therefore keep the trainer's max_steps at the absolute target.  The
    # *phase length* (= absolute - offset) is only used to scale the LR
    # scheduler horizon and self-forcing schedule progress.
    model_self_forcing_enabled = bool(
        cfg.config.model.params.get("self_forcing_enabled", False)
    )
    resume_step_offset = 0
    phase_max_steps = trainer_absolute_max_steps
    if cfg.train and model_self_forcing_enabled and cfg.resume_ckpt:
        resume_step_offset = load_resume_step_offset(cfg.resume_ckpt)
        phase_max_steps = resolve_runtime_max_steps(
            trainer_absolute_max_steps,
            resume_step_offset,
            self_forcing_enabled=model_self_forcing_enabled,
        )
        rank_zero_info(
            "[self_forcing runtime] "
            f"resume_step_offset={resume_step_offset} "
            f"absolute_target_step={trainer_absolute_max_steps} "
            f"phase_max_steps={phase_max_steps}"
        )
    scheduler_training_steps = int(
        OmegaConf.to_container(
            cfg.config.lr_scheduler.params,
            resolve=True,
        )["num_training_steps"]
    )
    runtime_scheduler_steps = resolve_runtime_scheduler_steps(
        scheduler_training_steps,
        absolute_target_step=trainer_absolute_max_steps,
        runtime_max_steps=phase_max_steps,
    )
    if runtime_scheduler_steps != scheduler_training_steps:
        OmegaConf.update(
            cfg.config,
            "lr_scheduler.params.num_training_steps",
            int(runtime_scheduler_steps),
        )
        rank_zero_info(
            "[self_forcing runtime] "
            f"lr_scheduler.num_training_steps={runtime_scheduler_steps} "
            f"(was {scheduler_training_steps})"
        )

    # lightning module, model is inside the lightning module
    model = CustomLightningModule(cfg=cfg.config)
    model.test_loader_tags = test_loader_tags
    model._resume_step_offset = int(resume_step_offset)

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.save_dir,
        filename="step_{ckpt_absolute_step:06.0f}",
        every_n_train_steps=cfg.validation.save_every_n_steps,
        save_top_k=cfg.validation.save_top_k,
        monitor="ckpt_absolute_step",
        mode="max",
        auto_insert_metric_name=False,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    if cfg.train:
        callbacks.append(checkpoint_callback)

    # Handle devices as either int or list
    num_devices = (
        cfg.trainer.devices
        if isinstance(cfg.trainer.devices, int)
        else len(cfg.trainer.devices)
    )
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    # Keep absolute max_steps for Lightning (it compares global_step which is
    # restored to ckpt's absolute value on resume).
    trainer_kwargs["max_steps"] = trainer_absolute_max_steps

    trainer = Trainer(
        **trainer_kwargs,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True)
        if num_devices > 1
        else "auto",
        callbacks=callbacks,
        default_root_dir=cfg.save_dir,
        val_check_interval=cfg.validation.validation_steps,
        check_val_every_n_epoch=None,
    )

    rank_zero_info("Validation test mode: inline")
    val_dataloaders = [val_dataloader] + test_probe_loaders

    if cfg.train:
        if not cfg.debug:
            pass
            # trainer.validate(
            #     model,
            #     dataloaders=val_dataloaders,
            #     ckpt_path=cfg.resume_ckpt if cfg.resume_ckpt else None,
            #     weights_only=False,
            # )
        trainer.fit(
            model,
            train_dataloader,
            val_dataloaders=val_dataloaders,
            ckpt_path=cfg.resume_ckpt,
            weights_only=False,
        )
    else:
        for i in range(cfg.config.val_repeat):
            # Set different seed for each validation run to get diverse results
            # But keep it deterministic: same i -> same seed -> same result
            seed_everything(cfg.seed + i)
            trainer.validate(
                model,
                dataloaders=val_dataloaders,
                ckpt_path=cfg.test_ckpt,
                weights_only=False,
            )
            model.cfg.test_setting.render = False  # only render once

    if not cfg.debug and logger is not None:
        wandb.finish()


if __name__ == "__main__":
    # train
    # train.py --config configs/ldf.yaml
    main()
