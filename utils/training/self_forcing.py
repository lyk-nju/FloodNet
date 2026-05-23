"""Self-forcing training strategy for streaming-aware diffusion model training.

Encapsulates the K-step scheduled rollout, manual optimization, cross-rank
DDP consensus, and checkpoint auto→manual progress mirroring that were
previously spread across CustomLightningModule and main().
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from lightning.pytorch.utilities import rank_zero_info

from .control_loss import compute_control_loss_xz
from .model_batch import prepare_model_input
from .module_step import compute_step_semantics

if TYPE_CHECKING:
    from train_ldf import CustomLightningModule


@dataclass(frozen=True)
class RolloutPlan:
    effective_k: int
    start_end_indices: torch.Tensor  # (B,) long
    phase_offset: torch.Tensor       # (B,) float32


class SelfForcingTrainer:
    """Orchestrates a self-forcing training step: plans a K-step rollout,
    executes it with no_grad for steps 0..K-2 and a supervised final step,
    then computes loss, backward, and optimizer step with manual optimization.

    Holds a reference to the owning LightningModule for device, logging,
    optimizer, and manual_backward access.
    """

    def __init__(self, module: CustomLightningModule):
        self._module = module
        self._preconditions_checked = False
        self._last_replace_diff: float | None = None
        self._grad_clip_val: float | None = None  # resolved lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _resolve_grad_clip(self) -> float:
        if self._grad_clip_val is not None:
            return self._grad_clip_val
        trainer = getattr(self._module, "trainer", None)
        try:
            clip_val = getattr(trainer, "gradient_clip_val", None)
        except RuntimeError:
            clip_val = None  # trainer not attached yet
        if clip_val is None or float(clip_val) <= 0:
            clip_val = float(self._module.cfg.get("self_forcing_grad_clip", 1.0))
        else:
            clip_val = float(clip_val)
        self._grad_clip_val = clip_val
        return clip_val

    def training_step(self, batch: dict) -> torch.Tensor:
        """Self-forcing K-step rollout training step.

        (T_B_01: the scheduled-sampling single-step branch + Bernoulli gate were
        removed — they were dead code since ``scheduled_sampling_prob`` defaulted
        to 0.0 in every config; replaced by history_corruption.apply_prob, see
        design.md §2.1.4.)
        """
        self._check_preconditions()

        model_batch = prepare_model_input(batch)
        return self._self_forcing_step(batch, model_batch)

    # ------------------------------------------------------------------
    # Self-forcing K-step rollout
    # ------------------------------------------------------------------

    def _self_forcing_step(
        self, batch: dict, model_batch: dict
    ) -> torch.Tensor:
        module = self._module
        net_start_time = time.time()
        semantics, runtime_metrics = self._build_runtime_metrics()
        optimizer = module.optimizers()
        lr_scheduler = module.lr_schedulers()
        lr_for_step = float(optimizer.param_groups[0]["lr"])
        self._log_metrics(runtime_metrics)

        optimizer.zero_grad(set_to_none=True)
        final_step_result, effective_k = self._run_rollout(
            model_batch, semantics.progress
        )
        runtime_metrics["self_forcing/k"] = float(effective_k)
        if self._last_replace_diff is not None:
            runtime_metrics["self_forcing/replace_abs_diff"] = float(
                self._last_replace_diff
            )
            self._last_replace_diff = None
        total_loss, step_diff_loss, step_control_loss = self._compute_losses(
            final_step_result, batch
        )

        module.manual_backward(total_loss)
        trainable = [p for p in module.model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self._resolve_grad_clip())
        runtime_metrics["self_forcing/grad_norm"] = float(grad_norm)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_dict = {"total": total_loss.detach(), "mse": step_diff_loss.detach()}
        if step_control_loss is not None:
            loss_dict["control"] = step_control_loss.detach()
        if lr_scheduler is not None:
            runtime_metrics["lr_next"] = float(optimizer.param_groups[0]["lr"])
        self._module._log_step_metrics(
            loss_dict,
            optimizer,
            net_start_time,
            extra_metrics=runtime_metrics,
            lr_value=lr_for_step,
        )
        return total_loss

    def on_load_checkpoint(self, checkpoint: dict) -> int:
        """Mirror automatic→manual optimizer progress so global_step stays
        consistent across the auto→manual switch on resume.

        Returns the resume_step_offset read from the checkpoint.
        """
        resume_step_offset = int(checkpoint.get("global_step", 0))
        rank_zero_info(
            f"[resume] loaded checkpoint global_step={resume_step_offset}"
        )

        if not self._module.automatic_optimization:
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
                    {
                        "total": {"ready": 0, "completed": 0},
                        "current": {"ready": 0, "completed": 0},
                    },
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
        return resume_step_offset

    def resolve_k(self, progress: float) -> int:
        """Resolve the rollout depth K from the schedule table at the given
        phase progress in [0, 1]."""
        schedule = self._module.model.self_forcing_k_schedule
        k = int(schedule[0][1])
        for threshold, candidate_k in schedule:
            if progress >= threshold:
                k = int(candidate_k)
            else:
                break
        return max(1, k)

    def plan_rollout(
        self, feature_length: torch.Tensor, device: torch.device, progress: float
    ) -> RolloutPlan:
        """Plan the rollout: K depth, per-sample start indices, and phase offsets.

        Includes cross-rank DDP consensus so all ranks execute the same number
        of rollout steps.
        """
        model = self._module.model
        target_k = self.resolve_k(progress)

        # Cross-rank consensus: shortest valid sequence wins.
        min_k_local = int(feature_length.min().item()) - model.chunk_size + 1
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

        max_start = (
            feature_length.to(device=device, dtype=torch.long) - effective_k + 1
        )
        start_end_indices = []
        for b in range(feature_length.shape[0]):
            low = 1
            high = int(max_start[b].item())
            if high < low:
                raise ValueError(
                    f"Invalid self-forcing start range for sample {b}: "
                    f"low={low}, high={high}, "
                    f"valid_len={int(feature_length[b].item())}, "
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

        batch_size = int(feature_length.shape[0])
        phase_offset = torch.empty(
            batch_size, device=device, dtype=torch.float32
        ).uniform_(0.0, 1.0 / model.chunk_size)

        return RolloutPlan(
            effective_k=effective_k,
            start_end_indices=start_end_indices,
            phase_offset=phase_offset,
        )

    def _run_rollout(self, model_batch: dict, progress: float):
        """Execute the K-step self-forcing rollout.

        Steps 0..K-2 run under no_grad and substitute predicted tokens back
        into the input. Step K-1 is the final supervised step that produces
        gradients.
        """
        model = self._module.model
        feature = model_batch["feature"]
        feature_length = model_batch["feature_length"]
        _, seq_len, _ = feature.shape
        device = feature.device

        text_dropped_flags = model._decide_text_dropout(feature.shape[0], device)
        all_text_context = model._prepare_text_context(model_batch, seq_len, device, text_dropped_flags)
        traj_dropped = model._decide_traj_dropout(device)
        plan = self.plan_rollout(feature_length, device, progress)
        traj_emb, traj_seq_lens, _ = model._prepare_traj_condition(
            model_batch, seq_len, device, traj_dropped_override=traj_dropped
        )

        current_feature = feature.clone()
        final_step_result = None
        for step_idx in range(plan.effective_k):
            end_indices = plan.start_end_indices + step_idx
            time_steps = (
                (end_indices.to(dtype=torch.float32) - 1.0) / model.chunk_size
                + plan.phase_offset
            )
            is_final_step = step_idx == plan.effective_k - 1
            if is_final_step:
                final_step_result = model._forward_single_window(
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
                rollout_result = model._forward_single_window(
                    model_batch,
                    current_feature,
                    time_steps,
                    all_text_context,
                    traj_emb,
                    traj_seq_lens,
                    traj_dropped,
                    enable_scheduled_sampling=False,
                )

            disable_replace = bool(
                self._module.cfg.get("self_forcing_disable_replace", False)
            )
            next_feature = current_feature.clone()
            replace_diffs = []
            if not disable_replace:
                for b in range(feature.shape[0]):
                    replace_idx = (
                        int(end_indices[b].item()) - model.chunk_size
                    )
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
                self._last_replace_diff = float(
                    sum(replace_diffs) / len(replace_diffs)
                )

        if final_step_result is None:
            raise RuntimeError(
                f"self-forcing expected at least one supervised step, "
                f"got effective_k={plan.effective_k}"
            )
        return final_step_result, plan.effective_k

    def _compute_losses(self, final_step_result: dict, batch: dict):
        """Compute total loss for the supervised final step, including the
        optional trajectory control loss."""
        step_diff_loss = final_step_result["loss"]
        total_loss = step_diff_loss
        step_control_loss = None
        control_weight = float(
            self._module.cfg.model.params.get("control_loss_weight", 1.0)
        )
        if control_weight > 0.0 and "traj" in batch:
            step_control_loss = _compute_control_loss(
                final_step_result["pred_x0_latent_list"],
                batch,
                self._module,
            )
            if step_control_loss is not None:
                total_loss = total_loss + control_weight * step_control_loss
        return total_loss, step_diff_loss, step_control_loss

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_preconditions(self):
        if self._preconditions_checked:
            return
        trainer = getattr(self._module, "trainer", None)
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
        self._preconditions_checked = True

    def _build_runtime_metrics(self):
        semantics = compute_step_semantics(self._module)
        runtime_metrics = {
            "self_forcing/enabled": 1.0,
            "self_forcing/active": 1.0,
            "self_forcing/progress": float(semantics.progress),
            "self_forcing/k": 0.0,
            "self_forcing/phase_step": float(semantics.phase_step),
            "self_forcing/absolute_step": float(semantics.absolute_step),
            "self_forcing/resume_step_offset": float(semantics.resume_step_offset),
            "self_forcing/phase_total_steps": float(semantics.phase_total_steps),
            "self_forcing/absolute_target_step": float(
                semantics.absolute_target_step
            ),
        }
        return semantics, runtime_metrics

    def _log_metrics(self, runtime_metrics: dict):
        log_every_n_steps = max(
            1,
            int(
                getattr(
                    getattr(self._module, "trainer", None), "log_every_n_steps", 100
                )
            ),
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


# ------------------------------------------------------------------
# Standalone helpers (called from main())
# ------------------------------------------------------------------


def _compute_control_loss(pred_list, batch, module):
    """Thin wrapper that resolves training-mode config then delegates to
    the pure XZ control-loss function."""
    if pred_list is None:
        return None
    traj_loss_gt = batch.get("traj_loss_gt", batch.get("traj"))
    if traj_loss_gt is None:
        return None
    traj = traj_loss_gt
    traj_mask = batch.get("traj_loss_mask", batch.get("traj_mask"))
    traj_length = batch["traj_length"]
    train_mode = module.cfg.get("control_loss_train_mode", 3)
    chunk_size_tokens = getattr(module.model, "chunk_size", None)
    return compute_control_loss_xz(
        pred_list,
        traj,
        traj_mask,
        traj_length,
        module.vae,
        module.device,
        train_mode=train_mode,
        chunk_size_tokens=chunk_size_tokens,
    )


def resolve_sf_runtime(
    absolute_target_step: int,
    resume_ckpt: str | None,
    model_self_forcing_enabled: bool,
    configured_num_training_steps: int,
):
    """Resolve phase_max_steps and runtime_scheduler_steps for self-forcing resume.

    Returns (resume_step_offset, phase_max_steps, runtime_scheduler_steps).
    """
    from .step_semantics import (
        load_resume_step_offset,
        resolve_runtime_max_steps,
        resolve_scheduler_steps,
    )

    resume_step_offset = 0
    phase_max_steps = absolute_target_step
    if resume_ckpt and model_self_forcing_enabled:
        resume_step_offset = load_resume_step_offset(resume_ckpt)
        phase_max_steps = resolve_runtime_max_steps(
            absolute_target_step,
            resume_step_offset,
            self_forcing_enabled=model_self_forcing_enabled,
        )
        rank_zero_info(
            "[self_forcing runtime] "
            f"resume_step_offset={resume_step_offset} "
            f"absolute_target_step={absolute_target_step} "
            f"phase_max_steps={phase_max_steps}"
        )

    runtime_scheduler_steps = resolve_scheduler_steps(
        configured_num_training_steps,
        absolute_target_step=absolute_target_step,
        runtime_max_steps=phase_max_steps,
    )
    return resume_step_offset, phase_max_steps, runtime_scheduler_steps
