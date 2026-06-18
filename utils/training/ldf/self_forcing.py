"""Self-forcing training strategy for streaming-aware diffusion model training.

Encapsulates the K-step scheduled rollout, manual optimization, cross-rank
DDP consensus, and checkpoint auto-to-manual progress mirroring that were
previously spread across CustomLightningModule and main().
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from lightning.pytorch.utilities import rank_zero_info

from utils.training.ldf.body_canonicalize import apply_body_window_canonicalize
from utils.training.ldf.conditioning import (
    PreparedCondition,
    prepare_text_condition,
    prepare_traj_condition,
    sample_traj_dropout,
)
from utils.training.ldf.history_corruption import (
    apply_history_corruption,
    should_apply_corruption,
)
from utils.training.ldf.horizon_sched import sample_random_horizon_tokens
from utils.training.ldf.losses import compute_body_aux_loss, compute_control_loss_xz
from utils.training.ldf.model_step import run_training_window
from utils.training.ldf.sample_creator import SampleCreator
from utils.training.ldf.self_forcing_config import (
    self_forcing_k_schedule,
    self_forcing_stride_tokens,
)
from utils.training.ldf.validation_eval_runtime import control_loss_train_mode
from utils.training.module_step import compute_step_semantics

if TYPE_CHECKING:
    from train_ldf import CustomLightningModule


@dataclass(frozen=True)
class RolloutPlan:
    effective_k: int
    start_end_indices: torch.Tensor  # (B,) long
    phase_offset: torch.Tensor       # (B,) float32


_ROLLING_LOSS_BATCH_KEYS = (
    "_window_global_start_token",
    "_window_local_latent_start_token",
    "_window_local_latent_valid_len",
    "_window_local_traj",
    "_window_sampling_active_left_token",
    "_window_sampling_history_tokens",
    "_window_sampling_horizon_tokens",
    "_window_sampling_rollout_span",
)

_PREFIX_LOSS_BATCH_KEYS = (
    "_window_global_start_token",
    "_window_local_latent_start_token",
    "_window_local_latent_valid_len",
    "_window_local_traj",
    "traj_cond_7d",
    "traj",
    "traj_mask",
    "traj_cond_mask",
    "traj_loss_mask",
    "traj_length",
    "traj_num_tokens",
    "traj_features_length",
)


def shifted_local_time_steps(
    local_end_indices: torch.Tensor,
    *,
    start_tokens: torch.Tensor | int | None = None,
    chunk_size: int,
    phase_offset: torch.Tensor,
) -> torch.Tensor:
    """Return window-local diffusion times equivalent to global prefix time.

    For a global window ``[S, E)``, the global time for active right boundary
    ``E`` is shifted by ``S / chunk_size`` before calling the local-window noise
    schedule. This keeps beta values identical to slicing the full-prefix
    schedule at ``[S:E]``.
    """
    local_end = local_end_indices.to(dtype=torch.float32)
    if start_tokens is None:
        start = torch.zeros_like(local_end)
    elif torch.is_tensor(start_tokens):
        start = start_tokens.to(
            device=local_end_indices.device,
            dtype=torch.float32,
        )
        if start.ndim == 0:
            start = start.expand_as(local_end)
    else:
        start = torch.full_like(local_end, float(start_tokens))
    phase = phase_offset.to(device=local_end_indices.device, dtype=torch.float32)
    if phase.ndim == 0:
        phase = phase.expand_as(local_end)
    global_end = start + local_end
    global_time = (global_end - 1.0) / float(chunk_size) + phase
    return global_time - start / float(chunk_size)


class SelfForcingTrainer:
    """Run self-forcing rollout, loss, backward, and optimizer step."""

    def __init__(self, module: CustomLightningModule):
        self._module = module
        self._preconditions_checked = False
        self._last_replace_diff: float | None = None
        self._grad_clip_val: float | None = None  # resolved lazily

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
        """Self-forcing K-step rollout training step."""
        self._check_preconditions()

        ldf_cfg = self._module.cfg.get("ldf_training", {}) or {}
        if str(ldf_cfg.get("window_policy", "prefix")) == "rolling":
            model = getattr(self._module, "model", None)
            default_context = getattr(model, "seq_len", batch["token"].shape[1])
            context_tokens = int(
                ldf_cfg.get(
                    "context_tokens",
                    default_context,
                )
            )
            window_sampling_cfg = ldf_cfg.get("window_sampling", {}) or {}
            window_sampling_enabled = bool(window_sampling_cfg.get("enabled", False))
            stride_tokens = self_forcing_stride_tokens(self._module.cfg)
            if window_sampling_enabled:
                semantics = compute_step_semantics(self._module)
                target_k = self.resolve_k(semantics.progress)
                rollout_span = max(0, (target_k - 1) * stride_tokens)
            else:
                rollout_span = 0
            horizon_tokens = int(ldf_cfg.get("horizon_tokens", 0))
            min_history_tokens = int(
                ldf_cfg.get("min_history_tokens", getattr(model, "chunk_size", 1))
            )
            model_batch = SampleCreator(
                stream_enabled=True,
                context_tokens=context_tokens,
                horizon_tokens=horizon_tokens,
                sample_policy=ldf_cfg.get("sample_policy", "variable_history"),
                min_history_tokens=min_history_tokens,
                window_sampling=(
                    window_sampling_cfg if window_sampling_enabled else None
                ),
                chunk_size=getattr(model, "chunk_size", None),
                rollout_span=rollout_span,
                force_start_token_zero=bool(
                    ldf_cfg.get("force_start_token_zero", False)
                ),
            ).create(batch, vae=getattr(self._module, "vae", None))
            loss_batch = batch.copy()
            for key in _ROLLING_LOSS_BATCH_KEYS:
                if key in model_batch:
                    loss_batch[key] = model_batch[key]
            if bool(model_batch.get("_window_local_traj", False)):
                loss_batch["traj_cond_7d"] = model_batch["traj_cond_7d"]
                loss_batch["traj_length"] = model_batch["traj_length"]
            return self._self_forcing_step(loss_batch, model_batch)

        prefix_creator = getattr(self._module, "build_prefix_sample_creator", None)
        semantics = compute_step_semantics(self._module)
        target_k = self.resolve_k(semantics.progress)
        min_prefix_tokens = self._required_prefix_tokens(target_k)
        if prefix_creator is not None:
            model_batch = prefix_creator(
                min_prefix_tokens=min_prefix_tokens,
            ).create(batch)
        else:
            model_batch = SampleCreator(
                min_prefix_tokens=min_prefix_tokens,
            ).create(batch)
        loss_batch = batch.copy()
        for key in _PREFIX_LOSS_BATCH_KEYS:
            if key in model_batch:
                loss_batch[key] = model_batch[key]
        return self._self_forcing_step(loss_batch, model_batch)

    def _self_forcing_step(
        self, batch: dict, model_batch: dict
    ) -> torch.Tensor:
        module = self._module
        net_start_time = time.time()
        semantics, runtime_metrics = self._build_runtime_metrics()
        runtime_metrics.update(_collect_window_local_metrics(model_batch))
        optimizer = module.optimizers()
        lr_scheduler = module.lr_schedulers()
        lr_for_step = float(optimizer.param_groups[0]["lr"])
        self._log_metrics(runtime_metrics)

        optimizer.zero_grad(set_to_none=True)
        final_step_result, effective_k = self._run_rollout(
            model_batch, semantics.progress
        )
        target_k = self.resolve_k(semantics.progress)
        runtime_metrics["self_forcing/k"] = float(effective_k)
        runtime_metrics["self_forcing/target_k"] = float(target_k)
        runtime_metrics["self_forcing/effective_k"] = float(effective_k)
        runtime_metrics["self_forcing/k_clipped"] = (
            1.0 if int(effective_k) < int(target_k) else 0.0
        )
        rollout_metrics = getattr(self, "_last_window_local_rollout_metrics", None)
        if rollout_metrics:
            runtime_metrics.update(rollout_metrics)
            self._last_window_local_rollout_metrics = None
        # Per-step corruption indicator; averaged by the logger across steps.
        runtime_metrics["history_corruption/applied"] = getattr(
            self, "_last_corruption_applied", 0.0
        )
        last_horizon_tokens = getattr(self, "_last_horizon_tokens", -1.0)
        ldf_cfg = module.cfg.get("ldf_training", {}) or {}
        rolling_enabled = str(ldf_cfg.get("window_policy", "prefix")) == "rolling"
        window_sampling_enabled = bool(
            (ldf_cfg.get("window_sampling", {}) or {}).get("enabled", False)
        )
        horizon_sim_enabled = bool(
            (module.cfg.get("horizon_sim", {}) or {}).get("enabled", False)
        )
        if rolling_enabled and window_sampling_enabled:
            runtime_metrics["ldf_training/runtime_horizon_tokens"] = (
                last_horizon_tokens
            )
        elif horizon_sim_enabled:
            runtime_metrics["horizon_sim/horizon_tokens"] = last_horizon_tokens
        elif rolling_enabled:
            runtime_metrics["ldf_training/runtime_horizon_tokens"] = (
                last_horizon_tokens
            )
        sample_loss_mask = getattr(self, "_last_sample_loss_mask", None)
        if sample_loss_mask is not None:
            runtime_metrics["anchor_canonicalize/valid_frac"] = float(
                sample_loss_mask.mean()
            )
        body_aux_terms = getattr(self, "_last_body_aux_terms", None)
        if body_aux_terms:
            for name, value in body_aux_terms.items():
                runtime_metrics[f"body_aux/{name}"] = float(value)
            self._last_body_aux_terms = None
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
        """Mirror automatic-to-manual optimizer progress on resume.

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
                        f"[resume] mirrored auto-to-manual optim_step_progress "
                        f"completed={completed} (was 0); keeps self.global_step "
                        f"aligned with ckpt"
                    )
            except (KeyError, TypeError) as exc:
                rank_zero_info(
                    f"[resume] could not mirror auto-to-manual progress ({exc!r}); "
                    f"global_step may start from 0"
                )
        return resume_step_offset

    def resolve_k(self, progress: float) -> int:
        """Resolve the rollout depth K from the schedule table at the given
        phase progress in [0, 1]."""
        schedule = self_forcing_k_schedule(self._module.cfg)
        rollout_depth = int(schedule[0][1])
        for threshold, candidate_k in schedule:
            if progress >= threshold:
                rollout_depth = int(candidate_k)
            else:
                break
        return max(1, rollout_depth)

    def _required_prefix_tokens(self, target_k: int) -> int:
        stride_tokens = self_forcing_stride_tokens(self._module.cfg)
        return 1 + (max(1, int(target_k)) - 1) * int(stride_tokens)

    def plan_rollout(
        self,
        feature_length: torch.Tensor,
        device: torch.device,
        progress: float,
        model_batch: dict | None = None,
    ) -> RolloutPlan:
        """Plan the rollout: K depth, per-sample start indices, and phase offsets.

        Includes cross-rank DDP consensus so all ranks execute the same number
        of rollout steps.
        """
        model = self._module.model
        target_k = self.resolve_k(progress)
        ldf_cfg = self._module.cfg.get("ldf_training", {}) or {}
        rolling_enabled = str(ldf_cfg.get("window_policy", "prefix")) == "rolling"
        window_sampling_enabled = bool(
            (ldf_cfg.get("window_sampling", {}) or {}).get("enabled", False)
        )
        if (
            not rolling_enabled
            and model_batch is not None
            and str(model_batch.get("_window_local_sample_policy", "")) == "prefix"
        ):
            feature_length_local = feature_length.to(
                device=device,
                dtype=torch.long,
            ).view(-1)
            stride_tokens = self_forcing_stride_tokens(self._module.cfg)
            min_active_end = 1
            max_k_per_sample = torch.div(
                (feature_length_local - min_active_end).clamp(min=-1),
                int(stride_tokens),
                rounding_mode="floor",
            ) + 1
            min_k_local = int(max_k_per_sample.min().item())
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                tmp = torch.tensor([min_k_local], device=device, dtype=torch.long)
                torch.distributed.all_reduce(tmp, op=torch.distributed.ReduceOp.MIN)
                min_k_supported = int(tmp.item())
            else:
                min_k_supported = min_k_local
            if min_k_supported < 1:
                raise ValueError(
                    "prefix self-forcing requires feature_length >= 1 for "
                    "every sample; "
                    f"feature_length={feature_length_local.tolist()}"
                )
            effective_k = min(target_k, min_k_supported)
            rollout_span = max(0, (effective_k - 1) * int(stride_tokens))
            start_end_indices = feature_length_local - rollout_span
            if bool((start_end_indices < min_active_end).any()):
                raise ValueError(
                    "prefix self-forcing rollout is inconsistent with latent "
                    "length; "
                    f"feature_length={feature_length_local.tolist()}, "
                    f"start_end_indices={start_end_indices.tolist()}, "
                    f"min_active_end={min_active_end}, rollout_span={rollout_span}"
                )
            batch_size = int(feature_length_local.shape[0])
            phase_offset = torch.empty(
                batch_size, device=device, dtype=torch.float32
            ).uniform_(0.0, 1.0 / model.chunk_size)
            return RolloutPlan(
                effective_k=effective_k,
                start_end_indices=start_end_indices,
                phase_offset=phase_offset,
            )
        if (
            rolling_enabled
            and window_sampling_enabled
            and model_batch is not None
            and "_window_sampling_history_tokens" in model_batch
        ):
            history_tokens = model_batch["_window_sampling_history_tokens"]
            if not torch.is_tensor(history_tokens):
                history_tokens = torch.as_tensor(
                    history_tokens, device=device, dtype=torch.long
                )
            else:
                history_tokens = history_tokens.to(device=device, dtype=torch.long)
            history_tokens = history_tokens.view(-1)
            if history_tokens.numel() == 1 and feature_length.numel() > 1:
                history_tokens = history_tokens.expand(feature_length.numel())
            if history_tokens.numel() != feature_length.numel():
                raise ValueError(
                    "_window_sampling_history_tokens must provide one value per "
                    f"sample; got {history_tokens.numel()} values for batch "
                    f"size {feature_length.numel()}"
                )
            stride_tokens = self_forcing_stride_tokens(self._module.cfg)
            rollout_span = max(0, (target_k - 1) * stride_tokens)
            start_end_indices = history_tokens + int(model.chunk_size)
            required_len = start_end_indices + rollout_span
            feature_length_local = feature_length.to(
                device=device,
                dtype=torch.long,
            ).view(-1)
            if bool((feature_length_local < required_len).any()):
                raise ValueError(
                    "window_sampling metadata is inconsistent with feature_length; "
                    f"feature_length={feature_length_local.tolist()}, "
                    f"required_final_active_right={required_len.tolist()}"
                )
            batch_size = int(feature_length_local.shape[0])
            phase_offset = torch.empty(
                batch_size, device=device, dtype=torch.float32
            ).uniform_(0.0, 1.0 / model.chunk_size)
            return RolloutPlan(
                effective_k=target_k,
                start_end_indices=start_end_indices,
                phase_offset=phase_offset,
            )
        min_history_tokens = 1
        if rolling_enabled:
            min_history_tokens = int(
                ldf_cfg.get("min_history_tokens", getattr(model, "chunk_size", 1))
            )
            if min_history_tokens < int(model.chunk_size):
                raise ValueError(
                    "ldf_training.min_history_tokens must be >= chunk_size "
                    "for rolling training; "
                    f"got min_history_tokens={min_history_tokens}, "
                    f"chunk_size={int(model.chunk_size)}"
                )

        # Cross-rank consensus: shortest valid sequence wins.
        min_k_local = int(feature_length.min().item()) - min_history_tokens + 1
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
        sample_policy = str(ldf_cfg.get("sample_policy", "variable_history"))
        for batch_idx in range(feature_length.shape[0]):
            low = min_history_tokens
            high = int(max_start[batch_idx].item())
            if high < low:
                raise ValueError(
                    f"Invalid self-forcing start range for sample {batch_idx}: "
                    f"low={low}, high={high}, "
                    f"valid_len={int(feature_length[batch_idx].item())}, "
                    f"effective_k={effective_k}"
                )
            if sample_policy == "fixed_window":
                start_end_indices.append(high)
            elif high == low:
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

        text_context, text_dropped_flags = prepare_text_condition(
            model, model_batch, seq_len, device
        )
        traj_dropped = sample_traj_dropout(model, device)
        condition = PreparedCondition(
            text_context=text_context,
            text_dropped_flags=text_dropped_flags,
            traj_emb=None,
            traj_seq_lens=None,
            traj_dropped=traj_dropped,
            traj_token_mask=None,
        )
        plan = self.plan_rollout(
            feature_length,
            device,
            progress,
            model_batch=model_batch,
        )
        self._last_window_local_rollout_metrics = _collect_window_local_rollout_metrics(
            model_batch, plan
        )

        # Re-anchor world-frame 7D trajectory features to the current body
        # window so training matches inference-time per-window conditioning.
        self._last_sample_loss_mask = None
        anchor_cfg = self._module.cfg.get("anchor_canonicalize", {}) or {}
        traj_features = model_batch.get("traj_features")
        if (
            anchor_cfg.get("enabled", True)
            and not bool(model_batch.get("_window_local_traj", False))
            and torch.is_tensor(traj_features)
            and traj_features.shape[-1] == 7
        ):
            traj_features_7d = traj_features.to(device)
            valid_len = model_batch.get("traj_length")
            if valid_len is None:
                valid_len = torch.full(
                    (traj_features_7d.shape[0],),
                    traj_features_7d.shape[1],
                    device=device,
                    dtype=torch.long,
                )
            gt_xyz = traj_features_7d[..., :3]
            gt_yaw = torch.atan2(traj_features_7d[..., 4], traj_features_7d[..., 3])
            canonical_traj_features, sample_loss_mask = apply_body_window_canonicalize(
                traj_features_7d,
                plan.start_end_indices,
                gt_xyz,
                gt_yaw,
                valid_len,
                body_window_tokens=seq_len,
            )
            model_batch = {**model_batch, "traj_features": canonical_traj_features}
            self._last_sample_loss_mask = sample_loss_mask

        ldf_cfg = self._module.cfg.get("ldf_training", {}) or {}
        rolling_enabled = str(ldf_cfg.get("window_policy", "prefix")) == "rolling"
        window_sampling_cfg = ldf_cfg.get("window_sampling", {}) or {}
        window_sampling_enabled = bool(window_sampling_cfg.get("enabled", False))
        use_window_sampling_horizon = (
            rolling_enabled
            and window_sampling_enabled
            and "_window_sampling_horizon_tokens" in model_batch
        )

        horizon_tokens = None
        horizon_active_end = 0
        if use_window_sampling_horizon:
            horizon_tokens = model_batch["_window_sampling_horizon_tokens"]
            if not torch.is_tensor(horizon_tokens):
                horizon_tokens = torch.as_tensor(
                    horizon_tokens, device=device, dtype=torch.long
                )
            else:
                horizon_tokens = horizon_tokens.to(device=device, dtype=torch.long)
            horizon_tokens = horizon_tokens.view(-1)
            if horizon_tokens.numel() == 1 and feature.shape[0] > 1:
                horizon_tokens = horizon_tokens.expand(feature.shape[0])
            self._last_horizon_tokens = float(
                horizon_tokens.to(dtype=torch.float32).mean().item()
            )
        else:
            # Prefix training reuses one horizon mask across the rollout.
            st_visible_horizon = None
            if rolling_enabled and not window_sampling_enabled:
                st_visible_horizon = int(ldf_cfg.get("horizon_tokens", 0))
                horizon_tokens = st_visible_horizon
            horizon_cfg = self._module.cfg.get("horizon_sim", {}) or {}
            if (not window_sampling_enabled) and horizon_cfg.get("enabled", False):
                sampled_horizon = sample_random_horizon_tokens(
                    progress,
                    1.0,
                    seq_len,
                    horizon_cfg,
                )
                if st_visible_horizon is None:
                    horizon_tokens = sampled_horizon
                else:
                    horizon_tokens = min(int(sampled_horizon), st_visible_horizon)
            if horizon_tokens is not None:
                local_active_end = (
                    plan.start_end_indices + (plan.effective_k - 1)
                ).to(device)
                horizon_active_end = _absolute_active_end_token(
                    model_batch, local_active_end, device
                )
            self._last_horizon_tokens = (
                float(horizon_tokens) if horizon_tokens is not None else -1.0
            )

            traj_emb, traj_seq_lens, traj_dropped, traj_token_mask = (
                prepare_traj_condition(
                    model,
                    model_batch,
                    seq_len,
                    device,
                    traj_dropped=condition.traj_dropped,
                    horizon_tokens=horizon_tokens,
                    horizon_active_end=horizon_active_end,
                )
            )
            condition = condition.with_traj(
                traj_emb, traj_seq_lens, traj_dropped, traj_token_mask
            )

        clean_feature_state = feature.clone()
        corruption_mask = None
        corrupted_feature_values = None
        # Corrupt the history region once at rollout start, then keep that view
        # fixed across all K steps.
        corruption_cfg = self._module.cfg.get("history_corruption", {}) or {}
        corruption_applied = False
        if should_apply_corruption(progress, 1.0, corruption_cfg):
            corrupted_initial = apply_history_corruption(
                clean_feature_state,
                plan.start_end_indices,
                mask_emb=model.model.mask_emb,
                z_std=model.model.z_std,
                chunk_size=model.chunk_size,
                alpha_mask=corruption_cfg.get("alpha_mask", 0.3),
                alpha_noisy=corruption_cfg.get("alpha_noisy", 0.3),
                noise_sigma_factor=corruption_cfg.get("noise_sigma_factor", 0.05),
            )
            corruption_mask = (corrupted_initial != clean_feature_state).any(
                dim=-1, keepdim=True
            )
            corrupted_feature_values = corrupted_initial
            corruption_applied = True
        self._last_corruption_applied = float(corruption_applied)

        final_step_result = None
        window_start_tokens = model_batch.get("_window_local_latent_start_token")
        stride_tokens = self_forcing_stride_tokens(self._module.cfg)
        for step_idx in range(plan.effective_k):
            current_feature = _apply_fixed_history_corruption_view(
                clean_feature_state, corruption_mask, corrupted_feature_values
            )
            end_indices = plan.start_end_indices + step_idx * stride_tokens
            time_steps = shifted_local_time_steps(
                end_indices,
                start_tokens=window_start_tokens,
                chunk_size=int(model.chunk_size),
                phase_offset=plan.phase_offset,
            )
            if use_window_sampling_horizon:
                step_horizon_active_end = _absolute_active_end_token(
                    model_batch, end_indices.to(device), device
                )
                traj_emb, traj_seq_lens, traj_dropped, traj_token_mask = (
                    prepare_traj_condition(
                        model,
                        model_batch,
                        seq_len,
                        device,
                        traj_dropped=condition.traj_dropped,
                        horizon_tokens=horizon_tokens,
                        horizon_active_end=step_horizon_active_end,
                    )
                )
                condition = condition.with_traj(
                    traj_emb, traj_seq_lens, traj_dropped, traj_token_mask
                )
            is_final_step = step_idx == plan.effective_k - 1
            if is_final_step:
                final_step_result = run_training_window(
                    model,
                    model_batch,
                    current_feature,
                    time_steps,
                    condition,
                )
                break

            with torch.no_grad():
                rollout_result = run_training_window(
                    model,
                    model_batch,
                    current_feature,
                    time_steps,
                    condition,
                )

            disable_replace = bool(
                self._module.cfg.get("self_forcing_disable_replace", False)
            )
            next_feature = clean_feature_state.clone()
            replace_diffs = []
            if not disable_replace:
                for batch_idx in range(feature.shape[0]):
                    replace_idx = (
                        int(end_indices[batch_idx].item()) - model.chunk_size
                    )
                    if replace_idx < 0:
                        continue
                    pred_seq = rollout_result["x0_latent_list"][batch_idx]
                    if replace_idx >= pred_seq.shape[0]:
                        continue
                    replacement = pred_seq[replace_idx].detach().to(
                        device=clean_feature_state.device,
                        dtype=clean_feature_state.dtype,
                    )
                    gt_token = clean_feature_state[batch_idx, replace_idx, :]
                    replace_diffs.append(
                        (replacement - gt_token).abs().mean().item()
                    )
                    next_feature[batch_idx, replace_idx, :] = replacement
                    if corruption_mask is not None:
                        corruption_mask = corruption_mask.clone()
                        corruption_mask[batch_idx, replace_idx, :] = False
            clean_feature_state = next_feature
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
        # 7D trajectories use body-aux loss; older 4D batches keep the xz loss.
        body_aux_cfg = self._module.cfg.get("body_aux_loss", {}) or {}
        use_body_aux = (
            bool(body_aux_cfg.get("enabled", False)) and "traj_cond_7d" in batch
        )
        if control_weight > 0.0 and use_body_aux:
            step_control_loss, self._last_body_aux_terms = _compute_body_aux_loss(
                final_step_result["pred_x0_latent_list"],
                batch,
                self._module,
                getattr(self, "_last_sample_loss_mask", None),
                body_aux_cfg,
            )
            if step_control_loss is not None:
                total_loss = total_loss + control_weight * step_control_loss
        elif control_weight > 0.0 and "traj" in batch:
            step_control_loss = _compute_control_loss(
                final_step_result["pred_x0_latent_list"],
                batch,
                self._module,
            )
            if step_control_loss is not None:
                total_loss = total_loss + control_weight * step_control_loss
        return total_loss, step_diff_loss, step_control_loss

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
        # 7D trajectory conditioning needs heading supervision.
        traj_in_dim = int(getattr(self._module.model, "traj_in_dim", 4))
        body_aux_enabled = bool(
            (self._module.cfg.get("body_aux_loss", {}) or {}).get("enabled", False)
        )
        if traj_in_dim == 7 and not body_aux_enabled:
            raise ValueError(
                "traj_encoder_in_dim=7 requires body_aux_loss.enabled=true "
                "(the 7D heading channels need supervision)."
            )
        self._preconditions_checked = True

    def _build_runtime_metrics(self):
        semantics = compute_step_semantics(self._module)
        runtime_metrics = {
            "self_forcing/enabled": 1.0,
            "self_forcing/active": 1.0,
            "self_forcing/progress": float(semantics.progress),
            "self_forcing/k": 0.0,
            "self_forcing/target_k": 0.0,
            "self_forcing/effective_k": 0.0,
            "self_forcing/k_clipped": 0.0,
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
            phase_step = int(runtime_metrics["self_forcing/phase_step"])
            absolute_step = int(runtime_metrics["self_forcing/absolute_step"])
            resume_step = int(runtime_metrics["self_forcing/resume_step_offset"])
            phase_total = int(runtime_metrics["self_forcing/phase_total_steps"])
            target_step = int(runtime_metrics["self_forcing/absolute_target_step"])
            rank_zero_info(
                "[self_forcing] "
                f"phase_step={phase_step} "
                f"absolute_step={absolute_step} "
                f"resume_step_offset={resume_step} "
                f"phase_total_steps={phase_total} "
                f"absolute_target_step={target_step} "
                f"active={int(runtime_metrics['self_forcing/active'])} "
                f"progress={runtime_metrics['self_forcing/progress']:.6f}"
            )


_DEFAULT_BODY_AUX_WEIGHTS = {
    "root_xz": 1.0,
    "root_y": 0.3,
    "heading": 0.5,
    "fwd_delta": 0.1,
    "yaw_delta": 0.1,
    "end_xz": 0.0,
}


def _compute_body_aux_loss(pred_list, batch, module, sample_loss_mask, body_aux_cfg):
    """Compute body-aux loss against clip-local 7D trajectory targets."""
    if pred_list is None or "traj_cond_7d" not in batch:
        return None, {}
    window_start_tokens = None
    weights = {
        **_DEFAULT_BODY_AUX_WEIGHTS,
        **(body_aux_cfg.get("weights", {}) or {}),
    }
    return compute_body_aux_loss(
        pred_list,
        batch["traj_cond_7d"],
        batch["traj_length"],
        module.vae,
        module.device,
        weights,
        chunk_size_tokens=getattr(module.model, "chunk_size", None),
        heading_form=body_aux_cfg.get("heading_form", "cosine"),
        sample_loss_mask=sample_loss_mask,
        window_start_tokens=window_start_tokens,
    )


def _collect_window_local_metrics(model_batch: dict) -> dict[str, float]:
    """Summarize window-local sampling state for training logs."""
    if not bool(model_batch.get("_window_local_traj", False)):
        return {}

    metrics: dict[str, float] = {"ldf_training/enabled": 1.0}
    sample_policy = str(
        model_batch.get("_window_local_sample_policy", "variable_history")
    )
    metrics["ldf_training/sample_policy_fixed_window"] = (
        1.0 if sample_policy == "fixed_window" else 0.0
    )

    starts = model_batch.get("_window_local_latent_start_token")
    if starts is not None:
        starts_t = torch.as_tensor(starts, dtype=torch.float32)
        metrics["ldf_training/window_start_mean"] = float(starts_t.mean().item())
    global_starts = model_batch.get("_window_global_start_token")
    if global_starts is not None:
        global_starts_t = torch.as_tensor(global_starts, dtype=torch.float32)
        metrics["ldf_training/global_window_start_mean"] = float(
            global_starts_t.mean().item()
        )
    lengths = model_batch.get("_window_local_latent_valid_len")
    if lengths is not None:
        lengths_t = torch.as_tensor(lengths, dtype=torch.float32)
        metrics["ldf_training/window_len_mean"] = float(lengths_t.mean().item())
        metrics["ldf_training/window_len_min"] = float(lengths_t.min().item())
        metrics["ldf_training/window_len_max"] = float(lengths_t.max().item())
    traj_tokens = model_batch.get("traj_num_tokens")
    if traj_tokens is not None:
        traj_tokens_t = torch.as_tensor(traj_tokens, dtype=torch.float32)
        metrics["ldf_training/traj_tokens_mean"] = float(traj_tokens_t.mean().item())
    return metrics


def _absolute_active_end_token(
    model_batch: dict,
    local_active_end: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    local_active_end = local_active_end.to(device=device, dtype=torch.long).view(-1)
    traj_start_token = model_batch.get(
        "traj_start_token", model_batch.get("_window_local_latent_start_token")
    )
    if traj_start_token is None:
        return local_active_end
    if not torch.is_tensor(traj_start_token):
        traj_start_token = torch.as_tensor(
            traj_start_token, device=device, dtype=torch.long
        )
    else:
        traj_start_token = traj_start_token.to(device=device, dtype=torch.long)
    traj_start_token = traj_start_token.view(-1)
    if traj_start_token.numel() == 1 and local_active_end.numel() > 1:
        traj_start_token = traj_start_token.expand_as(local_active_end)
    if traj_start_token.numel() != local_active_end.numel():
        raise ValueError(
            "traj_start_token must be scalar or match batch size; "
            f"got {traj_start_token.numel()} starts for "
            f"{local_active_end.numel()} active ends"
        )
    return traj_start_token + local_active_end


def _collect_window_local_rollout_metrics(
    model_batch: dict,
    plan: RolloutPlan,
) -> dict[str, float]:
    """Summarize the actual final active history selected by plan_rollout()."""
    if not bool(model_batch.get("_window_local_traj", False)):
        return {}
    rollout_span = model_batch.get("_window_sampling_rollout_span")
    if rollout_span is None:
        rollout_span = int(plan.effective_k) - 1
    else:
        rollout_span = int(rollout_span)
    final_active_end = plan.start_end_indices.to(dtype=torch.long) + rollout_span
    active_len = final_active_end.to(dtype=torch.float32)
    metrics = {
        "ldf_training/active_history_len_mean": float(active_len.mean().item()),
        "ldf_training/active_history_len_min": float(active_len.min().item()),
        "ldf_training/active_history_len_max": float(active_len.max().item()),
    }
    history_tokens = model_batch.get("_window_sampling_history_tokens")
    if history_tokens is not None:
        hist_t = torch.as_tensor(history_tokens, dtype=torch.float32).view(-1)
        metrics["ldf_training/history_tokens_mean"] = float(hist_t.mean().item())
        metrics["ldf_training/history_tokens_min"] = float(hist_t.min().item())
        metrics["ldf_training/history_tokens_max"] = float(hist_t.max().item())
    horizon_tokens = model_batch.get("_window_sampling_horizon_tokens")
    if horizon_tokens is not None:
        hor_t = torch.as_tensor(horizon_tokens, dtype=torch.float32).view(-1)
        metrics["ldf_training/horizon_tokens_mean"] = float(hor_t.mean().item())
        metrics["ldf_training/horizon_tokens_min"] = float(hor_t.min().item())
        metrics["ldf_training/horizon_tokens_max"] = float(hor_t.max().item())
    horizon_cap_clip = model_batch.get("_window_sampling_horizon_cap_clip")
    if horizon_cap_clip is not None:
        cap_t = torch.as_tensor(horizon_cap_clip, dtype=torch.float32).view(-1)
        metrics["ldf_training/horizon_cap_clip_mean"] = float(cap_t.mean().item())
    horizon_short_fallback = model_batch.get("_window_sampling_horizon_short_fallback")
    if horizon_short_fallback is not None:
        fallback_t = torch.as_tensor(
            horizon_short_fallback, dtype=torch.float32
        ).view(-1)
        metrics["ldf_training/horizon_short_fallback_rate"] = float(
            fallback_t.mean().item()
        )
    if "_window_sampling_rollout_span" in model_batch:
        metrics["ldf_training/rollout_span"] = float(rollout_span)
    if "_window_sampling_history_tokens_max_effective" in model_batch:
        metrics["ldf_training/history_tokens_max_effective"] = float(
            model_batch["_window_sampling_history_tokens_max_effective"]
        )
    lengths = model_batch.get("_window_local_latent_valid_len")
    if lengths is not None:
        final_len = torch.as_tensor(lengths, dtype=torch.float32).view(-1)
        metrics["ldf_training/final_visible_latent_len_mean"] = float(
            final_len.mean().item()
        )
    active_left = model_batch.get("_window_sampling_active_left_token")
    if active_left is not None:
        active_left_t = torch.as_tensor(active_left, dtype=torch.float32).view(-1)
        metrics["ldf_training/active_left_mean"] = float(active_left_t.mean().item())
    starts = model_batch.get("_window_local_latent_start_token")
    if starts is not None:
        starts_t = torch.as_tensor(
            starts, device=final_active_end.device, dtype=torch.long
        ).view(-1)
        if starts_t.numel() == 1 and final_active_end.numel() > 1:
            starts_t = starts_t.expand_as(final_active_end)
        if starts_t.numel() == final_active_end.numel():
            abs_end = starts_t + final_active_end
            metrics["ldf_training/active_abs_end_mean"] = float(
                abs_end.to(dtype=torch.float32).mean().item()
            )
    return metrics


def _apply_fixed_history_corruption_view(
    clean_feature_state: torch.Tensor,
    corruption_mask: torch.Tensor | None,
    corrupted_feature_values: torch.Tensor | None,
) -> torch.Tensor:
    """Overlay the fixed history-corruption view onto the clean latent state."""
    if corruption_mask is None or corrupted_feature_values is None:
        return clean_feature_state
    return torch.where(corruption_mask, corrupted_feature_values, clean_feature_state)


def _compute_control_loss(pred_list, batch, module):
    """Resolve training-mode config and delegate to the XZ control loss."""
    if pred_list is None:
        return None
    traj_loss_gt = batch.get("traj_loss_gt", batch.get("traj"))
    if traj_loss_gt is None:
        return None
    traj = traj_loss_gt
    traj_mask = batch.get("traj_loss_mask", batch.get("traj_mask"))
    traj_length = batch["traj_length"]
    train_mode = control_loss_train_mode(module.cfg)
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
    sf_enabled: bool,
    configured_num_training_steps: int,
    reset_optimizer_on_resume: bool = True,
):
    """Resolve phase_max_steps and runtime_scheduler_steps for self-forcing resume.

    Returns (resume_step_offset, phase_max_steps, runtime_scheduler_steps).
    """
    from utils.training.step_semantics import (
        load_resume_step_offset,
        resolve_runtime_max_steps,
        resolve_scheduler_steps,
    )

    resume_step_offset = 0
    phase_max_steps = absolute_target_step
    if resume_ckpt and sf_enabled:
        resume_step_offset = load_resume_step_offset(resume_ckpt)
        phase_max_steps = resolve_runtime_max_steps(
            absolute_target_step,
            resume_step_offset,
            self_forcing_enabled=sf_enabled,
        )
        rank_zero_info(
            "[self_forcing runtime] "
            f"resume_step_offset={resume_step_offset} "
            f"absolute_target_step={absolute_target_step} "
            f"phase_max_steps={phase_max_steps}"
        )

    if reset_optimizer_on_resume:
        runtime_scheduler_steps = resolve_scheduler_steps(
            configured_num_training_steps,
            absolute_target_step=absolute_target_step,
            runtime_max_steps=phase_max_steps,
        )
    else:
        # If Lightning restores optimizer/scheduler state from the checkpoint,
        # do not silently change the scheduler's construction horizon. LambdaLR
        # checkpoints store last_epoch/_last_lr but not the lambda closure inputs
        # (e.g. diffusers cosine num_training_steps), so rebuilding with a new
        # phase horizon produces a partially-restored scheduler and LR jumps.
        runtime_scheduler_steps = int(configured_num_training_steps)
    return resume_step_offset, phase_max_steps, runtime_scheduler_steps
