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

from utils.training.body_canonicalize import apply_body_window_canonicalize
from utils.training.history_corruption import (
    apply_history_corruption,
    should_apply_corruption,
)
from utils.training.horizon_sched import sample_random_horizon_tokens
from lightning.pytorch.utilities import rank_zero_info

from .control_loss import compute_body_aux_loss, compute_control_loss_xz
from .model_batch import prepare_model_input
from .window_local import build_window_local_model_batch
from .module_step import compute_step_semantics

if TYPE_CHECKING:
    from train_ldf import CustomLightningModule


@dataclass(frozen=True)
class RolloutPlan:
    effective_k: int
    start_end_indices: torch.Tensor  # (B,) long
    phase_offset: torch.Tensor       # (B,) float32


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

        st_cfg = self._module.cfg.get("stream_training", {}) or {}
        if bool(st_cfg.get("enabled", False)):
            module_model = getattr(self._module, "model", None)
            default_context = getattr(module_model, "seq_len", batch["token"].shape[1])
            context_tokens = int(
                st_cfg.get(
                    "context_tokens",
                    default_context,
                )
            )
            horizon_tokens = int(st_cfg.get("horizon_tokens", 0))
            min_history_tokens = int(
                st_cfg.get("min_history_tokens", getattr(module_model, "chunk_size", 1))
            )
            model_batch = build_window_local_model_batch(
                batch,
                context_tokens=context_tokens,
                horizon_tokens=horizon_tokens,
                sample_policy=st_cfg.get("sample_policy", "variable_history"),
                min_history_tokens=min_history_tokens,
            )
            loss_batch = batch.copy()
            for key in (
                "_window_local_latent_start_token",
                "_window_local_latent_valid_len",
                "_window_local_traj",
            ):
                if key in model_batch:
                    loss_batch[key] = model_batch[key]
            motion_aux_mode = st_cfg.get("motion_aux_loss", "latent_only")
            if motion_aux_mode in (False, None, "latent_only", "disabled"):
                for key in (
                    "traj", "traj_cond", "traj_cond_7d", "traj_mask",
                    "traj_cond_mask", "traj_loss_mask", "traj_features",
                    "traj_loss_gt",
                ):
                    loss_batch.pop(key, None)
            elif motion_aux_mode not in ("full_prefix",):
                raise ValueError(
                    "stream_training.motion_aux_loss must be 'latent_only', "
                    f"'disabled', or 'full_prefix'; got {motion_aux_mode!r}"
                )
            return self._self_forcing_step(loss_batch, model_batch)

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
        runtime_metrics.update(_collect_window_local_metrics(model_batch))
        optimizer = module.optimizers()
        lr_scheduler = module.lr_schedulers()
        lr_for_step = float(optimizer.param_groups[0]["lr"])
        self._log_metrics(runtime_metrics)

        optimizer.zero_grad(set_to_none=True)
        final_step_result, effective_k = self._run_rollout(
            model_batch, semantics.progress
        )
        runtime_metrics["self_forcing/k"] = float(effective_k)
        rollout_metrics = getattr(self, "_last_window_local_rollout_metrics", None)
        if rollout_metrics:
            runtime_metrics.update(rollout_metrics)
            self._last_window_local_rollout_metrics = None
        # T_B_03: per-step corruption indicator (0/1); averaged by the logger it
        # reports the effective corruption rate as the curriculum ramps.
        runtime_metrics["history_corruption/applied"] = getattr(
            self, "_last_corruption_applied", 0.0
        )
        # T_B_04: sampled horizon (tokens) this step; -1 when horizon_sim off.
        runtime_metrics["horizon_sim/horizon_tokens"] = getattr(
            self, "_last_horizon_tokens", -1.0
        )
        # T_B_05: fraction of samples with a valid canonicalize anchor (loss-weighted).
        _slm = getattr(self, "_last_sample_loss_mask", None)
        if _slm is not None:
            runtime_metrics["anchor_canonicalize/valid_frac"] = float(_slm.mean())
        # T_B_06: per-term body aux loss breakdown (heading visible in fine-tune log).
        _bat = getattr(self, "_last_body_aux_terms", None)
        if _bat:
            for _k, _v in _bat.items():
                runtime_metrics[f"body_aux/{_k}"] = float(_v)
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
        st_cfg = self._module.cfg.get("stream_training", {}) or {}
        stream_training_enabled = bool(st_cfg.get("enabled", False))
        min_history_tokens = 1
        if stream_training_enabled:
            min_history_tokens = int(
                st_cfg.get("min_history_tokens", getattr(model, "chunk_size", 1))
            )
            if min_history_tokens < int(model.chunk_size):
                raise ValueError(
                    "stream_training.min_history_tokens must be >= chunk_size; "
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
        sample_policy = str(st_cfg.get("sample_policy", "variable_history"))
        for b in range(feature_length.shape[0]):
            low = min_history_tokens
            high = int(max_start[b].item())
            if high < low:
                raise ValueError(
                    f"Invalid self-forcing start range for sample {b}: "
                    f"low={low}, high={high}, "
                    f"valid_len={int(feature_length[b].item())}, "
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

        text_dropped_flags = model._decide_text_dropout(feature.shape[0], device)
        all_text_context = model._prepare_text_context(model_batch, seq_len, device, text_dropped_flags)
        traj_dropped = model._decide_traj_dropout(device)
        plan = self.plan_rollout(feature_length, device, progress)
        self._last_window_local_rollout_metrics = _collect_window_local_rollout_metrics(
            model_batch, plan
        )

        # T_B_05: canonicalize the 7D world-frame traj_cond into the body-window-
        # local frame (anchor = body window leftmost = history0) so training
        # matches inference's per-window re-anchoring. v1: single canonicalize for
        # the K-step rollout using plan.start_end_indices; body_window_tokens =
        # seq_len; the GT anchor pose is read from traj_features itself (its xyz =
        # channels 0-2, heading = cos/sin in 3-4 ARE the unsmoothed GT root pose),
        # valid_len = traj_length. Only fires for 7D traj. sample_loss_mask is
        # produced here and consumed by the body aux loss (T_B_06).
        self._last_sample_loss_mask = None
        ac_cfg = self._module.cfg.get("anchor_canonicalize", {}) or {}
        traj_feats = model_batch.get("traj_features")
        if (ac_cfg.get("enabled", False)
                and not bool(model_batch.get("_window_local_traj", False))
                and torch.is_tensor(traj_feats)
                and traj_feats.shape[-1] == 7):
            feats7 = traj_feats.to(device)
            valid_len = model_batch.get("traj_length")
            if valid_len is None:
                valid_len = torch.full(
                    (feats7.shape[0],), feats7.shape[1], device=device, dtype=torch.long
                )
            gt_xyz = feats7[..., :3]
            gt_yaw = torch.atan2(feats7[..., 4], feats7[..., 3])
            canon, sample_loss_mask = apply_body_window_canonicalize(
                feats7, plan.start_end_indices, gt_xyz, gt_yaw, valid_len,
                body_window_tokens=seq_len,
            )
            model_batch = {**model_batch, "traj_features": canon}
            self._last_sample_loss_mask = sample_loss_mask

        # T_B_04 / B-P0-2: compute the horizon (token-level) in the outer loop and
        # pass it down — the model never reads global_step. The horizon is
        # truncated relative to the FINAL supervised step's active-window token
        # position (NOT clip start): horizon_active_end = start_end + (K-1), per
        # sample. Fixed across the K rollout steps so all steps see one
        # consistent traj mask. In stream_training mode, the configured
        # horizon_tokens is the visible horizon bound even when horizon_sim is
        # disabled; horizon_sim may sample a shorter value but not a longer one.
        # Outside stream_training, None horizon_tokens keeps the legacy
        # full-traj-cond behavior when horizon_sim is disabled.
        horizon_tokens = None
        horizon_active_end = 0
        st_cfg = self._module.cfg.get("stream_training", {}) or {}
        stream_training_enabled = bool(st_cfg.get("enabled", False))
        st_visible_horizon = None
        if stream_training_enabled:
            st_visible_horizon = int(st_cfg.get("horizon_tokens", 0))
            horizon_tokens = st_visible_horizon
        hs_cfg = self._module.cfg.get("horizon_sim", {}) or {}
        if hs_cfg.get("enabled", False):
            sampled_horizon = sample_random_horizon_tokens(
                progress, 1.0, seq_len, hs_cfg,
            )
            if st_visible_horizon is None:
                horizon_tokens = sampled_horizon
            else:
                horizon_tokens = min(int(sampled_horizon), st_visible_horizon)
        if horizon_tokens is not None:
            local_active_end = (
                plan.start_end_indices + (plan.effective_k - 1)
            ).to(device)
            traj_start_token = model_batch.get("traj_start_token")
            if traj_start_token is not None:
                if not torch.is_tensor(traj_start_token):
                    traj_start_token = torch.as_tensor(
                        traj_start_token, device=device, dtype=torch.long
                    )
                else:
                    traj_start_token = traj_start_token.to(device=device, dtype=torch.long)
                if traj_start_token.ndim == 0:
                    traj_start_token = traj_start_token.repeat(local_active_end.shape[0])
                horizon_active_end = traj_start_token + local_active_end
            else:
                horizon_active_end = local_active_end
        self._last_horizon_tokens = float(horizon_tokens) if horizon_tokens is not None else -1.0

        traj_emb, traj_seq_lens, _, traj_token_mask = model._prepare_traj_condition(
            model_batch, seq_len, device, traj_dropped_override=traj_dropped,
            horizon_tokens=horizon_tokens, horizon_active_end=horizon_active_end,
        )

        clean_feature_state = feature.clone()
        corruption_mask = None
        corrupted_feature_values = None
        # T_B_03 §2.1.5: optionally corrupt the history region ONCE at rollout
        # start, kept fixed across the K steps (so all steps see a consistent
        # "fake history"). Gated by the history_corruption cfg; `progress` is the
        # training progress in [0,1] used for the curriculum bands (passed as
        # global_step/total_steps = progress/1.0 so should_apply_corruption's
        # progress == progress).
        hc_cfg = self._module.cfg.get("history_corruption", {}) or {}
        corruption_applied = False
        if should_apply_corruption(progress, 1.0, hc_cfg):
            corrupted_initial = apply_history_corruption(
                clean_feature_state,
                plan.start_end_indices,
                mask_emb=model.model.mask_emb,
                z_std=model.model.z_std,
                chunk_size=model.chunk_size,
                alpha_mask=hc_cfg.get("alpha_mask", 0.3),
                alpha_noisy=hc_cfg.get("alpha_noisy", 0.3),
                noise_sigma_factor=hc_cfg.get("noise_sigma_factor", 0.05),
            )
            corruption_mask = (corrupted_initial != clean_feature_state).any(
                dim=-1, keepdim=True
            )
            corrupted_feature_values = corrupted_initial
            corruption_applied = True
        self._last_corruption_applied = float(corruption_applied)

        final_step_result = None
        window_start_tokens = model_batch.get("_window_local_latent_start_token")
        for step_idx in range(plan.effective_k):
            current_feature = _apply_fixed_history_corruption_view(
                clean_feature_state, corruption_mask, corrupted_feature_values
            )
            end_indices = plan.start_end_indices + step_idx
            time_steps = shifted_local_time_steps(
                end_indices,
                start_tokens=window_start_tokens,
                chunk_size=int(model.chunk_size),
                phase_offset=plan.phase_offset,
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
                    traj_token_mask=traj_token_mask,
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
                    traj_token_mask=traj_token_mask,
                )

            disable_replace = bool(
                self._module.cfg.get("self_forcing_disable_replace", False)
            )
            next_feature = clean_feature_state.clone()
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
                        device=clean_feature_state.device, dtype=clean_feature_state.dtype
                    )
                    gt_token = clean_feature_state[b, replace_idx, :]
                    replace_diffs.append(
                        (replacement - gt_token).abs().mean().item()
                    )
                    next_feature[b, replace_idx, :] = replacement
                    if corruption_mask is not None:
                        corruption_mask = corruption_mask.clone()
                        corruption_mask[b, replace_idx, :] = False
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
        # T_B_06: in 7D mode with body_aux_loss enabled, the body aux loss (5
        # terms incl. heading) replaces the legacy xz-only control loss. The 4D
        # path keeps using compute_control_loss_xz (gate below).
        ba_cfg = self._module.cfg.get("body_aux_loss", {}) or {}
        use_body_aux = bool(ba_cfg.get("enabled", False)) and "traj_cond_7d" in batch
        if control_weight > 0.0 and use_body_aux:
            step_control_loss, self._last_body_aux_terms = _compute_body_aux_loss(
                final_step_result["pred_x0_latent_list"],
                batch,
                self._module,
                getattr(self, "_last_sample_loss_mask", None),
                ba_cfg,
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
        # T_B_06: the new 7D traj heading channels MUST be supervised — refuse to
        # train a 7D model without body_aux_loss (heading) enabled.
        traj_in_dim = int(getattr(self._module.model, "traj_in_dim", 4))
        ba_enabled = bool((self._module.cfg.get("body_aux_loss", {}) or {}).get("enabled", False))
        if traj_in_dim == 7 and not ba_enabled:
            raise ValueError(
                "traj_encoder_in_dim=7 requires body_aux_loss.enabled=true "
                "(the new 7D heading channels need supervision); see T_B_06."
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


_DEFAULT_BODY_AUX_WEIGHTS = {
    "root_xz": 1.0, "root_y": 0.3, "heading": 0.5, "fwd_delta": 0.1, "yaw_delta": 0.1,
}


def _compute_body_aux_loss(pred_list, batch, module, sample_loss_mask, ba_cfg):
    """Resolve body-aux config then delegate to compute_body_aux_loss (T_B_06).

    GT is the clip-local raw 7D traj_cond (batch[traj_cond_7d]); it is in the same
    frame as the decoded pred (both clip-local recovery). Returns (loss, terms)."""
    if pred_list is None or "traj_cond_7d" not in batch:
        return None, {}
    if "_window_local_latent_start_token" in batch:
        window_start_tokens = batch["_window_local_latent_start_token"]
        pred_list = _splice_window_local_pred_to_prefix(
            pred_list, batch, module.device
        )
    else:
        window_start_tokens = None
    weights = {**_DEFAULT_BODY_AUX_WEIGHTS, **(ba_cfg.get("weights", {}) or {})}
    return compute_body_aux_loss(
        pred_list,
        batch["traj_cond_7d"],
        batch["traj_length"],
        module.vae,
        module.device,
        weights,
        chunk_size_tokens=getattr(module.model, "chunk_size", None),
        heading_form=ba_cfg.get("heading_form", "cosine"),
        sample_loss_mask=sample_loss_mask,
        window_start_tokens=window_start_tokens,
    )


def _collect_window_local_metrics(model_batch: dict) -> dict[str, float]:
    """Summarize window-local sampling state for training logs."""
    if not bool(model_batch.get("_window_local_traj", False)):
        return {}

    metrics: dict[str, float] = {"stream_training/enabled": 1.0}
    sample_policy = str(model_batch.get("_window_local_sample_policy", "variable_history"))
    metrics["stream_training/sample_policy_fixed_window"] = (
        1.0 if sample_policy == "fixed_window" else 0.0
    )

    starts = model_batch.get("_window_local_latent_start_token")
    if starts is not None:
        starts_t = torch.as_tensor(starts, dtype=torch.float32)
        metrics["stream_training/window_start_mean"] = float(starts_t.mean().item())
    lengths = model_batch.get("_window_local_latent_valid_len")
    if lengths is not None:
        lengths_t = torch.as_tensor(lengths, dtype=torch.float32)
        metrics["stream_training/window_len_mean"] = float(lengths_t.mean().item())
        metrics["stream_training/window_len_min"] = float(lengths_t.min().item())
        metrics["stream_training/window_len_max"] = float(lengths_t.max().item())
    traj_tokens = model_batch.get("traj_num_tokens")
    if traj_tokens is not None:
        traj_tokens_t = torch.as_tensor(traj_tokens, dtype=torch.float32)
        metrics["stream_training/traj_tokens_mean"] = float(traj_tokens_t.mean().item())
    return metrics


def _collect_window_local_rollout_metrics(
    model_batch: dict,
    plan: RolloutPlan,
) -> dict[str, float]:
    """Summarize the actual final active history selected by plan_rollout()."""
    if not bool(model_batch.get("_window_local_traj", False)):
        return {}
    final_active_end = (
        plan.start_end_indices.to(dtype=torch.long) + int(plan.effective_k) - 1
    )
    active_len = final_active_end.to(dtype=torch.float32)
    metrics = {
        "stream_training/active_history_len_mean": float(active_len.mean().item()),
        "stream_training/active_history_len_min": float(active_len.min().item()),
        "stream_training/active_history_len_max": float(active_len.max().item()),
    }
    starts = model_batch.get("_window_local_latent_start_token")
    if starts is not None:
        starts_t = torch.as_tensor(
            starts, device=final_active_end.device, dtype=torch.long
        ).view(-1)
        if starts_t.numel() == 1 and final_active_end.numel() > 1:
            starts_t = starts_t.expand_as(final_active_end)
        if starts_t.numel() == final_active_end.numel():
            abs_end = starts_t + final_active_end
            metrics["stream_training/active_abs_end_mean"] = float(
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


def _splice_window_local_pred_to_prefix(pred_list, batch, device):
    """Splice local predicted latents ``[S:E]`` back into full prefix ``[0:E]``.

    Motion-space auxiliary loss must decode with causal VAE prefix context. This
    helper preserves the original pre-window latent prefix and keeps gradients
    through the predicted local segment.
    """
    starts = batch["_window_local_latent_start_token"]
    if not torch.is_tensor(starts):
        starts = torch.as_tensor(starts, device=device, dtype=torch.long)
    else:
        starts = starts.to(device=device, dtype=torch.long)
    if starts.ndim == 0:
        starts = starts.repeat(len(pred_list))
    else:
        starts = starts.view(-1)
    if starts.numel() != len(pred_list):
        raise ValueError(
            "_window_local_latent_start_token must provide one value per pred "
            f"latent; got {starts.numel()} starts for {len(pred_list)} predictions"
        )
    token = batch["token"].to(device)
    if token.ndim != 3:
        raise ValueError(f"batch['token'] must be [B,T,D], got {tuple(token.shape)}")
    if token.shape[0] < len(pred_list):
        raise ValueError(
            "batch['token'] batch size must cover every prediction; "
            f"got token batch={token.shape[0]}, predictions={len(pred_list)}"
        )
    token_lengths = batch.get("token_length")
    if token_lengths is None:
        token_lengths = torch.full(
            (token.shape[0],), token.shape[1], device=device, dtype=torch.long
        )
    elif not torch.is_tensor(token_lengths):
        token_lengths = torch.as_tensor(token_lengths, device=device, dtype=torch.long)
    else:
        token_lengths = token_lengths.to(device=device, dtype=torch.long)
    token_lengths = token_lengths.view(-1)
    if token_lengths.numel() == 1 and len(pred_list) > 1:
        token_lengths = token_lengths.expand(len(pred_list))
    if token_lengths.numel() < len(pred_list):
        raise ValueError(
            "batch['token_length'] must cover every prediction; "
            f"got {token_lengths.numel()} lengths for {len(pred_list)} predictions"
        )
    valid_lengths = batch.get("_window_local_latent_valid_len")
    if valid_lengths is not None:
        if not torch.is_tensor(valid_lengths):
            valid_lengths = torch.as_tensor(
                valid_lengths, device=device, dtype=torch.long
            )
        else:
            valid_lengths = valid_lengths.to(device=device, dtype=torch.long)
        valid_lengths = valid_lengths.view(-1)
        if valid_lengths.numel() == 1 and len(pred_list) > 1:
            valid_lengths = valid_lengths.expand(len(pred_list))
        if valid_lengths.numel() != len(pred_list):
            raise ValueError(
                "_window_local_latent_valid_len must provide one value per pred "
                f"latent; got {valid_lengths.numel()} lengths for "
                f"{len(pred_list)} predictions"
            )
    out = []
    for i, pred_latent in enumerate(pred_list):
        start = int(starts[i].item())
        pred_latent = pred_latent.to(device=device, dtype=token.dtype)
        if start < 0:
            raise ValueError(f"window-local start token must be >= 0, got {start}")
        original_len = int(token_lengths[i].item())
        if start > original_len:
            raise ValueError(
                "window-local start token exceeds original token length; "
                f"sample={i}, start={start}, token_length={original_len}"
            )
        pred_len = int(pred_latent.shape[0])
        if valid_lengths is not None and pred_len > int(valid_lengths[i].item()):
            raise ValueError(
                "window-local prediction length exceeds window-local valid length; "
                f"sample={i}, pred_len={pred_len}, "
                f"valid_len={int(valid_lengths[i].item())}"
            )
        if start + pred_len > original_len:
            raise ValueError(
                "window-local prediction extends past original token length; "
                f"sample={i}, start={start}, pred_len={pred_len}, "
                f"token_length={original_len}"
            )
        if start == 0:
            out.append(pred_latent)
            continue
        prefix = token[i, :start, :].detach()
        out.append(torch.cat([prefix, pred_latent], dim=0))
    return out


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
