import json
import os
import random
import hashlib
from pathlib import Path

import numpy as np
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
from eval.eval_generation_metrics import (
    _average_control_metrics,
    _average_traj_metrics,
    _compute_deterministic_fwd_ctrl_loss_sample,
    _compute_omni_control_metrics,
    _compute_traj_metrics,
    _get_metric_statistics,
)
from utils.motion_process import extract_root_trajectory_263_torch
from utils.initialize import (
    compare_statedict_and_parameters,
    get_function,
    get_shared_run_time,
    instantiate,
    load_config,
    save_config_and_codes,
)
from utils.lightning_module import BasicLightningModule
from utils.traj_batch import root_to_traj_feats
from utils.visualize import (  # evaluate_video
    make_composite_compare_videos,
    render_video,
)

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomLightningModule(BasicLightningModule):
    def __init__(self, cfg):
        self._inline_eval_seen = {}
        super().__init__(cfg)

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

    @staticmethod
    def _flatten_summary_metrics(summary: dict, prefix: str) -> dict:
        flat_metrics = {}
        for key, value in summary.items():
            log_key = f"{prefix}/{key}"
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    if item is not None:
                        flat_metrics[f"{log_key}/slot_{idx}"] = float(item)
            else:
                flat_metrics[log_key] = float(value)
        return flat_metrics

    @staticmethod
    def _stable_eval_seed(base_seed: int, probe_tag: str, sample_name: str, run_idx: int) -> int:
        digest = hashlib.md5(f"{probe_tag}:{sample_name}:{run_idx}".encode("utf-8")).hexdigest()
        offset = int(digest[:8], 16)
        return int(base_seed) + offset

    @staticmethod
    def _seed_eval_locally(seed: int, device: torch.device | str):
        random.seed(seed)
        np.random.seed(seed % (2**32))
        gen = torch.Generator()
        gen.manual_seed(int(seed))
        torch.random.set_rng_state(gen.get_state())
        if torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed))

    @staticmethod
    def _slice_single_sample_batch(batch, sample_idx: int):
        sample_batch = {}
        batch_size = len(batch["name"])
        for key, value in batch.items():
            if torch.is_tensor(value):
                if value.ndim > 0 and value.shape[0] == batch_size:
                    sample_batch[key] = value[sample_idx : sample_idx + 1]
                else:
                    sample_batch[key] = value
            elif isinstance(value, list):
                if len(value) == batch_size:
                    sample_batch[key] = [value[sample_idx]]
                else:
                    sample_batch[key] = value
            else:
                sample_batch[key] = value
        return sample_batch

    @staticmethod
    def _compute_control_loss_xz(
        pred_list,
        traj,
        traj_mask,
        traj_length,
        vae,
        device,
        train_mode: int = 3,
        chunk_size_tokens: int | None = None,
        token_to_frame: int = 4,
    ):
        """
        XZ-plane trajectory control loss.  Behaviour is selected by train_mode:

          Mode 1 — active window, absolute coords, NO detach
                   Loss only on active window frames; gradient flows to ALL tokens
                   via cumsum (position[t] = Σvel[0..t]).
          Mode 2 — active window, absolute coords, detach past tokens
                   Past latents are detached before VAE decode; gradient is
                   confined to active window tokens, but absolute position has
                   a fixed offset from the detached past (old '20260419' style).
          Mode 3 — full sequence, absolute coords, NO detach  [DEFAULT]
                   Every frame compared; each token gets balanced gradient from
                   all future positions it affects.  Proven to converge (20260402).
          Mode 4 — full sequence, absolute coords, detach past tokens
                   Full comparison coverage but gradient only flows through the
                   active window tokens (past frames contribute to loss value,
                   not to gradient).
          Mode 5 — active window, relative displacement, pred anchor
                   Both shifted by pred's first frame; loss = |e[t]-e[0]|².
                   A fixed absolute offset gives zero loss (known limitation).
          Mode 6 — active window, relative displacement, GT anchor
                   pred shifted by pred's first frame, gt shifted by gt's first
                   frame; loss = |pred_rel[t]-gt_rel[t]|² — true displacement
                   error, decoupled from absolute position.
        """
        # Derive booleans from mode for readability
        use_active_window = train_mode in (1, 2, 5, 6) # slice to active window for comparison
        detach_past       = train_mode in (2, 4)        # detach past latents before decode
        relative_disp     = train_mode in (5, 6)        # subtract anchor frame
        relative_disp_gt_anchor = train_mode == 6       # use GT anchor (fixes Mode 5 bug)

        loss_control = 0.0
        n_valid = 0.0
        for i in range(len(pred_list)):
            pred_latent_full = pred_list[i].to(device)  # (T_token, z_dim)
            t_tok = pred_latent_full.size(0)

            # Active-window token/frame bounds (always computed; used conditionally)
            if chunk_size_tokens is not None and t_tok > chunk_size_tokens:
                start_tok = t_tok - chunk_size_tokens
                start_f   = 0 if start_tok == 0 else 4 * start_tok - 3
                end_f     = t_tok * token_to_frame
            else:
                start_tok = 0
                start_f   = 0
                end_f     = None

            # Optionally detach past latents so gradient stays in active window
            if detach_past and start_tok > 0:
                latent_for_decode = torch.cat(
                    [pred_latent_full[:start_tok].detach(), pred_latent_full[start_tok:]], dim=0
                )
            else:
                latent_for_decode = pred_latent_full

            decoded   = vae.decode(latent_for_decode.unsqueeze(0))[0].float()
            L_motion  = decoded.size(0)
            L_gt_total = min(int(traj_length[i].item()), traj.shape[1])

            # Frame slice to compare
            if use_active_window and end_f is not None:
                pred_sl = slice(min(start_f, L_motion),   min(end_f, L_motion))
                gt_sl   = slice(min(start_f, L_gt_total), min(end_f, L_gt_total))
            else:
                pred_sl = slice(0, L_motion)
                gt_sl   = slice(0, L_gt_total)

            L = min(pred_sl.stop - pred_sl.start, gt_sl.stop - gt_sl.start)
            if L <= 0:
                continue

            pred_traj_full = extract_root_trajectory_263_torch(decoded.unsqueeze(0))
            pred_traj = pred_traj_full[:, pred_sl, :][:, :L, :]
            gt_traj   = traj[i, gt_sl, :][:L].unsqueeze(0).to(pred_traj.device, dtype=pred_traj.dtype)
            mask      = traj_mask[i, gt_sl][:L].unsqueeze(0).to(pred_traj.device, dtype=pred_traj.dtype)

            pred_xz = pred_traj[..., [0, 2]]
            gt_xz   = gt_traj[...,  [0, 2]]

            if relative_disp:
                if relative_disp_gt_anchor:
                    # Mode 6: both pred and gt anchored to GT first frame.
                    # loss = ||(pred[t]-pred[0]) - (gt[t]-gt[0])||^2
                    # which equals ||pred_rel[t] - gt_rel[t]||^2 — true relative
                    # displacement error, decoupled from absolute position.
                    gt_anchor = gt_xz[:, 0:1, :].detach()
                    pred_xz   = pred_xz - pred_xz[:, 0:1, :].detach()
                    gt_xz     = gt_xz   - gt_anchor
                else:
                    # Mode 5: pred anchored to its own first frame (pred anchor).
                    # loss = ||(e[t] - e[0])||^2 — measures error change, not error
                    # itself; a fixed offset has zero loss (known design limitation).
                    anchor  = pred_xz[:, 0:1, :].detach()
                    pred_xz = pred_xz - anchor
                    gt_xz   = gt_xz   - gt_xz[:, 0:1, :]

            sq_err = ((pred_xz - gt_xz) ** 2).sum(dim=-1)
            loss_control = loss_control + (mask * sq_err).sum()
            n_valid += mask.sum().item()

        if n_valid <= 0:
            return None
        return loss_control / n_valid

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

    def _step(self, batch, is_training=True):
        # Create a copy and replace motion fields with token fields
        model_batch = batch.copy()
        model_batch["feature"] = batch["token"]
        model_batch["feature_length"] = batch["token_length"]
        if "token_text_end" in batch:
            model_batch["feature_text_end"] = batch["token_text_end"]
        self._copy_traj_fields_to_model_batch(batch, model_batch)
        out = self.model(model_batch)

        # MotionLCM-style control loss: explicit trajectory alignment in motion space
        if "control_aux" in out and "traj" in batch:
            control_weight = self.cfg.model.params.get("control_loss_weight", 1.0)
            if control_weight > 0:
                pred_list = out["control_aux"]["pred_x0_latent_list"]
                traj = batch["traj"]
                traj_mask = batch["traj_mask"]
                traj_length = batch["traj_length"]
                train_mode = self.cfg.get("control_loss_train_mode", 3)
                chunk_size_tokens = getattr(self.model, "chunk_size", None)
                loss_control = self._compute_control_loss_xz(
                    pred_list,
                    traj,
                    traj_mask,
                    traj_length,
                    self.vae,
                    self.device,
                    train_mode=train_mode,
                    chunk_size_tokens=chunk_size_tokens,
                )
                if loss_control is not None:
                    out["total"] = out["total"] + control_weight * loss_control
                    out["control"] = loss_control

        if "control_aux" in out:
            del out["control_aux"]
        return out

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
        # Fix seed for reproducible test generation, but save/restore the training RNG so
        # that training noise stays i.i.d. when the training loop resumes after validation.
        _py   = random.getstate()
        _np   = np.random.get_state()
        _cpu  = torch.random.get_rng_state()
        _cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        eval_cfg = self._get_generation_eval_cfg()
        eval_num_runs = max(eval_cfg["num_runs"], 1)
        eval_seg_size = eval_cfg["seg_size"]
        do_eval_metrics = eval_cfg["enabled"] and "traj" in batch and "traj_mask" in batch
        generation_num_runs = eval_num_runs if do_eval_metrics else 1
        probe_tag = self._resolve_test_probe_tag(test_loader_idx)
        step_tag = f"step_{int(self.global_step):06d}"
        try:
            local_payloads = []
            for sample_idx in range(len(batch["name"])):
                sample_batch = self._slice_single_sample_batch(batch, sample_idx)
                sample_name = sample_batch["name"][0]
                sample_dataset_id = sample_batch["dataset"][0]
                sample_text = ""
                token_run0 = None
                feature_run0 = None
                traj_xz = None
                traj_mask = None
                frames = None
                traj_runs = []
                control_runs = []
                fwd_stat = None

                if "feature_text_end" in sample_batch:
                    frames = np.asarray(sample_batch["feature_text_end"][0])

                if do_eval_metrics and eval_cfg["forward_ctrl_loss"] and "traj" in sample_batch:
                    try:
                        fwd_stat = _compute_deterministic_fwd_ctrl_loss_sample(
                            model=self.model,
                            sample_batch=sample_batch,
                            vae=self.vae,
                            device=self.device,
                            train_mode=int(self.cfg.get("control_loss_train_mode", 3)),
                            chunk_size_tokens=getattr(self.model, "chunk_size", None),
                            window_mode=eval_cfg["forward_ctrl_window_mode"],
                        )
                    except Exception as e:
                        rank_zero_info(
                            f"[inline fwd_ctrl_loss] sample={sample_name} deterministic eval failed: {e}"
                        )

                for run_idx in range(generation_num_runs):
                    sample_seed = self._stable_eval_seed(
                        self.cfg.seed, probe_tag, sample_name, run_idx
                    )
                    self._seed_eval_locally(sample_seed, self.device)
                    with self.ema.average_parameters([p for p in self.model.parameters() if p.requires_grad]):
                        model_batch = sample_batch.copy()
                        model_batch["feature"] = sample_batch["token"]
                        model_batch["feature_length"] = sample_batch["token_length"]
                        if "token_text_end" in sample_batch:
                            model_batch["feature_text_end"] = sample_batch["token_text_end"]
                        self._copy_traj_fields_to_model_batch(sample_batch, model_batch)
                        output = self.model.generate(model_batch)

                    single_generated = output["generated"][0]
                    decoded_single_generated = self.vae.decode(
                        single_generated[None, :].to(self.device)
                    )[0].float().detach()

                    if run_idx == 0:
                        sample_text = output["text"][0]
                        token_run0 = single_generated.float().cpu().numpy()
                        feature_run0 = decoded_single_generated.cpu().numpy()

                        L_feat = int(decoded_single_generated.shape[0])
                        if "traj_features" in sample_batch:
                            cond = sample_batch["traj_features"][0]
                            if torch.is_tensor(cond):
                                cond = cond.detach().cpu().numpy()
                            cond = np.asarray(cond)
                            if cond.ndim == 2 and cond.shape[1] >= 2:
                                traj_xz = cond[:L_feat, :2].astype(np.float32)
                        if traj_xz is None and "traj" in sample_batch:
                            tr = sample_batch["traj"][0]
                            if torch.is_tensor(tr):
                                tr = tr.detach().cpu().numpy()
                            tr = np.asarray(tr)[:L_feat]
                            if tr.ndim == 2 and tr.shape[1] >= 3:
                                traj_xz = root_to_traj_feats(tr)[:, :2].astype(np.float32)
                        if "traj_mask" in sample_batch:
                            traj_mask_i = sample_batch["traj_mask"][0]
                            if torch.is_tensor(traj_mask_i):
                                traj_mask_i = traj_mask_i.detach().cpu().numpy()
                            traj_mask = np.asarray(traj_mask_i).reshape(-1)[:L_feat]

                    if do_eval_metrics:
                        traj_runs.append(
                            _compute_traj_metrics(
                                decoded_single_generated, sample_batch, 0, seg_size=eval_seg_size
                            )
                        )
                        control_runs.append(
                            _compute_omni_control_metrics(decoded_single_generated, sample_batch, 0)
                        )

                rec = None
                if do_eval_metrics:
                    rec = {"name": sample_name, "num_runs": eval_num_runs, "probe_tag": probe_tag}
                    if traj_runs:
                        rec.update(_average_traj_metrics(traj_runs))
                    if control_runs:
                        rec.update(_average_control_metrics(control_runs))
                    if fwd_stat is not None:
                        rec["fwd_ctrl_loss"] = fwd_stat.get("loss", float("nan"))
                        rec["fwd_ctrl_loss_std"] = fwd_stat.get("loss_std", float("nan"))
                        rec["fwd_n_valid"] = fwd_stat.get("n_valid", float("nan"))
                        rec["fwd_win_len"] = fwd_stat.get("window_len", float("nan"))
                        rec["fwd_num_windows"] = fwd_stat.get("num_windows", 0)
                    rec["_traj_runs"] = traj_runs
                    rec["_control_runs"] = control_runs

                local_payloads.append(
                    {
                        "name": sample_name,
                        "dataset_id": sample_dataset_id,
                        "text": sample_text,
                        "token": token_run0,
                        "feature": feature_run0,
                        "traj_xz": traj_xz,
                        "traj_mask": traj_mask,
                        "frames": frames,
                        "record": rec,
                    }
                )

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                gathered_payloads = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(gathered_payloads, local_payloads)
                all_payloads = [
                    payload
                    for rank_payloads in gathered_payloads
                    for payload in (rank_payloads or [])
                ]
            else:
                all_payloads = local_payloads

            if self.trainer.global_rank == 0:
                seen = self._inline_eval_seen.setdefault((probe_tag, step_tag), set())
                for payload in all_payloads:
                    single_generated_id = payload["name"]
                    single_dataset_id = payload["dataset_id"]
                    dedupe_key = (single_dataset_id, single_generated_id)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)

                    text_dir = f"{self.cfg.save_dir}/{single_dataset_id}/text/{probe_tag}/{step_tag}"
                    token_dir = f"{self.cfg.save_dir}/{single_dataset_id}/token/{probe_tag}/{step_tag}"
                    feature_dir = f"{self.cfg.save_dir}/{single_dataset_id}/feature/{probe_tag}/{step_tag}"
                    cond_traj_dir = f"{self.cfg.save_dir}/{single_dataset_id}/traj_xz/{probe_tag}/{step_tag}"
                    traj_mask_dir = f"{self.cfg.save_dir}/{single_dataset_id}/traj_mask/{probe_tag}/{step_tag}"
                    frames_dir = f"{self.cfg.save_dir}/{single_dataset_id}/frames/{probe_tag}/{step_tag}"
                    metrics_dir = f"{self.cfg.save_dir}/{single_dataset_id}/metrics/{probe_tag}/{step_tag}"

                    try:
                        os.makedirs(text_dir, exist_ok=True)
                        with open(f"{text_dir}/{single_generated_id}.txt", "w") as f:
                            f.write(payload["text"])
                        os.makedirs(token_dir, exist_ok=True)
                        np.save(f"{token_dir}/{single_generated_id}.npy", payload["token"])
                        os.makedirs(feature_dir, exist_ok=True)
                        np.save(f"{feature_dir}/{single_generated_id}.npy", payload["feature"])

                        if payload["traj_xz"] is not None:
                            os.makedirs(cond_traj_dir, exist_ok=True)
                            np.save(f"{cond_traj_dir}/{single_generated_id}.npy", payload["traj_xz"])
                        if payload["traj_mask"] is not None:
                            os.makedirs(traj_mask_dir, exist_ok=True)
                            np.save(f"{traj_mask_dir}/{single_generated_id}.npy", payload["traj_mask"])
                        if payload["frames"] is not None:
                            os.makedirs(frames_dir, exist_ok=True)
                            np.save(f"{frames_dir}/{single_generated_id}.npy", payload["frames"])
                        if payload["record"] is not None:
                            os.makedirs(metrics_dir, exist_ok=True)
                            with open(f"{metrics_dir}/{single_generated_id}.json", "w") as f:
                                json.dump(payload["record"], f, indent=2)
                    except Exception as e:
                        rank_zero_info(
                            f"Error in saving motion {single_generated_id} of dataset {single_dataset_id}: {e}"
                        )

            return {"output": None}
        finally:
            # Restore training RNG state unconditionally (even if generate() raises).
            random.setstate(_py)
            np.random.set_state(_np)
            torch.random.set_rng_state(_cpu)
            if _cuda is not None:
                torch.cuda.set_rng_state_all(_cuda)

    def process_test_results(self):
        for dataset_id in os.listdir(self.cfg.save_dir):
            feature_root = Path(self.cfg.save_dir) / dataset_id / "feature"
            if not os.path.exists(feature_root):
                continue
            step_tag = f"step_{int(self.global_step):06d}"
            for probe_tag in self._get_test_probe_tags():
                feature_dir = feature_root / probe_tag / step_tag
                if not feature_dir.exists():
                    continue
                metrics_dir = Path(self.cfg.save_dir) / dataset_id / "metrics" / probe_tag / step_tag
                video_step_dir = Path(self.cfg.save_dir) / dataset_id / "video" / probe_tag / step_tag
                composite_step_dir = Path(self.cfg.save_dir) / dataset_id / "composite" / probe_tag / step_tag
                frames_dir = Path(self.cfg.save_dir) / dataset_id / "frames" / probe_tag / step_tag
                traj_mask_dir = Path(self.cfg.save_dir) / dataset_id / "traj_mask" / probe_tag / step_tag
                cond_traj_dir = Path(self.cfg.save_dir) / dataset_id / "traj_xz" / probe_tag / step_tag
                text_dir = Path(self.cfg.save_dir) / dataset_id / "text" / probe_tag / step_tag
                # render video and save
                if self.cfg.test_setting.render:
                    render_video(
                        motion_dir=str(feature_dir),
                        save_dir=str(video_step_dir),
                        render_setting=self.cfg.test_setting,
                        frames_dir=str(frames_dir),
                        traj_mask_dir=str(traj_mask_dir),
                        cond_traj_dir=str(cond_traj_dir),
                    )

                    # Create composite videos
                    make_composite_compare_videos(
                        result_folder=str(video_step_dir),
                        compare_folders=self.cfg.test_setting.get(dataset_id, {}).get(
                            "compare_folders", None
                        ),
                        compare_names=self.cfg.test_setting.get(dataset_id, {}).get(
                            "compare_names", None
                        ),
                        text_folder=str(text_dir),
                        save_dir=str(composite_step_dir),
                    )

                    # wandb log video
                    if (
                        not self.cfg.debug
                        and self.logger is not None
                        and isinstance(self.logger, WandbLogger)
                    ):
                        video_to_log = []
                        for video_path in sorted(os.listdir(composite_step_dir)):
                            video_to_log.append(
                                wandb.Video(
                                    str(composite_step_dir / video_path),
                                    format="gif",
                                )
                            )
                        wandb.log(
                            {f"{dataset_id}_{probe_tag}_video": video_to_log},
                            step=self.global_step,
                        )

                if not metrics_dir.exists():
                    continue
                sample_records = []
                for metric_file in sorted(os.listdir(metrics_dir)):
                    if not metric_file.endswith(".json"):
                        continue
                    with open(metrics_dir / metric_file, "r") as f:
                        sample_records.append(json.load(f))
                if not sample_records:
                    continue

                summary = {}
                valid_traj = [r for r in sample_records if "ade" in r and r["ade"] == r["ade"]]
                if valid_traj:
                    ades = [r["ade"] for r in valid_traj]
                    fdes = [r["fde"] for r in valid_traj]
                    mses = [r["mse"] for r in valid_traj]
                    summary["traj/ADE_mean"] = float(np.mean(ades))
                    summary["traj/ADE_std"] = float(np.std(ades))
                    summary["traj/FDE_mean"] = float(np.mean(fdes))
                    summary["traj/FDE_std"] = float(np.std(fdes))
                    summary["traj/MSE_mean"] = float(np.mean(mses))
                    summary["traj/MSE_std"] = float(np.std(mses))
                    summary["traj/n_samples"] = len(valid_traj)

                    max_segs = max(len(r.get("seg_mse", [])) for r in valid_traj)
                    seg_means = []
                    for s in range(max_segs):
                        vals = [r["seg_mse"][s] for r in valid_traj
                                if s < len(r.get("seg_mse", [])) and r["seg_mse"][s] is not None]
                        seg_means.append(float(np.mean(vals)) if vals else None)
                    summary["traj/seg_mse_per_slot"] = seg_means

                    max_pfx = max(len(r.get("prefix_mse", [])) for r in valid_traj)
                    pfx_means = []
                    for s in range(max_pfx):
                        vals = [r["prefix_mse"][s] for r in valid_traj
                                if s < len(r.get("prefix_mse", [])) and r["prefix_mse"][s] is not None]
                        pfx_means.append(float(np.mean(vals)) if vals else None)
                    summary["traj/prefix_mse_per_slot"] = pfx_means

                    jitter_vals = [r["traj_jitter"] for r in valid_traj
                                   if "traj_jitter" in r and r["traj_jitter"] == r["traj_jitter"]]
                    if jitter_vals:
                        summary["traj/jitter_mean"] = float(np.mean(jitter_vals))
                        summary["traj/jitter_std"] = float(np.std(jitter_vals))

                    path_arc_ades = [r["path_arc_ade"] for r in valid_traj
                                     if "path_arc_ade" in r and r["path_arc_ade"] == r["path_arc_ade"]]
                    if path_arc_ades:
                        summary["path/arc_ADE_mean"] = float(np.mean(path_arc_ades))
                        summary["path/arc_ADE_std"] = float(np.std(path_arc_ades))

                    path_chamfers = [r["path_chamfer"] for r in valid_traj
                                     if "path_chamfer" in r and r["path_chamfer"] == r["path_chamfer"]]
                    if path_chamfers:
                        summary["path/chamfer_mean"] = float(np.mean(path_chamfers))
                        summary["path/chamfer_std"] = float(np.std(path_chamfers))

                    fwd_vals = [r["fwd_ctrl_loss"] for r in valid_traj
                                if "fwd_ctrl_loss" in r and r["fwd_ctrl_loss"] == r["fwd_ctrl_loss"]]
                    if fwd_vals:
                        summary["traj/fwd_ctrl_loss_mean"] = float(np.mean(fwd_vals))
                        summary["traj/fwd_ctrl_loss_std"] = float(np.std(fwd_vals))
                        fwd_run_std_vals = [r["fwd_ctrl_loss_std"] for r in valid_traj
                                            if "fwd_ctrl_loss_std" in r and r["fwd_ctrl_loss_std"] == r["fwd_ctrl_loss_std"]]
                        if fwd_run_std_vals:
                            summary["traj/fwd_ctrl_loss_run_std_mean"] = float(np.mean(fwd_run_std_vals))

                control_metric_keys = (
                    "control_l2_dist",
                    "skating_ratio",
                    "traj_fail_20cm",
                    "traj_fail_50cm",
                    "kps_fail_20cm",
                    "kps_fail_50cm",
                    "kps_mean_err_m",
                )
                control_name_map = {
                    "control_l2_dist": "Control_L2_dist",
                    "skating_ratio": "Skating_Ratio",
                    "traj_fail_20cm": "traj_fail_20cm",
                    "traj_fail_50cm": "traj_fail_50cm",
                    "kps_fail_20cm": "kps_fail_20cm",
                    "kps_fail_50cm": "kps_fail_50cm",
                    "kps_mean_err_m": "kps_mean_err_m",
                }
                max_runs = max(len(r.get("_control_runs", [])) for r in sample_records)
                for key in control_metric_keys:
                    per_run_vals = []
                    for run_idx in range(max_runs):
                        vals = []
                        for r in sample_records:
                            control_runs = r.get("_control_runs", [])
                            if run_idx < len(control_runs):
                                v = control_runs[run_idx].get(key, float("nan"))
                                if v == v:
                                    vals.append(v)
                        if vals:
                            per_run_vals.append(float(np.mean(vals)))
                    if per_run_vals:
                        mean, std, conf = _get_metric_statistics(np.asarray(per_run_vals, dtype=np.float64), len(per_run_vals))
                        out_key = control_name_map[key]
                        summary[f"control/{out_key}_mean"] = float(mean)
                        summary[f"control/{out_key}_std"] = float(std)
                        summary[f"control/{out_key}_conf_interval"] = float(conf)
                        summary[f"control/{out_key}_num_runs"] = int(len(per_run_vals))

                summary_path = metrics_dir / "summary.json"
                with open(summary_path, "w") as f:
                    json.dump({"summary": summary, "samples": sample_records}, f, indent=2)

                rank_zero_info(
                    f"[eval][{dataset_id}][{probe_tag}][{step_tag}] "
                    f"ADE={summary.get('traj/ADE_mean', float('nan')):.4f} "
                    f"FDE={summary.get('traj/FDE_mean', float('nan')):.4f} "
                    f"PathArc={summary.get('path/arc_ADE_mean', float('nan')):.4f} "
                    f"ControlL2={summary.get('control/Control_L2_dist_mean', float('nan')):.4f}"
                )

                flat_metrics = {}
                for key, value in summary.items():
                    log_key = f"eval/{probe_tag}/{dataset_id}/{key}"
                    if isinstance(value, list):
                        for idx, item in enumerate(value):
                            if item is not None:
                                flat_metrics[f"{log_key}/slot_{idx}"] = float(item)
                    else:
                        flat_metrics[log_key] = float(value)
                if flat_metrics and self.logger is not None:
                    self.logger.log_metrics(flat_metrics, step=self.global_step)


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

    def build_test_probe_loaders():
        probe_cfg = cfg.data.get("test_probe_meta_paths", None)
        probe_specs = []
        if probe_cfg:
            for probe_tag, meta_paths in probe_cfg.items():
                probe_specs.append((str(probe_tag), list(meta_paths)))
        else:
            probe_specs.append(("test", list(cfg.data.test_meta_paths)))

        loaders = []
        tags = []
        total_probe_samples = 0
        test_target = cfg.data.get("test_target", cfg.data.target)
        for probe_tag, meta_paths in probe_specs:
            probe_cfg_obj = OmegaConf.create(OmegaConf.to_container(cfg.config, resolve=False))
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

    test_probe_loaders, test_loader_tags, total_probe_samples = build_test_probe_loaders()
    rank_zero_info(
        f"Train dataset: {len(train_dataset) if train_dataset is not None else 0}, "
        f"Val dataset: {len(val_dataset) if val_dataset is not None else 0}, "
        f"Test probe samples: {total_probe_samples}"
    )

    # lightning module, model is inside the lightning module
    model = CustomLightningModule(cfg=cfg.config)
    model.test_loader_tags = test_loader_tags

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.save_dir,
        filename="step_{step}",
        every_n_train_steps=cfg.validation.save_every_n_steps,
        save_top_k=cfg.validation.save_top_k,
        monitor="step",
        mode="max",
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

    trainer = Trainer(
        **cfg.trainer,
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
