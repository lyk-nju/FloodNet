import os

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
        self.t2m_metrics = T2MMetrics(self.cfg.metrics.t2m)

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
        t2m_output = self.t2m_metrics.compute(sanity_flag=self.trainer.sanity_checking)
        for key, value in t2m_output.items():
            self.log(f"metrics/t2m_metrics/{key}", value, sync_dist=True)

    def update_test(self, batch):
        # Fix seed for reproducible test generation, but save/restore the training RNG so
        # that training noise stays i.i.d. when the training loop resumes after validation.
        import random
        _py   = random.getstate()
        _np   = np.random.get_state()
        _cpu  = torch.random.get_rng_state()
        _cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        seed_everything(self.cfg.seed)
        try:
            with self.ema.average_parameters([p for p in self.model.parameters() if p.requires_grad]):
                model_batch = batch.copy()
                model_batch["feature"] = batch["token"]
                model_batch["feature_length"] = batch["token_length"]
                if "token_text_end" in batch:
                    model_batch["feature_text_end"] = batch["token_text_end"]
                self._copy_traj_fields_to_model_batch(batch, model_batch)
                output = self.model.generate(model_batch)
            generated = output["generated"]
            text = output["text"]
            # Save motion
            generated_id = batch["name"]  # [batch_size]
            dataset_id = batch["dataset"]  # [batch_size]
            step_tag = f"step_{int(self.global_step):06d}"
    
            # Decode motion to latent space
            # NOTE: inside the to() function, if will check current_tensor.device == target_device, then run this in each loop is fine.
            for i in range(len(generated)):
                single_generated = generated[i]
                single_generated_id = generated_id[i]
                single_dataset_id = dataset_id[i]
                single_text = text[i]
                text_dir = f"{self.cfg.save_dir}/{single_dataset_id}/text/{step_tag}"
                token_dir = f"{self.cfg.save_dir}/{single_dataset_id}/token/{step_tag}"
                feature_dir = f"{self.cfg.save_dir}/{single_dataset_id}/feature/{step_tag}"
                cond_traj_dir = f"{self.cfg.save_dir}/{single_dataset_id}/traj_xz/{step_tag}"
                traj_mask_dir = f"{self.cfg.save_dir}/{single_dataset_id}/traj_mask/{step_tag}"
                frames_dir = f"{self.cfg.save_dir}/{single_dataset_id}/frames/{step_tag}"
                if "feature_text_end" in batch:
                    single_feature_text_end = batch["feature_text_end"][i]
                    frames = np.array(single_feature_text_end)
                else:
                    frames = None
                try:
                    decoded_single_generated = self.vae.decode(
                        single_generated[None, :].to(self.device)
                    )[0]
                    os.makedirs(text_dir, exist_ok=True)
                    with open(
                        f"{text_dir}/{single_generated_id}.txt",
                        "w",
                    ) as f:
                        f.write(single_text)
                    os.makedirs(token_dir, exist_ok=True)
                    np.save(
                        f"{token_dir}/{single_generated_id}.npy",
                        single_generated.float().cpu().numpy(),
                    )
                    os.makedirs(feature_dir, exist_ok=True)
                    np.save(
                        f"{feature_dir}/{single_generated_id}.npy",
                        decoded_single_generated.float().cpu().numpy(),
                    )
    
                    # Save conditioning trajectory (T,2)=[x,z] for visualization (red overlay).
                    # Prefer frame-level traj_features (T,4)=[x,z,cos,sin]; else build from root traj xyz.
                    L_feat = int(decoded_single_generated.shape[0])
                    traj_xz = None
                    if "traj_features" in batch:
                        cond = batch["traj_features"][i]
                        if torch.is_tensor(cond):
                            cond = cond.detach().cpu().numpy()
                        cond = np.asarray(cond)
                        if cond.ndim == 2 and cond.shape[1] >= 2:
                            traj_xz = cond[:L_feat, :2].astype(np.float32)
                        else:
                            rank_zero_info(
                                f"Skip traj_xz for {single_generated_id}: "
                                f"traj_features bad shape {cond.shape}"
                            )
                    if traj_xz is None and "traj" in batch:
                        tr = batch["traj"][i]
                        if torch.is_tensor(tr):
                            tr = tr.detach().cpu().numpy()
                        tr = np.asarray(tr)[:L_feat]
                        if tr.ndim == 2 and tr.shape[1] >= 3:
                            traj_xz = root_to_traj_feats(tr)[:, :2].astype(
                                np.float32
                            )
                    if traj_xz is not None:
                        os.makedirs(cond_traj_dir, exist_ok=True)
                        np.save(
                            f"{cond_traj_dir}/{single_generated_id}.npy",
                            traj_xz,
                        )
    
                    # Save traj_mask (if provided by dataset) so we can mask the root trajectory in visualization.
                    if "traj_mask" in batch:
                        traj_mask_i = batch["traj_mask"][i]
                        if torch.is_tensor(traj_mask_i):
                            traj_mask_i = traj_mask_i.detach().cpu().numpy()
                        traj_mask_i = np.asarray(traj_mask_i).reshape(-1)[:L_feat]
                        os.makedirs(traj_mask_dir, exist_ok=True)
                        np.save(
                            f"{traj_mask_dir}/{single_generated_id}.npy",
                            traj_mask_i,
                        )
                    # Save text_end if available
                    if frames is not None:
                        os.makedirs(frames_dir, exist_ok=True)
                        np.save(
                            f"{frames_dir}/{single_generated_id}.npy",
                            frames,
                        )
                except Exception as e:
                    rank_zero_info(
                        f"Error in saving motion {single_generated_id} of dataset {single_dataset_id}: {e}"
                    )
    
            return {"output": output}
        finally:
            # Restore training RNG state unconditionally (even if generate() raises).
            random.setstate(_py)
            np.random.set_state(_np)
            torch.random.set_rng_state(_cpu)
            if _cuda is not None:
                torch.cuda.set_rng_state_all(_cuda)

    def process_test_results(self):
        for dataset_id in os.listdir(self.cfg.save_dir):
            feature_root = f"{self.cfg.save_dir}/{dataset_id}/feature"
            if not os.path.exists(feature_root):
                continue
            step_tag = f"step_{int(self.global_step):06d}"
            feature_dir = f"{feature_root}/{step_tag}"
            if not os.path.exists(feature_dir):
                continue
            video_step_dir = f"{self.cfg.save_dir}/{dataset_id}/video/{step_tag}"
            composite_step_dir = (
                f"{self.cfg.save_dir}/{dataset_id}/composite/{step_tag}"
            )
            frames_dir = f"{self.cfg.save_dir}/{dataset_id}/frames/{step_tag}"
            traj_mask_dir = f"{self.cfg.save_dir}/{dataset_id}/traj_mask/{step_tag}"
            cond_traj_dir = f"{self.cfg.save_dir}/{dataset_id}/traj_xz/{step_tag}"
            text_dir = f"{self.cfg.save_dir}/{dataset_id}/text/{step_tag}"
            # render video and save
            if self.cfg.test_setting.render:
                render_video(
                    motion_dir=feature_dir,
                    save_dir=video_step_dir,
                    render_setting=self.cfg.test_setting,
                    frames_dir=frames_dir,
                    traj_mask_dir=traj_mask_dir,
                    cond_traj_dir=cond_traj_dir,
                )

                # Create composite videos
                make_composite_compare_videos(
                    result_folder=video_step_dir,
                    compare_folders=self.cfg.test_setting.get(dataset_id, {}).get(
                        "compare_folders", None
                    ),
                    compare_names=self.cfg.test_setting.get(dataset_id, {}).get(
                        "compare_names", None
                    ),
                    text_folder=text_dir,
                    save_dir=composite_step_dir,
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
                                f"{composite_step_dir}/{video_path}",
                                format="gif",
                            )
                        )
                    wandb.log(
                        {f"{dataset_id}_video": video_to_log},
                        step=self.global_step,
                    )


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
    test_dataset = instantiate(
        cfg.data.get("test_target", cfg.data.target), cfg=cfg.config, split="test"
    )
    rank_zero_info(
        f"Train dataset: {len(train_dataset) if train_dataset is not None else 0}, Val dataset: {len(val_dataset) if val_dataset is not None else 0}, Test dataset: {len(test_dataset)}"
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
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.data.test_bs,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=False,
        prefetch_factor=8,
        collate_fn=collate_fn,
    )

    # lightning module, model is inside the lightning module
    model = CustomLightningModule(cfg=cfg.config)

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

    if cfg.train:
        trainer.fit(
            model,
            train_dataloader,
            val_dataloaders=[val_dataloader, test_dataloader],
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
                dataloaders=[val_dataloader, test_dataloader],
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
