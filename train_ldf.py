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
from utils.traj_batch import path_heading_features_from_root_xyz
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
        chunk_size_tokens: int | None = None,
        token_to_frame: int = 4,
        decode_mode: str = "chunk",
        align_start_xz: bool = False,
    ):
        """
        Motion-space control loss on ground plane only.

        - Decode each predicted x0 latent to 263D motion.
        - Extract root trajectory and compare only xz components against GT `traj`.
        - Normalize by valid masked frame count.
        """
        loss_control = 0.0
        n_valid = 0.0
        for i in range(len(pred_list)):
            pred_latent_full = pred_list[i].to(device)  # (T_token, z_dim)
            t_tok = pred_latent_full.size(0)
            if chunk_size_tokens is not None and t_tok > chunk_size_tokens:
                # Align with diffusion-forcing active window (last chunk_size tokens).
                pred_latent = (
                    pred_latent_full
                    if str(decode_mode).lower() == "full"
                    else pred_latent_full[-chunk_size_tokens:]
                )
                # Frame index range in GT corresponding to the same token window.
                start_f = (t_tok - chunk_size_tokens) * token_to_frame
                end_f = t_tok * token_to_frame
            else:
                pred_latent = pred_latent_full
                start_f = 0
                end_f = None

            decoded = vae.decode(pred_latent.unsqueeze(0))[0].float()
            L_motion = decoded.size(0)
            L_gt_total = min(int(traj_length[i].item()), traj.shape[1])
            if end_f is None:
                gt_slice = slice(0, L_gt_total)
            else:
                gt_slice = slice(min(start_f, L_gt_total), min(end_f, L_gt_total))
            L_gt = gt_slice.stop - gt_slice.start
            L = min(L_motion, L_gt)
            if L <= 0:
                continue
            pred_traj = extract_root_trajectory_263_torch(decoded[:L].unsqueeze(0))
            gt_traj = traj[i, gt_slice, :][:L].unsqueeze(0).to(
                pred_traj.device, dtype=pred_traj.dtype
            )
            mask = traj_mask[i, gt_slice][:L].unsqueeze(0).to(
                pred_traj.device, dtype=pred_traj.dtype
            )
            # Ground-plane only: align with xz trajectory conditioning (ignore root height y).
            pred_xz = pred_traj[..., [0, 2]]
            gt_xz = gt_traj[..., [0, 2]]
            # Optional: align window start to remove arbitrary translation when decoding only a chunk.
            if align_start_xz and pred_xz.size(1) > 0:
                pred_xz = pred_xz - pred_xz[:, :1, :] + gt_xz[:, :1, :]
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
            self.vae_ema = ExponentialMovingAverage(
                self.vae.parameters(), decay=self.cfg.test_vae.ema_decay
            )
            self.vae_ema.load_state_dict(vae_ckpt["ema_state"])
            self.vae_ema.copy_to(self.vae.parameters())
            rank_zero_info(f"Loaded VAE model from {self.cfg.test_vae_ckpt} with EMA")
        else:
            self.vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
            rank_zero_info(f"Loaded VAE model from {self.cfg.test_vae_ckpt} w/o EMA")

        compare_statedict_and_parameters(
            state_dict=self.vae.state_dict(),
            named_parameters=self.vae.named_parameters(),
            named_buffers=self.vae.named_buffers(),
        )

        # metric models
        self.recover_dim = self.cfg.metrics.dim
        self.t2m_metrics = T2MMetrics(self.cfg.metrics.t2m)

    def on_load_checkpoint(self, checkpoint):
        use_traj_cond = self.cfg.model.params.get("use_traj_cond", False)
        use_controlnet_traj = self.cfg.model.params.get("use_controlnet_traj", False)
        strict = not (use_traj_cond or use_controlnet_traj)
        result = self.model.load_state_dict(checkpoint["state_dict"], strict=strict)
        has_new_cond_params = (use_traj_cond or use_controlnet_traj) and result.missing_keys
        if (use_traj_cond or use_controlnet_traj) and not strict:
            if result.missing_keys:
                rank_zero_info(
                    "Loaded pretrained LDF with strict=False. "
                    f"Missing keys (new cond modules init from scratch): {result.missing_keys}"
                )
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
                self.model.parameters(), decay=self.cfg.model.ema_decay
            )
            rank_zero_info("init ema from current model weights")
        # When has_new_traj_params, optimizer/scheduler param groups mismatch -> skip restore
        # Set to empty lists (don't pop) so Lightning passes "key exists" check but restores nothing
        if has_new_cond_params:
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []
            rank_zero_info("Skip restoring optimizer/scheduler (new cond params)")
        compare_statedict_and_parameters(
            state_dict=self.model.state_dict(),
            named_parameters=self.model.named_parameters(),
            named_buffers=self.model.named_buffers(),
        )

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
                chunk_size_tokens = self.cfg.model.params.get("chunk_size", None)
                decode_mode = self.cfg.model.params.get("control_loss_decode_mode", "chunk")
                align_start_xz = bool(
                    self.cfg.model.params.get("control_loss_align_start_xz", False)
                )
                loss_control = self._compute_control_loss_xz(
                    pred_list,
                    traj,
                    traj_mask,
                    traj_length,
                    self.vae,
                    self.device,
                    chunk_size_tokens=chunk_size_tokens,
                    decode_mode=decode_mode,
                    align_start_xz=align_start_xz,
                )
                if loss_control is not None:
                    out["total"] = out["total"] + control_weight * loss_control
                    out["control"] = loss_control

        if "control_aux" in out:
            del out["control_aux"]
        return out

    def update_metrics(self, batch):
        with self.ema.average_parameters(self.model.parameters()):
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
        # Fix seed before each test generation so diffusion noise is reproducible.
        seed_everything(self.cfg.seed)
        
        with self.ema.average_parameters(self.model.parameters()):
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

        # Decode motion to latent space
        # NOTE: inside the to() function, if will check current_tensor.device == target_device, then run this in each loop is fine.
        for i in range(len(generated)):
            single_generated = generated[i]
            single_generated_id = generated_id[i]
            single_dataset_id = dataset_id[i]
            single_text = text[i]
            if "feature_text_end" in batch:
                single_feature_text_end = batch["feature_text_end"][i]
                frames = np.array(single_feature_text_end)
            else:
                frames = None
            try:
                decoded_single_generated = self.vae.decode(
                    single_generated[None, :].to(self.device)
                )[0]
                os.makedirs(
                    f"{self.cfg.save_dir}/{single_dataset_id}/text", exist_ok=True
                )
                with open(
                    f"{self.cfg.save_dir}/{single_dataset_id}/text/{single_generated_id}.txt",
                    "w",
                ) as f:
                    f.write(single_text)
                os.makedirs(
                    f"{self.cfg.save_dir}/{single_dataset_id}/token",
                    exist_ok=True,
                )
                np.save(
                    f"{self.cfg.save_dir}/{single_dataset_id}/token/{single_generated_id}.npy",
                    single_generated.float().cpu().numpy(),
                )
                os.makedirs(
                    f"{self.cfg.save_dir}/{single_dataset_id}/feature",
                    exist_ok=True,
                )
                np.save(
                    f"{self.cfg.save_dir}/{single_dataset_id}/feature/{single_generated_id}.npy",
                    decoded_single_generated.float().cpu().numpy(),
                )

                # Save conditioning trajectory (T,2)=[x,z] for visualization (red overlay).
                # Prefer frame-level traj_features (T,4)=[x,z,cos,sin]; else build from root traj xyz.
                L_feat = int(decoded_single_generated.shape[0])
                cond_traj = None
                if "traj_features" in batch:
                    cond = batch["traj_features"][i]
                    if torch.is_tensor(cond):
                        cond = cond.detach().cpu().numpy()
                    cond = np.asarray(cond)
                    if cond.ndim == 2 and cond.shape[1] >= 2:
                        cond_traj = cond[:L_feat, :2].astype(np.float32)
                    else:
                        rank_zero_info(
                            f"Skip cond_traj for {single_generated_id}: "
                            f"traj_features bad shape {cond.shape}"
                        )
                if cond_traj is None and "traj" in batch:
                    tr = batch["traj"][i]
                    if torch.is_tensor(tr):
                        tr = tr.detach().cpu().numpy()
                    tr = np.asarray(tr)[:L_feat]
                    if tr.ndim == 2 and tr.shape[1] >= 3:
                        cond_traj = path_heading_features_from_root_xyz(tr)[:, :2].astype(
                            np.float32
                        )
                if cond_traj is not None:
                    os.makedirs(
                        f"{self.cfg.save_dir}/{single_dataset_id}/cond_traj",
                        exist_ok=True,
                    )
                    np.save(
                        f"{self.cfg.save_dir}/{single_dataset_id}/cond_traj/{single_generated_id}.npy",
                        cond_traj,
                    )

                # Save traj_mask (if provided by dataset) so we can mask the root trajectory in visualization.
                if "traj_mask" in batch:
                    traj_mask_i = batch["traj_mask"][i]
                    if torch.is_tensor(traj_mask_i):
                        traj_mask_i = traj_mask_i.detach().cpu().numpy()
                    traj_mask_i = np.asarray(traj_mask_i).reshape(-1)[:L_feat]
                    os.makedirs(
                        f"{self.cfg.save_dir}/{single_dataset_id}/traj_mask",
                        exist_ok=True,
                    )
                    np.save(
                        f"{self.cfg.save_dir}/{single_dataset_id}/traj_mask/{single_generated_id}.npy",
                        traj_mask_i,
                    )
                # Save text_end if available
                if frames is not None:
                    os.makedirs(
                        f"{self.cfg.save_dir}/{single_dataset_id}/frames", exist_ok=True
                    )
                    np.save(
                        f"{self.cfg.save_dir}/{single_dataset_id}/frames/{single_generated_id}.npy",
                        frames,
                    )
            except Exception as e:
                rank_zero_info(
                    f"Error in saving motion {single_generated_id} of dataset {single_dataset_id}: {e}"
                )

        return {"output": output}

    def process_test_results(self):
        for dataset_id in os.listdir(self.cfg.save_dir):
            feature_dir = f"{self.cfg.save_dir}/{dataset_id}/feature"
            if not os.path.exists(feature_dir):
                continue
            # render video and save
            if self.cfg.test_setting.render:
                render_video(
                    motion_dir=feature_dir,
                    save_dir=f"{self.cfg.save_dir}/{dataset_id}/video",
                    render_setting=self.cfg.test_setting,
                    frames_dir=f"{self.cfg.save_dir}/{dataset_id}/frames",
                    traj_mask_dir=f"{self.cfg.save_dir}/{dataset_id}/traj_mask",
                    cond_traj_dir=f"{self.cfg.save_dir}/{dataset_id}/cond_traj",
                )

                # Create composite videos
                make_composite_compare_videos(
                    result_folder=f"{self.cfg.save_dir}/{dataset_id}/video",
                    compare_folders=self.cfg.test_setting.get(dataset_id, {}).get(
                        "compare_folders", None
                    ),
                    compare_names=self.cfg.test_setting.get(dataset_id, {}).get(
                        "compare_names", None
                    ),
                    text_folder=f"{self.cfg.save_dir}/{dataset_id}/text",
                    save_dir=f"{self.cfg.save_dir}/{dataset_id}/composite",
                )

                # wandb log video
                if (
                    not self.cfg.debug
                    and self.logger is not None
                    and isinstance(self.logger, WandbLogger)
                ):
                    video_to_log = []
                    for video_path in sorted(
                        os.listdir(f"{self.cfg.save_dir}/{dataset_id}/composite")
                    ):
                        video_to_log.append(
                            wandb.Video(
                                f"{self.cfg.save_dir}/{dataset_id}/composite/{video_path}",
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
        if not cfg.debug:
            trainer.validate(
                model,
                dataloaders=[val_dataloader, test_dataloader],
                ckpt_path=cfg.resume_ckpt,
                weights_only=False,
            )
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
