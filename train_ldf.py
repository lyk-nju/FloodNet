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
    async_test_mode_enabled,
    prepare_model_input,
    build_test_probe_loaders,
    build_test_probe_tags,
    build_val_dataloaders,
    compute_control_loss_xz,
    emit_async_test_request,
    emit_resume_ckpt_eval_request,
    compute_checkpoint_step_info,
    maybe_launch_async_eval_watcher,
    resolve_self_forcing_runtime_steps,
    SelfForcingTrainer,
)

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomLightningModule(BasicLightningModule):
    def __init__(self, cfg):
        self._inline_eval_dedup = {}
        self._resume_step_offset = 0
        super().__init__(cfg)
        # Must set AFTER super().__init__() because LightningModule.__init__
        # forces automatic_optimization=True; setting it before is silently
        # overwritten and crashes self.manual_backward at runtime.
        self_forcing_enabled = bool(
            cfg.model.params.get("self_forcing_enabled", False)
        )
        self.automatic_optimization = not self_forcing_enabled
        self._sf_trainer = (
            SelfForcingTrainer(self) if self_forcing_enabled else None
        )

    def _log_step_metrics(
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
            float(compute_checkpoint_step_info(self).metric_value),
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

    def _compute_control_loss(self, pred_list, batch):
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
        # Self-forcing checkpoint housekeeping: mirrors auto→manual optimizer
        # progress so global_step stays consistent across the switch.
        if self._sf_trainer is not None:
            self._resume_step_offset = self._sf_trainer.on_load_checkpoint(checkpoint)
        else:
            rank_zero_info(
                f"[resume] loaded checkpoint global_step={self._resume_step_offset}"
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
        emit_async_test_request(self)

    def train(self, mode: bool = True):
        super().train(mode)
        # VAE is a frozen decoder used only for L_control; keep it in eval mode regardless
        # of Lightning's train/eval switches so its parameters never enter train behaviour.
        if hasattr(self, "vae"):
            self.vae.eval()
        return self

    def _step(self, batch, is_training=True, model_batch=None):
        if model_batch is None:
            model_batch = prepare_model_input(batch)
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
                        step_loss = self._compute_control_loss(
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
                    loss_control = self._compute_control_loss(
                        pred_list, batch
                    )
                if loss_control is not None:
                    out["total"] = out["total"] + control_weight * loss_control
                    out["control"] = loss_control

        if "control_aux" in out:
            del out["control_aux"]
        return out

    def _standard_training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        if self._sf_trainer is not None:
            return self._sf_trainer.training_step(batch)
        return self._standard_training_step(batch, batch_idx)

    def update_metrics(self, batch):
        if not self.t2m_enabled or self.t2m_metrics is None:
            return
        with self.ema.average_parameters([p for p in self.model.parameters() if p.requires_grad]):
            model_batch = prepare_model_input(batch)
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
        if not async_test_mode_enabled(self.cfg):
            if (
                not self.trainer.sanity_checking
                and self.global_step > 0
                and self.global_step % self.cfg.validation.test_steps == 0
            ):
                self.on_test_epoch_end()
                self._inline_eval_dedup.clear()
        else:
            self._inline_eval_dedup.clear()
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
    maybe_launch_async_eval_watcher(cfg, save_dir)

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

    async_test_mode = async_test_mode_enabled(cfg)
    if async_test_mode:
        test_probe_loaders = []
        test_loader_tags = build_test_probe_tags(cfg)
        total_probe_samples = 0
        rank_zero_info(
            "Async test mode enabled: skip building test probe loaders inside training."
        )
    else:
        (
            test_probe_loaders,
            test_loader_tags,
            total_probe_samples,
        ) = build_test_probe_loaders(cfg, collate_fn)

    rank_zero_info(
        f"Train dataset: {len(train_dataset) if train_dataset is not None else 0}, "
        f"Val dataset: {len(val_dataset) if val_dataset is not None else 0}, "
        f"Test probe samples: {total_probe_samples}"
    )
    
    trainer_absolute_max_steps = int(cfg.trainer.max_steps)
    model_self_forcing_enabled = bool(
        cfg.config.model.params.get("self_forcing_enabled", False)
    )
    scheduler_training_steps = int(
        OmegaConf.to_container(
            cfg.config.lr_scheduler.params,
            resolve=True,
        )["num_training_steps"]
    )
    (
        resume_step_offset,
        phase_max_steps,
        runtime_scheduler_steps,
    ) = resolve_self_forcing_runtime_steps(
        trainer_absolute_max_steps,
        cfg.resume_ckpt if cfg.train else None,
        model_self_forcing_enabled,
        scheduler_training_steps,
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

    val_dataloaders = build_val_dataloaders(
        cfg, val_dataloader, test_probe_loaders
    )

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
