import os


for _thread_env_key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_key, "1")

import torch
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.initialize import (
    get_function,
    get_shared_run_time,
    instantiate,
    load_config,
    save_config_and_codes,
)
from utils.training.ldf import (
    build_probe_loaders,
    build_val_dataloaders,
    resolve_sf_runtime,
    self_forcing_enabled,
    validation_repeat_count,
)
from utils.training.ldf.lightning_module import LDFLightningModule

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


CustomLightningModule = LDFLightningModule


def main():
    """Build config, datasets, model, and launch training or validation."""
    ##############################
    # init
    ##############################
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

    ##############################
    # logger
    ##############################
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

    ##############################
    # dataloader
    ##############################
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

    dl_kwargs = dict(
        num_workers=cfg.data.num_workers,
        prefetch_factor=8 if cfg.data.num_workers > 0 else None,
        persistent_workers=cfg.data.num_workers > 0,
    )
    train_dataloader = (
        DataLoader(
            train_dataset,
            batch_size=cfg.data.train_bs,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
            **{k: v for k, v in dl_kwargs.items() if v is not None},
        )
        if cfg.train
        else None
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.data.val_bs,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        **{k: v for k, v in dl_kwargs.items() if v is not None},
    )

    (
        test_probe_loaders,
        test_loader_tags,
        total_probe_samples,
    ) = build_probe_loaders(cfg, collate_fn)

    rank_zero_info(
        f"Train dataset: {len(train_dataset) if train_dataset is not None else 0}, "
        f"Val dataset: {len(val_dataset) if val_dataset is not None else 0}, "
        f"Test probe samples: {total_probe_samples}"
    )

    ##############################
    # self-forcing runtime
    ##############################
    trainer_absolute_max_steps = int(cfg.trainer.max_steps)
    sf_enabled = self_forcing_enabled(cfg.config)
    lr_params = OmegaConf.to_container(
        cfg.config.lr_scheduler.params, resolve=True
    )
    scheduler_training_steps = int(
        lr_params.get("num_training_steps", lr_params.get("T_max", 0))
    )
    if scheduler_training_steps > 0:
        reset_optim_on_resume = bool(cfg.config.get("resume_reset_optimizer", False))
        (
            resume_step_offset,
            phase_max_steps,
            runtime_scheduler_steps,
        ) = resolve_sf_runtime(
            trainer_absolute_max_steps,
            cfg.resume_ckpt if cfg.train else None,
            sf_enabled,
            scheduler_training_steps,
            reset_optimizer_on_resume=reset_optim_on_resume,
        )
        if runtime_scheduler_steps != scheduler_training_steps:
            key = (
                "num_training_steps"
                if "num_training_steps" in lr_params
                else "T_max"
            )
            OmegaConf.update(
                cfg.config,
                f"lr_scheduler.params.{key}",
                int(runtime_scheduler_steps),
            )
            rank_zero_info(
                "[self_forcing runtime] "
                f"lr_scheduler.{key}={runtime_scheduler_steps} "
                f"(was {scheduler_training_steps})"
            )
    else:
        resume_step_offset = 0
        runtime_scheduler_steps = 0

    ##############################
    # lightning module
    ##############################
    model = LDFLightningModule(cfg=cfg.config)
    model.test_loader_tags = test_loader_tags
    model._resume_step_offset = int(resume_step_offset)

    ##############################
    # trainer
    ##############################
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

    ##############################
    # train or validate
    ##############################
    if cfg.train:
        # if cfg.resume_ckpt:
        #     rank_zero_info(
        #         f"[eval-on-resume] running test on resume ckpt: {cfg.resume_ckpt}"
        #     )
        #     trainer.test(
        #         model,
        #         dataloaders=test_probe_loaders,
        #         ckpt_path=cfg.resume_ckpt,
        #         weights_only=False,
        #     )
        trainer.fit(
            model,
            train_dataloader,
            val_dataloaders=val_dataloaders,
            ckpt_path=cfg.resume_ckpt,
            weights_only=False,
        )
    else:
        for i in range(validation_repeat_count(cfg.config)):
            # Set different seed for each validation run to get diverse results
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
