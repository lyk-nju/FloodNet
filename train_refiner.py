"""RootRefiner training entrypoint."""

from __future__ import annotations

import os
import multiprocessing as mp

import torch
import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.initialize import (
    get_function,
    get_shared_run_time,
    load_config,
    save_config_and_codes,
)
from utils.training.root_refiner import (
    apply_default_fixed_validation_dataset,
    apply_fixed_overfit_datasets,
    apply_training_schedule_to_cfg,
    build_datasets,
    TrainingSchedule,
    worker_init_fn,
)
from utils.training.root_refiner.config_validate import validate_refiner_config
from utils.training.root_refiner.lightning_module import RefinerLightningModule

# Backward-compatible import name used by tests and older scripts.
RootRefinerLightningModule = RefinerLightningModule

# Set tokenizers parallelism to false to avoid warnings in multiprocessing.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _resolve_cfg(cfg) -> dict:
    return OmegaConf.to_container(cfg.config, resolve=True)


def _load_initial_checkpoint(model: RefinerLightningModule, ckpt_path: str) -> None:
    """Load model weights from a Lightning checkpoint without optimizer state."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    rank_zero_info(f"Loaded initial RootRefiner weights from {ckpt_path}")


class RootRefinerTrainingScheduleCallback(Callback):
    """Apply RootRefiner sampling/freeze phases from trainer.global_step."""

    def __init__(self, schedule: TrainingSchedule, shared_phase):
        super().__init__()
        self.schedule = schedule
        self.shared_phase = shared_phase
        self._last_phase_index: int | None = None

    def on_fit_start(self, trainer, pl_module) -> None:
        self._apply(trainer, pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self._apply(trainer, pl_module)

    def _apply(self, trainer, pl_module) -> None:
        phase_index = self.schedule.phase_index_for_step(int(trainer.global_step))
        if self.shared_phase is not None:
            with self.shared_phase.get_lock():
                self.shared_phase.value = phase_index
        pl_module.apply_schedule_freeze(
            self.schedule.freeze_modules_for_phase_index(phase_index)
        )
        if phase_index == self._last_phase_index:
            return
        phase = self.schedule.phase(phase_index)
        rank_zero_info(
            f"RootRefiner training schedule phase={phase_index} "
            f"name={phase.name} steps=[{phase.start_step}, {phase.end_step}) "
            f"freeze={list(phase.freeze_refiner_modules)}"
        )
        self._last_phase_index = phase_index


class StepOffsetModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that displays an offset training step in filenames."""

    def __init__(self, *args, step_offset: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_offset = int(step_offset)

    def format_checkpoint_name(
        self,
        metrics,
        filename=None,
        ver=None,
        prefix=None,
    ) -> str:
        if self.step_offset:
            metrics = dict(metrics)
            step = metrics.get("step", torch.tensor(0))
            metrics["step"] = torch.as_tensor(step) + self.step_offset
        return super().format_checkpoint_name(
            metrics,
            filename=filename,
            ver=ver,
            prefix=prefix,
        )


def main():
    """Build config, datasets, model, and launch training or validation."""
    ##############################
    # init
    ##############################
    torch.set_float32_matmul_precision("high")
    cfg = load_config()
    training_schedule = apply_training_schedule_to_cfg(cfg.config)
    validate_refiner_config(cfg.config)

    seed_everything(cfg.seed, workers=True)
    torch.backends.cudnn.benchmark = not bool(cfg.get("deterministic", False))
    torch.backends.cudnn.deterministic = bool(cfg.get("deterministic", False))

    run_time = get_shared_run_time(cfg.save_dir)
    save_dir = os.path.join(cfg.save_dir, f"{run_time}_{cfg.exp_name}")
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.update(cfg.config, "save_dir", save_dir)

    cfg_dict = _resolve_cfg(cfg)
    wandb_key = None
    wandb_cfg = ((cfg_dict.get("logger") or {}).get("wandb") or {})
    if isinstance(wandb_cfg, dict):
        candidate = wandb_cfg.get("wandb_key")
        if isinstance(candidate, str) and candidate.strip():
            wandb_key = candidate
            wandb_cfg["wandb_key"] = None
            OmegaConf.update(cfg.config, "logger.wandb.wandb_key", None)

    rank_zero_info(
        f"Save dir: {save_dir}, current working dir: {os.getcwd()}, "
        f"exp_name: {cfg.exp_name}"
    )
    save_config_and_codes(cfg, cfg.save_dir)

    ##############################
    # logger
    ##############################
    logger = None
    if not cfg.debug:
        project = wandb_cfg.get("project")
        entity = wandb_cfg.get("entity")
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
            logger = WandbLogger(
                project=project,
                name=f"{cfg.exp_name}_{run_time}",
                entity=entity,
                config=cfg_dict,
                save_dir=cfg.save_dir,
            )
            rank_zero_info("WandB logging enabled")
        else:
            rank_zero_info("WandB API key not provided, skipping WandB logging")

    ##############################
    # dataloader
    ##############################
    cfg_dict = _resolve_cfg(cfg)
    train_ds, val_suites = build_datasets(cfg_dict, seed=int(cfg.seed))
    train_ds, val_suites = apply_fixed_overfit_datasets(train_ds, val_suites, cfg_dict)
    train_ds, val_suites = apply_default_fixed_validation_dataset(train_ds, val_suites)
    schedule_shared_phase = None
    if training_schedule is not None and train_ds is not None:
        schedule_shared_phase = mp.Value("i", 0)
        if hasattr(train_ds, "attach_training_schedule"):
            train_ds.attach_training_schedule(training_schedule, schedule_shared_phase)
        else:
            rank_zero_info(
                "RootRefiner training schedule is enabled, but the train dataset "
                "does not expose attach_training_schedule; sampling phases will "
                "not update this dataset."
            )
    OmegaConf.update(
        cfg.config,
        "validation.suites",
        [{"name": suite["name"]} for suite in val_suites],
    )
    cfg_dict = _resolve_cfg(cfg)

    data_cfg = cfg_dict["data"]
    collate_fn = get_function(data_cfg["collate_fn"])
    dl_kwargs = dict(
        num_workers=data_cfg["num_workers"],
        persistent_workers=data_cfg["num_workers"] > 0,
    )
    train_loader = (
        DataLoader(
            train_ds,
            batch_size=data_cfg["train_bs"],
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            **dl_kwargs,
        )
        if cfg.train
        else None
    )
    val_loaders = []
    for suite in val_suites:
        dataset = suite["dataset"]
        if len(dataset) == 0:
            continue
        val_loaders.append(
            DataLoader(
                dataset,
                batch_size=data_cfg["val_bs"],
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
                **dl_kwargs,
            )
        )
    val_dataloaders = val_loaders or None
    rank_zero_info(
        f"Train dataset: {len(train_ds) if train_ds is not None else 0}, "
        f"Val suites: {sum(len(suite['dataset']) for suite in val_suites)}"
    )

    ##############################
    # lightning module
    ##############################
    model = RefinerLightningModule(cfg_dict)
    init_ckpt = cfg.get("init_ckpt", None)
    if init_ckpt and cfg.resume_ckpt:
        raise ValueError(
            "Use either init_ckpt for weight-only initialization or resume_ckpt "
            "for full trainer-state resume, not both."
        )
    if init_ckpt:
        _load_initial_checkpoint(model, init_ckpt)

    ##############################
    # trainer
    ##############################
    callbacks = []
    checkpoint_cfg = cfg.get("checkpoint", {})
    checkpoint_callback = StepOffsetModelCheckpoint(
        dirpath=cfg.save_dir,
        filename="refiner_step_{step:06d}",
        step_offset=int(checkpoint_cfg.get("step_offset", 0) or 0),
        every_n_train_steps=cfg.validation.save_every_n_steps,
        save_top_k=cfg.validation.save_top_k,
        monitor=cfg.validation.get("monitor"),
        mode=cfg.validation.get("mode", "min"),
        auto_insert_metric_name=False,
        save_last=bool(cfg.validation.get("save_last", True)),
        save_on_train_epoch_end=False,
    )
    if cfg.train:
        callbacks.append(checkpoint_callback)
        if training_schedule is not None:
            callbacks.append(
                RootRefinerTrainingScheduleCallback(
                    training_schedule,
                    schedule_shared_phase,
                )
            )

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    devices = trainer_kwargs.get("devices", 1)
    num_devices = (
        devices
        if isinstance(devices, int) and devices >= 0
        else len(devices)
        if isinstance(devices, (list, tuple))
        else torch.cuda.device_count()
    )
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

    ##############################
    # train or validate
    ##############################
    if cfg.train:
        trainer.fit(
            model,
            train_loader,
            val_dataloaders=val_dataloaders,
            ckpt_path=cfg.resume_ckpt or None,
        )
    else:
        trainer.validate(
            model,
            dataloaders=val_dataloaders,
            ckpt_path=cfg.get("test_ckpt", cfg.get("resume_ckpt", None)),
        )

    if not cfg.debug and logger is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
