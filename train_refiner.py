"""RootRefiner training entrypoint.

Wires config loading, dataset construction, WandB, checkpointing, and Lightning
Trainer setup. Model-specific forward and loss logic lives in
``utils.refiner.lightning_module``.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import rank_zero_info
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.humanml3d_refiner import refiner_collate  # noqa: E402
from utils.initialize import (  # noqa: E402
    get_function,
    get_shared_run_time,
    instantiate,
    save_config_and_codes,
)
from utils.refiner.config_validate import validate_refiner_config  # noqa: E402
from utils.refiner.lightning_module import (  # noqa: E402
    RefinerLightningModule,
    RootRefinerLightningModule,
)
from utils.refiner.losses import (  # noqa: E402
    masked_mean,
    second_order_diff_l2,
    smooth_l1_masked,
)
from utils.text_encoder_resolver import FrozenStubTextEncoder  # noqa: E402,F401

log = logging.getLogger(__name__)


def _load_cfg(config_path: str) -> dict:
    import yaml

    path = Path(config_path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    base_config = cfg.pop("base_config", None)
    if not base_config:
        return cfg
    base_path = Path(base_config)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base = _load_cfg(str(base_path))
    return _deep_merge_dicts(base, cfg)


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dicts(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_paths_default() -> dict:
    """Read shared path and WandB interpolation values."""
    import yaml

    p = _REPO_ROOT / "configs" / "paths_default.yaml"
    if not p.is_file():
        return {}
    try:
        with p.open() as f:
            return yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001
        return {}


def resolve_cfg_interpolations(cfg: dict) -> dict:
    """Resolve OmegaConf-style interpolations after CLI overrides."""
    from omegaconf import OmegaConf

    extras = _load_paths_default()
    # Missing credential blocks should disable WandB, not abort local smoke runs.
    extras.setdefault("wandb_info", {})
    for key in ("key", "project", "entity"):
        extras["wandb_info"].setdefault(key, "")
    extras.setdefault("refiner_wandb_info", {})
    extras["refiner_wandb_info"].setdefault("project", "")

    injected = [key for key in extras if key not in cfg]
    merged = {**{k: extras[k] for k in injected}, **cfg}
    resolved = OmegaConf.to_container(OmegaConf.create(merged), resolve=True)
    # Keep credentials out of saved hparams, checkpoints, and WandB config.
    for k in set(injected) | {"wandb_info", "refiner_wandb_info"}:
        resolved.pop(k, None)
    return resolved


def resolve_seed(cfg: dict, cli_seed: int | None = None) -> int:
    """Seed precedence: CLI --seed > top-level cfg.seed > 1234 default."""
    if cli_seed is not None:
        return int(cli_seed)
    if cfg.get("seed") is not None:
        return int(cfg["seed"])
    return 1234


def resolve_resume_ckpt(cfg: dict, cli_ckpt: str | None = None) -> str | None:
    """Resume precedence: CLI --ckpt_path > cfg.resume_ckpt > None."""
    if cli_ckpt:
        return cli_ckpt
    rc = cfg.get("resume_ckpt")
    return rc or None


def _read_wandb_info_from_paths_default() -> dict:
    """Best-effort WandB credentials from configs/paths_default.yaml."""
    import yaml

    p = _REPO_ROOT / "configs" / "paths_default.yaml"
    if not p.is_file():
        return {}
    try:
        with p.open() as f:
            d = yaml.safe_load(f) or {}
        base = d.get("wandb_info", {}) or {}
        refiner = d.get("refiner_wandb_info", {}) or {}
        return {**base, **refiner}
    except Exception:  # noqa: BLE001
        return {}


def _literal_or_none(v):
    """Return usable literal strings and ignore unresolved interpolations."""
    if isinstance(v, str) and v.strip() and not v.startswith("${"):
        return v
    return None


def build_wandb_logger(cfg: dict, run_name: str, save_dir: str, api_key: str | None = None):
    """Build the optional WandB logger."""
    if cfg.get("debug", False):
        return None
    wb = (cfg.get("logger") or {}).get("wandb")
    if wb is None or wb.get("enabled") is False:
        return None
    info = _read_wandb_info_from_paths_default()
    key = (
        api_key
        or _literal_or_none(wb.get("wandb_key"))
        or info.get("key")
        or os.environ.get("WANDB_API_KEY")
    )
    project = _literal_or_none(wb.get("project")) or info.get("project")
    entity = _literal_or_none(wb.get("entity")) or info.get("entity")
    if not key:
        log.warning(
            "wandb requested but no API key was found; skipping wandb."
        )
        return None
    os.environ["WANDB_API_KEY"] = key
    from lightning.pytorch.loggers import WandbLogger

    safe_cfg = copy.deepcopy(cfg)
    try:
        safe_cfg["logger"]["wandb"].pop("wandb_key", None)
    except (KeyError, TypeError, AttributeError):
        pass
    return WandbLogger(
        project=project, entity=entity, name=run_name, save_dir=save_dir,
        config=safe_cfg,
    )


def build_checkpoint_callback(cfg: dict, output_dir: str):
    """Build periodic checkpointing from cfg.checkpoint or cfg.validation."""
    ck = cfg.get("checkpoint") or {}
    val = cfg.get("validation") or {}
    every = ck.get("save_every_n_steps", val.get("save_every_n_steps"))
    if not every:
        return None
    from lightning.pytorch.callbacks import ModelCheckpoint

    monitor = ck.get("monitor", val.get("monitor"))
    save_top_k = int(ck.get("save_top_k", val.get("save_top_k", -1)))
    if monitor is None and save_top_k not in (-1, 0):
        log.warning(
            "save_top_k=%d needs a monitor; keeping all periodic ckpts instead.",
            save_top_k,
        )
        save_top_k = -1

    return ModelCheckpoint(
        dirpath=ck.get("dirpath", output_dir),
        filename=ck.get("filename", "refiner_step_{step:06d}"),
        every_n_train_steps=int(every),
        save_top_k=save_top_k,
        monitor=monitor,
        mode=ck.get("mode", val.get("mode", "min")),
        save_last=bool(ck.get("save_last", val.get("save_last", True))),
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )


def _num_devices(devices) -> int:
    """Resolve the device count used to choose single-process vs DDP."""
    if isinstance(devices, (list, tuple)):
        return len(devices)
    if isinstance(devices, int) and devices >= 0:
        return devices
    try:
        return torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        return 1


def _parse_devices_arg(value):
    """Parse CLI --devices while preserving Lightning's int/list/auto forms."""
    if value is None:
        return None
    if isinstance(value, (int, list, tuple)):
        return value
    text = str(value).strip()
    if text.lower() == "auto":
        return "auto"
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if "," in text:
        try:
            return [int(part.strip()) for part in text.split(",") if part.strip()]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--devices list must contain integer device ids, got {value!r}"
            ) from exc
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--devices must be an integer count, a comma-separated id list, "
            f"or 'auto'; got {value!r}"
        ) from exc


def safe_precision(accelerator: str, precision, *, cuda_available: bool):
    """Use fp32 when a GPU-tuned mixed precision config runs on CPU."""
    if precision is None:
        return None
    on_cpu = accelerator == "cpu" or (accelerator == "auto" and not cuda_available)
    fp32 = {"32", "32-true", "64", "64-true", 32, 64}
    if on_cpu and precision not in fp32:
        log.warning(
            "precision=%s requested but the run resolves to CPU; using 32-true.",
            precision,
        )
        return "32-true"
    return precision


def _build_one_dataset(
    cfg: dict,
    split_file: str,
    *,
    seed: int | None = None,
    randomize_caption: bool = True,
    validation_dense_full: bool = False,
):
    """Build one RootRefiner dataset split."""
    from scripts.compute_5d_stats import load_clips_from_dir

    data_cfg = cfg.get("data", {})
    raw_dir = data_cfg["raw_data_dir"]
    stats_dir = data_cfg.get("stats_dir")
    normalize = bool(data_cfg.get("normalize", False))
    if normalize:
        if not stats_dir or not Path(stats_dir).is_dir():
            raise FileNotFoundError(
                f"data.normalize is true but stats_dir={stats_dir!r} does not exist. "
                f"Run scripts/compute_5d_stats.py first, or set data.normalize: false."
            )
    clips = load_clips_from_dir(
        raw_dir,
        dataset=data_cfg.get("dataset", "humanml3d"),
        split_file=split_file,
        feature_path=data_cfg.get("feature_path"),
        text_path=data_cfg.get("text_path"),
    )
    model_cfg = cfg["model"]["params"]
    sampling_cfg = cfg.get("sampling") or {}
    path_condition_cfg = (sampling_cfg.get("path_condition") or {})
    offset_cfg = (path_condition_cfg.get("offset_start") or {})
    sparse_cfg = (path_condition_cfg.get("sparse_path") or {})
    path_feature_stats_dir = data_cfg.get("path_feature_stats_dir") if normalize else None
    path_feature_stats_hash = None
    if path_feature_stats_dir is not None:
        from utils.refiner.path_feature_stats import compute_sampling_config_hash

        path_feature_stats_hash = compute_sampling_config_hash(cfg)
    dataset_target = data_cfg.get(
        "target",
        "datasets.humanml3d_refiner.HumanML3DRefinerDataset",
    )
    full_plan_ratio = sampling_cfg.get("full_plan_ratio", 0.5)
    horizon_policy = sampling_cfg.get("horizon_policy", "random")
    path_condition_policy = path_condition_cfg.get("policy", "dense_path")
    path_condition_ratios = path_condition_cfg.get("ratios")
    offset_start_enabled = bool(offset_cfg.get("enabled", False))
    offset_start_prob = float(offset_cfg.get("prob", 0.0))
    if validation_dense_full:
        full_plan_ratio = 1.0
        horizon_policy = "max"
        path_condition_policy = "dense_path"
        path_condition_ratios = None
        offset_start_enabled = False
        offset_start_prob = 0.0

    return instantiate(
        target=dataset_target,
        cfg=None,
        hfstyle=False,
        clips=clips,
        n_hist=model_cfg["n_hist"],
        n_path=model_cfg["n_path"],
        max_tokens=model_cfg["max_tokens"],
        min_tokens=model_cfg["min_tokens"],
        frames_per_token=model_cfg["frames_per_token"],
        full_plan_ratio=full_plan_ratio,
        horizon_policy=horizon_policy,
        path_condition_policy=path_condition_policy,
        path_condition_ratios=path_condition_ratios,
        offset_start_enabled=offset_start_enabled,
        offset_start_prob=offset_start_prob,
        offset_start_max_frames=int(offset_cfg.get("max_frames", 40)),
        offset_start_apply_to=tuple(
            offset_cfg.get("apply_to", ("dense_path", "sparse_path"))
        ),
        sparse_path_point_range=tuple(sparse_cfg.get("point_range", (3, 8))),
        normalize=normalize,
        stats_dir=stats_dir if normalize else None,
        path_feature_stats_dir=path_feature_stats_dir,
        sampling_config_hash=path_feature_stats_hash,
        seed=seed,
        randomize_caption=randomize_caption,
    )


def build_datasets(cfg: dict, seed: int | None = None):
    """Build train and optional validation datasets from cfg.data."""
    data_cfg = cfg.get("data", {})
    train_split = data_cfg.get("train_split_file", "train.txt")
    val_split = data_cfg.get("val_split_file")
    train_ds = _build_one_dataset(cfg, train_split, seed=seed)
    val_ds = (
        _build_one_dataset(
            cfg,
            val_split,
            seed=0,
            randomize_caption=False,
            validation_dense_full=True,
        )
        if val_split
        else None
    )
    return train_ds, val_ds


def apply_fixed_overfit_datasets(train_ds, val_ds, cfg: dict):
    """Replace train/val datasets with pre-sampled deterministic diagnostics."""
    fixed_cfg = cfg.get("fixed_overfit") or {}
    if not fixed_cfg.get("enabled", False):
        return train_ds, val_ds

    from datasets.humanml3d_refiner import (
        FixedRefinerSampleDataset,
        build_fixed_refiner_samples,
    )

    kwargs = {
        "num_samples": int(fixed_cfg.get("num_samples", 64)),
        "mode_policy": fixed_cfg.get("mode_policy", "mixed"),
        "force_no_path_aug": bool(fixed_cfg.get("force_no_path_aug", True)),
        "force_text_idx": fixed_cfg.get("force_text_idx", 0),
    }
    train_samples = build_fixed_refiner_samples(train_ds, **kwargs)
    fixed_train = FixedRefinerSampleDataset(train_samples)

    if bool(fixed_cfg.get("val_on_train", True)) or val_ds is None:
        fixed_val = FixedRefinerSampleDataset(train_samples)
    else:
        val_samples = build_fixed_refiner_samples(val_ds, **kwargs)
        fixed_val = FixedRefinerSampleDataset(val_samples)

    log.info(
        "fixed_overfit enabled: samples=%d mode_policy=%s force_no_path_aug=%s "
        "val_on_train=%s",
        len(fixed_train),
        kwargs["mode_policy"],
        kwargs["force_no_path_aug"],
        bool(fixed_cfg.get("val_on_train", True)),
    )
    return fixed_train, fixed_val


def apply_default_fixed_validation_dataset(train_ds, val_ds):
    """Replace validation with deterministic fixed samples."""
    if val_ds is None:
        return train_ds, val_ds
    from datasets.humanml3d_refiner import (
        FixedRefinerSampleDataset,
        build_fixed_refiner_samples,
    )
    if isinstance(val_ds, FixedRefinerSampleDataset):
        return train_ds, val_ds

    val_samples = build_fixed_refiner_samples(
        val_ds,
        num_samples=len(val_ds),
        mode_policy="random",
        force_no_path_aug=True,
        force_text_idx=0,
    )
    fixed_val = FixedRefinerSampleDataset(val_samples)
    log.info(
        "default fixed validation: samples=%d mode_policy=random force_no_path_aug=True",
        len(fixed_val),
    )
    return train_ds, fixed_val


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/root_refiner.yaml")
    parser.add_argument("--max_steps", type=int, default=None,
                         help="Override trainer.max_steps (smoke runs).")
    parser.add_argument("--devices", type=_parse_devices_arg, default=None,
                         help="Override trainer.devices, e.g. 4, auto, or 0,1,2,3.")
    parser.add_argument("--ckpt_path", type=str, default=None,
                         help="Resume-from-checkpoint path (overrides cfg.resume_ckpt).")
    parser.add_argument("--seed", type=int, default=None,
                         help="Override cfg.seed (RNG seed for torch + dataset aug).")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Checkpoint/log dir (overrides cfg.save_dir; "
                              "default outputs/root_refiner).")
    parser.add_argument("--raw_data_dir", type=str, default=None,
                         help="Override data.raw_data_dir (e.g. on a host where the "
                              "config's training-box path doesn't exist).")
    parser.add_argument("--stats_dir", type=str, default=None,
                         help="Override data.stats_dir.")
    parser.add_argument("--normalize", type=str, default=None,
                         choices=["true", "false"],
                         help="Override data.normalize (true/false).")
    args = parser.parse_args(argv)

    cfg = _load_cfg(args.config)
    cfg.setdefault("data", {})
    if args.raw_data_dir is not None:
        cfg["data"]["raw_data_dir"] = args.raw_data_dir
    if args.stats_dir is not None:
        cfg["data"]["stats_dir"] = args.stats_dir
    if args.normalize is not None:
        cfg["data"]["normalize"] = (args.normalize == "true")
    cfg = resolve_cfg_interpolations(cfg)
    validate_refiner_config(cfg)
    wandb_api_key = None
    _wb = (cfg.get("logger") or {}).get("wandb")
    if isinstance(_wb, dict):
        wandb_api_key = _literal_or_none(_wb.get("wandb_key"))
        _wb["wandb_key"] = None
    trainer_cfg = cfg.get("trainer") or {}
    max_steps = args.max_steps if args.max_steps is not None else trainer_cfg["max_steps"]
    base_output_dir = args.output_dir or cfg.get("save_dir") or "outputs"
    run_time = get_shared_run_time(base_output_dir)
    output_dir = str(Path(base_output_dir) / f"{run_time}_{cfg.get('exp_name', 'root_refiner')}")
    cfg["save_dir"] = output_dir
    from omegaconf import OmegaConf
    _config_snapshot = type(
        "_ConfigSnapshot",
        (),
        {"exp_name": cfg.get("exp_name", "root_refiner"), "config": OmegaConf.create(cfg)},
    )()
    save_config_and_codes(_config_snapshot, output_dir)
    rank_zero_info(
        f"Save dir: {output_dir}, current working dir: {os.getcwd()}, "
        f"exp_name: {cfg.get('exp_name', 'root_refiner')}"
    )

    seed = resolve_seed(cfg, args.seed)
    pl.seed_everything(seed, workers=True)
    if bool(cfg.get("deterministic", False)):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    log.info("seed=%d deterministic=%s", seed, bool(cfg.get("deterministic", False)))

    train_ds, val_ds = build_datasets(cfg, seed=seed)
    train_ds, val_ds = apply_fixed_overfit_datasets(train_ds, val_ds, cfg)
    train_ds, val_ds = apply_default_fixed_validation_dataset(train_ds, val_ds)
    from datasets.humanml3d_refiner import refiner_worker_init_fn
    data_cfg = cfg["data"]
    collate_fn = get_function(data_cfg["collate_fn"])
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["train_bs"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
        worker_init_fn=refiner_worker_init_fn,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=data_cfg["val_bs"],
            shuffle=False,
            num_workers=data_cfg["num_workers"],
            collate_fn=collate_fn,
            drop_last=False,
        )

    module = RefinerLightningModule(cfg)

    run_name = f"{cfg.get('exp_name', 'root_refiner')}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = build_wandb_logger(
        cfg,
        run_name=run_name,
        save_dir=output_dir,
        api_key=wandb_api_key,
    )
    callbacks = []
    ckpt_cb = build_checkpoint_callback(cfg, output_dir)
    if ckpt_cb is not None:
        callbacks.append(ckpt_cb)
    resume_ckpt = resolve_resume_ckpt(cfg, args.ckpt_path)
    if resume_ckpt:
        log.info("resuming from checkpoint: %s", resume_ckpt)

    accelerator = trainer_cfg.get("accelerator", "auto")
    devices = args.devices if args.devices is not None else trainer_cfg.get("devices", 1)
    strategy = "auto"
    if _num_devices(devices) > 1:
        from lightning.pytorch.strategies import DDPStrategy

        strategy = DDPStrategy(find_unused_parameters=True)
    trainer_kwargs = dict(
        max_steps=max_steps,
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
        default_root_dir=output_dir,
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
        logger=logger if logger is not None else True,
        callbacks=callbacks,
    )
    precision = safe_precision(
        accelerator,
        trainer_cfg.get("precision"),
        cuda_available=torch.cuda.is_available(),
    )
    if precision is not None:
        trainer_kwargs["precision"] = precision
    val_check_interval = (cfg.get("validation") or {}).get("validation_steps")
    if val_loader is not None and val_check_interval:
        trainer_kwargs["val_check_interval"] = val_check_interval
        trainer_kwargs["check_val_every_n_epoch"] = None

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module, train_loader, val_dataloaders=val_loader, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
