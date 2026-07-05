"""Validation dataloader and metric-runtime helpers for LDF training."""

from __future__ import annotations

from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.initialize import instantiate
from utils.training.ldf.t2m_generation_modes import T2M_GENERATION_MODES


def _resolve_eval_generation_mode(validation_cfg) -> str:
    mode = str(validation_cfg.get("eval_generation_mode", "stream_generate_step"))
    if mode not in T2M_GENERATION_MODES:
        raise ValueError(
            "validation.eval_generation_mode must be one of "
            f"{T2M_GENERATION_MODES}; got {mode!r}."
        )
    return mode


def build_generation_eval_cfg(cfg):
    validation_cfg = cfg.get("validation", {})
    return {
        "enabled": bool(validation_cfg.get("eval_generation_metrics", True)),
        "num_runs": int(validation_cfg.get("eval_num_runs", 10)),
        "seg_size": int(validation_cfg.get("eval_seg_size", 20)),
        "forward_ctrl_loss": bool(
            validation_cfg.get("eval_forward_control_loss", True)
        ),
        "forward_ctrl_window_mode": str(
            validation_cfg.get(
                "eval_forward_control_loss_window_mode",
                "mean_chunk_windows",
            )
        ),
        "eval_all_captions": bool(validation_cfg.get("eval_all_captions", False)),
        "condition_mode": str(
            validation_cfg.get("eval_condition_mode", "clip_start_local")
        ),
        "generation_mode": _resolve_eval_generation_mode(validation_cfg),
        "stream_history_length": int(
            validation_cfg.get("eval_stream_history_length", 30)
        ),
        "stream_traj_horizon_tokens": int(
            validation_cfg.get("eval_stream_traj_horizon_tokens", 20)
        ),
        "stream_token_dt": float(validation_cfg.get("eval_stream_token_dt", 0.20)),
        "stream_frames_per_token": int(
            validation_cfg.get("eval_stream_frames_per_token", 4)
        ),
        "stream_best_of_k": int(validation_cfg.get("eval_stream_best_of_k", 1)),
        "stream_best_of_k_score": str(
            validation_cfg.get("eval_stream_best_of_k_score", "xz")
        ),
        "stream_best_of_k_xz_weight": float(
            validation_cfg.get("eval_stream_best_of_k_xz_weight", 1.0)
        ),
        "stream_best_of_k_fde_weight": float(
            validation_cfg.get("eval_stream_best_of_k_fde_weight", 1.0)
        ),
        "stream_best_of_k_cont_weight": float(
            validation_cfg.get("eval_stream_best_of_k_cont_weight", 0.0)
        ),
        "stream_best_of_k_vel_weight": float(
            validation_cfg.get("eval_stream_best_of_k_vel_weight", 0.5)
        ),
        "stream_best_of_k_rel_margin": float(
            validation_cfg.get("eval_stream_best_of_k_rel_margin", 0.10)
        ),
        "stream_best_of_k_abs_margin": float(
            validation_cfg.get("eval_stream_best_of_k_abs_margin", 0.03)
        ),
        "stream_best_of_k_cont_tol": float(
            validation_cfg.get("eval_stream_best_of_k_cont_tol", 0.03)
        ),
        "stream_best_of_k_force_candidate0": bool(
            validation_cfg.get("eval_stream_best_of_k_force_candidate0", False)
        ),
        "stream_best_of_k_switch_cooldown_steps": int(
            validation_cfg.get("eval_stream_best_of_k_switch_cooldown_steps", 0)
        ),
        "num_denoise_steps": validation_cfg.get("eval_num_denoise_steps", None),
    }


def t2m_metric_enabled(cfg) -> bool:
    validation_cfg = cfg.get("validation", {})
    return bool(validation_cfg.get("t2m_metric", False))


def validation_repeat_count(cfg) -> int:
    validation_cfg = cfg.get("validation", {})
    return int(validation_cfg.get("val_repeat", 1))


def control_loss_train_mode(cfg) -> int:
    body_cfg = cfg.get("body_aux_loss", {}) or {}
    return int(body_cfg.get("control_loss_train_mode", 3))


def build_test_probe_tags(cfg):
    probe_cfg = cfg.data.get("test_probe_meta_paths", None)
    if probe_cfg:
        return [str(probe_tag) for probe_tag in probe_cfg.keys()]
    if cfg.data.get("test_meta_paths", None) is not None:
        return ["test"]
    return ["test"]


def build_val_dataloaders(cfg, val_dataloader, test_probe_loaders):
    rank_zero_info("Validation eval: local generation probes enabled")
    return [val_dataloader] + test_probe_loaders


def build_probe_loaders(cfg, collate_fn):
    probe_cfg = cfg.data.get("test_probe_meta_paths", None)
    probe_specs = []
    if probe_cfg:
        for probe_tag, meta_paths in probe_cfg.items():
            probe_specs.append((str(probe_tag), list(meta_paths)))
    else:
        test_meta_paths = cfg.data.get("test_meta_paths", None)
        if test_meta_paths is not None:
            probe_specs.append(("test", list(test_meta_paths)))
        else:
            probe_specs.append(("test", None))

    loaders, tags = [], []
    total_probe_samples = 0
    test_target = cfg.data.get("test_target", cfg.data.target)
    for probe_tag, meta_paths in probe_specs:
        probe_cfg_obj = OmegaConf.create(
            OmegaConf.to_container(cfg.config, resolve=False)
        )
        if meta_paths is not None:
            OmegaConf.update(probe_cfg_obj, "data.test_meta_paths", meta_paths)
        probe_dataset = instantiate(test_target, cfg=probe_cfg_obj, split="test")
        dataloader_kwargs = dict(
            num_workers=cfg.data.num_workers,
            prefetch_factor=8 if cfg.data.num_workers > 0 else None,
            persistent_workers=cfg.data.num_workers > 0,
        )
        probe_loader = DataLoader(
            probe_dataset,
            batch_size=cfg.data.test_bs,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            **{
                key: value
                for key, value in dataloader_kwargs.items()
                if value is not None
            },
        )
        loaders.append(probe_loader)
        tags.append(probe_tag)
        total_probe_samples += len(probe_dataset)
        rank_zero_info(f"Test probe[{probe_tag}]: {len(probe_dataset)} samples")
    return loaders, tags, total_probe_samples


def get_test_probe_tags(module) -> list[str]:
    tags = getattr(module, "test_loader_tags", None)
    if tags:
        return list(tags)
    return ["test"]


def resolve_test_probe_tag(module, test_loader_idx: int) -> str:
    tags = get_test_probe_tags(module)
    if 0 <= test_loader_idx < len(tags):
        return tags[test_loader_idx]
    return f"test_loader_{test_loader_idx}"


__all__ = [
    "build_generation_eval_cfg",
    "build_probe_loaders",
    "build_test_probe_tags",
    "build_val_dataloaders",
    "control_loss_train_mode",
    "get_test_probe_tags",
    "resolve_test_probe_tag",
    "t2m_metric_enabled",
    "validation_repeat_count",
]
