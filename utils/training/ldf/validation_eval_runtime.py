from __future__ import annotations

from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.initialize import instantiate


def build_generation_eval_cfg(cfg):
    val_cfg = cfg.get("validation", {})
    return {
        "enabled": bool(val_cfg.get("eval_generation_metrics", True)),
        "num_runs": int(val_cfg.get("eval_num_runs", 10)),
        "seg_size": int(val_cfg.get("eval_seg_size", 20)),
        "forward_ctrl_loss": bool(val_cfg.get("eval_forward_control_loss", True)),
        "forward_ctrl_window_mode": str(
            val_cfg.get("eval_forward_control_loss_window_mode", "mean_chunk_windows")
        ),
        "eval_all_captions": bool(val_cfg.get("eval_all_captions", False)),
    }


def t2m_metric_enabled(cfg) -> bool:
    val_cfg = cfg.get("validation", {})
    return bool(val_cfg.get("t2m_metric", False))


def validation_repeat_count(cfg) -> int:
    val_cfg = cfg.get("validation", {})
    return int(val_cfg.get("val_repeat", 1))


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
        dl_kwargs = dict(
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
            **{k: v for k, v in dl_kwargs.items() if v is not None},
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
