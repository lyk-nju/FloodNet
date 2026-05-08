from __future__ import annotations

from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.initialize import instantiate


def build_test_probe_tags(cfg):
    probe_cfg = cfg.data.get("test_probe_meta_paths", None)
    if probe_cfg:
        return [str(probe_tag) for probe_tag in probe_cfg.keys()]
    if cfg.data.get("test_meta_paths", None) is not None:
        return ["test"]
    return ["test"]


def build_val_dataloaders(cfg, val_dataloader, test_probe_loaders):
    test_mode = str(cfg.get("validation", {}).get("test_mode", "inline"))
    if test_mode == "async":
        rank_zero_info("Validation test mode: async (emit requests only)")
        return val_dataloader
    rank_zero_info("Validation test mode: inline")
    return [val_dataloader] + test_probe_loaders


def build_test_probe_loaders(cfg, collate_fn):
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
