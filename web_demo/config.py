"""Configuration helpers for the FloodNet web demo."""

from __future__ import annotations

import os

from omegaconf import OmegaConf


def load_traj_mask_cfg(path: str):
    """Load the web trajectory section from the main model config."""
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"Config not found: {path}. Using default trajectory settings.")
        return {}
    cfg = OmegaConf.load(path)
    if "traj_mask" in cfg:
        return OmegaConf.to_container(cfg.traj_mask, resolve=True)
    print(f"No traj_mask section in {path}. Using default trajectory settings.")
    return {}


def load_debug_preset_cfg(path: str):
    """Load optional web-demo debug preset config."""
    if not path or not os.path.exists(path):
        return {}
    cfg = OmegaConf.load(path)
    section = cfg.get("web_demo_debug", None)
    if section is None:
        return {}
    return OmegaConf.to_container(section, resolve=True)


__all__ = ["load_debug_preset_cfg", "load_traj_mask_cfg"]

