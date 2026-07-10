"""Model loading helpers for the web demo runtime."""

from __future__ import annotations

import os

import torch
from torch_ema import ExponentialMovingAverage

from utils.inference.condition_manager import ConditionManager
from utils.inference.stream_generator import StreamGenerator
from utils.initialize import instantiate, load_config
from utils.training.ldf.model_factory import instantiate_ldf_model

from .model_bundle import ModelBundle


def resolve_repo_path(path):
    if not path:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.abspath(os.path.join(parent_dir, path))


def load_ldf_models(config_path, device: str):
    """Load VAE and LDF model exactly as the legacy web demo did."""
    torch.set_float32_matmul_precision("high")
    original_dir = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    os.chdir(parent_dir)
    try:
        cfg = load_config(config_path=config_path)

        print("Loading VAE...")
        vae = instantiate(
            target=cfg.test_vae.target,
            cfg=None,
            hfstyle=False,
            **cfg.test_vae.params,
        )
        vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
        if "ema_state" in vae_ckpt:
            vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
            vae_ema = ExponentialMovingAverage(
                vae.parameters(), decay=cfg.test_vae.ema_decay
            )
            vae_ema.load_state_dict(vae_ckpt["ema_state"])
            vae_ema.copy_to(vae.parameters())
            print("Loaded VAE with EMA")
        else:
            vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
            print("Loaded VAE without EMA")
        vae.to(device)
        vae.eval()

        print("Loading diffusion model...")
        model = instantiate_ldf_model(cfg.model.target, cfg.model.params)
        checkpoint = torch.load(cfg.test_ckpt, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(checkpoint["state_dict"], strict=True)
        except RuntimeError as exc:
            print(
                "Strict checkpoint load failed; falling back to strict=False for backward compatibility."
            )
            print(f"Reason: {exc}")
            load_result = model.load_state_dict(checkpoint["state_dict"], strict=False)
            if load_result.missing_keys:
                print(f"Missing keys (initialized from current model): {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"Unexpected keys (ignored): {load_result.unexpected_keys}")

        if "ema_state" in checkpoint:
            shadow_count = len(checkpoint["ema_state"]["shadow_params"])
            ema_params = [p for p in model.parameters() if p.requires_grad]
            if len(ema_params) != shadow_count:
                ema_params = list(model.parameters())
            assert len(ema_params) == shadow_count, (
                f"EMA shadow_params count ({shadow_count}) does not match "
                f"trainable params ({len([p for p in model.parameters() if p.requires_grad])}) "
                f"or total params ({len(list(model.parameters()))}). "
                "Check freeze settings or EMA checkpoint."
            )
            ema = ExponentialMovingAverage(ema_params, decay=cfg.model.ema_decay)
            ema.load_state_dict(checkpoint["ema_state"])
            ema.copy_to(ema_params)
            print(f"Loaded model with EMA ({shadow_count} params)")
        else:
            print("Loaded model without EMA")
        model.to(device)
        model.eval()
        return vae, model, cfg
    finally:
        os.chdir(original_dir)


def reject_normalized_root_refiner_config(cfg) -> None:
    data_cfg = (cfg.get("data", {}) or {}) if isinstance(cfg, dict) else {}
    legacy_keys = {
        key
        for key in ("normalize", "stats_dir", "path_feature_stats_dir")
        if key in data_cfg
    }
    if not legacy_keys:
        return
    raise ValueError(
        "web_demo StreamGenerator currently expects physical RootRefiner output; "
        "normalized RootRefiner checkpoints need wp_mean/wp_std runtime support. "
        f"Remove legacy data config key(s): {sorted(legacy_keys)}"
    )


def load_root_refiner_modules(root_cfg: dict):
    refiner = None
    text_encoder = None
    sparse_point_range = (root_cfg.get("sparse_path", {}) or {}).get(
        "point_range", (3, 8)
    )
    if not bool(root_cfg.get("enabled", False)):
        print("RootRefiner modules disabled")
        return refiner, text_encoder, sparse_point_range

    refiner_config = resolve_repo_path(root_cfg.get("config_path") or root_cfg.get("config"))
    ckpt_path = resolve_repo_path(
        root_cfg.get("ckpt")
        or root_cfg.get("checkpoint")
        or root_cfg.get("checkpoint_path")
    )
    if refiner_config is None:
        raise ValueError("traj_mask.root_refiner.enabled=true requires config_path")
    if ckpt_path is None:
        raise ValueError("traj_mask.root_refiner.enabled=true requires ckpt")
    print(f"Loading RootRefiner modules: config={refiner_config}, ckpt={ckpt_path}")

    from omegaconf import OmegaConf

    from utils.initialize import load_config
    from utils.training.root_refiner.lightning_module import RootRefinerLightningModule

    cfg = OmegaConf.to_container(load_config(refiner_config).config, resolve=True)
    reject_normalized_root_refiner_config(cfg)
    module = RootRefinerLightningModule(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    try:
        module.load_state_dict(state_dict, strict=bool(root_cfg.get("strict", True)))
    except RuntimeError:
        if bool(root_cfg.get("strict", True)):
            raise
        module.load_state_dict(state_dict, strict=False)
    sparse_point_range = (
        (cfg.get("sampling", {}) or {})
        .get("path_condition", {})
        .get("sparse_path", {})
        .get("point_range", sparse_point_range)
    )
    print("Loaded RootRefiner modules")
    return module.refiner, module.text_encoder, sparse_point_range


def build_stream_generator(
    ldf_model,
    device: str,
    traj_mask_cfg=None,
    history_length: int = 30,
    vae=None,
):
    traj_mask_cfg = traj_mask_cfg or {}
    root_cfg = (traj_mask_cfg.get("root_refiner", {}) or {})
    refiner, text_encoder, sparse_point_range = load_root_refiner_modules(root_cfg)
    manager = ConditionManager(
        initial_text="",
        route_mode=str(root_cfg.get("route_mode", "relative_to_actor")),
        sparse_point_range=tuple(int(v) for v in sparse_point_range),
    )
    return StreamGenerator(
        ldf_model=ldf_model,
        condition_manager=manager,
        root_refiner=refiner,
        root_text_encoder=text_encoder,
        device=device,
        token_dt=float(traj_mask_cfg.get("token_dt", 0.20)),
        history_length=int(history_length),
        traj_horizon_tokens=int(traj_mask_cfg.get("horizon_tokens", 20)),
        vae=vae,
    )


def load_model_bundle(config_path, traj_mask_cfg=None, device="cpu") -> ModelBundle:
    vae, ldf_model, cfg = load_ldf_models(config_path, device)
    stream_generator = build_stream_generator(
        ldf_model,
        device,
        traj_mask_cfg=traj_mask_cfg,
        vae=vae,
    )
    return ModelBundle(
        vae=vae,
        ldf_model=ldf_model,
        cfg=cfg,
        device=device,
        stream_generator=stream_generator,
        root_refiner=stream_generator.root_refiner,
        root_text_encoder=stream_generator.root_text_encoder,
    )


__all__ = [
    "build_stream_generator",
    "load_ldf_models",
    "load_model_bundle",
    "load_root_refiner_modules",
    "reject_normalized_root_refiner_config",
    "resolve_repo_path",
]
