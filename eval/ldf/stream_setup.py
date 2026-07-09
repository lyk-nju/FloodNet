"""Setup helpers for LDF stream evaluation."""

from __future__ import annotations

import random
import types
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch_ema import ExponentialMovingAverage

from utils.initialize import get_function, instantiate, load_config
from utils.training.ldf.model_factory import instantiate_ldf_model


class InMemorySampleDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _parse_overrides(set_args: List[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in set_args:
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    torch.random.set_rng_state(gen.get_state())
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _infer_meta_tag(meta_paths) -> str:
    if not meta_paths:
        return "meta"
    stem = Path(str(meta_paths[0])).stem
    return stem[:-4] if stem.endswith("_min") else stem


def _resolve_meta_paths_and_probe_tag(args, cfg) -> tuple[list[str], str]:
    """Resolve stream-eval meta paths.

    Supports the standalone legacy `data.test_meta_paths` shape and the LDF
    probe layout `data.test_probe_meta_paths` used by run_eval.
    CLI `--meta_paths` remains the highest-priority override.
    """
    if args.meta_paths:
        meta_paths = list(args.meta_paths)
        probe_tag = (
            args.probe_tag
            or cfg.get("eval.probe_tag", None)
            or _infer_meta_tag(meta_paths)
        )
        return meta_paths, str(probe_tag)

    requested_probe = args.probe_tag or cfg.get("eval.probe_tag", None)
    data_cfg = cfg.get("data", {}) or {}
    probe_cfg = data_cfg.get("test_probe_meta_paths", None)
    if probe_cfg:
        probe_items = list(probe_cfg.items())
        if requested_probe is not None and str(requested_probe) in probe_cfg:
            return list(probe_cfg[str(requested_probe)]), str(requested_probe)
        first_tag, first_paths = probe_items[0]
        return list(first_paths), str(requested_probe or first_tag)

    test_meta_paths = data_cfg.get("test_meta_paths", None)
    if test_meta_paths is not None:
        meta_paths = list(test_meta_paths)
        probe_tag = requested_probe or _infer_meta_tag(meta_paths)
        return meta_paths, str(probe_tag)

    raise ValueError(
        "Stream eval requires either --meta_paths, data.test_meta_paths, or "
        "data.test_probe_meta_paths in the config."
    )


def _resolve_ema_params(model, checkpoint, cfg):
    n_shadow = len(checkpoint["ema_state"]["shadow_params"])
    all_params = list(model.parameters())
    backbone_params = list(model.model.parameters()) if getattr(model, "model", None) is not None else []
    state_dict = checkpoint.get("state_dict", {})
    legacy_split_traj_encoder = (
        any(key.startswith("local_traj_encoder.") for key in state_dict)
        and not any(key.startswith("traj_encoder.frame_encoder.") for key in state_dict)
    )
    if n_shadow == len(all_params):
        return all_params
    if backbone_params and n_shadow == len(backbone_params):
        return backbone_params
    if getattr(model, "freeze_backbone", False) and getattr(model, "controlnet", None) is not None:
        ema_params = list(model.controlnet.parameters())
        if getattr(model, "traj_encoder", None) is not None:
            if legacy_split_traj_encoder:
                # Old 7D stream ckpts stored EMA after the old module order:
                # controlnet -> traj_encoder(TokenTrajEncoder) -> local_traj_encoder(FrameTrajEncoder).
                # The current wrapper exposes frame_encoder before token_encoder, so keep the
                # checkpoint order when copying shadow params.
                token_encoder = getattr(model.traj_encoder, "token_encoder", None)
                frame_encoder = getattr(model.traj_encoder, "frame_encoder", None)
                if token_encoder is not None:
                    ema_params.extend(list(token_encoder.parameters()))
                if frame_encoder is not None:
                    ema_params.extend(list(frame_encoder.parameters()))
            else:
                ema_params.extend(list(model.traj_encoder.parameters()))
        if len(ema_params) == n_shadow:
            return ema_params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == n_shadow:
        return trainable_params
    raise ValueError(
        f"EMA shadow_params count ({n_shadow}) does not match any known param group. "
        "Check freeze settings or EMA checkpoint compatibility."
    )


def _remap_legacy_split_traj_encoder_state_dict(state_dict: dict) -> dict:
    """Map old 7D stream ckpt traj encoder keys onto the current wrapper.

    Older ckpts used two sibling modules:
      local_traj_encoder.*  -> frame-level Conv encoder
      traj_encoder.*        -> token-level MLP encoder

    Current code wraps those as:
      traj_encoder.frame_encoder.*
      traj_encoder.token_encoder.*
    """
    if not (
        any(key.startswith("local_traj_encoder.") for key in state_dict)
        and not any(key.startswith("traj_encoder.frame_encoder.") for key in state_dict)
    ):
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("local_traj_encoder."):
            remapped[
                "traj_encoder.frame_encoder." + key[len("local_traj_encoder."):]
            ] = value
        elif key.startswith("traj_encoder."):
            remapped[
                "traj_encoder.token_encoder." + key[len("traj_encoder."):]
            ] = value
        else:
            remapped[key] = value
    return remapped


def load_eval_model_and_vae(cfg, ckpt_path: str, vae_ckpt_path: str, device: torch.device, use_ema: bool):
    vae = instantiate(
        target=cfg.test_vae.target,
        cfg=None,
        hfstyle=False,
        **cfg.test_vae.params,
    )
    vae_ckpt = torch.load(vae_ckpt_path, map_location="cpu", weights_only=False)
    vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
    if "ema_state" in vae_ckpt:
        vae_ema = ExponentialMovingAverage(vae.parameters(), decay=cfg.test_vae.ema_decay)
        vae_ema.load_state_dict(vae_ckpt["ema_state"])
        vae_ema.copy_to(vae.parameters())
    vae.to(device).eval()

    model = instantiate_ldf_model(cfg.model.target, cfg.model.params)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint["state_dict"] = _remap_legacy_split_traj_encoder_state_dict(
        checkpoint["state_dict"]
    )
    ckpt_keys = set(checkpoint["state_dict"].keys())
    strict = any(key.startswith("controlnet.") for key in ckpt_keys)
    load_result = model.load_state_dict(checkpoint["state_dict"], strict=strict)
    if not strict and load_result.missing_keys and getattr(model, "controlnet", None) is not None:
        model.controlnet.init_from_backbone(model.model)

    if use_ema and "ema_state" in checkpoint:
        ema_params = _resolve_ema_params(model, checkpoint, cfg)
        ema = ExponentialMovingAverage(ema_params, decay=cfg.model.ema_decay)
        ema.load_state_dict(checkpoint["ema_state"])
        ema.copy_to(ema_params)

    model.to(device).eval()
    return model, vae


def enable_cpu_text_encoding(model):
    if getattr(model, "use_precomputed_text_emb", False) or getattr(model, "text_encoder", None) is None:
        return

    cpu_device = torch.device("cpu")
    model.text_encoder.model.to(cpu_device)

    def encode_text_with_cache_cpu(self, text_list, target_device):
        text_features = []
        indices_to_encode = []
        texts_to_encode = []

        for idx, text in enumerate(text_list):
            if text in self.text_cache:
                text_features.append(self.text_cache[text].to(target_device))
            else:
                text_features.append(None)
                indices_to_encode.append(idx)
                texts_to_encode.append(text)

        if texts_to_encode:
            self.text_encoder.model.to(cpu_device)
            encoded = self.text_encoder(texts_to_encode, cpu_device)
            for idx, text, feature in zip(indices_to_encode, texts_to_encode, encoded):
                cached_feature = feature.cpu()
                self.text_cache[text] = cached_feature
                text_features[idx] = cached_feature.to(target_device)

        return text_features

    model.encode_text_with_cache = types.MethodType(encode_text_with_cache_cpu, model)


def build_eval_dataloader(
    cfg,
    meta_paths=None,
    batch_size=None,
    num_workers=None,
    group_present_segments: bool = False,
):
    if meta_paths is not None:
        OmegaConf.update(cfg.config, "data.test_meta_paths", list(meta_paths), force_add=True)
    dataset_target = cfg.data.get("test_target", cfg.data.target)
    dataset = instantiate(dataset_target, cfg=cfg.config, split="test")
    if group_present_segments:
        if not hasattr(dataset, "build_present_segment_eval_samples"):
            raise NotImplementedError(
                f"{type(dataset).__name__} does not support present-segment regrouping"
            )
        dataset = InMemorySampleDataset(dataset.build_present_segment_eval_samples())
    collate_fn = get_function(cfg.data.collate_fn) if cfg.data.get("collate_fn", None) else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size or int(cfg.data.test_bs),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers if num_workers is not None else int(cfg.data.num_workers),
        persistent_workers=False,
        prefetch_factor=8 if (num_workers if num_workers is not None else int(cfg.data.num_workers)) > 0 else None,
        collate_fn=collate_fn,
    )
    return dataset, loader


def _parse_devices_arg(devices_arg) -> list[int]:
    if devices_arg is None:
        return [0]
    if isinstance(devices_arg, int):
        return list(range(devices_arg)) if devices_arg > 0 else []
    if isinstance(devices_arg, (list, tuple)):
        return [int(device) for device in devices_arg]

    text = str(devices_arg).strip()
    if text in {"", "none", "None"}:
        return [0]
    if text in {"auto", "-1"}:
        count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        return list(range(max(count, 1)))
    if text.startswith("[") and text.endswith("]"):
        values = json.loads(text)
        return [int(device) for device in values]
    if "," in text:
        return [int(part.strip()) for part in text.split(",") if part.strip()]

    count = int(text)
    return list(range(count)) if count > 0 else []


def _resolve_accelerator(args, device_ids: list[int]) -> str:
    if args.accelerator is not None:
        return args.accelerator
    return "gpu" if torch.cuda.is_available() and len(device_ids) > 0 else "cpu"


def _select_eval_device(accelerator: str, device_index: int | None) -> torch.device:
    if accelerator != "gpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    device_index = 0 if device_index is None else int(device_index)
    torch.cuda.set_device(device_index)
    return torch.device(f"cuda:{device_index}")


def _should_process_batch_on_rank(
    batch_idx: int,
    *,
    rank: int,
    world_size: int,
    max_batches: int,
    max_samples: int,
) -> bool:
    if max_batches > 0 and batch_idx >= max_batches:
        return False
    if max_samples > 0 and batch_idx >= max_samples:
        return False
    return batch_idx % max(world_size, 1) == rank



__all__ = [
    "InMemorySampleDataset",
    "_infer_meta_tag",
    "_parse_devices_arg",
    "_parse_overrides",
    "_resolve_accelerator",
    "_resolve_ema_params",
    "_resolve_meta_paths_and_probe_tag",
    "_select_eval_device",
    "_set_seed",
    "_should_process_batch_on_rank",
    "build_eval_dataloader",
    "enable_cpu_text_encoding",
    "load_eval_model_and_vae",
]
