"""Model and sample loading helpers for runtime benchmarks."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch_ema import ExponentialMovingAverage

from utils.initialize import instantiate
from utils.motion_process import extract_root_trajectory_263
from utils.training.ldf.model_factory import instantiate_ldf_model


def _load_vae(cfg, device):
    vae = instantiate(target=cfg.test_vae.target, cfg=None, hfstyle=False,
                      **cfg.test_vae.params)
    ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    if "ema_state" in ckpt:
        vae.load_state_dict(ckpt["state_dict"], strict=True)
        ema = ExponentialMovingAverage(vae.parameters(), decay=cfg.test_vae.ema_decay)
        ema.load_state_dict(ckpt["ema_state"])
        ema.copy_to(vae.parameters())
    else:
        vae.load_state_dict(ckpt["state_dict"], strict=True)
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def _load_model(cfg, ckpt_path, device):
    model = instantiate_ldf_model(cfg.model.target, cfg.model.params)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_keys = set(ckpt["state_dict"].keys())
    cn_missing = not any(k.startswith("controlnet.") for k in ckpt_keys)
    strict = not cn_missing
    result = model.load_state_dict(ckpt["state_dict"], strict=strict)
    if not strict and result.missing_keys:
        if any("controlnet." in k for k in result.missing_keys):
            model.controlnet.init_from_backbone(model.model)
    if "ema_state" in ckpt:
        n_shadow = len(ckpt["ema_state"]["shadow_params"])
        ema_params = [p for p in model.parameters() if p.requires_grad]
        if len(ema_params) != n_shadow:
            ema_params = list(model.parameters())
        ema = ExponentialMovingAverage(ema_params, decay=cfg.model.ema_decay)
        ema.load_state_dict(ckpt["ema_state"])
        ema.copy_to(ema_params)
    model.to(device).eval()
    return model


# ── sample loading ─────────────────────────────────────────────────────

def _load_humanml3d_sample(raw_data_dir, sample_id):
    data_dir = os.path.join(raw_data_dir, "HumanML3D")
    feat = np.load(os.path.join(data_dir, "new_joint_vecs", f"{sample_id}.npy")).astype(np.float32)
    txt_path = os.path.join(data_dir, "texts", f"{sample_id}.txt")
    text_data = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            text_data.append({
                "caption": parts[0],
                "tokens": parts[1].split(" ") if len(parts) > 1 else [],
                "f_tag": float(parts[2]) if len(parts) > 2 else 0.0,
                "to_tag": float(parts[3]) if len(parts) > 3 else 0.0,
            })
    traj_xyz = extract_root_trajectory_263(feat)
    token = np.load(os.path.join(
        data_dir, "TOKENS_20251030_085836_vae_wan_z4", f"{sample_id}.npy")).astype(np.float32)
    return {
        "name": sample_id, "dataset": "HumanML3D",
        "feature": torch.from_numpy(feat).float(), "feature_length": len(feat),
        "token": torch.from_numpy(token).float(), "token_length": len(token),
        "text": text_data[0]["caption"],
        "traj": torch.from_numpy(traj_xyz).float(), "traj_length": len(traj_xyz),
        "token_mask": torch.ones(len(token), dtype=torch.float32),
        "traj_mask": torch.ones(len(traj_xyz), dtype=torch.float32),
    }


def _load_babel_sample(raw_data_dir, sample_id):
    data_dir = os.path.join(raw_data_dir, "BABEL_streamed")
    feat = np.load(os.path.join(data_dir, "motions", f"{sample_id}.npy")).astype(np.float32)
    token = np.load(os.path.join(data_dir, "TOKENS_20251030_085836_vae_wan_z4",
                                 f"{sample_id}.npy")).astype(np.float32)
    txt_path = os.path.join(data_dir, "texts", f"{sample_id}.txt")
    text_data = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            ft = float(parts[2]) if len(parts) > 2 and parts[2].strip() else 0.0
            tt = float(parts[3]) if len(parts) > 3 and parts[3].strip() else 0.0
            text_data.append({"caption": parts[0].strip(),
                              "f_tag": 0.0 if np.isnan(ft) else ft,
                              "to_tag": 0.0 if np.isnan(tt) else tt})
    return {"feature": feat, "token": token, "text_data": text_data, "name": sample_id}


def _merge_babel(raw_data_dir, sample_ids):
    parts = [_load_babel_sample(raw_data_dir, sid) for sid in sample_ids]
    feat = np.concatenate([p["feature"] for p in parts], axis=0)
    token = np.concatenate([p["token"] for p in parts], axis=0)
    tf, tt = len(feat), len(token)
    text_data, feat_ofs = [], 0
    feat_fps = 20.0
    for p in parts:
        for td in p["text_data"]:
            ft, ttag = td["f_tag"], td["to_tag"]
            if ft == 0.0 and ttag == 0.0:
                af, at = feat_ofs / feat_fps, (feat_ofs + len(p["feature"])) / feat_fps
            else:
                af, at = feat_ofs / feat_fps + ft, feat_ofs / feat_fps + ttag
            text_data.append({"caption": td["caption"], "f_tag": af, "to_tag": at})
        feat_ofs += len(p["feature"])
    texts, fte, cursor = [], [], 0
    for td in text_data:
        a_start = max(0, int(td["f_tag"] * feat_fps + 0.5))
        a_end = int(td["to_tag"] * feat_fps + 0.5) if td["to_tag"] > 0 else tf
        if a_end <= a_start:
            continue
        if a_start > cursor:
            texts.append(""); fte.append(min(a_start, tf)); cursor = a_start
        texts.append(td["caption"]); fte.append(min(a_end, tf)); cursor = a_end
    if cursor < tf:
        texts.append(""); fte.append(tf)
    if not texts:
        texts = [td["caption"] or "" for td in text_data] or [""]; fte = [tf]
    token_te = [max(0, min(tt, (ef - 1 + 3) // 4 + 1)) for ef in fte]
    traj = extract_root_trajectory_263(feat)
    return {
        "name": sample_ids[0].rsplit("_", 1)[0], "dataset": "BABEL_streamed",
        "feature": torch.from_numpy(feat).float(), "feature_length": tf,
        "token": torch.from_numpy(token).float(), "token_length": tt,
        "text": texts, "traj": torch.from_numpy(traj).float(), "traj_length": len(traj),
        "token_text_end": token_te, "feature_text_end": fte,
        "token_mask": torch.ones(tt, dtype=torch.float32),
        "traj_mask": torch.ones(len(traj), dtype=torch.float32),
    }



__all__ = [
    "_load_babel_sample",
    "_load_humanml3d_sample",
    "_load_model",
    "_load_vae",
    "_merge_babel",
]
