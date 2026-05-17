"""Single-sample stream control diagnostic matrix.

Runs 6 modes on a fixed sample to isolate which dimension (inference path,
horizon, pred-root closed loop) causes trajectory control degradation.

Usage::

    cd FloodNet
    PYTHONPATH=. python eval/diagnose_stream_control.py \\
        --config configs/stream.yaml \\
        --ckpt /path/to/checkpoint.ckpt \\
        --vae_ckpt /path/to/vae.ckpt \\
        --sample_id 000021 \\
        --out_dir outputs/diagnose_stream/000021
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from any directory (e.g. ``python eval/diagnose_stream_control.py``).
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
from lightning import seed_everything
from torch_ema import ExponentialMovingAverage

from omegaconf import OmegaConf

from utils.initialize import (
    check_state_dict,
    instantiate,
    load_config,
)
from utils.motion_process import (
    StreamJointRecovery263,
    extract_root_trajectory_263,
)
from utils.stream_rollout import (
    StreamTextSegment,
    StreamTextRolloutController,
    build_stream_step_model_input,
    build_stream_suffix_conditioning,
    clip_traj_input_to_horizon,
)
from utils.stream_traj import (
    build_remaining_polyline,
    estimate_token_step_distance,
    resample_polyline,
)
from utils.traj_batch import root_to_traj_feats

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# helpers: model loading
# ---------------------------------------------------------------------------

def _load_vae(cfg, device):
    vae = instantiate(
        target=cfg.test_vae.target, cfg=None, hfstyle=False, **cfg.test_vae.params
    )
    vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    if "ema_state" in vae_ckpt:
        vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
        vae_ema = ExponentialMovingAverage(
            vae.parameters(), decay=cfg.test_vae.ema_decay
        )
        vae_ema.load_state_dict(vae_ckpt["ema_state"])
        vae_ema.copy_to(vae.parameters())
        print(f"Loaded VAE from {cfg.test_vae_ckpt} with EMA")
    else:
        vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
        print(f"Loaded VAE from {cfg.test_vae_ckpt} w/o EMA")
    check_state_dict(
        state_dict=vae.state_dict(),
        named_parameters=vae.named_parameters(),
        named_buffers=vae.named_buffers(),
    )
    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def _load_model(cfg, ckpt_path, device):
    model = instantiate(
        target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_keys = set(checkpoint["state_dict"].keys())
    controlnet_missing = not any(k.startswith("controlnet.") for k in ckpt_keys)
    strict = not controlnet_missing
    result = model.load_state_dict(checkpoint["state_dict"], strict=strict)
    if not strict and result.missing_keys:
        print(
            f"Loaded LDF with strict=False (no ControlNet in ckpt). "
            f"Missing keys: {result.missing_keys}"
        )
        if any("controlnet." in k for k in result.missing_keys):
            model.controlnet.init_from_backbone(model.model)
            print("Re-initialized ControlNet from backbone weights")
    if "ema_state" in checkpoint:
        n_shadow = len(checkpoint["ema_state"]["shadow_params"])
        ema_params = [p for p in model.parameters() if p.requires_grad]
        if len(ema_params) != n_shadow:
            ema_params = list(model.parameters())
        assert len(ema_params) == n_shadow, (
            f"EMA shadow count mismatch: {len(ema_params)} vs {n_shadow}"
        )
        ema = ExponentialMovingAverage(ema_params, decay=cfg.model.ema_decay)
        ema.load_state_dict(checkpoint["ema_state"])
        ema.copy_to(ema_params)
        print(f"Loaded model from {ckpt_path} with EMA ({n_shadow} params)")
    else:
        print(f"Loaded model from {ckpt_path} w/o EMA")
    check_state_dict(
        state_dict=model.state_dict(),
        named_parameters=model.named_parameters(),
        named_buffers=model.named_buffers(),
    )
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# helpers: sample loading
# ---------------------------------------------------------------------------

def _load_sample(raw_data_dir: str, sample_id: str, dataset: str = "humanml3d"):
    """Load a single sample, returning a flat batch dict.

    Args:
        raw_data_dir: root raw_data directory.
        sample_id: sample name without extension.
        dataset: ``"humanml3d"`` or ``"babel"``.
    """
    if dataset == "babel":
        return _load_babel_sample(raw_data_dir, sample_id)
    return _load_humanml3d_sample(raw_data_dir, sample_id)


def _load_humanml3d_sample(raw_data_dir: str, sample_id: str):
    """Load a single HumanML3D sample."""
    data_dir = os.path.join(raw_data_dir, "HumanML3D")

    feat_path = os.path.join(data_dir, "new_joint_vecs", f"{sample_id}.npy")
    feature = np.load(feat_path).astype(np.float32)
    feature_length = feature.shape[0]

    txt_path = os.path.join(data_dir, "texts", f"{sample_id}.txt")
    text_data = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            caption = parts[0]
            tokens = parts[1].split(" ") if len(parts) > 1 else []
            f_tag = float(parts[2]) if len(parts) > 2 else 0.0
            to_tag = float(parts[3]) if len(parts) > 3 else 0.0
            text_data.append(
                {"caption": caption, "tokens": tokens, "f_tag": f_tag, "to_tag": to_tag}
            )

    traj_xyz = extract_root_trajectory_263(feature)
    traj_features = root_to_traj_feats(traj_xyz)

    token_dir = os.path.join(data_dir, "TOKENS_20251030_085836_vae_wan_z4")
    token_path = os.path.join(token_dir, f"{sample_id}.npy")
    if os.path.exists(token_path):
        token = np.load(token_path).astype(np.float32)
        token_length = token.shape[0]
    else:
        downsample_factor = 4
        token_length = (feature_length + downsample_factor - 1) // downsample_factor + 1
        token = np.zeros((token_length, 4), dtype=np.float32)

    text_dict = text_data[0]
    text = text_dict["caption"]

    sample = {
        "name": sample_id,
        "dataset": "HumanML3D",
        "feature": torch.from_numpy(feature).float(),
        "feature_length": feature_length,
        "token": torch.from_numpy(token).float(),
        "token_length": token_length,
        "text": text,
        "text_all": [td["caption"] for td in text_data],
        "text_data": text_data,
        "text_tokens": text_dict["tokens"],
        "traj": torch.from_numpy(traj_xyz).float(),
        "traj_length": len(traj_xyz),
        "traj_features": torch.from_numpy(traj_features).float(),
        "token_text_end": [token_length],
        "feature_text_end": [feature_length],
        "token_mask": torch.ones(token_length, dtype=torch.float32),
        "traj_mask": torch.ones(len(traj_xyz), dtype=torch.float32),
    }
    return sample


def _load_babel_sample(raw_data_dir: str, sample_id: str):
    """Load a single BABEL sample with multi-segment text."""
    data_dir = os.path.join(raw_data_dir, "BABEL_streamed")

    feat_path = os.path.join(data_dir, "motions", f"{sample_id}.npy")
    feature = np.load(feat_path).astype(np.float32)
    feature_length = feature.shape[0]

    txt_path = os.path.join(data_dir, "texts", f"{sample_id}.txt")
    text_data = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            caption = parts[0].strip()
            tokens = parts[1].split(" ") if len(parts) > 1 else []
            f_tag = float(parts[2]) if len(parts) > 2 and parts[2].strip() else 0.0
            to_tag = float(parts[3]) if len(parts) > 3 and parts[3].strip() else 0.0
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            to_tag = 0.0 if np.isnan(to_tag) else to_tag
            text_data.append(
                {"caption": caption, "tokens": tokens, "f_tag": f_tag, "to_tag": to_tag}
            )

    traj_xyz = extract_root_trajectory_263(feature)
    traj_features = root_to_traj_feats(traj_xyz)

    token_dir = os.path.join(data_dir, "TOKENS_20251030_085836_vae_wan_z4")
    token_path = os.path.join(token_dir, f"{sample_id}.npy")
    if os.path.exists(token_path):
        token = np.load(token_path).astype(np.float32)
        token_length = token.shape[0]
    else:
        downsample_factor = 4
        token_length = (feature_length + downsample_factor - 1) // downsample_factor + 1
        token = np.zeros((token_length, 4), dtype=np.float32)

    # Build multi-segment text with feature_text_end / token_text_end.
    texts: list[str] = []
    feature_text_end: list[int] = []
    cursor = 0
    for td in text_data:
        f_tag, to_tag = td["f_tag"], td["to_tag"]
        if f_tag == 0.0 and to_tag == 0.0:
            # A full-duration caption — place after any existing gaps.
            if cursor < feature_length:
                texts.append(td["caption"])
                feature_text_end.append(feature_length)
            continue
        abs_start = max(0, int(f_tag * 20.0 + 0.5))
        abs_end = int(to_tag * 20.0 + 0.5) if to_tag > 0 else feature_length
        if abs_end <= abs_start:
            continue
        if abs_start > cursor:
            texts.append("")
            feature_text_end.append(abs_start)
            cursor = abs_start
        if abs_end <= cursor:
            continue
        texts.append(td["caption"])
        feature_text_end.append(abs_end)
        cursor = abs_end
    if cursor < feature_length:
        texts.append("")
        feature_text_end.append(feature_length)
    if not texts:
        texts = [td["caption"] for td in text_data] or [""]
        feature_text_end = [feature_length]

    # Convert frame ends → token ends (causal VAE convention).
    token_text_end: list[int] = []
    for ef in feature_text_end:
        last_frame = ef - 1
        tok_end = (last_frame + 3) // 4 + 1
        token_text_end.append(max(0, min(token_length, tok_end)))

    sample = {
        "name": sample_id,
        "dataset": "BABEL_streamed",
        "feature": torch.from_numpy(feature).float(),
        "feature_length": feature_length,
        "token": torch.from_numpy(token).float(),
        "token_length": token_length,
        "text": texts,
        "text_all": [td["caption"] for td in text_data],
        "text_data": text_data,
        "text_tokens": text_data[0]["tokens"] if text_data else [],
        "traj": torch.from_numpy(traj_xyz).float(),
        "traj_length": len(traj_xyz),
        "traj_features": torch.from_numpy(traj_features).float(),
        "token_text_end": token_text_end,
        "feature_text_end": feature_text_end,
        "token_mask": torch.ones(token_length, dtype=torch.float32),
        "traj_mask": torch.ones(len(traj_xyz), dtype=torch.float32),
    }
    return sample


def _wrap_flat_sample_for_suffix(sample: dict) -> dict:
    """Wrap flat-sample scalar fields into batch-style dict for \
    ``build_stream_suffix_conditioning``."""
    out = dict(sample)
    for key in ("token_length", "traj_length", "feature_length"):
        v = out.get(key)
        if v is not None and not hasattr(v, "shape"):
            out[key] = torch.tensor([v])
    for key in ("traj", "traj_features", "token_mask", "traj_mask"):
        v = out.get(key)
        if v is not None and torch.is_tensor(v) and v.ndim == 1:
            out[key] = v.unsqueeze(0)
    return out


# ---------------------------------------------------------------------------
# helpers: metrics
# ---------------------------------------------------------------------------

def _compute_ade(pred_root: np.ndarray, gt_root: np.ndarray) -> float:
    n = min(len(pred_root), len(gt_root))
    if n == 0:
        return float("nan")
    diff = pred_root[:n, [0, 2]] - gt_root[:n, [0, 2]]
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def _compute_fde(pred_root: np.ndarray, gt_root: np.ndarray) -> float:
    n = min(len(pred_root), len(gt_root))
    if n == 0:
        return float("nan")
    return float(np.linalg.norm(pred_root[n - 1, [0, 2]] - gt_root[n - 1, [0, 2]]))


def _compute_root_path_length(root: np.ndarray) -> float:
    if len(root) < 2:
        return 0.0
    if root.shape[1] == 2:
        return float(np.sum(np.linalg.norm(np.diff(root, axis=0), axis=1)))
    return float(np.sum(np.linalg.norm(np.diff(root[:, [0, 2]], axis=0), axis=1)))


# ---------------------------------------------------------------------------
# generation modes
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_generate_full(model, vae, sample, device):
    """Offline generate() baseline."""
    model_batch = _make_model_batch(sample, device)
    output = model.generate(model_batch, num_denoise_steps=10)
    generated = output["generated"][0]
    decoded = vae.decode(generated[None, :].to(device))[0].float().cpu().numpy()
    pred_root = extract_root_trajectory_263(decoded)
    return decoded, pred_root


@torch.no_grad()
def run_stream_generate_full(model, vae, sample, device):
    """Offline stream_generate() sanity check — full sequence."""
    model_batch = _make_model_batch(sample, device)
    vae.clear_cache()
    all_chunks = []
    for output in model.stream_generate(model_batch, num_denoise_steps=10):
        generated = output["generated"]
        if generated[0] is not None:
            decoded = (
                vae.stream_decode(
                    generated[0][None, :].to(device),
                    first_chunk=(len(all_chunks) == 0),
                )[0]
                .float()
                .cpu()
                .numpy()
            )
            all_chunks.append(decoded)
    vae.clear_cache()
    decoded_full = (
        np.concatenate(all_chunks, axis=0) if all_chunks else np.zeros((0, 263))
    )
    pred_root = extract_root_trajectory_263(decoded_full)
    return decoded_full, pred_root


@torch.no_grad()
def run_stream_step(
    model,
    vae,
    sample,
    device,
    *,
    history_length: int,
    num_denoise_steps: int,
    horizon_tokens: int | None,
    use_pred_root: bool,
):
    """Run stream_generate_step() with configurable horizon and root source.

    When *use_pred_root* is True, replicates web_demo closed-loop behaviour:
    each step decodes the predicted motion, accumulates the root position via
    ``StreamJointRecovery263``, projects it onto the GT polyline, and resamples
    *horizon_tokens* future positions — exactly as ``ModelManager._build_stream_traj_input`` does.
    """
    token_length = sample["token_length"]
    total_frames = 1 + 4 * (token_length - 1) if token_length > 1 else 1

    # Build text controller: handle both single-text (HumanML3D) and
    # multi-segment (BABEL) formats.
    if isinstance(sample["text"], list):
        segments = [
            StreamTextSegment(text=t, token_end=te)
            for t, te in zip(sample["text"], sample["token_text_end"])
        ]
    else:
        segments = [
            StreamTextSegment(text=sample["text"], token_end=token_length)
        ]
    text_ctrl = StreamTextRolloutController(segments)

    # build_stream_suffix_conditioning expects batch-style fields (tensor
    # with batch dim).  Wrap the flat sample once.
    _batch_sample = _wrap_flat_sample_for_suffix(sample)

    # For pred_root mode: maintain the same root-accumulation state as web_demo.
    if use_pred_root:
        stream_recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
        gt_polyline = sample["traj"].numpy()  # (T, 3) world-space GT polyline

    vae.clear_cache()
    model.init_generated(
        history_length, batch_size=1, num_denoise_steps=num_denoise_steps
    )

    all_decoded = []
    all_pred_root = []
    first_chunk = True

    # Pred-root closed-loop state (matches web_demo ModelManager).
    root_xz_history: list[np.ndarray] = []

    for commit_idx in range(token_length):
        current_text = text_ctrl.get_text_for_commit_index(commit_idx)

        if use_pred_root:
            # Replicate web_demo's _build_stream_traj_input pipeline:
            #   1. current_root = stream_recovery.r_pos_accum
            #   2. project onto GT polyline → build_remaining_polyline
            #   3. estimate token_step from predicted velocity history
            #   4. resample H future tokens at that step
            # H shrinks with remaining tokens so that step_full_xyz_predroot
            # and step_full_xyz_gtroot differ ONLY in root source, not horizon.
            if horizon_tokens is not None:
                h = horizon_tokens
            else:
                h = max(1, token_length - commit_idx)
            current_root = np.zeros(3, dtype=np.float32)
            current_root[[0, 2]] = stream_recovery.r_pos_accum[[0, 2]].astype(np.float32)
            polyline = build_remaining_polyline(current_root, gt_polyline)
            token_step = estimate_token_step_distance(root_xz_history)
            future_traj = resample_polyline(polyline, h, token_step)
            traj_input = {
                "traj": torch.from_numpy(future_traj).float().unsqueeze(0),
                "token_mask": torch.ones(1, future_traj.shape[0]),
            }
        else:
            # GT-root path: use suffix from dataset.
            traj_input = build_stream_suffix_conditioning(
                _batch_sample, commit_idx, prefer_xyz=True
            )
            if horizon_tokens is not None:
                traj_input = clip_traj_input_to_horizon(traj_input, horizon_tokens)

        step_payload = build_stream_step_model_input(current_text, traj_input=traj_input)

        output = model.stream_generate_step(step_payload, first_chunk=first_chunk)
        generated = output["generated"]
        decoded = (
            vae.stream_decode(
                generated[0][None, :].to(device), first_chunk=first_chunk
            )[0]
            .float()
            .cpu()
            .numpy()
        )
        first_chunk = False

        # Accumulate root position for pred_root closed loop (matching web_demo).
        if use_pred_root:
            for frame in decoded:
                stream_recovery.process_frame(frame)
                root_xz_history.append(
                    stream_recovery.r_pos_accum[[0, 2]].astype(np.float32).copy()
                )

        all_decoded.append(decoded)
        pred_root_chunk = extract_root_trajectory_263(decoded)
        all_pred_root.append(pred_root_chunk)

    vae.clear_cache()

    decoded_full = (
        np.concatenate(all_decoded, axis=0)[:total_frames]
        if all_decoded
        else np.zeros((0, 263))
    )
    pred_root_full = (
        np.concatenate(all_pred_root, axis=0)[:total_frames]
        if all_pred_root
        else np.zeros((0, 3))
    )
    return decoded_full, pred_root_full


def _make_model_batch(sample, device):
    mb = {
        "feature": sample["token"].unsqueeze(0).to(device),
        "feature_length": torch.tensor([sample["token_length"]], device=device),
        "text": [sample["text"]],  # str → [str]; List[str] → [List[str]]
    }
    # Multi-segment text fields (used by BABEL).
    if isinstance(sample["text"], list):
        mb["feature_text_end"] = [sample["feature_text_end"]]
        mb["token_text_end"] = [sample["token_text_end"]]
    if sample.get("traj") is not None:
        mb["traj"] = sample["traj"].unsqueeze(0).to(device)
        mb["traj_features"] = sample["traj_features"].unsqueeze(0).to(device)
        mb["traj_length"] = torch.tensor([sample["traj_length"]])
        mb["token_mask"] = sample["token_mask"].unsqueeze(0).to(device)
        mb["traj_mask"] = sample["traj_mask"].unsqueeze(0).to(device)
    return mb


# ---------------------------------------------------------------------------
# artifact saving
# ---------------------------------------------------------------------------

def _save_artifacts(out_dir, pred_motion, pred_root, gt_root, target_traj, metrics):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "pred_motion.npy"), pred_motion)
    np.save(os.path.join(out_dir, "pred_root.npy"), pred_root)
    np.save(os.path.join(out_dir, "target_root.npy"), gt_root)
    np.save(os.path.join(out_dir, "target_traj.npy"), target_traj)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def _build_mode_metrics(pred_root, gt_root, target_traj, mode_name, horizon, root_source,
                        traj_encoder_path=""):
    n = min(len(pred_root), len(gt_root))
    return {
        "mode": mode_name,
        "horizon": "full" if horizon is None else int(horizon),
        "root_source": root_source,
        "ADE": _compute_ade(pred_root[:n], gt_root[:n]),
        "FDE": _compute_fde(pred_root[:n], gt_root[:n]),
        "pred_root_path_length": _compute_root_path_length(pred_root[:n]),
        "gt_root_path_length": _compute_root_path_length(gt_root[:n]),
        "target_traj_path_length": _compute_root_path_length(target_traj[:n]),
        "pred_frames": int(len(pred_root)),
        "gt_frames": int(len(gt_root)),
        "traj_encoder_path": traj_encoder_path,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stream control diagnostic matrix on a single sample"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--vae_ckpt", required=True)
    parser.add_argument("--sample_id", default="000021")
    parser.add_argument("--out_dir", default="outputs/diagnose_stream")
    parser.add_argument("--history_length", type=int, default=30)
    parser.add_argument(
        "--ablate_history",
        type=str,
        default=None,
        help="Comma-separated history_length values for context-window ablation. "
        "When set, runs step_full_xyz_gtroot at each value (bypasses the full matrix).",
    )
    parser.add_argument("--traj_horizon_tokens", type=int, default=20)
    parser.add_argument("--num_denoise_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--raw_data_dir", required=True)
    parser.add_argument(
        "--dataset",
        default="humanml3d",
        choices=["humanml3d", "babel"],
        help="Dataset to load the sample from (default: humanml3d)",
    )
    parser.add_argument(
        "--precomputed_text_emb_path",
        default=None,
        help="Path to pre-tokenized T5 embeddings .pt file. "
        "When set, enables use_precomputed_text_emb and skips live T5.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(config_path=args.config)
    OmegaConf.update(cfg.config, "test_vae_ckpt", args.vae_ckpt)
    if args.precomputed_text_emb_path:
        OmegaConf.update(
            cfg.config, "model.params.use_precomputed_text_emb", True
        )
        OmegaConf.update(
            cfg.config,
            "model.params.precomputed_text_emb_path",
            args.precomputed_text_emb_path,
        )
    out_root = os.path.join(args.out_dir, args.sample_id)

    print(f"Loading VAE from {args.vae_ckpt} ...")
    vae = _load_vae(cfg, device)
    print(f"Loading model from {args.ckpt} ...")
    model = _load_model(cfg, args.ckpt, device)
    print(f"Loading sample {args.sample_id} (dataset={args.dataset}) ...")
    sample = _load_sample(args.raw_data_dir, args.sample_id, dataset=args.dataset)
    _text_preview = (
        sample["text"][0][:60]
        if isinstance(sample["text"], list)
        else sample["text"][:60]
    )
    print(
        f"  feature: {sample['feature'].shape}, token: {sample['token'].shape}, "
        f"text: {_text_preview}..."
    )

    gt_root = extract_root_trajectory_263(sample["feature"].numpy())
    target_traj = sample["traj"].numpy()  # (T, 3) xyz

    # --- history-length ablation (bypasses full matrix) ---
    if args.ablate_history:
        hl_values = [int(x.strip()) for x in args.ablate_history.split(",")]
        print(f"\nHistory-length ablation: {hl_values}")
        all_records = []
        for hl in hl_values:
            mode_name = f"step_full_xyz_gtroot_hl{hl}"
            print(f"\n--- {mode_name} ---")
            pred_motion, pred_root = run_stream_step(
                model, vae, sample, device,
                history_length=hl,
                num_denoise_steps=args.num_denoise_steps,
                horizon_tokens=None,
                use_pred_root=False,
            )
            metrics = _build_mode_metrics(
                pred_root, gt_root, target_traj, mode_name, None, "gt",
                traj_encoder_path="",
            )
            print(f"  ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}")
            mode_dir = os.path.join(out_root, mode_name)
            _save_artifacts(
                mode_dir, pred_motion, pred_root,
                gt_root[:len(pred_root)], target_traj[:len(pred_root)], metrics,
            )
            all_records.append(metrics)
        summary = {
            "sample_id": args.sample_id, "dataset": args.dataset,
            "ckpt": args.ckpt, "vae_ckpt": args.vae_ckpt, "config": args.config,
            "ablation": "history_length",
            "modes": all_records,
        }
        summary_path = os.path.join(out_root, "summary_hl_ablation.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nAblation summary saved to {summary_path}")
        print(f"\n{'Mode':<35} {'ADE':>8} {'FDE':>8}")
        print("-" * 53)
        for rec in all_records:
            print(f"{rec['mode']:<35} {rec['ADE']:>8.4f} {rec['FDE']:>8.4f}")
        return

    # Encoding path labels per mode family.
    _TPATH_GENERATE = (
        "traj_features (frame-level) -> _build_traj_emb"
        " -> frames_to_tokens -> LocalTrajEncoder -> TrajEncoder"
        " (no anchor-subtract)"
    )
    _TPATH_STEP = (
        "traj (xyz) -> TrajStreamBuffer._build_from_xyz"
        " (anchor-subtract) -> token-level -> frames_to_tokens"
        " -> LocalTrajEncoder -> TrajEncoder"
    )

    # Diagnostic matrix: (mode_name, horizon, root_source, use_pred_root, traj_encoder_path)
    modes = [
        ("generate_full", None, "gt", False, _TPATH_GENERATE),
        ("stream_generate_full", None, "gt", False, _TPATH_GENERATE),
        ("step_full_xyz_gtroot", None, "gt", False, _TPATH_STEP),
        ("step_full_xyz_predroot", None, "pred", True, _TPATH_STEP),
        (f"step_h{args.traj_horizon_tokens}_xyz_gtroot",
         args.traj_horizon_tokens, "gt", False, _TPATH_STEP),
        (f"step_h{args.traj_horizon_tokens}_xyz_predroot",
         args.traj_horizon_tokens, "pred", True, _TPATH_STEP),
    ]

    all_records = []
    for mode_name, horizon, root_source, use_pred_root, traj_path in modes:
        print(f"\n--- {mode_name} ---")
        if mode_name == "generate_full":
            pred_motion, pred_root = run_generate_full(model, vae, sample, device)
        elif mode_name == "stream_generate_full":
            pred_motion, pred_root = run_stream_generate_full(model, vae, sample, device)
        else:
            pred_motion, pred_root = run_stream_step(
                model, vae, sample, device,
                history_length=args.history_length,
                num_denoise_steps=args.num_denoise_steps,
                horizon_tokens=horizon,
                use_pred_root=use_pred_root,
            )

        metrics = _build_mode_metrics(
            pred_root, gt_root, target_traj, mode_name, horizon, root_source,
            traj_encoder_path=traj_path,
        )
        print(f"  ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}")

        mode_dir = os.path.join(out_root, mode_name)
        _save_artifacts(
            mode_dir, pred_motion, pred_root,
            gt_root[: len(pred_root)], target_traj[: len(pred_root)], metrics,
        )
        all_records.append(metrics)

    # Summary
    summary = {
        "sample_id": args.sample_id,
        "dataset": args.dataset,
        "ckpt": args.ckpt,
        "vae_ckpt": args.vae_ckpt,
        "config": args.config,
        "history_length": args.history_length,
        "traj_horizon_tokens": args.traj_horizon_tokens,
        "num_denoise_steps": args.num_denoise_steps,
        "seed": args.seed,
        "modes": all_records,
    }
    summary_path = os.path.join(out_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    print("\nDiagnostic Matrix:")
    print(f"{'Mode':<35} {'ADE':>8} {'FDE':>8}")
    print("-" * 53)
    for rec in all_records:
        print(f"{rec['mode']:<35} {rec['ADE']:>8.4f} {rec['FDE']:>8.4f}")


if __name__ == "__main__":
    main()
