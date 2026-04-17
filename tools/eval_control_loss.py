import argparse
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

# Ensure project root is importable when running as:
#   python tools/eval_control_loss.py --config ...
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.initialize import Config, get_function, instantiate
from utils.motion_process import extract_root_trajectory_263_torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate control loss on test split with train-identical pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ldf.yaml",
        help="Path to yaml config (e.g., configs/ldf.yaml).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Model checkpoint path. If omitted, use config test_ckpt then resume_ckpt.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override test batch size. Default uses cfg.data.test_bs.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override test dataloader workers. Default uses cfg.data.num_workers.",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Do not load/apply EMA state even if checkpoint contains it.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="If >0, additionally print top-k hardest samples by control loss.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for deterministic/reproducible control-loss evaluation.",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="forward",
        choices=["generate", "forward"],
        help=(
            "Evaluation path: "
            "'generate' matches validation visualization (model.generate), "
            "'forward' uses training forward/control_aux active-window loss."
        ),
    )
    parser.add_argument(
        "--set",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "OmegaConf dot-path overrides, e.g. "
            "--set model.params.cfg_scale_text=3.0 model.params.cfg_scale_traj=5.0"
        ),
    )
    return parser.parse_args()


def _copy_traj_fields_to_model_batch(batch: Dict[str, Any], model_batch: Dict[str, Any]):
    if "traj" in batch:
        model_batch["traj"] = batch["traj"]
        model_batch["traj_length"] = batch["traj_length"]
        model_batch["traj_mask"] = batch["traj_mask"]
    if "token_mask" in batch:
        model_batch["token_mask"] = batch["token_mask"]
    if "traj_features" in batch:
        model_batch["traj_features"] = batch["traj_features"]


def _to_device(obj: Any, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    return obj


def _compute_control_loss_xz_stats(
    pred_list,
    traj,
    traj_mask,
    traj_length,
    vae,
    device,
    chunk_size_tokens: int | None = None,
    token_to_frame: int = 4,
) -> Tuple[torch.Tensor, float]:
    """Same math as train_ldf.py::_compute_control_loss_xz, but returns numerator/denominator."""
    loss_control_sum = torch.tensor(0.0, device=device)
    n_valid = 0.0
    for i in range(len(pred_list)):
        pred_latent_full = pred_list[i].to(device)  # (T_token, z_dim)
        t_tok = pred_latent_full.size(0)

        if chunk_size_tokens is not None and t_tok > chunk_size_tokens:
            start_tok = t_tok - chunk_size_tokens
            start_f = 0 if start_tok == 0 else 4 * start_tok - 3
            end_f = t_tok * token_to_frame  # clamped to L_motion = 4*(t_tok-1)+1
        else:
            start_f = 0
            end_f = None

        decoded = vae.decode(pred_latent_full.unsqueeze(0))[0].float()
        l_motion = decoded.size(0)
        l_gt_total = min(int(traj_length[i].item()), traj.shape[1])

        if end_f is None:
            pred_sl = slice(0, l_motion)
            gt_sl = slice(0, l_gt_total)
        else:
            pred_sl = slice(min(start_f, l_motion), min(end_f, l_motion))
            gt_sl = slice(min(start_f, l_gt_total), min(end_f, l_gt_total))

        l = min(pred_sl.stop - pred_sl.start, gt_sl.stop - gt_sl.start)
        if l <= 0:
            continue

        pred_traj_full = extract_root_trajectory_263_torch(decoded.unsqueeze(0))
        pred_traj = pred_traj_full[:, pred_sl, :][:, :l, :]
        gt_traj = traj[i, gt_sl, :][:l].unsqueeze(0).to(
            pred_traj.device, dtype=pred_traj.dtype
        )
        mask = traj_mask[i, gt_sl][:l].unsqueeze(0).to(
            pred_traj.device, dtype=pred_traj.dtype
        )

        pred_xz = pred_traj[..., [0, 2]]
        gt_xz = gt_traj[..., [0, 2]]
        sq_err = ((pred_xz - gt_xz) ** 2).sum(dim=-1)
        loss_control_sum = loss_control_sum + (mask * sq_err).sum()
        n_valid += mask.sum().item()
    return loss_control_sum, n_valid


def _load_model(cfg, ckpt_path: str, device: torch.device, use_ema: bool):
    model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    strict = not (
        cfg.model.params.get("use_traj_cond", False)
        or cfg.model.params.get("use_controlnet_traj", False)
    )
    load_result = model.load_state_dict(checkpoint["state_dict"], strict=strict)
    if not strict:
        if load_result.missing_keys:
            print(
                "[load] strict=False missing keys (new modules expected): "
                f"{len(load_result.missing_keys)}"
            )
        if load_result.unexpected_keys:
            print(
                "[load] strict=False unexpected keys ignored: "
                f"{len(load_result.unexpected_keys)}"
            )
        # If ckpt is from pre-ControlNet backbone, re-copy loaded backbone params into
        # ControlNet so initialization behavior matches "copy-from-backbone" intent.
        if (
            getattr(model, "controlnet", None) is not None
            and bool(getattr(model, "controlnet_init_from_backbone", False))
            and load_result.missing_keys
        ):
            model.controlnet.init_from_backbone(model.model)
            print("[load] Re-initialized ControlNet from loaded backbone weights.")
    if use_ema and ("ema_state" in checkpoint):
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_decay)
        try:
            ema.load_state_dict(checkpoint["ema_state"])
            ema.copy_to(model.parameters())
            print("[load] Applied EMA weights from checkpoint.")
        except ValueError as e:
            print(
                "[load] EMA state incompatible with current model params "
                f"(likely architecture changed, e.g. new ControlNet). "
                f"Skip EMA and continue with model weights. Detail: {e}"
            )
    model.to(device)
    model.eval()
    return model


def _load_vae(cfg, device: torch.device):
    vae = instantiate(target=cfg.test_vae.target, cfg=None, hfstyle=False, **cfg.test_vae.params)
    vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
    vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
    if "ema_state" in vae_ckpt:
        vae_ema = ExponentialMovingAverage(vae.parameters(), decay=cfg.test_vae.ema_decay)
        vae_ema.load_state_dict(vae_ckpt["ema_state"])
        vae_ema.copy_to(vae.parameters())
    vae.to(device)
    vae.eval()
    return vae


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Keep behavior aligned with train script.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _compute_control_loss_xz_per_sample(
    pred_list,
    traj,
    traj_mask,
    traj_length,
    vae,
    device,
    chunk_size_tokens: int | None = None,
    token_to_frame: int = 4,
) -> List[Dict[str, Any]]:
    """Per-sample stats aligned with train_ldf control-loss math.

    Returns per-sample stats with debug fields (t_tok / valid_len / window length).
    """
    out = []
    for i in range(len(pred_list)):
        pred_latent_full = pred_list[i].to(device)
        t_tok = pred_latent_full.size(0)
        if chunk_size_tokens is not None and t_tok > chunk_size_tokens:
            start_tok = t_tok - chunk_size_tokens
            start_f = 0 if start_tok == 0 else 4 * start_tok - 3
            end_f = t_tok * token_to_frame  # clamped to L_motion = 4*(t_tok-1)+1
        else:
            start_f = 0
            end_f = None

        decoded = vae.decode(pred_latent_full.unsqueeze(0))[0].float()
        l_motion = decoded.size(0)
        l_gt_total = min(int(traj_length[i].item()), traj.shape[1])
        if end_f is None:
            pred_sl = slice(0, l_motion)
            gt_sl = slice(0, l_gt_total)
        else:
            pred_sl = slice(min(start_f, l_motion), min(end_f, l_motion))
            gt_sl = slice(min(start_f, l_gt_total), min(end_f, l_gt_total))
        l = min(pred_sl.stop - pred_sl.start, gt_sl.stop - gt_sl.start)
        if l <= 0:
            out.append(
                {
                    "loss_sum": torch.tensor(0.0, device=device),
                    "n_valid": 0.0,
                    "t_tok": int(t_tok),
                    "decoded_len": int(l_motion),
                    "pred_window_len": int(pred_sl.stop - pred_sl.start),
                    "gt_window_len": int(gt_sl.stop - gt_sl.start),
                    "valid_len": int(0),
                }
            )
            continue

        pred_traj_full = extract_root_trajectory_263_torch(decoded.unsqueeze(0))
        pred_traj = pred_traj_full[:, pred_sl, :][:, :l, :]
        gt_traj = traj[i, gt_sl, :][:l].unsqueeze(0).to(
            pred_traj.device, dtype=pred_traj.dtype
        )
        mask = traj_mask[i, gt_sl][:l].unsqueeze(0).to(
            pred_traj.device, dtype=pred_traj.dtype
        )
        pred_xz = pred_traj[..., [0, 2]]
        gt_xz = gt_traj[..., [0, 2]]
        sq_err = ((pred_xz - gt_xz) ** 2).sum(dim=-1)
        out.append(
            {
                "loss_sum": (mask * sq_err).sum(),
                "n_valid": float(mask.sum().item()),
                "t_tok": int(t_tok),
                "decoded_len": int(l_motion),
                "pred_window_len": int(pred_sl.stop - pred_sl.start),
                "gt_window_len": int(gt_sl.stop - gt_sl.start),
                "valid_len": int(l),
            }
        )
    return out


def _compute_generated_control_loss_xz_per_sample(
    generated_list,
    traj,
    traj_mask,
    traj_length,
    vae,
    device,
) -> List[Dict[str, Any]]:
    """Per-sample control loss from validation-style generated latents.

    This follows visualization path: model.generate -> decode full generated motion,
    then compare root xz against GT traj on full valid range.
    """
    out = []
    for i in range(len(generated_list)):
        pred_latent = generated_list[i].to(device)  # (T_token, z_dim)
        decoded = vae.decode(pred_latent.unsqueeze(0))[0].float()  # (T_motion, 263)
        pred_traj = extract_root_trajectory_263_torch(decoded.unsqueeze(0))  # (1, T, 3)
        l_motion = pred_traj.size(1)
        l_gt_total = min(int(traj_length[i].item()), traj.shape[1])
        l = min(l_motion, l_gt_total)
        if l <= 0:
            out.append(
                {
                    "loss_sum": torch.tensor(0.0, device=device),
                    "n_valid": 0.0,
                    "t_tok": int(pred_latent.size(0)),
                    "decoded_len": int(l_motion),
                    "pred_window_len": int(l_motion),
                    "gt_window_len": int(l_gt_total),
                    "valid_len": int(0),
                }
            )
            continue

        pred_xz = pred_traj[:, :l, [0, 2]]
        gt_xz = traj[i, :l, :][:, [0, 2]].unsqueeze(0).to(
            pred_xz.device, dtype=pred_xz.dtype
        )
        mask = traj_mask[i, :l].unsqueeze(0).to(pred_xz.device, dtype=pred_xz.dtype)
        sq_err = ((pred_xz - gt_xz) ** 2).sum(dim=-1)
        out.append(
            {
                "loss_sum": (mask * sq_err).sum(),
                "n_valid": float(mask.sum().item()),
                "t_tok": int(pred_latent.size(0)),
                "decoded_len": int(l_motion),
                "pred_window_len": int(l_motion),
                "gt_window_len": int(l_gt_total),
                "valid_len": int(l),
            }
        )
    return out


def main():
    args = parse_args()
    _set_seed(args.seed)
    cfg = Config(args.config).config

    # Apply --set overrides (dot-path KEY=VALUE pairs)
    for kv in (args.set or []):
        if "=" not in kv:
            raise ValueError(f"--set expects KEY=VALUE, got: {kv!r}")
        key, val = kv.split("=", 1)
        # Try to cast to int/float/bool; fall back to string
        for cast in (int, float, lambda x: {"true": True, "false": False}[x.lower()]):
            try:
                val = cast(val)
                break
            except (ValueError, KeyError):
                pass
        OmegaConf.update(cfg, key, val, merge=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt or OmegaConf.select(cfg, "test_ckpt") or OmegaConf.select(cfg, "resume_ckpt")
    if not ckpt_path:
        raise ValueError("No checkpoint specified. Please set --ckpt or config test_ckpt/resume_ckpt.")

    collate_fn = get_function(cfg.data.collate_fn) if cfg.data.get("collate_fn", None) else None
    test_dataset = instantiate(cfg.data.get("test_target", cfg.data.target), cfg=cfg, split="test")
    test_bs = int(args.batch_size or cfg.data.test_bs)
    num_workers = int(args.num_workers if args.num_workers is not None else cfg.data.num_workers)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=False,
        prefetch_factor=8 if num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    model = _load_model(cfg, ckpt_path, device, use_ema=not args.no_ema)
    vae = _load_vae(cfg, device)
    chunk_size_tokens = getattr(model, "chunk_size", None)

    total_loss_sum = torch.tensor(0.0, device=device)
    total_valid = 0.0
    total_batches = 0
    used_batches = 0
    per_sample_rows = []

    with torch.no_grad():
        for batch in test_loader:
            total_batches += 1
            model_batch = batch.copy()
            model_batch["feature"] = batch["token"]
            model_batch["feature_length"] = batch["token_length"]
            if "token_text_end" in batch:
                model_batch["feature_text_end"] = batch["token_text_end"]
            _copy_traj_fields_to_model_batch(batch, model_batch)
            model_batch = _to_device(model_batch, device)

            if args.eval_mode == "generate":
                out = model.generate(model_batch)
                if "generated" not in out or "traj" not in batch:
                    continue
                sample_stats = _compute_generated_control_loss_xz_per_sample(
                    generated_list=out["generated"],
                    traj=model_batch["traj"],
                    traj_mask=model_batch["traj_mask"],
                    traj_length=model_batch["traj_length"],
                    vae=vae,
                    device=device,
                )
            else:
                out = model(model_batch)
                if "control_aux" not in out or "traj" not in batch:
                    continue
                pred_list = out["control_aux"]["pred_x0_latent_list"]
                sample_stats = _compute_control_loss_xz_per_sample(
                    pred_list=pred_list,
                    traj=model_batch["traj"],
                    traj_mask=model_batch["traj_mask"],
                    traj_length=model_batch["traj_length"],
                    vae=vae,
                    device=device,
                    chunk_size_tokens=chunk_size_tokens,
                )

            if "traj" not in batch:
                continue
            loss_sum = torch.tensor(0.0, device=device)
            n_valid = 0.0
            names = batch.get("name", None)
            if names is None:
                names = [f"sample_{total_batches:04d}_{i:03d}" for i in range(len(sample_stats))]
            for i, stat in enumerate(sample_stats):
                loss_i = stat["loss_sum"]
                n_i = stat["n_valid"]
                name_i = names[i] if i < len(names) else f"sample_{total_batches:04d}_{i:03d}"
                avg_i = (loss_i / n_i).item() if n_i > 0 else float("nan")
                loss_sum = loss_sum + loss_i
                n_valid += n_i
                per_sample_rows.append(
                    {
                        "name": str(name_i),
                        "loss_sum": float(loss_i.item()),
                        "n_valid": float(n_i),
                        "loss": float(avg_i),
                        "t_tok": int(stat["t_tok"]),
                        "decoded_len": int(stat["decoded_len"]),
                        "pred_window_len": int(stat["pred_window_len"]),
                        "gt_window_len": int(stat["gt_window_len"]),
                        "valid_len": int(stat["valid_len"]),
                    }
                )
            if n_valid > 0:
                used_batches += 1
                total_loss_sum = total_loss_sum + loss_sum
                total_valid += n_valid

    if total_valid <= 0:
        print("No valid control-loss samples found (check traj/traj_mask/control_aux settings).")
        return

    avg_control_loss = (total_loss_sum / total_valid).item()
    print("===== Control Loss Evaluation =====")
    print(f"config: {args.config}")
    print(f"seed: {args.seed}")
    print(f"eval_mode: {args.eval_mode}")
    print(f"checkpoint: {ckpt_path}")
    print(f"split: test (from cfg.data.test_meta_paths)")
    print(f"batches: {total_batches}, used_batches: {used_batches}")
    print(f"valid_points: {total_valid:.0f}")
    print(f"control_loss_xz: {avg_control_loss:.8f}")
    print("----- Per-sample control loss -----")
    for row in per_sample_rows:
        if row["n_valid"] <= 0:
            print(f"{row['name']}\tcontrol_loss_xz=nan\tvalid_points=0")
        else:
            print(
                f"{row['name']}\tcontrol_loss_xz={row['loss']:.8f}\t"
                f"valid_points={int(row['n_valid'])}\t"
                f"t_tok={row['t_tok']}\tdecoded_len={row['decoded_len']}\t"
                f"pred_window_len={row['pred_window_len']}\t"
                f"gt_window_len={row['gt_window_len']}\tvalid_len={row['valid_len']}"
            )
    if args.topk > 0:
        hard = [r for r in per_sample_rows if r["n_valid"] > 0 and r["loss"] == r["loss"]]
        hard = sorted(hard, key=lambda r: r["loss"], reverse=True)[: args.topk]
        print(f"----- Top-{args.topk} hardest samples -----")
        for row in hard:
            print(
                f"{row['name']}\tcontrol_loss_xz={row['loss']:.8f}\t"
                f"valid_points={int(row['n_valid'])}"
            )


if __name__ == "__main__":
    main()

