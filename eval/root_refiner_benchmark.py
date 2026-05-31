"""Refiner standalone benchmark (T_A_10, Benchmark A).

Evaluates RootRefiner predictions against RefinerDataset targets WITHOUT the
body model. Computes the metric suite from docs/TODO.md §T_A_10:

    num_token_top1_accuracy / num_token_top3_accuracy / num_token_MAE
    xyz_ADE / xyz_FDE
    heading_error_deg (median)
    fwd_speed_MAE      (per-frame fwd_delta channel; "speed" is the metric label)
    lateral_speed_MAE  (metric only — perpendicular drift)
    yaw_rate_MAE       (per-frame yaw_delta channel)
    smoothness_acc_mean

Outputs a JSON summary + per-sample CSV.

⚠ Done-criteria thresholds (num_token top-1 > 0.5, heading_error_deg < 30°
median) require a TRAINED checkpoint (T_A_09). With random weights the pipeline
runs end-to-end but the numbers are meaningless — the smoke test only checks
that metrics are finite and the report is written.

References:
- docs/TODO.md §T_A_10 lines 1441-1478.
- docs/design.md §10.3 (Benchmark A metric definitions).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from torch import Tensor

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.local_frame import wrap_angle   # noqa: E402
from utils.refiner.path_feature_stats import compute_sampling_config_hash  # noqa: E402

_RAD2DEG = 180.0 / math.pi


# ---------------------------------------------------------------------------
# Per-sample metric helpers (operate on a single sample's valid frames)
# ---------------------------------------------------------------------------


def _heading_to_yaw(cos_sin: Tensor) -> Tensor:
    """[..., 2] (cos, sin) → yaw angle via atan2(sin, cos)."""
    return torch.atan2(cos_sin[..., 1], cos_sin[..., 0])


def _lateral_component(xyz: Tensor, yaw: Tensor) -> Tensor:
    """Per-frame lateral (perpendicular-to-heading) displacement magnitude.

    xyz: [T, 3] world/local positions; yaw: [T] heading per frame.
    Returns [T-1] lateral displacement magnitudes (frame t uses heading[t-1]).
    """
    if xyz.shape[0] < 2:
        return xyz.new_zeros(0)
    delta_xz = xyz[1:, [0, 2]] - xyz[:-1, [0, 2]]            # [T-1, 2]
    # perpendicular to heading_dir_xz(yaw) = [sin, cos]; perp = [cos, -sin]
    yaw_prev = yaw[:-1]
    perp = torch.stack([torch.cos(yaw_prev), -torch.sin(yaw_prev)], dim=-1)  # [T-1, 2]
    lateral = (delta_xz * perp).sum(-1).abs()                # [T-1]
    return lateral


def compute_sample_metrics(pred_wp: Tensor, gt_wp: Tensor, mask: Tensor) -> dict:
    """Metrics for a single sample over valid frames.

    pred_wp / gt_wp: [T, 7]; mask: [T] bool. Returns dict of python floats.
    Frames where mask is False are excluded. Empty-valid → metrics are NaN.
    """
    valid = mask.bool()
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return {k: float("nan") for k in (
            "xyz_ADE", "xyz_FDE", "heading_error_deg",
            "fwd_speed_MAE", "yaw_rate_MAE", "lateral_speed_MAE", "smoothness",
        )}

    pv = pred_wp[valid]
    gv = gt_wp[valid]

    # xyz ADE / FDE.
    xyz_err = (pv[:, 0:3] - gv[:, 0:3]).norm(dim=-1)          # [n_valid]
    ade = xyz_err.mean().item()
    fde = xyz_err[-1].item()

    # heading error (deg).
    pred_yaw = _heading_to_yaw(pv[:, 3:5])
    gt_yaw = _heading_to_yaw(gv[:, 3:5])
    head_err = wrap_angle(pred_yaw - gt_yaw).abs() * _RAD2DEG
    heading_error_deg = head_err.median().item()

    # fwd_delta / yaw_delta MAE (channels 5, 6).
    fwd_mae = (pv[:, 5] - gv[:, 5]).abs().mean().item()
    yaw_mae = (pv[:, 6] - gv[:, 6]).abs().mean().item()

    # lateral drift MAE (metric only) — compare pred vs gt lateral displacement.
    pred_lat = _lateral_component(pv[:, 0:3], pred_yaw)
    gt_lat = _lateral_component(gv[:, 0:3], gt_yaw)
    if pred_lat.numel() > 0:
        lateral_mae = (pred_lat - gt_lat).abs().mean().item()
    else:
        lateral_mae = float("nan")

    # smoothness: mean L2 of 2nd-order diff of predicted [fwd_delta, yaw_delta].
    if pv.shape[0] >= 3:
        dyn = pv[:, 5:7]
        diff2 = dyn[2:] - 2 * dyn[1:-1] + dyn[:-2]
        smoothness = (diff2 ** 2).sum(-1).mean().item()
    else:
        smoothness = 0.0

    return {
        "xyz_ADE": ade,
        "xyz_FDE": fde,
        "heading_error_deg": heading_error_deg,
        "fwd_speed_MAE": fwd_mae,
        "yaw_rate_MAE": yaw_mae,
        "lateral_speed_MAE": lateral_mae,
        "smoothness": smoothness,
    }


# ---------------------------------------------------------------------------
# Aggregate benchmark
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_benchmark(model, dataset, text_encoder, device="cpu",
                   max_samples: int = -1, oracle_duration: bool = False) -> dict:
    """Run inference over `dataset` and aggregate the metric suite.

    `model` is a RootRefiner (eval mode). `text_encoder` must have
    `.encode(list[str], device=...) -> [B, text_emb_dim]`.

    `oracle_duration`: if True, feed the GT num_tokens to the model so the
    trajectory metrics (xyz_ADE/FDE, heading, ...) measure the WAYPOINT DECODER
    ALONE under the correct horizon, isolating it from num_token-head prediction
    error. Default False = real inference (model picks its own argmax horizon, so
    trajectory metrics conflate duration + waypoint quality). num_token_* metrics
    are always reported against the head's argmax regardless of this flag.

    Returns dict with `summary` (aggregate metrics) and `per_sample` (list).
    """
    model = model.to(device).eval()
    min_tokens = model.min_tokens

    # Reproducibility: get_sample advances the dataset RNG every call (mode/anchor/
    # num_tokens dice), so reset it to the base seed → repeat runs (and oracle vs
    # normal) see the identical sample sequence.
    if hasattr(dataset, "reset_rng"):
        dataset.reset_rng()

    n = len(dataset) if max_samples < 0 else min(max_samples, len(dataset))

    per_sample = []
    num_top1 = 0
    num_top3 = 0
    num_mae_sum = 0.0

    for idx in range(n):
        # ⚠ Use get_sample(force_no_path_aug=True) so the benchmark measures
        # CLEAN inference: dataset[idx] would apply random trim/sparse path
        # augmentation, making metrics non-deterministic and inflating errors.
        # force_text_idx=0 pins the canonical first caption (training randomizes
        # over all captions for augmentation; eval must stay comparable).
        if hasattr(dataset, "get_sample"):
            sample = dataset.get_sample(idx, force_no_path_aug=True, force_text_idx=0)
        else:
            sample = dataset[idx]
        text_emb = text_encoder.encode([sample["text"]], device=device)
        # oracle_duration: teacher-force the GT horizon so the trajectory metrics
        # isolate the waypoint decoder (the model honors num_tokens when given,
        # regardless of eval mode — see RootRefiner.forward gate).
        oracle_nt = (
            sample["num_tokens"].reshape(1).to(device) if oracle_duration else None
        )
        out = model(
            text_emb=text_emb,
            # R2.4: feed the new shim keys. The legacy aliases mapped
            # path_stats(3-dim) → path_features, which now mismatches the
            # duration head's path_features_dim (5). The shim emits the 5-dim
            # physical path_features and a unit-aligned geometry `path`.
            path=sample["path"].unsqueeze(0).to(device),
            path_valid_mask=sample["path_valid_mask"].unsqueeze(0).to(device),
            path_features=sample["path_features"].unsqueeze(0).to(device),
            history_motion=sample["history_motion"].unsqueeze(0).to(device),
            history_mask=sample["history_mask"].unsqueeze(0).to(device),
            num_tokens=oracle_nt,
        )
        logits = out["num_token_logits"][0]                  # [K]
        # Model emits NORMALIZED 5D. Assemble physical 7D at the boundary
        # (unnormalize xyz → unit heading → append fwd_delta / yaw_delta) so
        # metrics are computed in physical space. GT `target_waypoints` is
        # PHYSICAL-then-z-scored, so unnormalize its xyz the same way and
        # re-derive its deltas (do NOT trust target_wp[..., 5:7] — those are
        # z-scored physical deltas that would mismatch the freshly-derived
        # physical pred deltas in scale and offset).
        from utils.motion_process import build_physical_7d_from_normalized_5d
        wp_mean = getattr(dataset, "_wp_mean", None)
        wp_std = getattr(dataset, "_wp_std", None)
        wp_norm_idx = getattr(dataset, "_wp_norm_idx", None)
        pred_wp = build_physical_7d_from_normalized_5d(
            out["waypoints"][0].cpu(), wp_mean, wp_std, wp_norm_idx,
        )                                                    # [max_frames, 7] physical
        gt5_norm = sample["target_waypoints"][..., :5]
        gt_wp = build_physical_7d_from_normalized_5d(
            gt5_norm, wp_mean, wp_std, wp_norm_idx,
        )                                                    # [max_frames, 7] physical
        mask = sample["target_mask"]

        # num_token metrics.
        gt_class = int(sample["num_tokens"].item()) - min_tokens
        gt_class = max(0, min(gt_class, logits.shape[-1] - 1))
        pred_class = int(logits.argmax().item())
        top3 = torch.topk(logits, k=min(3, logits.shape[-1])).indices.tolist()
        num_top1 += int(pred_class == gt_class)
        num_top3 += int(gt_class in top3)
        num_mae_sum += abs(pred_class - gt_class)

        m = compute_sample_metrics(pred_wp, gt_wp, mask)
        m["idx"] = idx
        m["pred_num_tokens"] = pred_class + min_tokens
        m["gt_num_tokens"] = gt_class + min_tokens
        per_sample.append(m)

    # Aggregate (nan-safe mean over per-sample metrics).
    def _nanmean(key):
        vals = [s[key] for s in per_sample if not math.isnan(s[key])]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def _nanmedian(key):
        vals = sorted(s[key] for s in per_sample if not math.isnan(s[key]))
        if not vals:
            return float("nan")
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            return float(vals[mid])
        return float((vals[mid - 1] + vals[mid]) / 2)

    summary = {
        "n_samples": n,
        "oracle_duration": oracle_duration,
        "num_token_top1_accuracy": num_top1 / n if n else float("nan"),
        "num_token_top3_accuracy": num_top3 / n if n else float("nan"),
        "num_token_MAE": num_mae_sum / n if n else float("nan"),
        "xyz_ADE": _nanmean("xyz_ADE"),
        "xyz_FDE": _nanmean("xyz_FDE"),
        "heading_error_deg": _nanmedian("heading_error_deg"),
        "fwd_speed_MAE": _nanmean("fwd_speed_MAE"),
        "lateral_speed_MAE": _nanmean("lateral_speed_MAE"),
        "yaw_rate_MAE": _nanmean("yaw_rate_MAE"),
        "smoothness_acc_mean": _nanmean("smoothness"),
    }
    return {"summary": summary, "per_sample": per_sample}


def write_report(result: dict, output_dir: str | Path) -> None:
    """Write summary.json + per_sample.csv."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.json").open("w") as f:
        json.dump(result["summary"], f, indent=2)
    per_sample = result["per_sample"]
    if per_sample:
        keys = list(per_sample[0].keys())
        with (out / "per_sample.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_sample)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_model_from_ckpt(ckpt_path: str, device: str):
    """Load RootRefiner weights from a Lightning checkpoint or raw state_dict."""
    from train_refiner import RefinerLightningModule

    ckpt = torch.load(ckpt_path, map_location=device)
    if "hyper_parameters" in ckpt and "cfg" in ckpt["hyper_parameters"]:
        cfg = ckpt["hyper_parameters"]["cfg"]
        module = RefinerLightningModule(cfg)
        module.load_state_dict(ckpt["state_dict"])
        return module.refiner, module.text_encoder
    raise ValueError(
        f"checkpoint {ckpt_path} missing hyper_parameters.cfg; "
        "pass a Lightning checkpoint saved by train_refiner.py"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/root_refiner.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/refiner_bench")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--split_file", type=str, default=None,
                         help="Eval split; defaults to data.val_split_file or the dataset default.")
    args = parser.parse_args(argv)

    from datasets.humanml3d_refiner import HumanML3DRefinerDataset as RefinerDataset
    from scripts.compute_5d_stats import load_clips_from_dir

    from train_refiner import resolve_cfg_interpolations

    from train_refiner import _load_cfg
    cfg = _load_cfg(args.config)
    # A-P0-1: resolve ${data.raw_data_dir} etc. (the model cfg comes from the
    # ckpt's saved hparams, which train_refiner already resolved).
    cfg = resolve_cfg_interpolations(cfg)

    model, text_encoder = _load_model_from_ckpt(args.ckpt, args.device)

    data_cfg = cfg.get("data", {})
    split_file = args.split_file or data_cfg.get("val_split_file")
    clips = load_clips_from_dir(
        data_cfg["raw_data_dir"],
        dataset=data_cfg.get("dataset", "humanml3d"),
        split_file=split_file,
        feature_path=data_cfg.get("feature_path"),
        text_path=data_cfg.get("text_path"),
    )
    model_cfg = cfg["model"]["params"]
    # P0-2: follow data.normalize exactly (like train_refiner.py); do NOT force
    # normalize just because a stats_dir is present in the config — that loaded
    # stats even when normalize:false and crashed if the dir was missing.
    normalize = bool(data_cfg.get("normalize", False))
    dataset = RefinerDataset(
        clips,
        n_hist=model_cfg["n_hist"], n_path=model_cfg["n_path"],
        max_tokens=model_cfg["max_tokens"], min_tokens=model_cfg["min_tokens"],
        frames_per_token=model_cfg["frames_per_token"],
        normalize=normalize,
        stats_dir=data_cfg.get("stats_dir") if normalize else None,
        # R2.5: must mirror training's path-feature normalization, else the
        # duration head (which reads path_features via the raw skip) sees an
        # out-of-distribution scale at eval and num_token metrics go garbage.
        path_feature_stats_dir=(
            data_cfg.get("path_feature_stats_dir") if normalize else None
        ),
        sampling_config_hash=(
            compute_sampling_config_hash(cfg)
            if normalize and data_cfg.get("path_feature_stats_dir")
            else None
        ),
        seed=0,
    )

    result = run_benchmark(model, dataset, text_encoder, device=args.device,
                            max_samples=args.max_samples)
    write_report(result, args.output_dir)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
