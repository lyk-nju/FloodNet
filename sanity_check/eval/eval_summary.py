import json
import os
from pathlib import Path

import numpy as np
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_info

try:
    from FloodNet.metrics.traj import _get_metric_statistics
    from FloodNet.utils.training import (
        ckpt_step_info,
        compute_step_semantics,
        get_test_probe_tags,
    )
    from FloodNet.utils.visualize import make_composite_compare_videos, render_video
except ImportError:  # pragma: no cover - script entrypoints use top-level imports
    from metrics.traj import _get_metric_statistics
    from utils.training import (
        ckpt_step_info,
        compute_step_semantics,
        get_test_probe_tags,
    )
    from utils.visualize import make_composite_compare_videos, render_video


def flatten_inline_eval_summary(summary: dict, prefix: str) -> dict:
    flat_metrics = {}
    for key, value in summary.items():
        log_key = f"{prefix}/{key}"
        if isinstance(value, list):
            for idx, item in enumerate(value):
                if item is not None:
                    flat_metrics[f"{log_key}/slot_{idx}"] = float(item)
        else:
            flat_metrics[log_key] = float(value)
    return flat_metrics


def build_inline_eval_summary(sample_records):
    summary = {}
    valid_traj = [r for r in sample_records if "ade" in r and r["ade"] == r["ade"]]
    if valid_traj:
        ades = [r["ade"] for r in valid_traj]
        fdes = [r["fde"] for r in valid_traj]
        mses = [r["mse"] for r in valid_traj]
        summary["traj/ADE_mean"] = float(np.mean(ades))
        summary["traj/ADE_std"] = float(np.std(ades))
        summary["traj/FDE_mean"] = float(np.mean(fdes))
        summary["traj/FDE_std"] = float(np.std(fdes))
        summary["traj/MSE_mean"] = float(np.mean(mses))
        summary["traj/MSE_std"] = float(np.std(mses))
        summary["traj/n_samples"] = len(valid_traj)

        max_segs = max(len(r.get("seg_mse", [])) for r in valid_traj)
        seg_means = []
        for seg_idx in range(max_segs):
            vals = [
                r["seg_mse"][seg_idx]
                for r in valid_traj
                if seg_idx < len(r.get("seg_mse", [])) and r["seg_mse"][seg_idx] is not None
            ]
            seg_means.append(float(np.mean(vals)) if vals else None)
        summary["traj/seg_mse_per_slot"] = seg_means

        max_pfx = max(len(r.get("prefix_mse", [])) for r in valid_traj)
        prefix_means = []
        for prefix_idx in range(max_pfx):
            vals = [
                r["prefix_mse"][prefix_idx]
                for r in valid_traj
                if prefix_idx < len(r.get("prefix_mse", []))
                and r["prefix_mse"][prefix_idx] is not None
            ]
            prefix_means.append(float(np.mean(vals)) if vals else None)
        summary["traj/prefix_mse_per_slot"] = prefix_means

        jitter_vals = [
            r["traj_jitter"]
            for r in valid_traj
            if "traj_jitter" in r and r["traj_jitter"] == r["traj_jitter"]
        ]
        if jitter_vals:
            summary["traj/jitter_mean"] = float(np.mean(jitter_vals))
            summary["traj/jitter_std"] = float(np.std(jitter_vals))

        path_arc_ades = [
            r["path_arc_ade"]
            for r in valid_traj
            if "path_arc_ade" in r and r["path_arc_ade"] == r["path_arc_ade"]
        ]
        if path_arc_ades:
            summary["path/arc_ADE_mean"] = float(np.mean(path_arc_ades))
            summary["path/arc_ADE_std"] = float(np.std(path_arc_ades))

        path_chamfers = [
            r["path_chamfer"]
            for r in valid_traj
            if "path_chamfer" in r and r["path_chamfer"] == r["path_chamfer"]
        ]
        if path_chamfers:
            summary["path/chamfer_mean"] = float(np.mean(path_chamfers))
            summary["path/chamfer_std"] = float(np.std(path_chamfers))

        fwd_vals = [
            r["fwd_ctrl_loss"]
            for r in valid_traj
            if "fwd_ctrl_loss" in r and r["fwd_ctrl_loss"] == r["fwd_ctrl_loss"]
        ]
        if fwd_vals:
            summary["traj/fwd_ctrl_loss_mean"] = float(np.mean(fwd_vals))
            summary["traj/fwd_ctrl_loss_std"] = float(np.std(fwd_vals))
            fwd_run_std_vals = [
                r["fwd_ctrl_loss_std"]
                for r in valid_traj
                if "fwd_ctrl_loss_std" in r and r["fwd_ctrl_loss_std"] == r["fwd_ctrl_loss_std"]
            ]
            if fwd_run_std_vals:
                summary["traj/fwd_ctrl_loss_run_std_mean"] = float(np.mean(fwd_run_std_vals))

    control_metric_keys = (
        "control_l2_dist",
        "skating_ratio",
        "traj_fail_20cm",
        "traj_fail_50cm",
        "kps_fail_20cm",
        "kps_fail_50cm",
        "kps_mean_err_m",
    )
    control_name_map = {
        "control_l2_dist": "Control_L2_dist",
        "skating_ratio": "Skating_Ratio",
        "traj_fail_20cm": "traj_fail_20cm",
        "traj_fail_50cm": "traj_fail_50cm",
        "kps_fail_20cm": "kps_fail_20cm",
        "kps_fail_50cm": "kps_fail_50cm",
        "kps_mean_err_m": "kps_mean_err_m",
    }
    max_runs = max(len(r.get("_control_runs", [])) for r in sample_records)
    for key in control_metric_keys:
        per_run_vals = []
        for run_idx in range(max_runs):
            vals = []
            for record in sample_records:
                control_runs = record.get("_control_runs", [])
                if run_idx < len(control_runs):
                    val = control_runs[run_idx].get(key, float("nan"))
                    if val == val:
                        vals.append(val)
            if vals:
                per_run_vals.append(float(np.mean(vals)))
        if per_run_vals:
            mean, std, conf = _get_metric_statistics(
                np.asarray(per_run_vals, dtype=np.float64), len(per_run_vals)
            )
            out_key = control_name_map[key]
            summary[f"control/{out_key}_mean"] = float(mean)
            summary[f"control/{out_key}_std"] = float(std)
            summary[f"control/{out_key}_conf_interval"] = float(conf)
            summary[f"control/{out_key}_num_runs"] = int(len(per_run_vals))
    return summary


def _render_probe_outputs(module, dataset_id, probe_tag, artifact_dirs):
    if not module.cfg.test_setting.render:
        return

    render_video(
        motion_dir=str(artifact_dirs["feature"]),
        save_dir=str(artifact_dirs["video"]),
        render_setting=module.cfg.test_setting,
        frames_dir=str(artifact_dirs["frames"]),
        traj_mask_dir=str(artifact_dirs["traj_mask"]),
        cond_traj_dir=str(artifact_dirs["traj_xz"]),
    )

    make_composite_compare_videos(
        result_folder=str(artifact_dirs["video"]),
        compare_folders=module.cfg.test_setting.get(dataset_id, {}).get(
            "compare_folders", None
        ),
        compare_names=module.cfg.test_setting.get(dataset_id, {}).get(
            "compare_names", None
        ),
        text_folder=str(artifact_dirs["text"]),
        save_dir=str(artifact_dirs["composite"]),
    )

    if (
        not module.cfg.debug
        and module.logger is not None
        and isinstance(module.logger, WandbLogger)
    ):
        video_to_log = []
        for video_path in sorted(os.listdir(artifact_dirs["composite"])):
            video_to_log.append(
                wandb.Video(
                    str(artifact_dirs["composite"] / video_path),
                    format="gif",
                )
            )
        wandb.log(
            {f"{dataset_id}_{probe_tag}_video": video_to_log},
            step=compute_step_semantics(module).absolute_step,
        )


def _save_summary(metrics_dir, summary, sample_records):
    summary_path = Path(metrics_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"summary": summary, "samples": sample_records}, f, indent=2)


def _log_probe_summary(module, dataset_id, probe_tag, step_tag, summary):
    rank_zero_info(
        f"[eval][{dataset_id}][{probe_tag}][{step_tag}] "
        f"ADE={summary.get('traj/ADE_mean', float('nan')):.4f} "
        f"FDE={summary.get('traj/FDE_mean', float('nan')):.4f} "
        f"PathArc={summary.get('path/arc_ADE_mean', float('nan')):.4f} "
        f"ControlL2={summary.get('control/Control_L2_dist_mean', float('nan')):.4f}"
    )

    flat_metrics = flatten_inline_eval_summary(
        summary,
        prefix=f"eval/{probe_tag}/{dataset_id}",
    )
    if flat_metrics and module.logger is not None:
        module.logger.log_metrics(
            flat_metrics,
            step=compute_step_semantics(module).absolute_step,
        )


def process_inline_generation_results(module):
    step_tag = ckpt_step_info(module).step_tag
    for dataset_id in os.listdir(module.cfg.save_dir):
        feature_root = Path(module.cfg.save_dir) / dataset_id / "feature"
        if not os.path.exists(feature_root):
            continue
        for probe_tag in get_test_probe_tags(module):
            artifact_dirs = build_inline_eval_artifact_dirs(
                module.cfg.save_dir,
                dataset_id,
                probe_tag,
                step_tag,
            )
            if not artifact_dirs["feature"].exists():
                continue

            _render_probe_outputs(module, dataset_id, probe_tag, artifact_dirs)

            if not artifact_dirs["metrics"].exists():
                continue
            sample_records = load_inline_eval_sample_records(artifact_dirs["metrics"])
            if not sample_records:
                continue

            summary = build_inline_eval_summary(sample_records)
            _save_summary(artifact_dirs["metrics"], summary, sample_records)
            _log_probe_summary(module, dataset_id, probe_tag, step_tag, summary)


# ------------------------------------------------------------------
# Artifact I/O (moved from inline_eval_artifacts.py)
# ------------------------------------------------------------------


def build_inline_eval_artifact_dirs(save_dir, dataset_id, probe_tag, step_tag):
    base_dir = Path(save_dir) / dataset_id
    return {
        "text": base_dir / "text" / probe_tag / step_tag,
        "token": base_dir / "token" / probe_tag / step_tag,
        "feature": base_dir / "feature" / probe_tag / step_tag,
        "traj_xz": base_dir / "traj_xz" / probe_tag / step_tag,
        "traj_mask": base_dir / "traj_mask" / probe_tag / step_tag,
        "frames": base_dir / "frames" / probe_tag / step_tag,
        "metrics": base_dir / "metrics" / probe_tag / step_tag,
        "video": base_dir / "video" / probe_tag / step_tag,
        "composite": base_dir / "composite" / probe_tag / step_tag,
    }


def save_inline_eval_payloads(module, payloads, probe_tag, step_tag):
    if module.trainer.global_rank != 0:
        return

    seen = module._inline_eval_dedup.setdefault((probe_tag, step_tag), set())
    for payload in payloads:
        sample_id = payload["name"]
        dataset_id = payload["dataset_id"]
        dedupe_key = (dataset_id, sample_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        dirs = build_inline_eval_artifact_dirs(
            module.cfg.save_dir,
            dataset_id,
            probe_tag,
            step_tag,
        )

        try:
            os.makedirs(dirs["text"], exist_ok=True)
            with open(dirs["text"] / f"{sample_id}.txt", "w") as f:
                f.write(payload["text"])

            os.makedirs(dirs["token"], exist_ok=True)
            np.save(dirs["token"] / f"{sample_id}.npy", payload["token"])

            os.makedirs(dirs["feature"], exist_ok=True)
            np.save(dirs["feature"] / f"{sample_id}.npy", payload["feature"])

            if payload["traj_xz"] is not None:
                os.makedirs(dirs["traj_xz"], exist_ok=True)
                np.save(dirs["traj_xz"] / f"{sample_id}.npy", payload["traj_xz"])
            if payload["traj_mask"] is not None:
                os.makedirs(dirs["traj_mask"], exist_ok=True)
                np.save(dirs["traj_mask"] / f"{sample_id}.npy", payload["traj_mask"])
            if payload["frames"] is not None:
                os.makedirs(dirs["frames"], exist_ok=True)
                np.save(dirs["frames"] / f"{sample_id}.npy", payload["frames"])
            if payload["record"] is not None:
                os.makedirs(dirs["metrics"], exist_ok=True)
                with open(dirs["metrics"] / f"{sample_id}.json", "w") as f:
                    json.dump(payload["record"], f, indent=2)
        except Exception as e:
            rank_zero_info(
                f"Error in saving motion {sample_id} of dataset {dataset_id}: {e}"
            )


def load_inline_eval_sample_records(metrics_dir):
    sample_records = []
    for metric_file in sorted(os.listdir(metrics_dir)):
        if not metric_file.endswith(".json"):
            continue
        with open(Path(metrics_dir) / metric_file, "r") as f:
            sample_records.append(json.load(f))
    return sample_records
