import json
import os
from pathlib import Path

import numpy as np
from lightning.pytorch.utilities import rank_zero_info


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

    seen = module._inline_eval_seen.setdefault((probe_tag, step_tag), set())
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
