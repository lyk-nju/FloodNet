import numpy as np
import torch
import random
from lightning.pytorch.utilities import rank_zero_info

from .eval_summary import save_inline_eval_payloads

try:
    from FloodNet.metrics.traj import (
        _average_control_metrics,
        _average_traj_metrics,
        _compute_deterministic_fwd_ctrl_loss_sample,
        _compute_omni_control_metrics,
        _compute_traj_metrics,
        _seed_eval_locally,
        _slice_single_sample_batch,
        _stable_eval_seed,
    )
    from FloodNet.utils.traj_batch import root_to_traj_feats
    from FloodNet.utils.training import (
        build_generation_eval_cfg,
        prepare_model_input,
        compute_checkpoint_step_info,
        resolve_test_probe_tag,
    )
except ImportError:  # pragma: no cover - script entrypoints use top-level imports
    from metrics.traj import (
        _average_control_metrics,
        _average_traj_metrics,
        _compute_deterministic_fwd_ctrl_loss_sample,
        _compute_omni_control_metrics,
        _compute_traj_metrics,
        _seed_eval_locally,
        _slice_single_sample_batch,
        _stable_eval_seed,
    )
    from utils.traj_batch import root_to_traj_feats
    from utils.training import (
        build_generation_eval_cfg,
        prepare_model_input,
        compute_checkpoint_step_info,
        resolve_test_probe_tag,
    )


def _gather_payloads(local_payloads):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered_payloads = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_payloads, local_payloads)
        return [
            payload
            for rank_payloads in gathered_payloads
            for payload in (rank_payloads or [])
        ]
    return local_payloads


def run_inline_generation_eval(module, batch, batch_idx=None, test_loader_idx=0):
    # Fix seed for reproducible test generation, but save/restore the training RNG so
    # that training noise stays i.i.d. when the training loop resumes after validation.
    py_state = random.getstate()
    np_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    eval_cfg = build_generation_eval_cfg(module.cfg)
    eval_num_runs = max(eval_cfg["num_runs"], 1)
    eval_seg_size = eval_cfg["seg_size"]
    do_eval_metrics = eval_cfg["enabled"] and "traj" in batch and "traj_mask" in batch
    generation_num_runs = eval_num_runs if do_eval_metrics else 1
    probe_tag = resolve_test_probe_tag(module, test_loader_idx)
    step_tag = compute_checkpoint_step_info(module).step_tag

    try:
        local_payloads = []
        for sample_idx in range(len(batch["name"])):
            sample_batch = _slice_single_sample_batch(batch, sample_idx)
            sample_name = sample_batch["name"][0]
            sample_dataset_id = sample_batch["dataset"][0]
            sample_text = ""
            token_run0 = None
            feature_run0 = None
            traj_xz = None
            traj_mask = None
            frames = None
            traj_runs = []
            control_runs = []
            fwd_stat = None

            if "feature_text_end" in sample_batch:
                frames = np.asarray(sample_batch["feature_text_end"][0])

            if do_eval_metrics and eval_cfg["forward_ctrl_loss"] and "traj" in sample_batch:
                try:
                    fwd_stat = _compute_deterministic_fwd_ctrl_loss_sample(
                        model=module.model,
                        sample_batch=sample_batch,
                        vae=module.vae,
                        device=module.device,
                        train_mode=int(module.cfg.get("control_loss_train_mode", 3)),
                        chunk_size_tokens=getattr(module.model, "chunk_size", None),
                        window_mode=eval_cfg["forward_ctrl_window_mode"],
                    )
                except Exception as e:
                    rank_zero_info(
                        f"[inline fwd_ctrl_loss] sample={sample_name} deterministic eval failed: {e}"
                    )

            for run_idx in range(generation_num_runs):
                sample_seed = _stable_eval_seed(
                    module.cfg.seed, probe_tag, sample_name, run_idx
                )
                _seed_eval_locally(sample_seed)
                with module.ema.average_parameters(
                    [p for p in module.model.parameters() if p.requires_grad]
                ):
                    model_batch = prepare_model_input(sample_batch)
                    output = module.model.generate(model_batch)

                single_generated = output["generated"][0]
                decoded_single_generated = module.vae.decode(
                    single_generated[None, :].to(module.device)
                )[0].float().detach()

                if run_idx == 0:
                    sample_text = output["text"][0]
                    token_run0 = single_generated.float().cpu().numpy()
                    feature_run0 = decoded_single_generated.cpu().numpy()

                    feat_len = int(decoded_single_generated.shape[0])
                    if "traj_features" in sample_batch:
                        cond = sample_batch["traj_features"][0]
                        if torch.is_tensor(cond):
                            cond = cond.detach().cpu().numpy()
                        cond = np.asarray(cond)
                        if cond.ndim == 2 and cond.shape[1] >= 2:
                            traj_xz = cond[:feat_len, :2].astype(np.float32)
                    if traj_xz is None and "traj" in sample_batch:
                        traj = sample_batch["traj"][0]
                        if torch.is_tensor(traj):
                            traj = traj.detach().cpu().numpy()
                        traj = np.asarray(traj)[:feat_len]
                        if traj.ndim == 2 and traj.shape[1] >= 3:
                            traj_xz = root_to_traj_feats(traj)[:, :2].astype(np.float32)
                    if "traj_mask" in sample_batch:
                        traj_mask_i = sample_batch["traj_mask"][0]
                        if torch.is_tensor(traj_mask_i):
                            traj_mask_i = traj_mask_i.detach().cpu().numpy()
                        traj_mask = np.asarray(traj_mask_i).reshape(-1)[:feat_len]

                if do_eval_metrics:
                    traj_runs.append(
                        _compute_traj_metrics(
                            decoded_single_generated, sample_batch, 0, seg_size=eval_seg_size
                        )
                    )
                    control_runs.append(
                        _compute_omni_control_metrics(decoded_single_generated, sample_batch, 0)
                    )

            record = None
            if do_eval_metrics:
                record = {
                    "name": sample_name,
                    "num_runs": eval_num_runs,
                    "probe_tag": probe_tag,
                }
                if traj_runs:
                    record.update(_average_traj_metrics(traj_runs))
                if control_runs:
                    record.update(_average_control_metrics(control_runs))
                if fwd_stat is not None:
                    record["fwd_ctrl_loss"] = fwd_stat.get("loss", float("nan"))
                    record["fwd_ctrl_loss_std"] = fwd_stat.get("loss_std", float("nan"))
                    record["fwd_n_valid"] = fwd_stat.get("n_valid", float("nan"))
                    record["fwd_win_len"] = fwd_stat.get("window_len", float("nan"))
                    record["fwd_num_windows"] = fwd_stat.get("num_windows", 0)
                record["_traj_runs"] = traj_runs
                record["_control_runs"] = control_runs

            local_payloads.append(
                {
                    "name": sample_name,
                    "dataset_id": sample_dataset_id,
                    "text": sample_text,
                    "token": token_run0,
                    "feature": feature_run0,
                    "traj_xz": traj_xz,
                    "traj_mask": traj_mask,
                    "frames": frames,
                    "record": record,
                }
            )

        all_payloads = _gather_payloads(local_payloads)
        save_inline_eval_payloads(module, all_payloads, probe_tag, step_tag)
        return {"output": None}
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
