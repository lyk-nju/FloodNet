import os
import numpy as np
import torch
import random
from lightning.pytorch.utilities import rank_zero_info

from .eval_summary import save_eval_payloads

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
    from FloodNet.eval.ldf.conditioning import prepare_ldf_eval_model_batch
    from FloodNet.utils.traj_batch import root_to_traj_feats
    from FloodNet.utils.training import (
        build_generation_eval_cfg,
        ckpt_step_info,
        control_loss_train_mode,
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
    from eval.ldf.conditioning import prepare_ldf_eval_model_batch
    from utils.traj_batch import root_to_traj_feats
    from utils.training import (
        build_generation_eval_cfg,
        ckpt_step_info,
        control_loss_train_mode,
        resolve_test_probe_tag,
    )


def _hash_sd(sd):
    import hashlib
    _h = hashlib.sha256()
    for _k in sorted(sd.keys()):
        _h.update(sd[_k].detach().cpu().numpy().tobytes())
    return _h.hexdigest()[:16]

def _hash_ema(ema):
    import hashlib
    _h = hashlib.sha256()
    for _s in ema.shadow_params:
        _h.update(_s.detach().cpu().numpy().tobytes())
    return _h.hexdigest()[:16]

def _dump_eval_debug(module, model_batch, sample_seed, step_tag):
    """Write comprehensive debug state to FLOODNET_DEBUG_DIR/eval_state.log."""
    import hashlib, time

    _dbg_dir = os.environ.get("FLOODNET_DEBUG_DIR", "/tmp")
    _dbg_file = os.path.join(_dbg_dir, "eval_state.log")
    _lines = []
    _w = _lines.append

    _w(f"=== {step_tag} seed={sample_seed} time={time.time():.3f} ===")

    # 1. Model state_dict hash
    _w(f"sd_hash={_hash_sd(module.model.state_dict())}")

    # 2. EMA shadow hash
    _w(f"ema_hash={_hash_ema(module.ema)}")

    # 3. Key param spot-checks
    for _name in (
        "model.blocks.0.self_attn.q.weight",
        "model.blocks.0.cross_attn.q.weight",
        "controlnet.blocks.0.self_attn.q.weight",
        "controlnet.zero_out.0.weight",
    ):
        _p = dict(module.model.named_parameters()).get(_name)
        if _p is not None:
            _w(f"param.{_name} abs_mean={_p.detach().abs().mean().item():.10f}")

    # 4. model_batch
    _feat = model_batch.get("feature")
    if _feat is not None:
        _h = hashlib.sha256()
        _h.update(_feat.detach().cpu().numpy().tobytes())
        _w(f"feature shape={tuple(_feat.shape)} abs_mean={_feat.abs().mean():.6f} hash={_h.hexdigest()[:16]}")
    _text = model_batch.get("text", [None])
    _w(f"text={_text[0] if _text else 'N/A'}")

    _traj = model_batch.get("traj_features", model_batch.get("traj"))
    if _traj is not None:
        _h = hashlib.sha256()
        _h.update(_traj.detach().cpu().numpy().tobytes())
        _w(f"traj shape={tuple(_traj.shape)} abs_mean={_traj.abs().mean():.6f} hash={_h.hexdigest()[:16]}")

    # 5. Config knobs
    _w(f"cfg_text={module.cfg.model.params.get('cfg_scale_text')} cfg_traj={module.cfg.model.params.get('cfg_scale_traj')}")
    _w(f"chunk={module.model.chunk_size} noise_steps={module.model.noise_steps} pred={module.model.prediction_type}")
    _w(f"device={module.device} inf_mode={torch.is_inference_mode_enabled()}")

    with open(_dbg_file, "a") as _f:
        _f.write("\n".join(_lines) + "\n\n")


def _check_ckpt_consistency(module, step_tag):
    """Compare email-applied model state against the saved ckpt hash.
    Uses EMA-applied weights (what generation actually uses), not raw state_dict.
    Always writes to {save_dir}/ckpt_consistency.txt."""
    import hashlib, json
    try:
        _hash_path = os.path.join(module.cfg.save_dir, "ckpt_hash.txt")
        if not os.path.exists(_hash_path):
            return
        with open(_hash_path) as _fh:
            _saved = json.load(_fh)
        # Build EMA-applied state_dict: replace trainable params with EMA shadows
        _sd = {k: v.clone() for k, v in module.model.state_dict().items()}
        _trainable_names = [
            n for n, p in module.model.named_parameters() if p.requires_grad
        ]
        for _name, _s in zip(_trainable_names, module.ema.shadow_params):
            if _name in _sd:
                _sd[_name] = _s.clone()
        _h = hashlib.sha256()
        for _k, _v in sorted(_sd.items()):
            _h.update(_v.cpu().numpy().tobytes())
        _eval_hash = _h.hexdigest()
        _match = "MATCH" if _eval_hash == _saved["hash"] else "MISMATCH"
        _out_path = os.path.join(module.cfg.save_dir, "ckpt_consistency.txt")
        with open(_out_path, "a") as _f:
            _f.write(
                f"{step_tag} {_match} "
                f"saved_hash={_saved['hash'][:16]} "
                f"eval_hash={_eval_hash[:16]}\n"
            )
    except Exception:
        pass


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


def run_validation_generation_eval(module, batch, batch_idx=None, test_loader_idx=0):
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
    step_tag = ckpt_step_info(module).step_tag

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
            _all_cap_traj = []
            _all_cap_ctrl = []
            _all_flat_traj = []
            _all_flat_ctrl = []
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
                        train_mode=control_loss_train_mode(module.cfg),
                        chunk_size_tokens=getattr(module.model, "chunk_size", None),
                        window_mode=eval_cfg["forward_ctrl_window_mode"],
                        model_batch_builder=prepare_ldf_eval_model_batch,
                    )
                except Exception as e:
                    rank_zero_info(
                        f"[validation fwd_ctrl_loss] sample={sample_name} deterministic eval failed: {e}"
                    )

            _debug = os.environ.get("FLOODNET_DEBUG", "") == "1"
            _all = sample_batch.get("text_all", None)
            _eval_all = eval_cfg.get("eval_all_captions", False)
            # text_all is a list-of-lists from collate; extract inner list.
            _all_captions = _all[0] if (_all and _eval_all) else [sample_batch["text"][0]]
            for _cap_idx, _cap_text in enumerate(_all_captions):
                _cap_batch = {k: v for k, v in sample_batch.items()}
                _cap_batch["text"] = [_cap_text]
                _cap_traj = []
                _cap_ctrl = []

                for run_idx in range(generation_num_runs):
                    sample_seed = _stable_eval_seed(
                        module.cfg.seed, probe_tag, sample_name,
                        _cap_idx * generation_num_runs + run_idx
                    )
                    _seed_eval_locally(sample_seed)
                    if run_idx == 0 and _cap_idx == 0 and sample_idx == 0:
                        _check_ckpt_consistency(module, step_tag)
                    if _debug and run_idx == 0 and _cap_idx == 0 and sample_idx == 0:
                        _dbg_file = os.path.join(
                            os.environ.get("FLOODNET_DEBUG_DIR", "/tmp"), "eval_state.log")
                        _pre_sd = _hash_sd(module.model.state_dict())
                        _pre_ema = _hash_ema(module.ema)
                        with open(_dbg_file, "a") as _f:
                            _f.write(
                                f"[PRE-EMA {step_tag}] sd_hash={_pre_sd} "
                                f"ema_hash={_pre_ema}\n"
                            )
                    with module.ema.average_parameters(
                        [p for p in module.model.parameters() if p.requires_grad]
                    ):
                        model_batch = prepare_ldf_eval_model_batch(
                            _cap_batch,
                            module.device,
                            model=module.model,
                        )
                        if _debug and run_idx == 0 and _cap_idx == 0 and sample_idx == 0:
                            _dump_eval_debug(module, model_batch, sample_seed, step_tag)
                        output = module.model.generate(model_batch)
                    if _debug and run_idx == 0 and _cap_idx == 0 and sample_idx == 0:
                        _post_sd = _hash_sd(module.model.state_dict())
                        _post_ema = _hash_ema(module.ema)
                        with open(_dbg_file, "a") as _f:
                            _f.write(
                                f"[POST-EMA {step_tag}] sd_hash={_post_sd} "
                                f"ema_hash={_post_ema} "
                                f"sd_restored={_pre_sd == _post_sd}\n"
                            )

                    single_generated = output["generated"][0]
                    decoded_single_generated = module.vae.decode(
                        single_generated[None, :].to(module.device)
                    )[0].float().detach()

                    if run_idx == 0 and _cap_idx == 0:
                        sample_text = output["text"][0]
                        token_run0 = single_generated.float().cpu().numpy()
                        feature_run0 = decoded_single_generated.cpu().numpy()
                        if _debug:
                            _debug_token_path = os.path.join(
                                os.environ.get("FLOODNET_DEBUG_DIR", "/tmp"),
                                f"debug_token_{sample_name}.npy",
                            )
                            np.save(_debug_token_path, token_run0)
                            rank_zero_info(
                                f"[DEBUG token] saved {sample_name} "
                                f"token shape={token_run0.shape} "
                                f"abs_mean={np.abs(token_run0).mean():.6f}"
                            )

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
                        _cap_traj.append(
                            _compute_traj_metrics(
                                decoded_single_generated, sample_batch, 0, seg_size=eval_seg_size
                            )
                        )
                        _cap_ctrl.append(
                            _compute_omni_control_metrics(decoded_single_generated, sample_batch, 0)
                        )

                if do_eval_metrics:
                    _all_cap_traj.append(_average_traj_metrics(_cap_traj))
                    _all_cap_ctrl.append(_average_control_metrics(_cap_ctrl))
                    _all_flat_traj.extend(_cap_traj)
                    _all_flat_ctrl.extend(_cap_ctrl)

            record = None
            if do_eval_metrics:
                record = {
                    "name": sample_name,
                    "num_runs": eval_num_runs,
                    "num_captions": len(_all_captions),
                    "probe_tag": probe_tag,
                    "text": sample_text,
                    "text_all": _all_captions,
                }
                # Aggregate per-caption metrics: for each scalar field,
                # store caption-mean and cross-caption mean/std.
                _scalar_fields = (
                    "ade", "fde", "mse", "traj_jitter",
                    "path_arc_ade", "path_chamfer",
                    "control_l2_dist", "skating_ratio",
                    "traj_fail_20cm", "traj_fail_50cm",
                    "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err_m",
                    "T", "masked_ratio",
                )
                for _field in _scalar_fields:
                    _vals = [m.get(_field, float("nan")) for m in _all_cap_traj]
                    _vals = [v for v in _vals if v == v]  # filter nan
                    if _vals:
                        record[_field] = float(np.mean(_vals))
                        record[f"{_field}_std"] = float(np.std(_vals))
                for _field in _scalar_fields:
                    _vals = [m.get(_field, float("nan")) for m in _all_cap_ctrl]
                    _vals = [v for v in _vals if v == v]
                    if _vals:
                        record[f"caption_ctrl_{_field}"] = float(np.mean(_vals))
                        record[f"caption_ctrl_{_field}_std"] = float(np.std(_vals))
                record["caption_ade_mean"] = record.get("ade", float("nan"))
                record["caption_ade_std"] = record.get("ade_std", float("nan"))
                # List fields: element-wise mean/std across captions
                for _field in ("seg_mse", "prefix_mse"):
                    _lists = [m.get(_field, []) for m in _all_cap_traj]
                    if _lists and _lists[0]:
                        _n = max(len(l) for l in _lists)
                        _means = []
                        for _i in range(_n):
                            _vals = [l[_i] for l in _lists if _i < len(l) and l[_i] is not None]
                            _means.append(float(np.mean(_vals)) if _vals else None)
                        record[_field] = _means
                # Per-caption raw dicts (summary can aggregate across samples)
                record["_caption_traj"] = _all_cap_traj
                record["_caption_ctrl"] = _all_cap_ctrl
                if fwd_stat is not None:
                    record["fwd_ctrl_loss"] = fwd_stat.get("loss", float("nan"))
                    record["fwd_ctrl_loss_std"] = fwd_stat.get("loss_std", float("nan"))
                    record["fwd_n_valid"] = fwd_stat.get("n_valid", float("nan"))
                    record["fwd_win_len"] = fwd_stat.get("window_len", float("nan"))
                    record["fwd_num_windows"] = fwd_stat.get("num_windows", 0)

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
        save_eval_payloads(module, all_payloads, probe_tag, step_tag)
        return {"output": None}
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
