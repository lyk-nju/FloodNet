import argparse
import json
import os
import random
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Multi-GPU stream eval spawns one Python process per device. Cap BLAS/OpenMP
# thread pools before importing numpy/torch so workers do not exhaust RLIMIT_NPROC.
for _thread_env_key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_key, "1")

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch_ema import ExponentialMovingAverage

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from FloodNet.metrics.stream import (
        compute_stream_boundary_metrics,
        compute_root_path_yaw_error,
        compute_stream_vs_offline_metrics,
        decode_stream_chunks,
        summarize_stream_records,
    )
    from FloodNet.metrics.traj import (
        _average_control_metrics,
        _average_traj_metrics,
        _compute_omni_control_metrics,
        _compute_traj_metrics,
        _seed_eval_locally,
        _slice_single_sample_batch,
        _stable_eval_seed,
    )
    from FloodNet.eval.ldf.conditioning import (
        LdfEvalStreamConditioner,
        prepare_ldf_eval_model_batch,
    )
    from FloodNet.eval.common.visualization import (
        plot_xz_trajectories,
        plot_yaw_series,
        render_motion_video,
        yaw_from_7d,
        yaw_from_root_path,
    )
    from FloodNet.utils.initialize import get_function, instantiate, load_config
    from FloodNet.utils.motion_process import (
        StreamJointRecovery263,
        extract_root_trajectory_263_torch,
    )
    from FloodNet.utils.stream_rollout import (
        StreamTextRolloutController,
        build_stream_step_model_input,
        build_stream_suffix_conditioning,
        clip_traj_input_to_horizon,
    )
except ImportError:  # pragma: no cover - script entrypoints use top-level imports
    from metrics.stream import (
        compute_stream_boundary_metrics,
        compute_root_path_yaw_error,
        compute_stream_vs_offline_metrics,
        decode_stream_chunks,
        summarize_stream_records,
    )
    from metrics.traj import (
        _average_control_metrics,
        _average_traj_metrics,
        _compute_omni_control_metrics,
        _compute_traj_metrics,
        _seed_eval_locally,
        _slice_single_sample_batch,
        _stable_eval_seed,
    )
    from eval.ldf.conditioning import LdfEvalStreamConditioner, prepare_ldf_eval_model_batch
    from eval.common.visualization import (
        plot_xz_trajectories,
        plot_yaw_series,
        render_motion_video,
        yaw_from_7d,
        yaw_from_root_path,
    )
    from utils.initialize import get_function, instantiate, load_config
    from utils.motion_process import (
        StreamJointRecovery263,
        extract_root_trajectory_263_torch,
    )
    from utils.stream_rollout import (
        StreamTextRolloutController,
        build_stream_step_model_input,
        build_stream_suffix_conditioning,
        clip_traj_input_to_horizon,
    )


class InMemorySampleDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate BABEL long-horizon streaming generation via stream_generate()."
    )
    parser.add_argument("--config", type=str, default="configs/eval_babel_stream.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--vae_ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--stream_mode",
        type=str,
        choices=["stream_generate", "stream_generate_step"],
        default=None,
    )
    parser.add_argument("--num_denoise_steps", type=int, default=None)
    parser.add_argument("--compute_offline_baseline", action="store_true")
    parser.add_argument("--no_compute_offline_baseline", action="store_true")
    parser.add_argument("--compute_no_traj_baseline", action="store_true")
    parser.add_argument("--no_compute_no_traj_baseline", action="store_true")
    parser.add_argument("--save_feature_npy", action="store_true")
    parser.add_argument("--save_latent_npy", action="store_true")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--no_save_plots", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--render_offline_video", action="store_true")
    parser.add_argument("--render_no_traj_video", action="store_true")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=(
            "Optional deterministic run directory name under --out_dir. "
            "Defaults to a timestamped ckpt/probe tag."
        ),
    )
    parser.add_argument("--probe_tag", type=str, default=None)
    parser.add_argument("--meta_paths", nargs="+", default=None)
    parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help=(
            "Number of GPU worker processes to launch, or a comma-separated "
            "visible device list such as 0,1,2,3. Stream eval shards samples "
            "across workers; each worker still runs batch_size=1."
        ),
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        choices=["gpu", "cpu"],
        default=None,
        help="Execution device type. Defaults to gpu when CUDA is available.",
    )
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument(
        "--set",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="OmegaConf dot-path overrides, e.g. --set model.params.cfg_scale_traj=3.0",
    )
    return parser.parse_args(argv)


def parse_args_from_list(argv):
    return parse_args(argv)


def resolve_eval_path_flags(args, cfg) -> tuple[bool, bool]:
    """Resolve LDF stream eval diagnostic paths.

    The design-default LDF suite runs stream GT, offline GT, and stream no-traj.
    CLI disable flags are explicit opt-outs.
    """
    compute_offline_baseline = bool(
        args.compute_offline_baseline
        or cfg.get("eval.compute_offline_baseline", True)
    )
    compute_no_traj_baseline = bool(
        args.compute_no_traj_baseline
        or cfg.get("eval.compute_no_traj_baseline", True)
    )
    if getattr(args, "no_compute_offline_baseline", False):
        compute_offline_baseline = False
    if args.no_compute_no_traj_baseline:
        compute_no_traj_baseline = False
    return compute_offline_baseline, compute_no_traj_baseline


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
    if n_shadow == len(all_params):
        return all_params
    if backbone_params and n_shadow == len(backbone_params):
        return backbone_params
    if getattr(model, "freeze_backbone", False) and getattr(model, "controlnet", None) is not None:
        ema_params = list(model.controlnet.parameters())
        if getattr(model, "traj_encoder", None) is not None:
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

    model = instantiate(
        target=cfg.model.target,
        cfg=None,
        hfstyle=False,
        **cfg.model.params,
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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


def build_stream_input(sample_batch: Dict, device: torch.device) -> Dict:
    return prepare_ldf_eval_model_batch(sample_batch, device)


def _to_python_int(value) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)




def run_stream_generate_sample(model, vae, sample_batch: Dict, device: torch.device, num_denoise_steps: Optional[int]):
    model_batch = build_stream_input(sample_batch, device)
    latent_chunks: List[torch.Tensor] = []
    for output in model.stream_generate(model_batch, num_denoise_steps=num_denoise_steps):
        latent_chunk = output["generated"][0]
        if latent_chunk is None or latent_chunk.shape[0] == 0:
            continue
        latent_chunks.append(latent_chunk.detach())

    decoded_feature, decoded_chunks, chunk_frame_ends = decode_stream_chunks(vae, latent_chunks)
    latent_stream = (
        torch.cat([chunk.detach().cpu() for chunk in latent_chunks], dim=0)
        if latent_chunks
        else torch.zeros((0, model.input_dim), dtype=torch.float32)
    )
    return {
        "decoded_feature": decoded_feature,
        "decoded_chunks": decoded_chunks,
        "chunk_frame_ends": chunk_frame_ends,
        "latent_stream": latent_stream,
    }


def run_stream_generate_step_sample(
    model,
    vae,
    sample_batch: Dict,
    device: torch.device,
    history_length: int,
    num_denoise_steps: Optional[int],
    traj_horizon_tokens: Optional[int] = None,
    token_dt: float = 0.20,
    frames_per_token: int = 4,
):
    total_tokens = _to_python_int(sample_batch["token_length"][0])
    total_frames = _to_python_int(sample_batch["feature_length"][0])
    step_count = total_tokens

    if num_denoise_steps is None:
        num_denoise_steps = int(getattr(model, "noise_steps"))

    model.init_generated(
        history_length,
        batch_size=1,
        num_denoise_steps=num_denoise_steps,
    )
    vae.clear_cache()

    text_rollout = StreamTextRolloutController.from_sample_batch(sample_batch)
    stream_conditioner = (
        LdfEvalStreamConditioner(
            sample_batch,
            history_length=history_length,
            traj_horizon_tokens=int(traj_horizon_tokens or 0),
            token_dt=float(token_dt),
            frames_per_token=int(frames_per_token),
            device=device,
        )
        if "traj_cond_7d" in sample_batch and sample_batch["traj_cond_7d"] is not None
        else None
    )
    stream_recovery = (
        StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
        if stream_conditioner is not None
        else None
    )
    first_chunk = True
    latent_tokens: List[torch.Tensor] = []
    decoded_chunks: List[torch.Tensor] = []
    chunk_frame_ends: List[int] = []
    generated_frames = 0

    try:
        for commit_index in range(step_count):
            current_text = text_rollout.get_text_for_commit_index(commit_index)
            if stream_conditioner is not None:
                local_commit_index = int(getattr(model, "commit_index", commit_index))
                chunk_size = int(getattr(model, "chunk_size", 1))
                traj_input = stream_conditioner.build_step_payload(
                    local_commit_index=local_commit_index,
                    absolute_commit_index=commit_index,
                    chunk_size=chunk_size,
                )
            else:
                traj_input = build_stream_suffix_conditioning(sample_batch, commit_index)
                if traj_horizon_tokens is not None and traj_horizon_tokens > 0:
                    traj_input = clip_traj_input_to_horizon(traj_input, traj_horizon_tokens)
            step_payload = build_stream_step_model_input(
                current_text,
                traj_input=traj_input,
            )
            output = model.stream_generate_step(step_payload, first_chunk=first_chunk)
            latent_token = output["generated"][0].detach().cpu()
            decoded_chunk = vae.stream_decode(
                output["generated"][0][None, :], first_chunk=first_chunk
            )[0].float().detach().cpu()
            first_chunk = False

            latent_tokens.append(latent_token)
            decoded_chunks.append(decoded_chunk)
            generated_frames += decoded_chunk.shape[0]
            chunk_frame_ends.append(min(generated_frames, total_frames))
            if stream_conditioner is not None and stream_recovery is not None:
                stream_conditioner.append_decoded(
                    decoded_chunk,
                    commit_idx=commit_index + 1,
                    recovery=stream_recovery,
                )
    finally:
        vae.clear_cache()

    decoded_feature = (
        torch.cat(decoded_chunks, dim=0)[:total_frames]
        if decoded_chunks
        else torch.zeros((0, 263), dtype=torch.float32)
    )
    latent_stream = (
        torch.cat(latent_tokens, dim=0)
        if latent_tokens
        else torch.zeros((0, model.input_dim), dtype=torch.float32)
    )
    return {
        "decoded_feature": decoded_feature,
        "decoded_chunks": decoded_chunks,
        "chunk_frame_ends": chunk_frame_ends,
        "latent_stream": latent_stream,
    }


def run_offline_generate_sample(model, vae, sample_batch: Dict, device: torch.device, num_denoise_steps: Optional[int]):
    model_batch = build_stream_input(sample_batch, device)
    output = model.generate(model_batch, num_denoise_steps=num_denoise_steps)
    latent = output["generated"][0].detach()
    decoded = vae.decode(latent.unsqueeze(0))[0].float().detach().cpu()
    return {
        "decoded_feature": decoded,
        "latent": latent.detach().cpu(),
    }


def _format_text(sample_batch: Dict) -> str:
    lines = []
    segment_names = sample_batch.get("segment_names", None)
    if isinstance(segment_names, list) and len(segment_names) == 1 and isinstance(segment_names[0], list):
        segment_names = segment_names[0]
    if segment_names:
        lines.append("segments: " + ", ".join(str(name) for name in segment_names))
    text_value = sample_batch.get("text", [""])[0]
    if isinstance(text_value, list):
        end_list = sample_batch.get("feature_text_end", [[]])[0]
        for idx, segment in enumerate(text_value):
            end_frame = end_list[idx] if idx < len(end_list) else None
            lines.append(f"[{idx}] end={end_frame}: {segment}")
        return "\n".join(lines)
    lines.append(str(text_value))
    return "\n".join(lines)


_TRAJECTORY_BATCH_KEYS = {
    "traj_cond_7d",
    "traj_cond",
    "traj",
    "traj_features",
    "traj_length",
    "traj_cond_mask",
    "traj_mask",
    "traj_loss_mask",
    "token_mask",
}


def _remove_trajectory_conditioning(sample_batch: Dict) -> Dict:
    return {
        key: value
        for key, value in sample_batch.items()
        if key not in _TRAJECTORY_BATCH_KEYS
    }


def _root_numpy(feature: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if feature is None:
        return None
    with torch.no_grad():
        return extract_root_trajectory_263_torch(feature[None, :])[0].cpu().numpy()


def _root_path_yaw_error(
    pred_feature: torch.Tensor,
    sample_batch: Dict,
) -> float:
    gt_feature = sample_batch.get("feature")
    if gt_feature is None:
        return float("nan")
    gt_single = gt_feature[0].float().cpu() if torch.is_tensor(gt_feature) and gt_feature.ndim == 3 else gt_feature
    pred_root = _root_numpy(pred_feature)
    gt_root = _root_numpy(gt_single)
    if pred_root is None or gt_root is None:
        return float("nan")
    return compute_root_path_yaw_error(pred_root, gt_root)


def _condition_root_numpy(sample_batch: Dict) -> Optional[np.ndarray]:
    traj7 = sample_batch.get("traj_cond_7d")
    if traj7 is not None:
        value = traj7[0] if torch.is_tensor(traj7) and traj7.ndim == 3 else traj7
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if arr.ndim == 2 and arr.shape[-1] >= 3:
            return arr[:, :3].astype(np.float32)
    traj = sample_batch.get("traj")
    if traj is not None:
        value = traj[0] if torch.is_tensor(traj) and traj.ndim == 3 else traj
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if arr.ndim == 2 and arr.shape[-1] >= 3:
            return arr[:, :3].astype(np.float32)
    return None


def _condition_yaw_numpy(sample_batch: Dict) -> Optional[np.ndarray]:
    traj7 = sample_batch.get("traj_cond_7d")
    if traj7 is None:
        return None
    value = traj7[0] if torch.is_tensor(traj7) and traj7.ndim == 3 else traj7
    yaw = yaw_from_7d(value)
    return yaw if yaw.shape[0] > 0 else None


def _save_sample_outputs(
    sample_dir: Path,
    sample_batch: Dict,
    sample_record: Dict,
    stream_feature: torch.Tensor,
    gt_feature: Optional[torch.Tensor],
    offline_feature: Optional[torch.Tensor],
    stream_no_traj_feature: Optional[torch.Tensor] = None,
    stream_latent: Optional[torch.Tensor] = None,
    offline_latent: Optional[torch.Tensor] = None,
    save_feature_npy: bool = True,
    save_latent_npy: bool = False,
    save_plots: bool = True,
    render_video: bool = False,
    render_offline_video: bool = False,
    render_no_traj_video: bool = False,
):
    sample_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_dir / "text.txt", "w") as f:
        f.write(_format_text(sample_batch))
    with open(sample_dir / "metrics.json", "w") as f:
        json.dump(sample_record, f, indent=2)

    gt_root = _root_numpy(gt_feature)
    stream_root = _root_numpy(stream_feature)
    offline_root = _root_numpy(offline_feature)
    no_traj_root = _root_numpy(stream_no_traj_feature)
    condition_root = _condition_root_numpy(sample_batch)

    if gt_root is not None:
        np.save(sample_dir / "gt_root.npy", gt_root.astype(np.float32))
    if condition_root is not None:
        np.save(sample_dir / "condition_root.npy", condition_root.astype(np.float32))
    if stream_root is not None:
        np.save(sample_dir / "stream_gt_root.npy", stream_root.astype(np.float32))
    if offline_root is not None:
        np.save(sample_dir / "offline_gt_root.npy", offline_root.astype(np.float32))
    if no_traj_root is not None:
        np.save(sample_dir / "stream_no_traj_root.npy", no_traj_root.astype(np.float32))

    if save_plots:
        plot_xz_trajectories(
            sample_dir / "plot_xz.png",
            {
                "gt_root": gt_root,
                "condition_root": condition_root,
                "stream_gt": stream_root,
                "offline_gt": offline_root,
                "stream_no_traj": no_traj_root,
            },
            title=str(sample_batch.get("name", ["sample"])[0]),
        )
        plot_yaw_series(
            sample_dir / "plot_yaw.png",
            {
                "gt_yaw": yaw_from_root_path(gt_root),
                "condition_yaw": _condition_yaw_numpy(sample_batch),
                "stream_gt_yaw": yaw_from_root_path(stream_root),
                "offline_gt_yaw": yaw_from_root_path(offline_root),
                "stream_no_traj_yaw": yaw_from_root_path(no_traj_root),
            },
            title=str(sample_batch.get("name", ["sample"])[0]),
        )

    if save_feature_npy:
        np.save(sample_dir / "stream_feature.npy", stream_feature.cpu().numpy())
        if gt_feature is not None:
            np.save(sample_dir / "gt_feature.npy", gt_feature.cpu().numpy())
        if offline_feature is not None:
            np.save(sample_dir / "offline_feature.npy", offline_feature.cpu().numpy())
        if stream_no_traj_feature is not None:
            np.save(
                sample_dir / "stream_no_traj_feature.npy",
                stream_no_traj_feature.cpu().numpy(),
            )

    if save_latent_npy and stream_latent is not None:
        np.save(sample_dir / "stream_latent.npy", stream_latent.cpu().numpy())
        if offline_latent is not None:
            np.save(sample_dir / "offline_latent.npy", offline_latent.cpu().numpy())

    if render_video:
        render_motion_video(
            stream_feature,
            sample_dir / "video_stream_gt.mp4",
            dim=263,
            traj_xz=condition_root[:, [0, 2]] if condition_root is not None else None,
        )
    if render_offline_video and offline_feature is not None:
        render_motion_video(
            offline_feature,
            sample_dir / "video_offline_gt.mp4",
            dim=263,
            traj_xz=condition_root[:, [0, 2]] if condition_root is not None else None,
        )
    if render_no_traj_video and stream_no_traj_feature is not None:
        render_motion_video(
            stream_no_traj_feature,
            sample_dir / "video_stream_no_traj.mp4",
            dim=263,
            traj_xz=condition_root[:, [0, 2]] if condition_root is not None else None,
        )


def _stream_eval_artifact_dirs(out_root: Path, dataset_id: str, probe_tag: str, step_tag: str) -> Dict[str, Path]:
    base_dir = Path(out_root) / dataset_id
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
        "condition_compare": base_dir / "condition_compare" / probe_tag / step_tag,
    }


def _to_numpy(value) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _frames_numpy(sample_batch: Dict) -> Optional[np.ndarray]:
    frames = sample_batch.get("feature_text_end")
    if frames is None:
        return None
    value = frames[0] if isinstance(frames, list) and len(frames) > 0 else frames
    arr = _to_numpy(value).reshape(-1)
    return arr.astype(np.int64) if arr.size > 0 else None


def _traj_mask_numpy(sample_batch: Dict, feat_len: int) -> Optional[np.ndarray]:
    mask = sample_batch.get("traj_mask")
    if mask is None:
        return None
    value = mask[0] if torch.is_tensor(mask) and mask.ndim >= 2 else mask
    arr = _to_numpy(value).reshape(-1)
    if feat_len > 0:
        arr = arr[:feat_len]
    return arr.astype(np.float32)


def _save_eval_style_sample_outputs(
    *,
    out_root: Path,
    dataset_id: str,
    probe_tag: str,
    step_tag: str,
    sample_name: str,
    sample_batch: Dict,
    sample_record: Dict,
    stream_feature: Optional[torch.Tensor],
    stream_latent: Optional[torch.Tensor] = None,
    gt_feature: Optional[torch.Tensor] = None,
    offline_feature: Optional[torch.Tensor] = None,
    stream_no_traj_feature: Optional[torch.Tensor] = None,
) -> None:
    dirs = _stream_eval_artifact_dirs(Path(out_root), dataset_id, probe_tag, step_tag)
    for out_dir in dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    with open(dirs["text"] / f"{sample_name}.txt", "w") as f:
        f.write(_format_text(sample_batch))

    with open(dirs["metrics"] / f"{sample_name}.json", "w") as f:
        json.dump(sample_record, f, indent=2)

    feat_len = 0
    if stream_feature is not None:
        stream_np = _to_numpy(stream_feature).astype(np.float32)
        feat_len = int(stream_np.shape[0]) if stream_np.ndim > 0 else 0
        np.save(dirs["feature"] / f"{sample_name}.npy", stream_np)

    if stream_latent is not None:
        np.save(
            dirs["token"] / f"{sample_name}.npy",
            _to_numpy(stream_latent).astype(np.float32),
        )

    condition_root = _condition_root_numpy(sample_batch)
    if condition_root is not None:
        cond_xz = condition_root[:, [0, 2]].astype(np.float32)
        if feat_len > 0:
            cond_xz = cond_xz[:feat_len]
        np.save(dirs["traj_xz"] / f"{sample_name}.npy", cond_xz)

    traj_mask = _traj_mask_numpy(sample_batch, feat_len)
    if traj_mask is not None:
        np.save(dirs["traj_mask"] / f"{sample_name}.npy", traj_mask)

    frames = _frames_numpy(sample_batch)
    if frames is not None:
        np.save(dirs["frames"] / f"{sample_name}.npy", frames)

    plot_xz_trajectories(
        dirs["condition_compare"] / f"{sample_name}.png",
        {
            "gt_root": _root_numpy(gt_feature),
            "condition_root": condition_root,
            "stream_gt": _root_numpy(stream_feature),
            "offline_gt": _root_numpy(offline_feature),
            "stream_no_traj": _root_numpy(stream_no_traj_feature),
        },
        title=str(sample_name),
    )


def _build_run_name(ckpt_path: str, probe_tag: str, stream_mode: str) -> str:
    ckpt_tag = Path(ckpt_path).stem.replace("=", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{probe_tag}_{stream_mode}_{ckpt_tag}"


def _resolve_run_name(
    *,
    ckpt_path: str,
    probe_tag: str,
    stream_mode: str,
    requested_run_name: str | None,
) -> str:
    if requested_run_name:
        return requested_run_name
    return _build_run_name(ckpt_path, probe_tag, stream_mode)


def _average_scalar_metric(run_metrics: List[Dict], key: str) -> float:
    vals = [metric[key] for metric in run_metrics if key in metric and metric[key] == metric[key]]
    return float(np.mean(vals)) if vals else float("nan")


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


def _rank_payload_path(run_dir: Path, rank: int) -> Path:
    return run_dir / "_rank_records" / f"rank_{rank}.json"


def _write_summary_payload(run_dir: Path, payload: Dict) -> None:
    with open(run_dir / "summary.json", "w") as f:
        json.dump(payload, f, indent=2)


def _write_eval_style_summaries(
    *,
    out_root: Path,
    payload: Dict,
    probe_tag: str,
    step_tag: str,
) -> None:
    records_by_dataset: Dict[str, List[Dict]] = {}
    for record in payload.get("samples", []):
        dataset_id = record.get("dataset") or record.get("dataset_id")
        if dataset_id is None:
            continue
        records_by_dataset.setdefault(str(dataset_id), []).append(record)

    for dataset_id, records in records_by_dataset.items():
        dirs = _stream_eval_artifact_dirs(Path(out_root), dataset_id, probe_tag, step_tag)
        dirs["metrics"].mkdir(parents=True, exist_ok=True)
        summary = summarize_stream_records(records)
        with open(dirs["metrics"] / "summary.json", "w") as f:
            json.dump({"summary": summary, "samples": records}, f, indent=2)


def _render_eval_style_outputs(
    *,
    cfg,
    out_root: Path,
    sample_records: List[Dict],
    probe_tag: str,
    step_tag: str,
    enabled: bool,
) -> None:
    if not enabled:
        return
    try:
        from utils.visualize import (
            make_composite_compare_videos,
            render_video as render_eval_video,
        )
    except Exception as exc:
        print(f"[stream-eval render] imports failed: {exc}")
        return

    render_setting = cfg.get("test_setting", {}) or {}
    try:
        render_setting = OmegaConf.to_container(render_setting, resolve=True)
    except Exception:
        render_setting = dict(render_setting)
    render_setting.setdefault("recover_dim", 263)

    dataset_ids = sorted(
        {
            str(record.get("dataset") or record.get("dataset_id"))
            for record in sample_records
            if record.get("dataset") or record.get("dataset_id")
        }
    )
    for dataset_id in dataset_ids:
        dirs = _stream_eval_artifact_dirs(Path(out_root), dataset_id, probe_tag, step_tag)
        if not dirs["feature"].exists():
            continue
        try:
            render_eval_video(
                motion_dir=str(dirs["feature"]),
                save_dir=str(dirs["video"]),
                render_setting=render_setting,
                frames_dir=str(dirs["frames"]),
                traj_mask_dir=str(dirs["traj_mask"]),
                cond_traj_dir=str(dirs["traj_xz"]),
            )
            make_composite_compare_videos(
                result_folder=str(dirs["video"]),
                compare_folders=render_setting.get(dataset_id, {}).get(
                    "compare_folders", None
                ),
                compare_names=render_setting.get(dataset_id, {}).get(
                    "compare_names", None
                ),
                text_folder=str(dirs["text"]),
                save_dir=str(dirs["composite"]),
            )
        except Exception as exc:
            print(f"[stream-eval render] dataset={dataset_id} failed: {exc}")


def _aggregate_rank_payloads(
    *,
    run_dir: Path,
    world_size: int,
    probe_tag: str,
    ckpt_path: str,
    vae_ckpt_path: str,
    stream_mode: str,
    num_runs: int,
    out_root: Optional[Path] = None,
    step_tag: Optional[str] = None,
) -> Dict:
    sample_records = []
    for rank in range(world_size):
        rank_path = _rank_payload_path(run_dir, rank)
        if not rank_path.is_file():
            raise FileNotFoundError(f"Missing stream eval rank payload: {rank_path}")
        with open(rank_path) as f:
            rank_payload = json.load(f)
        sample_records.extend(rank_payload.get("samples", []))

    sample_records.sort(key=lambda record: int(record.get("_sample_index", 0)))
    summary = summarize_stream_records(sample_records)
    payload = {
        "probe_tag": probe_tag,
        "ckpt": ckpt_path,
        "vae_ckpt": vae_ckpt_path,
        "stream_mode": stream_mode,
        "num_samples": len(sample_records),
        "num_runs": num_runs,
        "summary": summary,
        "samples": sample_records,
    }
    _write_summary_payload(run_dir, payload)
    if out_root is not None and step_tag is not None:
        _write_eval_style_summaries(
            out_root=Path(out_root),
            payload=payload,
            probe_tag=probe_tag,
            step_tag=step_tag,
        )
    return payload


def _resolve_launch_context(args) -> Dict:
    overrides = _parse_overrides(args.set)
    cfg = load_config(config_path=args.config, override_args=overrides)
    ckpt_path = args.ckpt or cfg.get("test_ckpt", None) or cfg.get("resume_ckpt", None)
    vae_ckpt_path = args.vae_ckpt or cfg.get("test_vae_ckpt", None)
    if ckpt_path is None:
        raise ValueError("No checkpoint provided via --ckpt / cfg.test_ckpt / cfg.resume_ckpt")
    if vae_ckpt_path is None:
        raise ValueError("No VAE checkpoint provided via --vae_ckpt / cfg.test_vae_ckpt")

    meta_paths, probe_tag = _resolve_meta_paths_and_probe_tag(args, cfg)
    stream_mode = args.stream_mode or cfg.get("eval.stream_mode", "stream_generate")
    num_runs = args.num_runs or int(cfg.get("eval.num_runs", 1))
    out_root = Path(args.out_dir or cfg.get("eval.out_dir", "./outputs_stream_eval"))
    run_name = _resolve_run_name(
        ckpt_path=ckpt_path,
        probe_tag=probe_tag,
        stream_mode=stream_mode,
        requested_run_name=args.run_name,
    )
    step_tag = Path(str(ckpt_path)).stem.replace("=", "_")
    return {
        "cfg": cfg,
        "ckpt_path": ckpt_path,
        "vae_ckpt_path": vae_ckpt_path,
        "meta_paths": meta_paths,
        "probe_tag": probe_tag,
        "stream_mode": stream_mode,
        "num_runs": num_runs,
        "out_root": out_root,
        "run_name": run_name,
        "step_tag": step_tag,
        "run_dir": out_root / run_name,
    }


def _run_stream_eval(
    args,
    *,
    rank: int = 0,
    world_size: int = 1,
    accelerator: str | None = None,
    device_index: int | None = None,
    write_summary: bool = True,
) -> Dict:
    context = _resolve_launch_context(args)
    cfg = context["cfg"]
    ckpt_path = context["ckpt_path"]
    vae_ckpt_path = context["vae_ckpt_path"]
    meta_paths = context["meta_paths"]
    probe_tag = context["probe_tag"]
    stream_mode = context["stream_mode"]
    num_runs = context["num_runs"]
    out_root = context["out_root"]
    step_tag = context["step_tag"]
    run_dir = context["run_dir"]

    _set_seed(args.seed + rank)
    if accelerator is None:
        accelerator = _resolve_accelerator(
            args,
            [device_index] if device_index is not None else [0],
        )
    device = _select_eval_device(accelerator, device_index)

    batch_size = args.batch_size or int(cfg.data.test_bs)
    if batch_size != 1:
        raise NotImplementedError(
            "Streaming evaluator currently requires batch_size=1 because rollout and VAE streaming decode are single-sample."
        )

    num_workers = args.num_workers if args.num_workers is not None else int(cfg.data.num_workers)
    seg_size = int(cfg.get("eval.seg_size", 20))
    num_denoise_steps = (
        args.num_denoise_steps
        if args.num_denoise_steps is not None
        else cfg.get("eval.num_denoise_steps", None)
    )
    compute_offline_baseline, compute_no_traj_baseline = resolve_eval_path_flags(
        args, cfg
    )
    save_feature_npy = bool(args.save_feature_npy or cfg.get("eval.save_feature_npy", True))
    save_latent_npy = bool(args.save_latent_npy or cfg.get("eval.save_latent_npy", False))
    save_plots = bool(cfg.get("eval.save_plots", True) or args.save_plots)
    if args.no_save_plots:
        save_plots = False
    render_video = bool(args.render_video or cfg.get("eval.render_video", False))
    render_offline_video = bool(
        args.render_offline_video or cfg.get("eval.render_offline_video", False)
    )
    render_no_traj_video = bool(
        args.render_no_traj_video or cfg.get("eval.render_no_traj_video", False)
    )
    max_batches = args.max_batches or int(cfg.get("eval.max_batches", 0))
    max_samples = args.max_samples or int(cfg.get("eval.max_samples", 0))
    text_device = str(cfg.get("eval.text_device", "cpu")).lower()
    history_length = int(cfg.get("eval.history_length", 30))
    group_present_segments = bool(cfg.get("eval.group_present_segments", False))
    _traj_horizon_raw = cfg.get("eval.traj_horizon_tokens", None)
    traj_horizon_tokens = int(_traj_horizon_raw) if _traj_horizon_raw is not None else None
    token_dt = float(cfg.get("eval.token_dt", cfg.get("stream.token_dt", 0.20)))
    frames_per_token = int(cfg.get("eval.frames_per_token", cfg.get("data.frames_per_token", 4)))

    sample_root = run_dir / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        OmegaConf.save(cfg.config, run_dir / "config.yaml")

    model, vae = load_eval_model_and_vae(
        cfg,
        ckpt_path=ckpt_path,
        vae_ckpt_path=vae_ckpt_path,
        device=device,
        use_ema=not args.no_ema,
    )
    if text_device == "cpu":
        enable_cpu_text_encoding(model)
    _, dataloader = build_eval_dataloader(
        cfg,
        meta_paths=meta_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        group_present_segments=group_present_segments,
    )

    print(
        f"[stream-eval][rank {rank}/{world_size}] ckpt={ckpt_path} "
        f"probe={probe_tag} stream_mode={stream_mode} device={device} "
        f"num_runs={num_runs} batch_size={batch_size} text_device={text_device} "
        f"group_present_segments={int(group_present_segments)} history_length={history_length} "
        f"traj_horizon_tokens={traj_horizon_tokens} out_dir={run_dir}"
    )

    sample_records = []
    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        if max_samples > 0 and batch_idx >= max_samples:
            break
        if not _should_process_batch_on_rank(
            batch_idx,
            rank=rank,
            world_size=world_size,
            max_batches=max_batches,
            max_samples=max_samples,
        ):
            continue
        sample_batch = _slice_single_sample_batch(batch, 0)
        sample_name = sample_batch["name"][0]
        sample_dataset = sample_batch["dataset"][0]

        traj_runs = []
        control_runs = []
        stream_runs = []
        stream_feature_run0 = None
        stream_latent_run0 = None
        offline_feature_run0 = None
        offline_latent_run0 = None
        no_traj_runs = []
        stream_no_traj_feature_run0 = None

        for run_idx in range(num_runs):
            sample_seed = _stable_eval_seed(args.seed, probe_tag, sample_name, run_idx)
            _seed_eval_locally(sample_seed)

            with torch.no_grad():
                if stream_mode == "stream_generate":
                    stream_out = run_stream_generate_sample(
                        model=model,
                        vae=vae,
                        sample_batch=sample_batch,
                        device=device,
                        num_denoise_steps=num_denoise_steps,
                    )
                elif stream_mode == "stream_generate_step":
                    stream_out = run_stream_generate_step_sample(
                        model=model,
                        vae=vae,
                        sample_batch=sample_batch,
                        device=device,
                        history_length=history_length,
                        num_denoise_steps=num_denoise_steps,
                        traj_horizon_tokens=traj_horizon_tokens,
                        token_dt=token_dt,
                        frames_per_token=frames_per_token,
                    )
                else:
                    raise ValueError(f"Unsupported stream_mode: {stream_mode}")

            decoded_stream = stream_out["decoded_feature"]
            traj_runs.append(_compute_traj_metrics(decoded_stream, sample_batch, 0, seg_size=seg_size))
            control_runs.append(_compute_omni_control_metrics(decoded_stream, sample_batch, 0))

            boundary_metrics = compute_stream_boundary_metrics(
                decoded_stream,
                stream_out["chunk_frame_ends"],
            )
            stream_metric = {
                "stream_root_jump_mean": boundary_metrics["root_jump_mean"],
                "stream_root_jump_max": boundary_metrics["root_jump_max"],
                "stream_joint_jump_mean": boundary_metrics["joint_jump_mean"],
                "stream_num_boundaries": boundary_metrics["n_boundaries"],
                "stream_yaw_error": _root_path_yaw_error(
                    decoded_stream,
                    sample_batch,
                ),
            }

            if compute_offline_baseline:
                _seed_eval_locally(sample_seed)
                with torch.no_grad():
                    offline_out = run_offline_generate_sample(
                        model=model,
                        vae=vae,
                        sample_batch=sample_batch,
                        device=device,
                        num_denoise_steps=num_denoise_steps,
                    )
                offline_cmp = compute_stream_vs_offline_metrics(
                    decoded_stream,
                    offline_out["decoded_feature"],
                )
                stream_metric["stream_offline_feature_l2_mean"] = offline_cmp["feature_l2_mean"]
                stream_metric["stream_offline_feature_l2_max"] = offline_cmp["feature_l2_max"]
                stream_metric["stream_offline_root_ade"] = offline_cmp["root_ade"]
                stream_metric["stream_offline_length_delta"] = offline_cmp["length_delta"]
                if run_idx == 0:
                    offline_feature_run0 = offline_out["decoded_feature"]
                    offline_latent_run0 = offline_out["latent"]

            if compute_no_traj_baseline:
                no_traj_batch = _remove_trajectory_conditioning(sample_batch)
                _seed_eval_locally(sample_seed)
                with torch.no_grad():
                    if stream_mode == "stream_generate":
                        no_traj_out = run_stream_generate_sample(
                            model=model,
                            vae=vae,
                            sample_batch=no_traj_batch,
                            device=device,
                            num_denoise_steps=num_denoise_steps,
                        )
                    elif stream_mode == "stream_generate_step":
                        no_traj_out = run_stream_generate_step_sample(
                            model=model,
                            vae=vae,
                            sample_batch=no_traj_batch,
                            device=device,
                            history_length=history_length,
                            num_denoise_steps=num_denoise_steps,
                            traj_horizon_tokens=traj_horizon_tokens,
                            token_dt=token_dt,
                            frames_per_token=frames_per_token,
                        )
                    else:
                        raise ValueError(f"Unsupported stream_mode: {stream_mode}")
                no_traj_decoded = no_traj_out["decoded_feature"]
                no_traj_metrics = _compute_traj_metrics(
                    no_traj_decoded,
                    sample_batch,
                    0,
                    seg_size=seg_size,
                )
                no_traj_runs.append(no_traj_metrics)
                if run_idx == 0:
                    stream_no_traj_feature_run0 = no_traj_decoded

            if run_idx == 0:
                stream_feature_run0 = decoded_stream
                stream_latent_run0 = stream_out["latent_stream"]
            stream_runs.append(stream_metric)

        sample_record = {
            "_sample_index": batch_idx,
            "name": sample_name,
            "dataset": sample_dataset,
            "stream_mode": stream_mode,
            "num_runs": num_runs,
        }
        if "segment_names" in sample_batch:
            segment_names = sample_batch["segment_names"][0]
            sample_record["segment_names"] = list(segment_names)
        sample_record.update(_average_traj_metrics(traj_runs))
        sample_record.update(_average_control_metrics(control_runs))
        sample_record["stream_root_jump_mean"] = _average_scalar_metric(stream_runs, "stream_root_jump_mean")
        sample_record["stream_root_jump_max"] = _average_scalar_metric(stream_runs, "stream_root_jump_max")
        sample_record["stream_joint_jump_mean"] = _average_scalar_metric(stream_runs, "stream_joint_jump_mean")
        sample_record["stream_num_boundaries"] = _average_scalar_metric(stream_runs, "stream_num_boundaries")
        sample_record["stream_yaw_error"] = _average_scalar_metric(stream_runs, "stream_yaw_error")
        if compute_offline_baseline:
            sample_record["stream_offline_feature_l2_mean"] = _average_scalar_metric(stream_runs, "stream_offline_feature_l2_mean")
            sample_record["stream_offline_feature_l2_max"] = _average_scalar_metric(stream_runs, "stream_offline_feature_l2_max")
            sample_record["stream_offline_root_ade"] = _average_scalar_metric(stream_runs, "stream_offline_root_ade")
            sample_record["stream_offline_length_delta"] = _average_scalar_metric(stream_runs, "stream_offline_length_delta")
        if compute_no_traj_baseline and no_traj_runs:
            no_traj_avg = _average_traj_metrics(no_traj_runs)
            for key, value in no_traj_avg.items():
                sample_record[f"stream_no_traj/{key}"] = value
        sample_record["_traj_runs"] = traj_runs
        sample_record["_control_runs"] = control_runs
        sample_record["_stream_runs"] = stream_runs
        if compute_no_traj_baseline:
            sample_record["_stream_no_traj_runs"] = no_traj_runs
        sample_records.append(sample_record)

        gt_feature = sample_batch["feature"][0].float().cpu() if "feature" in sample_batch else None
        _save_sample_outputs(
            sample_dir=sample_root / sample_name,
            sample_batch=sample_batch,
            sample_record=sample_record,
            stream_feature=stream_feature_run0,
            gt_feature=gt_feature,
            offline_feature=offline_feature_run0,
            stream_no_traj_feature=stream_no_traj_feature_run0,
            stream_latent=stream_latent_run0,
            offline_latent=offline_latent_run0,
            save_feature_npy=save_feature_npy,
            save_latent_npy=save_latent_npy,
            save_plots=save_plots,
            render_video=render_video,
            render_offline_video=render_offline_video,
            render_no_traj_video=render_no_traj_video,
        )
        _save_eval_style_sample_outputs(
            out_root=out_root,
            dataset_id=sample_dataset,
            probe_tag=probe_tag,
            step_tag=step_tag,
            sample_name=sample_name,
            sample_batch=sample_batch,
            sample_record=sample_record,
            stream_feature=stream_feature_run0,
            stream_latent=stream_latent_run0,
            gt_feature=gt_feature,
            offline_feature=offline_feature_run0,
            stream_no_traj_feature=stream_no_traj_feature_run0,
        )

    summary = summarize_stream_records(sample_records)
    payload = {
        "probe_tag": probe_tag,
        "ckpt": ckpt_path,
        "vae_ckpt": vae_ckpt_path,
        "stream_mode": stream_mode,
        "num_samples": len(sample_records),
        "num_runs": num_runs,
        "summary": summary,
        "samples": sample_records,
    }
    if world_size > 1:
        rank_path = _rank_payload_path(run_dir, rank)
        rank_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rank_path, "w") as f:
            json.dump(payload, f, indent=2)
    elif write_summary:
        _write_summary_payload(run_dir, payload)
        _write_eval_style_summaries(
            out_root=out_root,
            payload=payload,
            probe_tag=probe_tag,
            step_tag=step_tag,
        )
        _render_eval_style_outputs(
            cfg=cfg,
            out_root=out_root,
            sample_records=sample_records,
            probe_tag=probe_tag,
            step_tag=step_tag,
            enabled=render_video,
        )

    print(
        f"[stream-eval][rank {rank}/{world_size}] finished {len(sample_records)} samples | "
        f"ADE={summary.get('traj/ADE_mean', float('nan')):.4f} "
        f"FDE={summary.get('traj/FDE_mean', float('nan')):.4f} "
        f"RootJump={summary.get('stream_boundary/root_jump_mean', float('nan')):.4f}"
    )
    return payload


def _distributed_worker(rank: int, args, device_ids: list[int]) -> None:
    _run_stream_eval(
        args,
        rank=rank,
        world_size=len(device_ids),
        accelerator="gpu",
        device_index=device_ids[rank],
        write_summary=False,
    )


def main():
    args = parse_args()
    device_ids = _parse_devices_arg(args.devices)
    accelerator = _resolve_accelerator(args, device_ids)
    if accelerator == "gpu" and len(device_ids) > 1:
        context = _resolve_launch_context(args)
        args.run_name = context["run_name"]
        context["run_dir"].mkdir(parents=True, exist_ok=True)
        torch.multiprocessing.spawn(
            _distributed_worker,
            args=(args, device_ids),
            nprocs=len(device_ids),
            join=True,
        )
        payload = _aggregate_rank_payloads(
            run_dir=context["run_dir"],
            world_size=len(device_ids),
            probe_tag=context["probe_tag"],
            ckpt_path=context["ckpt_path"],
            vae_ckpt_path=context["vae_ckpt_path"],
            stream_mode=context["stream_mode"],
            num_runs=context["num_runs"],
            out_root=context["out_root"],
            step_tag=context["step_tag"],
        )
        render_video = bool(
            args.render_video
            or context["cfg"].get("eval.render_video", False)
        )
        _render_eval_style_outputs(
            cfg=context["cfg"],
            out_root=context["out_root"],
            sample_records=payload["samples"],
            probe_tag=context["probe_tag"],
            step_tag=context["step_tag"],
            enabled=render_video,
        )
        summary = payload["summary"]
        print(
            f"[stream-eval] merged {payload['num_samples']} samples from "
            f"{len(device_ids)} ranks | "
            f"ADE={summary.get('traj/ADE_mean', float('nan')):.4f} "
            f"FDE={summary.get('traj/FDE_mean', float('nan')):.4f} "
            f"RootJump={summary.get('stream_boundary/root_jump_mean', float('nan')):.4f}"
        )
        return

    device_index = device_ids[0] if accelerator == "gpu" and device_ids else None
    _run_stream_eval(
        args,
        rank=0,
        world_size=1,
        accelerator=accelerator,
        device_index=device_index,
    )


if __name__ == "__main__":
    main()
