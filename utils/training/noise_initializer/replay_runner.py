"""Two-stage snapshot replay training for the residual NoiseInitializer."""

from __future__ import annotations

import copy
import json
import time
from collections import Counter
from pathlib import Path

import torch

from utils.inference.stream_generator import StreamGenerator
from utils.motion_process import StreamJointRecovery263
from utils.token_frame import token_range_to_frame_slice
from utils.training.noise_initializer.shadow_rollout import (
    restore_stream_state,
    restore_vae_cache,
    snapshot_stream_state,
    snapshot_vae_cache,
)
from utils.training.noise_initializer.snapshot_replay import (
    NoiseInitializerReplaySnapshot,
    ReplaySamplingConfig,
    WeightedSnapshotSampler,
)


def resolve_replay_candidate_commits(
    *,
    target_tokens: int,
    max_commits: int,
    optimize_every_tokens: int,
) -> list[int]:
    """Return absolute decision commits available for snapshot collection."""

    limit = min(max(0, int(target_tokens)), max(0, int(max_commits)))
    interval = max(1, int(optimize_every_tokens))
    return list(range(0, limit, interval))


def normalize_replay_seeds(value) -> list[int]:
    """Normalize YAML values and string-valued CLI overrides into seed integers."""

    if isinstance(value, str):
        stripped = value.strip().strip("[]")
        values = [] if not stripped else stripped.split(",")
    elif isinstance(value, (int, float)):
        values = [value]
    else:
        values = list(value or [])
    return [int(item.strip() if isinstance(item, str) else item) for item in values]


def seeded_initial_noise(
    shape: tuple[int, ...],
    *,
    seed: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate reproducible Gaussian stream initialization without global RNG state."""

    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(int(seed))
    return torch.randn(
        shape,
        generator=generator,
        device=resolved_device,
        dtype=dtype,
    )


def resolve_affected_frame_counts(
    *,
    target_mask: torch.Tensor,
    history_frames: int,
    requested_window_frames: int | None = None,
) -> tuple[int, int]:
    """Count valid and requested frames after the causal no-effect prefix."""

    mask = target_mask.reshape(-1)
    start = min(max(0, int(history_frames)), int(mask.numel()))
    affected = mask[start:]
    requested_total = (
        int(mask.numel())
        if requested_window_frames is None
        else max(0, int(requested_window_frames))
    )
    requested = max(0, requested_total - max(0, int(history_frames)))
    return int((affected > 0).sum().item()), requested


def _snapshot_pool_summary(
    snapshots: list[NoiseInitializerReplaySnapshot],
) -> dict:
    return {
        "size": len(snapshots),
        "by_seed": dict(sorted(Counter(int(item.seed) for item in snapshots).items())),
        "by_commit": dict(
            sorted(Counter(int(item.commit_index) for item in snapshots).items())
        ),
        "by_source": dict(sorted(Counter(item.source for item in snapshots).items())),
    }


def collect_replay_snapshots(
    *,
    cfg: dict,
    model,
    vae,
    initializer,
    initializer_text_encoder,
    sample_batch: dict,
    target_xz: torch.Tensor,
    target_mask: torch.Tensor,
    target_tokens: int,
    device: torch.device,
    source: str,
) -> list[NoiseInitializerReplaySnapshot]:
    """Roll each seed once and capture valid decision-point stream snapshots."""

    from eval.ldf.conditioning import LdfEvalStreamConditioner
    from eval.ldf.latent_initializer.optimize_stream_chunk_noise import _build_step
    from eval.ldf.stream_generation import StreamTextRolloutController
    from utils.training.noise_initializer.overfit_runner import (
        _build_initializer_context_for_commit,
        _build_runtime_traj_payload,
        affected_history_frames,
        advance_model_token_update_count,
        apply_initializer_to_stream_state,
    )

    if source not in {"gaussian", "initializer"}:
        raise ValueError(f"unknown replay snapshot source: {source!r}")
    replay_cfg = dict(cfg.get("replay", {}) or {})
    seeds = normalize_replay_seeds(replay_cfg.get("train_seeds", []))
    history_tokens = int(cfg.get("history_tokens", cfg.get("history_length", 30)))
    traj_horizon_tokens = int(cfg.get("traj_horizon_tokens", 20))
    frames_per_token = int(cfg.get("frames_per_token", 4))
    loss_horizon_tokens = int(cfg.get("loss_horizon_tokens", 20))
    frontier_tokens = int(cfg.get("frontier_tokens", 5))
    max_commits = min(int(cfg.get("max_commits", target_tokens)), int(target_tokens))
    optimize_every = int(cfg.get("optimize_every_tokens", 5))
    candidate_commits = set(
        resolve_replay_candidate_commits(
            target_tokens=int(target_tokens),
            max_commits=max_commits,
            optimize_every_tokens=optimize_every,
        )
    )
    min_valid_frames = int(replay_cfg.get("min_valid_affected_frames", 1))
    require_zero_update_count = bool(cfg.get("require_zero_update_count", False))
    snapshots: list[NoiseInitializerReplaySnapshot] = []

    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        initial_shape = (
            1,
            history_tokens * 2 + int(model.chunk_size),
            int(model.input_dim),
        )
        model.init_generated(
            history_tokens,
            batch_size=1,
            num_denoise_steps=cfg.get("num_denoise_steps", None),
            traj_buffer=None,
            initial_generated=seeded_initial_noise(
                initial_shape,
                seed=seed,
                device=device,
            ),
        )
        model.token_update_count = torch.zeros(
            int(model.generated.shape[2]),
            device=model.generated.device,
            dtype=torch.long,
        )
        stream = StreamGenerator(
            ldf_model=model,
            device=device,
            history_length=history_tokens,
            traj_horizon_tokens=traj_horizon_tokens,
            token_dt=float(cfg.get("token_dt", 0.20)),
        )
        text_rollout = StreamTextRolloutController.from_sample_batch(sample_batch)
        conditioner = LdfEvalStreamConditioner(
            sample_batch,
            history_length=history_tokens,
            traj_horizon_tokens=traj_horizon_tokens,
            token_dt=float(cfg.get("token_dt", 0.20)),
            frames_per_token=frames_per_token,
            device=device,
        )
        recovery = StreamJointRecovery263(joints_num=22, smoothing_alpha=1.0)
        vae.clear_cache()
        first_chunk = True
        committed_latents: list[torch.Tensor] = []

        for commit_index in range(max_commits):
            runtime_payload = _build_runtime_traj_payload(
                model=model,
                conditioner=conditioner,
                commit_index=commit_index,
            )
            context = None
            if commit_index in candidate_commits:
                context = _build_initializer_context_for_commit(
                    model=model,
                    sample_batch=sample_batch,
                    text_rollout=text_rollout,
                    commit_index=commit_index,
                    device=device,
                    text_encoder=initializer_text_encoder,
                    history_tokens=history_tokens,
                    frontier_tokens=frontier_tokens,
                    traj_horizon_tokens=traj_horizon_tokens,
                    frames_per_token=frames_per_token,
                    beta_threshold=float(cfg.get("beta_threshold", 0.999)),
                    traj_payload=runtime_payload,
                    token_update_count=getattr(model, "token_update_count", None),
                    require_zero_update_count=require_zero_update_count,
                )
                target_slice = token_range_to_frame_slice(
                    commit_index,
                    loss_horizon_tokens,
                    frames_per_token,
                )
                local_target_xz = target_xz[target_slice]
                local_target_mask = target_mask[target_slice]
                history_frames = affected_history_frames(
                    context,
                    frames_per_token=frames_per_token,
                )
                valid_frames, requested_frames = resolve_affected_frame_counts(
                    target_mask=local_target_mask,
                    history_frames=history_frames,
                    requested_window_frames=int(target_slice.stop)
                    - int(target_slice.start),
                )
                if valid_frames >= min_valid_frames and requested_frames > 0:
                    prefix = (
                        torch.cat(committed_latents, dim=1)
                        if committed_latents
                        else context.frontier_base_zT[:, :0]
                    )
                    batch = {
                        "context": context,
                        "target_xz": local_target_xz,
                        "target_mask": local_target_mask,
                        "history_frames": history_frames,
                        "first_chunk": first_chunk,
                        "generated_anchor_xz": (
                            conditioner.timeline.head.world_xz.detach().clone()
                        ),
                        "committed_prefix_latents": prefix.detach().clone(),
                        "commit_index": int(commit_index),
                        "frames_per_token": frames_per_token,
                    }
                    snapshots.append(
                        NoiseInitializerReplaySnapshot(
                            seed=seed,
                            commit_index=int(commit_index),
                            source=source,
                            valid_affected_frames=valid_frames,
                            requested_affected_frames=requested_frames,
                            model_state=snapshot_stream_state(model),
                            vae_state=snapshot_vae_cache(vae),
                            context=context,
                            batch=batch,
                            conditioner=copy.deepcopy(conditioner),
                            recovery=copy.deepcopy(recovery),
                            first_chunk=bool(first_chunk),
                        )
                    )

            if source == "initializer" and commit_index in candidate_commits:
                if context is None:
                    raise RuntimeError("initializer roll-in requires a decision context")
                apply_initializer_to_stream_state(
                    model=model,
                    initializer=initializer,
                    context=context,
                    alpha=float(cfg.get("alpha", 1.0)),
                    max_delta_norm_ratio=cfg.get("max_delta_norm_ratio", None),
                )

            step_payload, condition_provider = _build_step(
                model=model,
                stream=stream,
                text_rollout=text_rollout,
                conditioner=conditioner,
                commit_index=commit_index,
                first_chunk=first_chunk,
                device=device,
            )
            with torch.no_grad():
                start_step = int(model.current_step)
                start_commit = int(model.commit_index)
                output = model.stream_generate_step(
                    step_payload,
                    first_chunk=first_chunk,
                    condition=condition_provider,
                )
                advance_model_token_update_count(
                    model,
                    start_step=start_step,
                    start_commit=start_commit,
                )
                committed_latents.append(output["generated"].detach().clone())
                decoded = vae.stream_decode(
                    output["generated"].detach(),
                    first_chunk=first_chunk,
                )[0].float().detach().cpu()
                conditioner.append_decoded(
                    decoded,
                    commit_idx=commit_index + 1,
                    recovery=recovery,
                )
            first_chunk = False
            model.generated = model.generated.detach()

    return snapshots


def _train_replay_stage(
    *,
    stage: int,
    steps: int,
    sampler: WeightedSnapshotSampler,
    cfg: dict,
    model,
    vae,
    training_vae,
    initializer,
    lightning,
    optimizer,
    sample_batch: dict,
    device: torch.device,
    rows: list[dict],
) -> dict:
    from eval.ldf.stream_generation import StreamTextRolloutController
    from utils.training.noise_initializer.overfit_runner import (
        format_train_progress_log,
        make_ldf_shadow_rollout_fn,
        sync_vae_decode_cache,
    )

    history_tokens = int(cfg.get("history_tokens", cfg.get("history_length", 30)))
    traj_horizon_tokens = int(cfg.get("traj_horizon_tokens", 20))
    stream = StreamGenerator(
        ldf_model=model,
        device=device,
        history_length=history_tokens,
        traj_horizon_tokens=traj_horizon_tokens,
        token_dt=float(cfg.get("token_dt", 0.20)),
    )
    text_rollout = StreamTextRolloutController.from_sample_batch(sample_batch)
    log_every = int(cfg.get("log_every_train_steps", 0) or 0)
    sample_counts = Counter()

    for local_step in range(int(steps)):
        snapshot = sampler.sample(stage=int(stage))
        restore_stream_state(model, snapshot.model_state)
        restore_vae_cache(vae, snapshot.vae_state)
        sync_vae_decode_cache(vae, training_vae)
        lightning.rollout_fn = make_ldf_shadow_rollout_fn(
            vae=training_vae,
            stream=stream,
            text_rollout=text_rollout,
            conditioner=snapshot.conditioner,
            recovery=snapshot.recovery,
            start_commit=int(snapshot.commit_index),
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = lightning.training_step(snapshot.batch, local_step)
        loss.backward()
        grad_norm = sum(
            float(parameter.grad.detach().float().norm().cpu().item())
            for parameter in initializer.parameters()
            if parameter.grad is not None
        )
        optimizer.step()
        global_step = len(rows)
        row = {
            "stage": int(stage),
            "source": snapshot.source,
            "seed": int(snapshot.seed),
            "commit_index": int(snapshot.commit_index),
            "inner_step": int(local_step),
            "train_step": int(global_step),
            "loss": float(loss.detach().cpu().item()),
            "grad_norm_sum": grad_norm,
            "history_frames": int(snapshot.batch["history_frames"]),
            "valid_affected_frames": int(snapshot.valid_affected_frames),
            "requested_affected_frames": int(snapshot.requested_affected_frames),
        }
        row.update(lightning.last_step_diagnostics)
        rows.append(row)
        sample_counts[(snapshot.source, snapshot.seed, snapshot.commit_index)] += 1
        completed = len(rows)
        if local_step == 0 or (log_every > 0 and completed % log_every == 0):
            payload = format_train_progress_log(train_step=global_step, row=row)
            payload.update(
                {
                    "stage": int(stage),
                    "source": snapshot.source,
                    "seed": int(snapshot.seed),
                    "valid_affected_frames": int(snapshot.valid_affected_frames),
                }
            )
            print(json.dumps(payload, sort_keys=True), flush=True)

    return {
        f"{source}:{seed}:{commit}": count
        for (source, seed, commit), count in sorted(sample_counts.items())
    }


def run_two_stage_snapshot_replay(
    *,
    cfg: dict,
    model,
    vae,
    training_vae,
    initializer,
    initializer_text_encoder,
    lightning,
    optimizer,
    sample_batch: dict,
    target_xz: torch.Tensor,
    target_mask: torch.Tensor,
    target_tokens: int,
    device: torch.device,
) -> dict:
    """Collect replay pools and train one initializer in two consecutive stages."""

    replay = dict(cfg.get("replay", {}) or {})
    sampling_cfg = ReplaySamplingConfig(
        late_progress_start=float(replay.get("late_progress_start", 0.55)),
        late_weight_multiplier=float(replay.get("late_weight_multiplier", 3.0)),
        initializer_probability=float(replay.get("initializer_probability", 0.3)),
    )
    gaussian_snapshots = collect_replay_snapshots(
        cfg=cfg,
        model=model,
        vae=vae,
        initializer=initializer,
        initializer_text_encoder=initializer_text_encoder,
        sample_batch=sample_batch,
        target_xz=target_xz,
        target_mask=target_mask,
        target_tokens=target_tokens,
        device=device,
        source="gaussian",
    )
    sampler_seed = int(replay.get("sampler_seed", cfg.get("seed", 1234)))
    stage1_sampler = WeightedSnapshotSampler(
        gaussian_snapshots,
        [],
        cfg=sampling_cfg,
        seed=sampler_seed,
    )
    rows: list[dict] = []
    stage1_counts = _train_replay_stage(
        stage=1,
        steps=int(replay.get("stage1_steps", 0)),
        sampler=stage1_sampler,
        cfg=cfg,
        model=model,
        vae=vae,
        training_vae=training_vae,
        initializer=initializer,
        lightning=lightning,
        optimizer=optimizer,
        sample_batch=sample_batch,
        device=device,
        rows=rows,
    )

    out_dir = Path(str(cfg.get("out_dir", "eval/out_eval/noise_initializer_replay")))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("%Y%m%d_%H%M%S")
    stage1_path = out_dir / f"noise_initializer_stage1_{tag}.pt"
    torch.save({"state_dict": initializer.state_dict(), "cfg": dict(cfg)}, stage1_path)

    initializer_snapshots: list[NoiseInitializerReplaySnapshot] = []
    stage2_steps = int(replay.get("stage2_steps", 0))
    stage2_counts = {}
    if stage2_steps > 0:
        initializer_snapshots = collect_replay_snapshots(
            cfg=cfg,
            model=model,
            vae=vae,
            initializer=initializer,
            initializer_text_encoder=initializer_text_encoder,
            sample_batch=sample_batch,
            target_xz=target_xz,
            target_mask=target_mask,
            target_tokens=target_tokens,
            device=device,
            source="initializer",
        )
        stage2_sampler = WeightedSnapshotSampler(
            gaussian_snapshots,
            initializer_snapshots,
            cfg=sampling_cfg,
            seed=sampler_seed + 1,
        )
        stage2_counts = _train_replay_stage(
            stage=2,
            steps=stage2_steps,
            sampler=stage2_sampler,
            cfg=cfg,
            model=model,
            vae=vae,
            training_vae=training_vae,
            initializer=initializer,
            lightning=lightning,
            optimizer=optimizer,
            sample_batch=sample_batch,
            device=device,
            rows=rows,
        )

    final_path = out_dir / f"noise_initializer_stage2_{tag}.pt"
    debug_path = out_dir / f"debug_replay_{tag}.json"
    torch.save({"state_dict": initializer.state_dict(), "cfg": dict(cfg)}, final_path)
    summary = {
        "checkpoint": str(final_path),
        "stage1_checkpoint": str(stage1_path),
        "debug_json": str(debug_path),
        "training_mode": "two_stage_snapshot_replay",
        "num_train_steps": len(rows),
        "target_tokens": int(target_tokens),
        "gaussian_pool": _snapshot_pool_summary(gaussian_snapshots),
        "initializer_pool": _snapshot_pool_summary(initializer_snapshots),
        "stage1_sample_counts": stage1_counts,
        "stage2_sample_counts": stage2_counts,
        "loss_curve": rows,
    }
    debug_path.write_text(json.dumps(summary, indent=2))
    return summary


__all__ = [
    "collect_replay_snapshots",
    "normalize_replay_seeds",
    "resolve_affected_frame_counts",
    "resolve_replay_candidate_commits",
    "run_two_stage_snapshot_replay",
    "seeded_initial_noise",
]
