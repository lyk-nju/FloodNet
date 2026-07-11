from __future__ import annotations

from dataclasses import replace

import torch

from utils.training.noise_initializer.snapshot_replay import (
    NoiseInitializerReplaySnapshot,
    ReplaySamplingConfig,
    WeightedSnapshotSampler,
    snapshot_sampling_weight,
)
from utils.training.noise_initializer.config_validate import (
    validate_noise_initializer_overfit_config,
)
from utils.training.noise_initializer.replay_runner import (
    normalize_replay_seeds,
    resolve_affected_frame_counts,
    resolve_replay_candidate_commits,
    seeded_initial_noise,
)


def _snapshot(
    *,
    seed: int,
    commit: int,
    source: str = "gaussian",
    valid: int = 24,
    requested: int = 24,
) -> NoiseInitializerReplaySnapshot:
    return NoiseInitializerReplaySnapshot(
        seed=seed,
        commit_index=commit,
        source=source,
        valid_affected_frames=valid,
        requested_affected_frames=requested,
        model_state={},
        vae_state=None,
        context=None,
        batch={},
        conditioner=None,
        recovery=None,
        first_chunk=False,
    )


def test_snapshot_sampling_weight_prioritizes_late_valid_commits():
    cfg = ReplaySamplingConfig(late_progress_start=0.55, late_weight_multiplier=3.0)
    early = _snapshot(seed=1, commit=10)
    late = _snapshot(seed=1, commit=30)
    late_short = replace(late, commit_index=40, valid_affected_frames=8)

    assert snapshot_sampling_weight(early, max_commit=40, cfg=cfg) == 1.0
    assert snapshot_sampling_weight(late, max_commit=40, cfg=cfg) == 3.0
    assert snapshot_sampling_weight(late_short, max_commit=40, cfg=cfg) == 1.0


def test_weighted_snapshot_sampler_is_seeded_and_respects_stage2_source_mix():
    gaussian = [_snapshot(seed=1, commit=10), _snapshot(seed=2, commit=30)]
    initializer = [
        _snapshot(seed=1, commit=10, source="initializer"),
        _snapshot(seed=2, commit=30, source="initializer"),
    ]
    cfg = ReplaySamplingConfig(initializer_probability=0.3)
    sampler_a = WeightedSnapshotSampler(gaussian, initializer, cfg=cfg, seed=17)
    sampler_b = WeightedSnapshotSampler(gaussian, initializer, cfg=cfg, seed=17)

    draws_a = [sampler_a.sample(stage=2) for _ in range(1000)]
    draws_b = [sampler_b.sample(stage=2) for _ in range(1000)]

    assert [(x.source, x.seed, x.commit_index) for x in draws_a] == [
        (x.source, x.seed, x.commit_index) for x in draws_b
    ]
    initializer_count = sum(x.source == "initializer" for x in draws_a)
    assert 250 <= initializer_count <= 350
    assert all(sampler_a.sample(stage=1).source == "gaussian" for _ in range(20))


def test_weighted_snapshot_sampler_rejects_snapshots_without_valid_affected_frames():
    invalid = _snapshot(seed=1, commit=40, valid=0)

    try:
        WeightedSnapshotSampler([invalid], [], cfg=ReplaySamplingConfig(), seed=1)
    except ValueError as exc:
        assert "valid affected" in str(exc)
    else:
        raise AssertionError("expected invalid replay pool to be rejected")


def test_two_stage_replay_config_requires_seeds_and_valid_source_probability():
    cfg = {
        "ckpt": "model.ckpt",
        "meta_path": "meta.txt",
        "training_mode": "two_stage_snapshot_replay",
        "model": {"params": {"latent_dim": 4, "text_dim": 4096, "frontier_tokens": 5}},
        "replay": {
            "train_seeds": [1, 2, 3],
            "stage1_steps": 10,
            "stage2_steps": 5,
            "initializer_probability": 0.3,
        },
    }

    validate_noise_initializer_overfit_config(cfg)

    cfg["replay"]["train_seeds"] = []
    try:
        validate_noise_initializer_overfit_config(cfg)
    except ValueError as exc:
        assert "train_seeds" in str(exc)
    else:
        raise AssertionError("expected replay seed validation error")


def test_replay_candidate_commits_and_seeded_noise_are_deterministic():
    assert resolve_replay_candidate_commits(
        target_tokens=46,
        max_commits=46,
        optimize_every_tokens=5,
    ) == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

    first = seeded_initial_noise((1, 4, 2), seed=1234, device="cpu")
    second = seeded_initial_noise((1, 4, 2), seed=1234, device="cpu")
    other = seeded_initial_noise((1, 4, 2), seed=1235, device="cpu")
    assert first.equal(second)
    assert not first.equal(other)


def test_affected_frame_counts_exclude_causal_prefix_and_invalid_tail():
    valid, requested = resolve_affected_frame_counts(
        target_mask=torch.tensor([1, 1, 1, 0, 0], dtype=torch.float32),
        history_frames=2,
        requested_window_frames=8,
    )

    assert valid == 1
    assert requested == 6


def test_normalize_replay_seeds_accepts_cli_list_strings():
    assert normalize_replay_seeds("[2234,2235]") == [2234, 2235]
    assert normalize_replay_seeds(2234) == [2234]
