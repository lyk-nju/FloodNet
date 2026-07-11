#!/usr/bin/env python3
"""Compare direct and compatibility stream-runtime entry points."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.inference.stream_runtime import (
    RootSourceProposal,
    SetRootFeedback,
    SetRootSource,
    SetText,
    SpaceContract,
)
from utils.initialize import load_config
from web_demo.runtime.model_loader import load_model_bundle


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sample", default="001168")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--root-feedback",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _traj_cfg(config_path: str) -> dict:
    cfg = load_config(config_path)
    raw_cfg = getattr(cfg, "config", cfg)
    if OmegaConf.is_config(raw_cfg):
        raw = OmegaConf.to_container(raw_cfg, resolve=True)
    elif isinstance(raw_cfg, dict):
        raw = raw_cfg
    else:
        traj_cfg = dict(cfg.get("traj_mask", {}) or {})
        root_refiner = dict(traj_cfg.get("root_refiner", {}) or {})
        root_refiner["enabled"] = False
        traj_cfg["root_refiner"] = root_refiner
        return traj_cfg
    traj_cfg = dict((raw or {}).get("traj_mask", {}) or {})
    root_refiner = dict(traj_cfg.get("root_refiner", {}) or {})
    root_refiner["enabled"] = False
    traj_cfg["root_refiner"] = root_refiner
    return traj_cfg


def _straight_proposal(num_frames: int = 512) -> RootSourceProposal:
    future = torch.zeros(num_frames, 7)
    future[:, 2] = torch.arange(1, num_frames + 1, dtype=torch.float32) * 0.02
    future[:, 3] = 1.0
    future[:, 6] = 0.02
    return RootSourceProposal(
        future_traj7=future,
        future_frame_mask=torch.ones(num_frames, dtype=torch.bool),
        source_id="parity-straight-route",
        version=1,
        metadata={"sample": "synthetic-parity"},
    )


def _prepare_bundle(args):
    bundle = load_model_bundle(
        args.config,
        traj_mask_cfg=_traj_cfg(args.config),
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    if args.checkpoint is not None:
        configured = str(load_config(args.config).test_ckpt)
        if Path(configured).expanduser().resolve() != Path(args.checkpoint).expanduser().resolve():
            raise ValueError(
                "--checkpoint must match config test_ckpt; checkpoint overrides are "
                "not applied implicitly because they would make the parity inputs differ"
            )
    session = bundle.runtime_session
    session.submit(SetText(version=1, requested_commit_abs=0, text="walk forward"))
    session.submit(
        SetRootFeedback(
            version=2,
            requested_commit_abs=0,
            enabled=bool(args.root_feedback),
            xz_blend_alpha=0.5,
        )
    )
    session.submit(
        SetRootSource(
            version=3,
            requested_commit_abs=0,
            proposal=_straight_proposal(),
            space_contract=SpaceContract.WORLD_ROUTE,
        )
    )
    return bundle


def _equal(left: Any, right: Any, path: str = "event") -> str | None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return path
        return None if torch.equal(left.cpu(), right.cpu()) else path
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping)):
            return path
        if set(left) != set(right):
            return f"{path}.keys"
        for key in sorted(left):
            mismatch = _equal(left[key], right[key], f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if not isinstance(left, (tuple, list)) or not isinstance(right, (tuple, list)):
            return path
        if len(left) != len(right):
            return f"{path}.length"
        for index, (lhs, rhs) in enumerate(zip(left, right)):
            mismatch = _equal(lhs, rhs, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    return None if left == right else path


def _event_fields(event) -> dict:
    return {
        "absolute_commit_before": event.absolute_commit_before,
        "absolute_commit_after": event.absolute_commit_after,
        "local_commit_before": event.local_commit_before,
        "local_commit_after": event.local_commit_after,
        "latent_buffer_start_commit_abs": event.latent_buffer_start_commit_abs,
        "latent_buffer_epoch": event.latent_buffer_epoch,
        "committed_latent": event.committed_latent,
        "decoded_chunk": event.decoded_chunk,
        "joint_frames": event.joint_frames,
        "root_frames_start_abs": event.root_frames_start_abs,
        "root_frames": event.root_frames,
        "timeline_world_xz": event.timeline_state.world_xz,
        "timeline_world_yaw": event.timeline_state.world_yaw,
        "actual_payload": None if event.actual_payload is None else dict(event.actual_payload),
        "source_id": event.source_id,
        "source_version": event.source_version,
        "actual_activation_commit": event.actual_activation_commit,
        "lifecycle_events": event.lifecycle_events,
        "route_status": event.route_status.value,
    }


def main() -> int:
    args = _parse_args()
    if args.tokens <= 0:
        raise ValueError("--tokens must be > 0")

    _seed_everything(args.seed)
    direct_bundle = _prepare_bundle(args)
    _seed_everything(args.seed)
    direct_events = [direct_bundle.runtime_session.step() for _ in range(args.tokens)]
    del direct_bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _seed_everything(args.seed)
    compat_bundle = _prepare_bundle(args)
    _seed_everything(args.seed)
    compat_events = [
        compat_bundle.stream_generator.execute_step() for _ in range(args.tokens)
    ]

    mismatch = None
    mismatch_token = None
    for index, (direct, compat) in enumerate(zip(direct_events, compat_events)):
        mismatch = _equal(_event_fields(direct), _event_fields(compat))
        if mismatch is not None:
            mismatch_token = index
            break
    report = {
        "matched": mismatch is None,
        "sample": str(args.sample),
        "seed": int(args.seed),
        "tokens": int(args.tokens),
        "root_feedback": bool(args.root_feedback),
        "first_mismatch_token": mismatch_token,
        "first_mismatch_field": mismatch,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
