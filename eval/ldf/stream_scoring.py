"""Scoring helpers for conservative stream best-of-K selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch

from utils.motion_process import recover_root_rot_pos


@dataclass(frozen=True)
class ConservativeGateConfig:
    fde_weight: float = 1.0
    vel_weight: float = 0.5
    rel_margin: float = 0.10
    abs_margin: float = 0.03
    cont_tol: float = 0.03
    force_candidate0: bool = False
    switch_cooldown_steps: int = 0


@dataclass(frozen=True)
class CandidateScore:
    index: int
    xz_ade: float
    xz_fde: float
    pos_cont: float
    vel_cont: float

    def track(self, cfg: ConservativeGateConfig) -> float:
        return float(self.xz_ade) + float(cfg.fde_weight) * float(self.xz_fde)

    def continuity(self, cfg: ConservativeGateConfig) -> float:
        return float(self.pos_cont) + float(cfg.vel_weight) * float(self.vel_cont)


@dataclass(frozen=True)
class SwitchDecision:
    selected_index: int
    reason: str
    selected_track: float
    candidate0_track: float
    selected_continuity: float
    candidate0_continuity: float


def _sample_traj7(sample_batch: Dict) -> torch.Tensor:
    traj7 = sample_batch.get("traj_cond_7d")
    if traj7 is None:
        raise ValueError("root feedback requires sample_batch['traj_cond_7d']")
    value = traj7[0] if torch.is_tensor(traj7) and traj7.ndim == 3 else traj7
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32)
    return value.float()


def target_xz_slice(sample_batch: Dict, frame_range: tuple[int, int]) -> torch.Tensor:
    start_frame, end_frame = frame_range
    start = int(start_frame)
    end = int(end_frame)
    target = _sample_traj7(sample_batch)
    if target.shape[0] < end:
        pad = target[-1:].expand(end - target.shape[0], -1)
        target = torch.cat([target, pad], dim=0)
    return target[start:end, [0, 2]].float().cpu()


def decoded_chunk_root_xz(
    decoded_chunk: torch.Tensor,
    previous_decoded_chunks: Optional[Sequence[torch.Tensor]] = None,
) -> torch.Tensor:
    chunks = list(previous_decoded_chunks or []) + [decoded_chunk]
    full = torch.cat(chunks, dim=0)
    _, root_xyz = recover_root_rot_pos(full.unsqueeze(0))
    start = int(full.shape[0] - decoded_chunk.shape[0])
    return root_xyz[0, start : start + decoded_chunk.shape[0], [0, 2]].float().cpu()


def previous_root_xz(
    previous_decoded_chunks: Optional[Sequence[torch.Tensor]],
) -> torch.Tensor:
    if not previous_decoded_chunks:
        return torch.empty((0, 2), dtype=torch.float32)
    full = torch.cat(list(previous_decoded_chunks), dim=0)
    _, root_xyz = recover_root_rot_pos(full.unsqueeze(0))
    return root_xyz[0, :, [0, 2]].float().cpu()


def compute_chunk_xz_score(pred_xz: torch.Tensor, target_xz: torch.Tensor) -> tuple[float, float]:
    valid = min(int(pred_xz.shape[0]), int(target_xz.shape[0]))
    if valid <= 0:
        return 0.0, 0.0
    pred = pred_xz[:valid].float()
    target = target_xz[:valid].float()
    per_frame = torch.linalg.norm(pred - target, dim=-1)
    return float(per_frame.mean().item()), float(per_frame[-1].item())


def compute_continuity_score(
    previous_xz: Optional[torch.Tensor],
    candidate_xz: torch.Tensor,
) -> tuple[float, float]:
    if previous_xz is None or int(previous_xz.shape[0]) <= 0 or int(candidate_xz.shape[0]) <= 0:
        return 0.0, 0.0
    previous = previous_xz.float()
    candidate = candidate_xz.float()
    pos_jump = float(torch.linalg.norm(candidate[0] - previous[-1]).item())
    vel_jump = 0.0
    if int(previous.shape[0]) >= 2 and int(candidate.shape[0]) >= 2:
        previous_velocity = previous[-1] - previous[-2]
        candidate_velocity = candidate[1] - candidate[0]
        vel_jump = float(torch.linalg.norm(candidate_velocity - previous_velocity).item())
    return pos_jump, vel_jump


def _decision(
    selected: CandidateScore,
    candidate0: CandidateScore,
    cfg: ConservativeGateConfig,
    reason: str,
) -> SwitchDecision:
    return SwitchDecision(
        selected_index=int(selected.index),
        reason=reason,
        selected_track=selected.track(cfg),
        candidate0_track=candidate0.track(cfg),
        selected_continuity=selected.continuity(cfg),
        candidate0_continuity=candidate0.continuity(cfg),
    )


def choose_candidate(
    scores: Sequence[CandidateScore],
    cfg: ConservativeGateConfig,
    *,
    steps_since_switch: int,
) -> SwitchDecision:
    if not scores:
        raise ValueError("choose_candidate requires at least one candidate score")

    candidate0 = next((score for score in scores if int(score.index) == 0), None)
    if candidate0 is None:
        raise ValueError("choose_candidate requires a score with index 0")

    if bool(cfg.force_candidate0):
        return _decision(candidate0, candidate0, cfg, "force_candidate0")
    if (
        int(cfg.switch_cooldown_steps) > 0
        and int(steps_since_switch) < int(cfg.switch_cooldown_steps)
    ):
        return _decision(candidate0, candidate0, cfg, "cooldown")

    track0 = candidate0.track(cfg)
    cont0 = candidate0.continuity(cfg)
    required_gain = max(float(cfg.abs_margin), float(cfg.rel_margin) * track0)
    best = candidate0
    best_track = track0
    for score in scores:
        if int(score.index) == 0:
            continue
        track = score.track(cfg)
        continuity = score.continuity(cfg)
        if track < track0 - required_gain and continuity <= cont0 + float(cfg.cont_tol):
            if best is candidate0 or track < best_track:
                best = score
                best_track = track

    if best is candidate0:
        return _decision(candidate0, candidate0, cfg, "keep_candidate0")
    return _decision(best, candidate0, cfg, "switch_improved_continuous")
