"""Serial state-fork best-of-K selection for LDF stream generation."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from eval.ldf.stream_scoring import (
    CandidateScore,
    ConservativeGateConfig,
    choose_candidate,
    compute_chunk_xz_score,
    compute_continuity_score,
    decoded_chunk_root_xz,
    previous_root_xz,
    target_xz_slice,
)
from eval.ldf.stream_state import (
    CandidateState,
    StepOutput,
    capture_runtime_snapshot,
    restore_runtime_snapshot,
)
from eval.ldf.stream_step import run_one_stream_step


@dataclass(frozen=True)
class StreamBestOfKConfig:
    """Config for serial stream best-of-K.

    xz_weight and cont_weight are accepted for config compatibility with the
    old selector, but conservative gate v1 scores tracking as ADE + FDE weight
    and continuity as position + velocity weight.
    """

    k: int = 1
    score: str = "xz"
    xz_weight: float = 1.0
    fde_weight: float = 1.0
    cont_weight: float = 0.0
    vel_weight: float = 0.5
    rel_margin: float = 0.10
    abs_margin: float = 0.03
    cont_tol: float = 0.03
    force_candidate0: bool = False
    switch_cooldown_steps: int = 0
    debug: bool = False

    @classmethod
    def from_values(
        cls,
        *,
        k: int = 1,
        score: str = "xz",
        xz_weight: float = 1.0,
        fde_weight: float = 1.0,
        cont_weight: float = 0.0,
        vel_weight: float = 0.5,
        rel_margin: float = 0.10,
        abs_margin: float = 0.03,
        cont_tol: float = 0.03,
        force_candidate0: bool = False,
        switch_cooldown_steps: int = 0,
        debug: bool = False,
    ) -> "StreamBestOfKConfig":
        out = cls(
            k=max(1, int(k)),
            score=str(score),
            xz_weight=float(xz_weight),
            fde_weight=float(fde_weight),
            cont_weight=float(cont_weight),
            vel_weight=float(vel_weight),
            rel_margin=float(rel_margin),
            abs_margin=float(abs_margin),
            cont_tol=float(cont_tol),
            force_candidate0=bool(force_candidate0),
            switch_cooldown_steps=int(switch_cooldown_steps),
            debug=bool(debug),
        )
        if out.score != "xz":
            raise ValueError(
                "Only eval_stream_best_of_k_score='xz' is implemented; "
                f"got {out.score!r}."
            )
        return out

    @property
    def enabled(self) -> bool:
        return int(self.k) > 1


def _seed_extra_candidate(
    base_seed: int,
    local_commit_index: int,
    candidate_idx: int,
) -> None:
    seed = int(base_seed) + int(local_commit_index) * 9176 + int(candidate_idx) * 1000003
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _randomize_extra_candidate_proposal_noise(model: Any, local_commit_index: int) -> None:
    generated = getattr(model, "generated", None)
    if not torch.is_tensor(generated):
        return
    start = max(0, int(local_commit_index))
    if start >= int(generated.shape[2]):
        return
    generated[:, :, start:, ...] = torch.randn_like(generated[:, :, start:, ...])


def _score_step_output(
    *,
    candidate_idx: int,
    step_output: StepOutput,
    sample_batch: Dict,
    previous_decoded_chunks: list[torch.Tensor],
) -> tuple[CandidateScore, dict]:
    pred_xz = decoded_chunk_root_xz(
        step_output.decoded_chunk,
        previous_decoded_chunks,
    )
    target_xz = target_xz_slice(
        sample_batch,
        step_output.target_xz_frame_range,
    )
    xz_ade, xz_fde = compute_chunk_xz_score(pred_xz, target_xz)
    prev_xz = previous_root_xz(previous_decoded_chunks)
    pos_cont, vel_cont = compute_continuity_score(prev_xz, pred_xz)
    score = CandidateScore(
        index=int(candidate_idx),
        xz_ade=float(xz_ade),
        xz_fde=float(xz_fde),
        pos_cont=float(pos_cont),
        vel_cont=float(vel_cont),
    )
    debug = {
        "index": int(candidate_idx),
        "xz_ade": float(xz_ade),
        "xz_fde": float(xz_fde),
        "pos_cont": float(pos_cont),
        "vel_cont": float(vel_cont),
        "commit_token_range": list(step_output.commit_token_range),
        "commit_frame_range": list(step_output.commit_frame_range),
        "decoded_chunk_frame_range": list(step_output.decoded_chunk_frame_range),
        "target_xz_frame_range": list(step_output.target_xz_frame_range),
        "decoded_chunk_length": int(step_output.decoded_chunk.shape[0]),
        "commit_frame_range_matches_candidate0": True,
        "decoded_chunk_frame_range_matches_candidate0": True,
        "target_xz_frame_range_matches_candidate0": True,
        "decoded_chunk_length_matches_candidate0": True,
        "frame_range_matches_candidate0": True,
    }
    return score, debug


def _score_to_record(
    score: CandidateScore,
    debug: dict,
    gate_cfg: ConservativeGateConfig,
) -> dict:
    return {
        **debug,
        "track": float(score.track(gate_cfg)),
        "continuity": float(score.continuity(gate_cfg)),
    }


def run_best_of_k_step(
    *,
    model: Any,
    vae: Any,
    stream: Any,
    stream_conditioner: Any,
    step_payload: dict,
    sample_batch: dict,
    first_chunk: bool,
    device: torch.device,
    local_commit_index: int,
    generated_frames: int,
    previous_decoded_chunks: list[torch.Tensor],
    chunk_frame_ends: list[int],
    frames_per_token: int,
    cfg: StreamBestOfKConfig,
    steps_since_switch: int,
) -> tuple[StepOutput, dict]:
    started = time.perf_counter()
    base_snapshot = capture_runtime_snapshot(
        model=model,
        vae=vae,
        stream_conditioner=stream_conditioner,
        first_chunk=first_chunk,
        generated_frames=generated_frames,
        chunk_frame_ends=chunk_frame_ends,
    )
    base_seed = int(torch.initial_seed())
    candidates: list[CandidateState] = []
    scores: list[CandidateScore] = []
    score_debugs: list[dict] = []
    force_candidate0 = bool(cfg.force_candidate0)

    try:
        for candidate_idx in range(int(cfg.k)):
            restore_runtime_snapshot(
                model=model,
                vae=vae,
                stream_conditioner=stream_conditioner,
                snapshot=base_snapshot,
            )
            if int(candidate_idx) > 0:
                _seed_extra_candidate(base_seed, local_commit_index, candidate_idx)
                _randomize_extra_candidate_proposal_noise(
                    model,
                    local_commit_index=local_commit_index,
                )
            step_output = run_one_stream_step(
                model=model,
                vae=vae,
                stream=stream,
                step_payload=step_payload,
                first_chunk=first_chunk,
                device=device,
                local_commit_index=local_commit_index,
                generated_frames=generated_frames,
                frames_per_token=frames_per_token,
            )
            chunk_frames = int(step_output.decoded_chunk.shape[0])
            post_snapshot = capture_runtime_snapshot(
                model=model,
                vae=vae,
                stream_conditioner=stream_conditioner,
                first_chunk=False,
                generated_frames=int(generated_frames) + chunk_frames,
                chunk_frame_ends=list(chunk_frame_ends)
                + [int(step_output.commit_frame_range[1])],
            )
            score, debug = _score_step_output(
                candidate_idx=candidate_idx,
                step_output=step_output,
                sample_batch=sample_batch,
                previous_decoded_chunks=previous_decoded_chunks,
            )
            candidates.append(
                CandidateState(
                    index=int(candidate_idx),
                    step_output=step_output,
                    snapshot=post_snapshot,
                )
            )
            scores.append(score)
            score_debugs.append(debug)

        candidate0_output = candidates[0].step_output
        candidate0_latent = candidate0_output.clean_committed_latent.float()
        candidate0_commit_range = tuple(candidate0_output.commit_frame_range)
        candidate0_decoded_range = tuple(candidate0_output.decoded_chunk_frame_range)
        candidate0_target_range = tuple(candidate0_output.target_xz_frame_range)
        candidate0_length = int(candidate0_output.decoded_chunk.shape[0])
        for candidate, debug in zip(candidates, score_debugs):
            output = candidate.step_output
            candidate_latent = output.clean_committed_latent.float()
            if tuple(candidate_latent.shape) == tuple(candidate0_latent.shape):
                latent_diff = float(
                    (candidate_latent - candidate0_latent).abs().max().item()
                )
            else:
                latent_diff = float("nan")
            commit_matches = tuple(output.commit_frame_range) == candidate0_commit_range
            decoded_matches = tuple(output.decoded_chunk_frame_range) == candidate0_decoded_range
            target_matches = tuple(output.target_xz_frame_range) == candidate0_target_range
            length_matches = int(output.decoded_chunk.shape[0]) == candidate0_length
            aggregate_matches = (
                commit_matches
                and decoded_matches
                and target_matches
                and length_matches
            )
            debug["commit_frame_range_matches_candidate0"] = bool(commit_matches)
            debug["decoded_chunk_frame_range_matches_candidate0"] = bool(decoded_matches)
            debug["target_xz_frame_range_matches_candidate0"] = bool(target_matches)
            debug["decoded_chunk_length_matches_candidate0"] = bool(length_matches)
            debug["frame_range_matches_candidate0"] = bool(aggregate_matches)
            debug["latent_max_abs_diff_from_candidate0"] = latent_diff
            if not aggregate_matches:
                force_candidate0 = True

        gate_cfg = ConservativeGateConfig(
            fde_weight=float(cfg.fde_weight),
            vel_weight=float(cfg.vel_weight),
            rel_margin=float(cfg.rel_margin),
            abs_margin=float(cfg.abs_margin),
            cont_tol=float(cfg.cont_tol),
            force_candidate0=force_candidate0,
            switch_cooldown_steps=int(cfg.switch_cooldown_steps),
        )
        decision = choose_candidate(
            scores,
            gate_cfg,
            steps_since_switch=int(steps_since_switch),
        )
        selected_idx = int(decision.selected_index)
        selected = candidates[selected_idx]
        restore_runtime_snapshot(
            model=model,
            vae=vae,
            stream_conditioner=stream_conditioner,
            snapshot=selected.snapshot,
        )
        candidate_records = [
            _score_to_record(score, debug, gate_cfg)
            for score, debug in zip(scores, score_debugs)
        ]
        record = {
            "selected_idx": selected_idx,
            "switch_reason": str(decision.reason),
            "candidate_scores": candidate_records,
            "scores": [float(score.track(gate_cfg)) for score in scores],
            "selected_score": float(scores[selected_idx].track(gate_cfg)),
            "candidate0_track": float(decision.candidate0_track),
            "selected_track": float(decision.selected_track),
            "candidate0_continuity": float(decision.candidate0_continuity),
            "selected_continuity": float(decision.selected_continuity),
            "elapsed_sec": float(time.perf_counter() - started),
        }
        return selected.step_output, record
    except Exception:
        restore_runtime_snapshot(
            model=model,
            vae=vae,
            stream_conditioner=stream_conditioner,
            snapshot=base_snapshot,
        )
        raise
