"""RefinerDataset (T_A_04 final v2).

HumanML3D raw 263D → B-full canonical 7D + history + sparse-simulated path +
num_token target. Used by `train_refiner.py` (T_A_08).

References:
- docs/TODO.md §T_A_04 lines 840-1098 — full spec + Required changes +
  unit tests T01-T15.
- docs/design.md §0.3 (Anchor Convention v1) and §3.4.1 (path simulation).

Key design rules (HARD CONSTRAINTS):
1. anchor = duplicate (§0.3 v1): same frame is both current_motion's last
   slot AND target_waypoints[0].
2. full-plan mode: anchor = motion[0], `valid_history_frames = 1` (scalar,
   not a 1..4 range).
3. Sampling order: mode → anchor_frame → num_tokens (round 8 P0-5; the
   `target_mask.sum() == num_frames_for_tokens(num_tokens)` lock-in needs
   token-aligned `target_frame_count`, hence we derive frame_count from
   num_tokens after selecting anchor).
4. Sliding eligibility: only clips with `T >= (N_hist - 1) +
   num_frames_for_tokens(min_tokens)` may sample sliding mode (round 8 P0-4);
   shorter clips force full mode regardless of dice roll.
5. motion_5d_world via `torch.cat` of [root_xyz, cos(yaw)[..., None],
   sin(yaw)[..., None]] (round 8 P0-6 shape bug guard).
6. 7D construction via `utils.motion_process.root_to_traj_feats_7d`
   (T_B_05 helper; first-frame fwd_delta/yaw_delta = 0).
7. cos / sin channels are NOT z-scored (unit-vector invariant). T_A_06's
   `compute_5d_stats` produces `*_norm_indices.npy` listing only the channels
   to z-score (xyz + fwd_delta + yaw_delta); this dataset honors them.

The dataset returns dict with the schema documented in `__getitem__`.
"""

from __future__ import annotations

import logging
import os
import random as random_module
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from utils.local_frame import (
    canonicalize_5d,
    canonicalize_7d,
    root_quat_to_physical_yaw,
)
from utils.motion_process import recover_root_rot_pos, root_to_traj_feats_7d
from utils.path_arclength import arclength_resample
from utils.token_frame import num_frames_for_tokens

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sparse_sample_arclength(points_xz: Tensor, K: int) -> Tensor:
    """Pick K points from `points_xz` keeping start + end + (K-2) middles
    spaced by arclength. K must satisfy 2 <= K <= len(points_xz).
    """
    M = points_xz.shape[0]
    if K >= M:
        return points_xz.clone()
    if K <= 2:
        # endpoints only
        return torch.stack([points_xz[0], points_xz[-1]], dim=0)
    # Cumulative arclength
    seg = points_xz[1:] - points_xz[:-1]
    seg_len = seg.norm(dim=-1)
    cum = torch.cat([torch.zeros(1, device=points_xz.device, dtype=points_xz.dtype),
                       torch.cumsum(seg_len, dim=0)])
    total = float(cum[-1].item())
    if total < 1e-9:
        # degenerate: just stride-pick
        idxs = torch.linspace(0, M - 1, K).round().long()
        return points_xz[idxs]
    # Target arclengths for K samples, including endpoints
    targets = torch.linspace(0.0, total, K, device=points_xz.device, dtype=points_xz.dtype)
    # Find for each target the nearest cum index (using searchsorted, then
    # snap to nearest point to keep them as actual input points).
    idxs = torch.searchsorted(cum, targets).clamp(max=M - 1)
    # Deduplicate while preserving order; ensure start and end present.
    seen, dedup = set(), []
    for v in idxs.tolist():
        if v not in seen:
            seen.add(v)
            dedup.append(v)
    if 0 not in seen:
        dedup.insert(0, 0)
    if M - 1 not in seen:
        dedup.append(M - 1)
    dedup = sorted(set(dedup))
    return points_xz[dedup]


def _pad_or_truncate(x: Tensor, target_len: int) -> Tensor:
    """Zero-pad along dim 0 (or truncate) so dim 0 == target_len."""
    cur = x.shape[0]
    if cur == target_len:
        return x
    if cur > target_len:
        return x[:target_len]
    pad_shape = list(x.shape)
    pad_shape[0] = target_len - cur
    pad = x.new_zeros(*pad_shape)
    return torch.cat([x, pad], dim=0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RefinerDataset(Dataset):
    """Final v2 RefinerDataset.

    Parameters
    ----------
    clips : list[dict]
        Each item must have keys `motion_263` (np.ndarray or Tensor [T, 263])
        and `text` (str). T must be >= num_frames_for_tokens(min_tokens) for
        the clip to be `full_eligible`; sliding eligibility is stricter.
    n_hist, n_path, max_tokens, min_tokens, frames_per_token :
        Model-side schema constants; match `configs/root_refiner.yaml`.
    full_plan_ratio : probability of choosing full mode (when sliding-eligible).
    path_trim_prob / path_trim_max_frames : random trim aug on path input.
    path_sparse_prob / path_sparse_range : random sparse aug on path input.
    normalize : if True, apply selective z-score using `stats_dir`'s mean/std
        and `*_norm_indices.npy` files; if False, raw values are returned
        (T_A_06 stats computation uses normalize=False to avoid double-norm).
    stats_dir : directory containing `current_motion_mean.npy`,
        `current_motion_std.npy`, `current_motion_norm_indices.npy`,
        `waypoint_mean.npy`, `waypoint_std.npy`, `waypoint_norm_indices.npy`.
    seed : optional fixed seed for reproducibility (testing).
    """

    def __init__(
        self,
        clips: list[dict],
        *,
        n_hist: int = 20,
        n_path: int = 64,
        max_tokens: int = 49,
        min_tokens: int = 4,
        frames_per_token: int = 4,
        full_plan_ratio: float = 0.5,
        path_trim_prob: float = 0.3,
        path_trim_max_frames: int = 10,
        path_sparse_prob: float = 0.5,
        path_sparse_range: tuple[int, int] = (3, 8),
        normalize: bool = False,
        stats_dir: str | os.PathLike | None = None,
        seed: int | None = None,
        randomize_caption: bool = True,
    ):
        self._clips = clips
        self.n_hist = n_hist
        self.n_path = n_path
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.frames_per_token = frames_per_token
        self.full_plan_ratio = full_plan_ratio
        self.path_trim_prob = path_trim_prob
        self.path_trim_max_frames = path_trim_max_frames
        self.path_sparse_prob = path_sparse_prob
        self.path_sparse_range = path_sparse_range
        self.normalize = normalize
        # When False (e.g. validation / benchmark), always use the first caption
        # so the metric is comparable across epochs; when True (training), draw a
        # random caption per sample for text augmentation. See _choose_caption.
        self.randomize_caption = randomize_caption

        # Pre-compute clip lengths.
        self._clip_lengths = [int(self._motion_of(c).shape[0]) for c in clips]

        # Eligibility split (round 8 P0-4).
        min_full = num_frames_for_tokens(min_tokens, frames_per_token)
        min_sliding = (n_hist - 1) + min_full

        self.full_eligible_indices: list[int] = [
            i for i, T in enumerate(self._clip_lengths) if T >= min_full
        ]
        self.sliding_eligible_indices: set[int] = {
            i for i, T in enumerate(self._clip_lengths) if T >= min_sliding
        }
        # T07: full-eligibility is the dataset-level filter; short clips never
        # appear via __getitem__.
        self.valid_indices: list[int] = self.full_eligible_indices

        # Normalization stats (optional).
        self._cm_mean = None
        self._cm_std = None
        self._cm_norm_idx = None
        self._wp_mean = None
        self._wp_std = None
        self._wp_norm_idx = None
        if self.normalize:
            if stats_dir is None:
                raise ValueError("normalize=True requires stats_dir")
            self._load_stats(Path(stats_dir))

        self._seed = seed
        self._rng = random_module.Random(seed) if seed is not None else random_module.Random()

    def set_worker_seed(self) -> None:
        """P1-3: reseed this worker's augmentation RNG to a per-worker-unique
        value. Without this, fork()ed DataLoader workers inherit one identical
        `random.Random` state (it's a Python RNG, NOT auto-seeded per worker like
        torch's), so path-trim / sparse augmentation correlates across workers.
        torch.initial_seed() is set distinctly per worker by the DataLoader, so we
        derive from it (combined with the dataset's base seed when fixed).
        """
        base = int(torch.initial_seed()) % (2 ** 31)
        if self._seed is not None:
            base = (base + int(self._seed)) % (2 ** 31)
        self._rng = random_module.Random(base)

    def reset_rng(self) -> None:
        """Reset the sampling/augmentation RNG to the dataset's base `seed` so a
        repeat single-process pass (e.g. a benchmark run) draws the IDENTICAL
        mode / anchor / num_tokens / path-aug sequence. `get_sample` advances the
        RNG every call (force_no_path_aug only disables PATH aug, not the
        mode/anchor/num_tokens dice), so without this two passes over the same
        dataset object diverge. Reproducible only when constructed with a fixed
        `seed` (seed=None → reseeds from OS entropy, matching construction)."""
        self._rng = random_module.Random(self._seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip_idx = self.valid_indices[idx]
        return self._make_sample(clip_idx)

    # ------------------------------------------------------------------
    # Helpers (kept public for unit tests / debug)
    # ------------------------------------------------------------------

    @staticmethod
    def _motion_of(clip: dict) -> Tensor:
        m = clip["motion_263"]
        if not isinstance(m, torch.Tensor):
            m = torch.as_tensor(m, dtype=torch.float32)
        elif m.dtype != torch.float32:
            m = m.float()
        return m

    def _choose_caption(self, clip: dict, force_text_idx: int | None) -> str:
        """Pick the caption for this sample. A clip may carry `texts` (all
        distinct captions, from load_clips_from_dir); HumanML3D gives 3-5 per
        clip. With `randomize_caption` (training default) a random caption is
        drawn each sample — paraphrases become text augmentation, and the frozen
        T5 cache holds every caption so the encoder lookup always hits. With it
        off (val / benchmark) the first caption is used so the metric stays
        comparable across epochs; that path consumes NO RNG, so the mode/anchor/
        num_tokens draw order matches the legacy single-caption behavior.
        `force_text_idx` pins a specific caption (benchmark / tests) and takes
        precedence. Falls back to the single `text` key when `texts` is
        absent/empty (synthetic test clips), keeping the legacy schema working.
        """
        texts = clip.get("texts")
        if texts:
            if force_text_idx is not None:
                return texts[int(force_text_idx) % len(texts)]
            if self.randomize_caption:
                return self._rng.choice(texts)
            return texts[0]
        return clip.get("text", "")

    def get_sample(self, idx_in_valid: int, *,
                    force_mode: str | None = None,
                    force_num_tokens: int | None = None,
                    force_anchor_frame: int | None = None,
                    force_no_path_aug: bool = False,
                    force_text_idx: int | None = None) -> dict[str, Any]:
        """Debug / unit-test entry point: bypasses random by forcing decisions.

        `force_mode`: "full" | "sliding" | None (use full_plan_ratio dice).
        `force_num_tokens`: override num_tokens; must be in [min_tokens, max_tokens].
        `force_anchor_frame`: override anchor_frame.
        `force_no_path_aug`: skip trim / sparse augmentation (deterministic path).
        `force_text_idx`: pick `clip["texts"][i]` instead of a random caption
            (no-op when the clip has no `texts` list).
        """
        clip_idx = self.valid_indices[idx_in_valid]
        return self._make_sample(
            clip_idx,
            force_mode=force_mode,
            force_num_tokens=force_num_tokens,
            force_anchor_frame=force_anchor_frame,
            force_no_path_aug=force_no_path_aug,
            force_text_idx=force_text_idx,
        )

    # ------------------------------------------------------------------
    # Stats loader (z-score normalization helper)
    # ------------------------------------------------------------------

    def _load_stats(self, stats_dir: Path):
        self._cm_mean = torch.as_tensor(
            np.load(stats_dir / "current_motion_mean.npy"), dtype=torch.float32,
        )
        self._cm_std = torch.as_tensor(
            np.load(stats_dir / "current_motion_std.npy"), dtype=torch.float32,
        ).clamp(min=1e-6)
        self._cm_norm_idx = torch.as_tensor(
            np.load(stats_dir / "current_motion_norm_indices.npy"), dtype=torch.long,
        )
        self._wp_mean = torch.as_tensor(
            np.load(stats_dir / "waypoint_mean.npy"), dtype=torch.float32,
        )
        self._wp_std = torch.as_tensor(
            np.load(stats_dir / "waypoint_std.npy"), dtype=torch.float32,
        ).clamp(min=1e-6)
        self._wp_norm_idx = torch.as_tensor(
            np.load(stats_dir / "waypoint_norm_indices.npy"), dtype=torch.long,
        )
        # ⚠ Validate shapes / index ranges up front so a stale or mismatched
        # stats_dir fails loudly here instead of via a cryptic IndexError deep
        # in __getitem__ (or, worse, silently normalizing the wrong channels).
        if self._cm_mean.shape != (5,) or self._cm_std.shape != (5,):
            raise ValueError(
                f"current_motion stats must be shape (5,), got "
                f"mean={tuple(self._cm_mean.shape)} std={tuple(self._cm_std.shape)}"
            )
        if self._wp_mean.shape != (7,) or self._wp_std.shape != (7,):
            raise ValueError(
                f"waypoint stats must be shape (7,), got "
                f"mean={tuple(self._wp_mean.shape)} std={tuple(self._wp_std.shape)}"
            )
        if self._cm_norm_idx.numel() and int(self._cm_norm_idx.max()) >= 5:
            raise ValueError(f"current_motion_norm_indices out of range for dim 5: "
                             f"{self._cm_norm_idx.tolist()}")
        # Heading channels 3/4 (cos/sin yaw) are unit-vector invariant and MUST
        # NOT be z-scored (rule 7) — same invariant as the waypoint check below.
        # current_motion is only a model input (hist_proj), so a z-scored heading
        # wouldn't break a loss, but it silently diverges from the design's
        # unit-heading convention; fail loudly on a stale/hand-edited stats file.
        if set(self._cm_norm_idx.tolist()) & {3, 4}:
            raise ValueError(
                "current_motion_norm_indices must NOT include heading channels 3/4 "
                "(cos/sin yaw are unit-vector invariant; rule 7): "
                f"{self._cm_norm_idx.tolist()}"
            )
        if self._wp_norm_idx.numel() and int(self._wp_norm_idx.max()) >= 7:
            raise ValueError(f"waypoint_norm_indices out of range for dim 7: "
                             f"{self._wp_norm_idx.tolist()}")
        # Heading channels 3/4 (cos/sin yaw) are unit-vector invariant and MUST
        # NOT be z-scored (rule 7): the cosine heading loss assumes a unit-norm GT
        # heading, which a z-scored target would violate. Fail loudly on a stale
        # stats file rather than silently breaking heading supervision.
        if set(self._wp_norm_idx.tolist()) & {3, 4}:
            raise ValueError(
                "waypoint_norm_indices must NOT include heading channels 3/4 "
                "(cos/sin yaw are unit-vector invariant; z-scoring them breaks the "
                "unit-norm GT the cosine heading loss assumes): "
                f"{self._wp_norm_idx.tolist()}"
            )

    def _apply_zscore(self, tensor: Tensor, mean: Tensor, std: Tensor,
                       norm_idx: Tensor) -> Tensor:
        """Z-score only the channels listed in norm_idx; leave others (cos/sin)
        bit-for-bit unchanged. Operates per-frame on the last dim.

        mean/std are moved to `tensor`'s device/dtype so this works even if the
        sample tensor lives on CUDA / is non-fp32 (stats are loaded as CPU fp32).
        """
        out = tensor.clone()
        mean = mean.to(device=tensor.device, dtype=tensor.dtype)
        std = std.to(device=tensor.device, dtype=tensor.dtype)
        for c in norm_idx.tolist():
            out[..., c] = (out[..., c] - mean[c]) / std[c]
        return out

    # ------------------------------------------------------------------
    # Sample builder (the actual data flow)
    # ------------------------------------------------------------------

    def _make_sample(self,
                      clip_idx: int,
                      *,
                      force_mode: str | None = None,
                      force_num_tokens: int | None = None,
                      force_anchor_frame: int | None = None,
                      force_no_path_aug: bool = False,
                      force_text_idx: int | None = None) -> dict[str, Any]:
        clip = self._clips[clip_idx]
        motion_263 = self._motion_of(clip).unsqueeze(0)            # [1, T, 263]
        T = motion_263.shape[1]
        text = self._choose_caption(clip, force_text_idx)

        # Step 1: 263D recovery → (root_quat, root_xyz, root_yaw)
        root_quat, root_xyz = recover_root_rot_pos(motion_263)
        root_quat = root_quat[0]                                    # [T, 4]
        root_xyz = root_xyz[0]                                      # [T, 3]
        root_yaw = root_quat_to_physical_yaw(root_quat)             # [T]

        # Step 2: compose world 5D + 7D (round 8 P0-6 cat with explicit unsqueeze).
        motion_5d_world = torch.cat(
            [
                root_xyz,                                  # [T, 3]
                torch.cos(root_yaw)[..., None],            # [T, 1]
                torch.sin(root_yaw)[..., None],            # [T, 1]
            ],
            dim=-1,
        )                                                  # [T, 5]
        motion_7d_world = root_to_traj_feats_7d(
            root_quat.unsqueeze(0), root_xyz.unsqueeze(0),
        )[0]                                               # [T, 7]

        # Step 3: mode + anchor + num_tokens (round 8 P0-5 order).
        min_full = num_frames_for_tokens(self.min_tokens, self.frames_per_token)

        # Choose mode.
        if force_mode is not None:
            mode_drawn = force_mode
        elif self._rng.random() < self.full_plan_ratio:
            mode_drawn = "full"
        else:
            mode_drawn = "sliding"
        if mode_drawn == "sliding" and clip_idx not in self.sliding_eligible_indices:
            mode = "full"   # P0-4 fallback
        else:
            mode = mode_drawn

        # ⚠ max_valid_tokens must satisfy `num_frames_for_tokens(N) <= remaining_frames`,
        # i.e. N <= (remaining_frames + frames_per_token - 1) // frames_per_token.
        # The TODO doc's inline formula `frame_idx_to_token_idx(R-1) + 1` counts
        # the partial-cover token at the tail and over-shoots — fixed here so the
        # downstream `assert anchor_frame + target_frame_count <= T` cannot fire
        # on clips whose length is not exactly a token boundary.
        def _max_tokens_in_frames(remaining: int) -> int:
            if remaining <= 0:
                return 0
            return (remaining + self.frames_per_token - 1) // self.frames_per_token

        if mode == "full":
            anchor_frame = 0 if force_anchor_frame is None else int(force_anchor_frame)
            valid_history_frames = 1
            history_frame_indices = [anchor_frame]
            # ⚠ bound by frames REMAINING after the anchor (T - anchor_frame), not
            # the whole clip length T. With anchor_frame=0 (normal full mode) these
            # are identical, but a forced/non-zero anchor must not let
            # target_frame_count overrun the clip (else the assert below fires).
            max_valid_tokens = min(self.max_tokens, _max_tokens_in_frames(T - anchor_frame))
        else:
            lo = self.n_hist - 1
            hi = T - min_full                              # inclusive upper bound
            if hi < lo:
                # Defensive: should not happen for sliding_eligible clips.
                raise RuntimeError(
                    f"sliding anchor range invalid: lo={lo} hi={hi} T={T} "
                    f"clip_idx={clip_idx}"
                )
            anchor_frame = (
                int(force_anchor_frame) if force_anchor_frame is not None
                else self._rng.randint(lo, hi)
            )
            valid_history_frames = self.n_hist
            history_frame_indices = list(
                range(anchor_frame - self.n_hist + 1, anchor_frame + 1),
            )
            remaining_frames = T - anchor_frame
            max_valid_tokens = min(
                self.max_tokens, _max_tokens_in_frames(remaining_frames),
            )

        if force_num_tokens is not None:
            num_tokens = int(force_num_tokens)
        else:
            num_tokens = self._rng.randint(self.min_tokens, max_valid_tokens)
        num_tokens = max(self.min_tokens, min(num_tokens, max_valid_tokens))
        target_frame_count = num_frames_for_tokens(num_tokens, self.frames_per_token)
        assert anchor_frame + target_frame_count <= T, (
            f"anchor_frame={anchor_frame} + target_frame_count={target_frame_count} > T={T}"
        )

        # Step 4: canonicalize.
        anchor_xz = root_xyz[anchor_frame, [0, 2]]                  # [2]
        anchor_yaw = root_yaw[anchor_frame]                          # scalar

        # current_motion (anchor at last slot).
        current_motion_world = motion_5d_world[history_frame_indices]   # [valid_hist, 5]
        current_motion_local = canonicalize_5d(
            current_motion_world, anchor_xz, anchor_yaw,
        )
        # Pad to [n_hist, 5] with leading zeros (history_mask flags valid slots).
        if valid_history_frames < self.n_hist:
            pad = current_motion_local.new_zeros(
                self.n_hist - valid_history_frames, 5,
            )
            current_motion = torch.cat([pad, current_motion_local], dim=0)
        else:
            current_motion = current_motion_local
        history_mask = torch.zeros(self.n_hist, dtype=torch.bool)
        history_mask[self.n_hist - valid_history_frames :] = True

        # target_waypoints (anchor at first slot, token-aligned length).
        target_world = motion_7d_world[anchor_frame : anchor_frame + target_frame_count]
        target_local = canonicalize_7d(target_world, anchor_xz, anchor_yaw)
        max_frames = num_frames_for_tokens(self.max_tokens, self.frames_per_token)
        target_waypoints = _pad_or_truncate(target_local, max_frames)
        target_mask = torch.zeros(max_frames, dtype=torch.bool)
        target_mask[:target_frame_count] = True

        # Step 5: path input construction.
        dense_path_local = target_local[:target_frame_count, [0, 2]]   # first point = (0, 0)

        do_aug = not force_no_path_aug
        # Trim aug.
        if do_aug and self._rng.random() < self.path_trim_prob:
            max_trim = min(self.path_trim_max_frames, dense_path_local.shape[0] - 2)
            trim_frames = self._rng.randint(0, max(0, max_trim))
            user_path_source = dense_path_local[trim_frames:]
        else:
            user_path_source = dense_path_local
        if user_path_source.shape[0] < 2:
            user_path_source = dense_path_local   # fallback

        # Sparse aug.
        if do_aug and self._rng.random() < self.path_sparse_prob:
            K = self._rng.randint(*self.path_sparse_range)
            K = max(2, min(K, user_path_source.shape[0]))
            control_points = _sparse_sample_arclength(user_path_source, K)
        else:
            control_points = user_path_source

        path_start_gap = float(user_path_source[0].norm().item())

        # Prepend synthetic anchor (0, 0).
        zero_pt = control_points.new_zeros(1, 2)
        control_points = torch.cat([zero_pt, control_points], dim=0)

        # Arclength resample to N_path.
        cp_np = control_points.detach().cpu().numpy().astype(np.float64)
        arc = arclength_resample(cp_np, n_points=self.n_path)
        xz_path = torch.as_tensor(arc.points_xz, dtype=torch.float32)
        # arc.points_xz[0] is the prepended (0, 0) when control_points is non-degenerate.
        path_mask = torch.as_tensor(arc.mask, dtype=torch.bool)
        path_length = float(arc.total_length)
        chord_length = float(torch.tensor(
            arc.points_xz[-1] - arc.points_xz[0], dtype=torch.float64,
        ).norm().item())
        path_stats = torch.tensor(
            [path_length, path_start_gap, chord_length], dtype=torch.float32,
        )

        # Step 6: optional z-score (selective).
        if self.normalize:
            current_motion = self._apply_zscore(
                current_motion, self._cm_mean, self._cm_std, self._cm_norm_idx,
            )
            target_waypoints = self._apply_zscore(
                target_waypoints, self._wp_mean, self._wp_std, self._wp_norm_idx,
            )
            # Path xz uses waypoint xz mean/std (indices 0, 2 of waypoint_norm_indices
            # are x and z). Apply same z-score per axis.
            if (self._wp_mean is not None and self._wp_std is not None
                    and self._wp_norm_idx is not None):
                idx_set = set(self._wp_norm_idx.tolist())
                if 0 in idx_set:
                    xz_path[..., 0] = (xz_path[..., 0] - self._wp_mean[0]) / self._wp_std[0]
                if 2 in idx_set:
                    xz_path[..., 1] = (xz_path[..., 1] - self._wp_mean[2]) / self._wp_std[2]

        return {
            "text": text,
            "xz_path": xz_path,                                         # [N_path, 2]
            "path_mask": path_mask,                                     # [N_path]
            "path_stats": path_stats,                                   # [3]
            "current_motion": current_motion,                           # [n_hist, 5]
            "history_mask": history_mask,                               # [n_hist]
            "target_waypoints": target_waypoints,                       # [max_frames, 7]
            "target_mask": target_mask,                                 # [max_frames]
            "num_tokens": torch.tensor(num_tokens, dtype=torch.long),
            "mode": mode,
            # debug only
            "anchor_frame": anchor_frame,
            "anchor_xz_world": anchor_xz.detach().clone(),
            "anchor_yaw_world": anchor_yaw.detach().clone(),
        }


def refiner_worker_init_fn(worker_id: int) -> None:
    """DataLoader worker_init_fn: give each worker a distinct augmentation RNG
    (P1-3). Pass to DataLoader(worker_init_fn=refiner_worker_init_fn)."""
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "set_worker_seed"):
        info.dataset.set_worker_seed()


__all__ = ["RefinerDataset", "refiner_worker_init_fn"]
