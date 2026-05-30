"""HumanML3D-backed RootRefiner datasets.

Self-contained module: shared helpers, the base `RefinerDataset`, the
HumanML3D subclass, the batch collate, the worker-init fn, and the
deterministic fixed-sample tools used by overfit/validation diagnostics.
"""

from __future__ import annotations

import copy
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
from utils.token_frame import num_frames_for_tokens

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        num_token_policy: str = "random",
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
        self.num_token_policy = str(num_token_policy)
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
        torch's), so offset-start path augmentation correlates across workers.
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
        `force_no_path_aug`: skip offset-start path augmentation (deterministic path).
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
        elif self.num_token_policy == "max":
            num_tokens = max_valid_tokens
        elif self.num_token_policy == "random":
            num_tokens = self._rng.randint(self.min_tokens, max_valid_tokens)
        else:
            raise ValueError(
                "num_token_policy must be 'random' or 'max', "
                f"got {self.num_token_policy!r}"
            )
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
        # R2.5: keep an UN-z-scored copy so the path condition (and its physical
        # path_features) can be built in physical space by the shim. The base
        # z-scores `target_waypoints` in place below; the shim reads physical xz
        # from this field so compute_path_features runs on metres, not z-units.
        target_waypoints_physical = target_waypoints.clone()
        target_mask = torch.zeros(max_frames, dtype=torch.bool)
        target_mask[:target_frame_count] = True

        # Step 5: optional z-score (selective).
        if self.normalize:
            current_motion = self._apply_zscore(
                current_motion, self._cm_mean, self._cm_std, self._cm_norm_idx,
            )
            target_waypoints = self._apply_zscore(
                target_waypoints, self._wp_mean, self._wp_std, self._wp_norm_idx,
            )

        return {
            "text": text,
            "current_motion": current_motion,                           # [n_hist, 5]
            "history_mask": history_mask,                               # [n_hist]
            "target_waypoints": target_waypoints,                       # [max_frames, 7]
            "target_waypoints_physical": target_waypoints_physical,     # [max_frames, 7] pre-z-score (R2.5)
            "target_mask": target_mask,                                 # [max_frames]
            "num_tokens": torch.tensor(num_tokens, dtype=torch.long),
            "mode": mode,
            # debug only
            "anchor_frame": anchor_frame,
            "anchor_xz_world": anchor_xz.detach().clone(),
            "anchor_yaw_world": anchor_yaw.detach().clone(),
        }


class HumanML3DRefinerDataset(RefinerDataset):
    """RootRefiner dataset using the new batch contract.

    This class initially reuses the legacy HumanML3D-to-refiner motion pipeline
    and exposes the redesigned field names and masks. Path-mode construction is
    implemented in `utils.refiner.path_condition` and can be integrated without
    changing the public contract.
    """

    def __init__(
        self,
        *args,
        horizon_policy: str | None = None,
        path_condition_policy: str = "dense_path",
        path_condition_ratios: dict[str, float] | None = None,
        offset_start_enabled: bool = False,
        offset_start_prob: float = 0.0,
        offset_start_max_frames: int = 40,
        offset_start_apply_to: tuple[str, ...] | list[str] = ("dense_path", "sparse_path"),
        sparse_path_point_range: tuple[int, int] = (3, 8),
        path_feature_stats_dir: str | None = None,
        sampling_config_hash: str | None = None,
        **kwargs,
    ):
        if horizon_policy is not None and "num_token_policy" not in kwargs:
            kwargs["num_token_policy"] = horizon_policy
        super().__init__(*args, **kwargs)
        self.path_condition_policy = str(path_condition_policy)
        self.path_condition_ratios = dict(
            path_condition_ratios
            or {"dense_path": 0.5, "sparse_path": 0.3, "goal_point": 0.2}
        )
        self.offset_start_enabled = bool(offset_start_enabled)
        self.offset_start_prob = float(offset_start_prob)
        self.offset_start_max_frames = int(offset_start_max_frames)
        self.offset_start_apply_to = tuple(offset_start_apply_to)
        self.sparse_path_point_range = tuple(int(v) for v in sparse_path_point_range)
        self.max_frames = self.max_frames if hasattr(self, "max_frames") else None
        # R2.5: optional PHYSICAL path-feature normalization stats (own mean/std,
        # NOT the waypoint stats). When normalize=True and a stats dir is given,
        # path_features are z-scored by these; otherwise they stay raw-physical.
        self._pf_mean = None
        self._pf_std = None
        if path_feature_stats_dir is not None:
            if sampling_config_hash is None:
                raise ValueError(
                    "path_feature_stats_dir was set but sampling_config_hash is "
                    "None; strong hash validation is mandatory when using "
                    "path-feature stats (pass the hash from "
                    "compute_sampling_config_hash(cfg))."
                )
            from utils.refiner.path_feature_stats import load_path_feature_stats

            stats = load_path_feature_stats(
                path_feature_stats_dir, expected_hash=sampling_config_hash,
            )
            self._pf_mean = stats.mean
            self._pf_std = stats.std

    def _zscore_path_xz(self, path_xz: torch.Tensor) -> torch.Tensor:
        """Z-score path geometry [N, 2] with the WAYPOINT x/z stats (indices 0,2),
        so path tokens match the z-scored waypoints used by the control loss."""
        wp_mean, wp_std = self._wp_mean, self._wp_std
        wp_idx = self._wp_norm_idx
        if wp_mean is None or wp_std is None or wp_idx is None:
            return path_xz
        out = path_xz.clone()
        idx_set = set(wp_idx.tolist())
        if 0 in idx_set:
            out[..., 0] = (out[..., 0] - wp_mean[0]) / wp_std[0]
        if 2 in idx_set:
            out[..., 1] = (out[..., 1] - wp_mean[2]) / wp_std[2]
        return out

    def _normalize_path_features(self, features: torch.Tensor) -> torch.Tensor:
        """Z-score physical path_features by their OWN stats, if loaded."""
        if self._pf_mean is None or self._pf_std is None:
            return features
        return (features - self._pf_mean) / self._pf_std

    @property
    def max_frames(self) -> int:
        from utils.token_frame import num_frames_for_tokens

        return num_frames_for_tokens(self.max_tokens, self.frames_per_token)

    @max_frames.setter
    def max_frames(self, value: int | None) -> None:
        # Compatibility no-op; the legacy class computes this ad hoc.
        pass

    def get_sample(
        self,
        idx_in_valid: int,
        *,
        force_path_mode: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        force_no_path_aug = bool(kwargs.get("force_no_path_aug", False))
        sample = super().get_sample(idx_in_valid, **kwargs)
        return self._convert_legacy_sample(
            sample,
            force_path_mode=force_path_mode,
            force_no_path_aug=force_no_path_aug,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip_idx = self.valid_indices[idx]
        sample = self._make_sample(clip_idx)
        return self._convert_legacy_sample(
            sample,
            force_path_mode=None,
            force_no_path_aug=False,
        )

    def _convert_legacy_sample(
        self,
        sample: dict[str, Any],
        *,
        force_path_mode: str | None,
        force_no_path_aug: bool,
    ) -> dict[str, Any]:
        from utils.refiner.path_condition import build_path_condition

        waypoints = sample["target_waypoints"][..., :5]
        waypoints_mask = sample["target_mask"]
        valid_frame_count = int(waypoints_mask.sum().item())
        # R2.5: build the path condition (and its physical path_features) from the
        # PHYSICAL pre-z-score waypoints, not the z-scored `target_waypoints`.
        # Reading z-scored xz here made compute_path_features run in anisotropic
        # z-units → non-physical path_length and the duration-head overfit. The
        # geometry `path` tokens are re-z-scored below to stay in the waypoint
        # space the control loss compares against; only path_features stay physical.
        if "target_waypoints_physical" in sample:
            physical_wp = sample["target_waypoints_physical"]
        elif getattr(self, "normalize", False):
            # When normalizing, the base MUST hand us the pre-z-score waypoints.
            # Silently falling back to the z-scored `target_waypoints` would
            # rebuild path_features in z-score space — the exact R2.5 overfit
            # bug — so fail loudly (e.g. a sample cache built before R2.5).
            raise KeyError(
                "target_waypoints_physical missing while normalize=True; the "
                "base dataset must emit pre-z-score waypoints. Rebuild any stale "
                "sample cache — a z-scored fallback would silently reintroduce "
                "the R2.5 path-feature-space bug."
            )
        else:
            # normalize=False: target_waypoints is already physical (never z-scored).
            physical_wp = sample["target_waypoints"]
        future_xz = physical_wp[:valid_frame_count, [0, 2]]
        path_mode = force_path_mode or self._sample_path_mode()
        offset_start_frames = 0
        if (
            not force_no_path_aug
            and self.offset_start_enabled
            and path_mode in self.offset_start_apply_to
            and self._rng.random() < self.offset_start_prob
        ):
            max_offset = min(self.offset_start_max_frames, max(0, valid_frame_count - 2))
            offset_start_frames = self._rng.randint(0, max_offset) if max_offset > 0 else 0

        condition = build_path_condition(
            future_xz,
            n_path=self.n_path,
            valid_frame_count=valid_frame_count,
            max_frames=self.max_frames,
            path_mode=path_mode,
            offset_start_frames=offset_start_frames,
            sparse_point_range=self.sparse_path_point_range,
            rng=self._rng,
        )

        out = dict(sample)
        # R2.5: path was built in physical space. Re-apply the waypoint xz z-score
        # to the GEOMETRY tokens (only) so they live in the same space as the
        # z-scored `waypoints` the dense/sparse control loss compares them against.
        # path_features stay PHYSICAL (optionally normalized by their OWN stats).
        path_tokens = condition.path
        path_features = condition.path_features_raw
        if getattr(self, "normalize", False):
            path_tokens = self._zscore_path_xz(path_tokens)
            path_features = self._normalize_path_features(path_features)
        out.update(
            {
                "path": path_tokens,
                "path_valid_mask": condition.path_valid_mask,
                "path_control_mask": condition.path_control_mask,
                "path_features": path_features,
                "path_mode": condition.path_mode,
                "history_motion": sample["current_motion"],
                "waypoints": waypoints,
                "waypoints_mask": waypoints_mask,
                "path_supervision_mask": condition.path_supervision_mask,
                "offset_start_frames": torch.tensor(
                    condition.offset_start_frames,
                    dtype=torch.long,
                ),
            }
        )
        return out

    def _sample_path_mode(self) -> str:
        policy = self.path_condition_policy
        if policy in {"dense_path", "sparse_path", "goal_point"}:
            return policy
        if policy != "mixed":
            raise ValueError(f"unknown path_condition_policy {policy!r}")
        total = sum(max(0.0, float(v)) for v in self.path_condition_ratios.values())
        if total <= 0:
            return "dense_path"
        pick = self._rng.random() * total
        acc = 0.0
        for mode in ("dense_path", "sparse_path", "goal_point"):
            acc += max(0.0, float(self.path_condition_ratios.get(mode, 0.0)))
            if pick <= acc:
                return mode
        return "goal_point"

    @staticmethod
    def _path_features(path: torch.Tensor) -> torch.Tensor:
        from utils.refiner.path_condition import compute_path_features

        return compute_path_features(path)


def refiner_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "text": [sample["text"] for sample in batch],
        "path_mode": [sample["path_mode"] for sample in batch],
    }
    if "mode" in batch[0]:
        out["mode"] = [sample["mode"] for sample in batch]
    for key in (
        "path",
        "path_valid_mask",
        "path_control_mask",
        "path_features",
        "history_motion",
        "history_mask",
        "waypoints",
        "waypoints_mask",
        "path_supervision_mask",
        "offset_start_frames",
        "num_tokens",
    ):
        out[key] = torch.stack([sample[key] for sample in batch])
    return out


def refiner_worker_init_fn(worker_id: int) -> None:
    """DataLoader worker_init_fn: give each worker a distinct augmentation RNG
    (P1-3). Pass to DataLoader(worker_init_fn=refiner_worker_init_fn)."""
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "set_worker_seed"):
        info.dataset.set_worker_seed()



def clone_refiner_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a Refiner sample, cloning tensors explicitly."""
    out: dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        else:
            out[key] = copy.deepcopy(value)
    return out


class FixedRefinerSampleDataset(Dataset):
    """Dataset backed by pre-sampled RefinerDataset outputs."""

    def __init__(self, samples: list[dict[str, Any]]):
        if not samples:
            raise ValueError("FixedRefinerSampleDataset requires at least one sample")
        self._samples = [clone_refiner_sample(sample) for sample in samples]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return clone_refiner_sample(self._samples[int(idx)])


def _mode_for_index(mode_policy: str, idx: int) -> str | None:
    mode_policy = str(mode_policy).lower()
    if mode_policy in ("random", "dataset", "none"):
        return None
    if mode_policy in ("full", "sliding"):
        return mode_policy
    if mode_policy in ("mixed", "alternate", "alternating"):
        return "full" if idx % 2 == 0 else "sliding"
    raise ValueError(
        "mode_policy must be one of full, sliding, mixed, or random; "
        f"got {mode_policy!r}"
    )


def build_fixed_refiner_samples(
    source,
    *,
    num_samples: int,
    mode_policy: str = "mixed",
    force_no_path_aug: bool = True,
    force_text_idx: int | None = 0,
) -> list[dict[str, Any]]:
    """Pre-sample deterministic Refiner tasks from *source*.

    Anchor frame and num_tokens are intentionally not forced here: they are drawn
    once from the source dataset RNG and then frozen. This preserves realistic
    task diversity while making overfit/validation deterministic.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    if len(source) == 0:
        raise ValueError("source dataset is empty")
    if hasattr(source, "reset_rng"):
        source.reset_rng()

    samples: list[dict[str, Any]] = []
    for out_idx in range(int(num_samples)):
        src_idx = out_idx % len(source)
        if hasattr(source, "get_sample"):
            sample = source.get_sample(
                src_idx,
                force_mode=_mode_for_index(mode_policy, out_idx),
                force_no_path_aug=bool(force_no_path_aug),
                force_text_idx=force_text_idx,
            )
        else:
            sample = source[src_idx]
        samples.append(clone_refiner_sample(sample))
    return samples

# Back-compat alias: this name was exported when the fixed tools lived in
# datasets/refiner_fixed and were re-exported here under this alias.
FixedHumanML3DRefinerDataset = FixedRefinerSampleDataset


__all__ = [
    "RefinerDataset",
    "HumanML3DRefinerDataset",
    "FixedRefinerSampleDataset",
    "FixedHumanML3DRefinerDataset",
    "build_fixed_refiner_samples",
    "clone_refiner_sample",
    "refiner_collate",
    "refiner_worker_init_fn",
]
