"""RootRefiner runtime boundary for streaming LDF inference.

This module converts the web-demo contract (text + user route + current root
anchor) into the exact RootRefiner forward inputs, then converts RootRefiner's
normalized 5D output into the physical 7D RootPlan consumed by LDF.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from utils.local_frame import transform_xz_world_to_local
from utils.local_frame import canonicalize_5d
from utils.motion_process import build_physical_7d_from_normalized_5d
from utils.refiner.path_condition import build_path_condition
from utils.root_plan import RootPlan
from utils.token_frame import num_frames_for_tokens


def _state_dict_has_pace_duration(state_dict) -> bool:
    return any(str(key).startswith("refiner.pace_head.") for key in state_dict.keys())


class RootRefinerRuntime:
    def __init__(
        self,
        refiner,
        text_encoder,
        *,
        device,
        wp_mean=None,
        wp_std=None,
        wp_norm_idx=None,
        cm_mean=None,
        cm_std=None,
        cm_norm_idx=None,
        pf_mean=None,
        pf_std=None,
        path_mode: str = "dense_path",
        sparse_point_range=(3, 8),
    ):
        self.refiner = refiner.to(device).eval()
        self.text_encoder = text_encoder.to(device).eval()
        self.device = torch.device(device)
        self.wp_mean = self._to_tensor_or_none(wp_mean)
        self.wp_std = self._to_tensor_or_none(wp_std)
        self.wp_norm_idx = self._to_long_or_none(wp_norm_idx)
        self.cm_mean = self._to_tensor_or_none(cm_mean)
        self.cm_std = self._to_tensor_or_none(cm_std)
        self.cm_norm_idx = self._to_long_or_none(cm_norm_idx)
        self.pf_mean = self._to_tensor_or_none(pf_mean)
        self.pf_std = self._to_tensor_or_none(pf_std)
        self.path_mode = str(path_mode or "dense_path")
        self.sparse_point_range = tuple(int(v) for v in sparse_point_range)

    @classmethod
    def from_config(
        cls,
        *,
        config_path: str,
        ckpt_path: str,
        device,
        strict: bool = True,
        path_mode: str = "dense_path",
    ) -> "RootRefinerRuntime":
        from train_refiner import _load_cfg, resolve_cfg_interpolations
        from utils.refiner.lightning_module import RootRefinerLightningModule

        cfg = resolve_cfg_interpolations(_load_cfg(config_path))
        module = RootRefinerLightningModule(cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        has_pace_duration = _state_dict_has_pace_duration(state_dict)
        try:
            module.load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            if strict:
                raise
            module.load_state_dict(state_dict, strict=False)
        if not has_pace_duration and hasattr(module.refiner, "use_pace_duration"):
            module.refiner.use_pace_duration = False

        data_cfg = cfg.get("data", {}) or {}
        cm_mean = cm_std = cm_norm_idx = None
        if bool(data_cfg.get("normalize", False)):
            stats_dir = Path(data_cfg["stats_dir"])
            cm_mean = torch.as_tensor(
                np.load(stats_dir / "current_motion_mean.npy"), dtype=torch.float32
            )
            cm_std = torch.as_tensor(
                np.load(stats_dir / "current_motion_std.npy"), dtype=torch.float32
            ).clamp(min=1e-6)
            cm_norm_idx = torch.as_tensor(
                np.load(stats_dir / "current_motion_norm_indices.npy"), dtype=torch.long
            )

        pf_mean = pf_std = None
        pf_stats_dir = data_cfg.get("path_feature_stats_dir")
        if pf_stats_dir:
            from utils.refiner.path_feature_stats import (
                compute_sampling_config_hash,
                load_path_feature_stats,
            )

            stats = load_path_feature_stats(
                pf_stats_dir,
                expected_hash=compute_sampling_config_hash(cfg),
            )
            pf_mean, pf_std = stats.mean, stats.std

        return cls(
            module.refiner,
            module.text_encoder,
            device=device,
            wp_mean=getattr(module, "_wp_mean", None),
            wp_std=getattr(module, "_wp_std", None),
            wp_norm_idx=getattr(module, "_wp_norm_idx", None),
            cm_mean=cm_mean,
            cm_std=cm_std,
            cm_norm_idx=cm_norm_idx,
            pf_mean=pf_mean,
            pf_std=pf_std,
            path_mode=path_mode,
            sparse_point_range=(
                (cfg.get("sampling", {}) or {})
                .get("path_condition", {})
                .get("sparse_path", {})
                .get("point_range", (3, 8))
            ),
        )

    @staticmethod
    def _to_tensor_or_none(value):
        if value is None:
            return None
        return torch.as_tensor(value, dtype=torch.float32)

    @staticmethod
    def _to_long_or_none(value):
        if value is None:
            return None
        return torch.as_tensor(value, dtype=torch.long)

    def _zscore_selected(self, tensor, mean, std, norm_idx):
        if mean is None or std is None or norm_idx is None:
            return tensor
        out = tensor.clone()
        mean = mean.to(device=out.device, dtype=out.dtype)
        std = std.to(device=out.device, dtype=out.dtype)
        for c in norm_idx.tolist():
            out[..., c] = (out[..., c] - mean[c]) / std[c]
        return out

    def _normalize_path_xz(self, path_xz):
        if self.wp_mean is None or self.wp_std is None or self.wp_norm_idx is None:
            return path_xz
        out = path_xz.clone()
        idx = set(int(i) for i in self.wp_norm_idx.tolist())
        mean = self.wp_mean.to(device=out.device, dtype=out.dtype)
        std = self.wp_std.to(device=out.device, dtype=out.dtype)
        if 0 in idx:
            out[..., 0] = (out[..., 0] - mean[0]) / std[0]
        if 2 in idx:
            out[..., 1] = (out[..., 1] - mean[2]) / std[2]
        return out

    def _normalize_path_features(self, path_features):
        if self.pf_mean is None or self.pf_std is None:
            return path_features
        return (
            path_features
            - self.pf_mean.to(device=path_features.device, dtype=path_features.dtype)
        ) / self.pf_std.to(device=path_features.device, dtype=path_features.dtype)

    def _default_root_height(self):
        for stats in (self.cm_mean, self.wp_mean):
            if stats is None or int(stats.numel()) <= 1:
                continue
            return stats.to(device=self.device, dtype=torch.float32)[1]
        return None

    def _history_anchor_only(self):
        n_hist = int(self.refiner.n_hist)
        hist = torch.zeros(n_hist, 5, device=self.device, dtype=torch.float32)
        default_y = self._default_root_height()
        if default_y is not None:
            hist[-1, 1] = default_y
        hist[-1, 3] = 1.0
        hist_mask = torch.zeros(n_hist, device=self.device, dtype=torch.bool)
        hist_mask[-1] = True
        hist = self._zscore_selected(hist, self.cm_mean, self.cm_std, self.cm_norm_idx)
        return hist.unsqueeze(0), hist_mask.unsqueeze(0)

    def _history_from_world_5d(self, history_motion_world_5d, anchor_xz, anchor_yaw):
        if history_motion_world_5d is None:
            return self._history_anchor_only()
        hist_world = torch.as_tensor(
            history_motion_world_5d,
            device=self.device,
            dtype=torch.float32,
        )
        if hist_world.ndim != 2 or hist_world.shape[-1] != 5:
            raise ValueError(
                "history_motion_world_5d must be [T,5], got "
                f"{tuple(hist_world.shape)}"
            )
        if hist_world.shape[0] <= 0:
            return self._history_anchor_only()

        n_hist = int(self.refiner.n_hist)
        hist_world = hist_world[-n_hist:]
        hist_local = canonicalize_5d(hist_world, anchor_xz, anchor_yaw)
        valid = int(hist_local.shape[0])
        if valid < n_hist:
            pad = hist_local.new_zeros(n_hist - valid, 5)
            hist_local = torch.cat([pad, hist_local], dim=0)
        hist_mask = torch.zeros(n_hist, device=self.device, dtype=torch.bool)
        hist_mask[n_hist - valid :] = True
        hist_local = self._zscore_selected(
            hist_local,
            self.cm_mean,
            self.cm_std,
            self.cm_norm_idx,
        )
        return hist_local.unsqueeze(0), hist_mask.unsqueeze(0)

    @torch.no_grad()
    def build_root_plan(
        self,
        *,
        text: str,
        plan,
        anchor_state,
        token_dt: float,
        history_motion_world_5d=None,
        forced_num_tokens: int | None = None,
    ) -> RootPlan:
        points = torch.as_tensor(
            plan.points_xyz, device=self.device, dtype=torch.float32
        )
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(
                f"RootRefinerRuntime expects plan.points_xyz [N,3], got {tuple(points.shape)}"
            )

        anchor_xz = anchor_state.world_xz.to(device=self.device, dtype=torch.float32)
        anchor_yaw = anchor_state.world_yaw.to(device=self.device, dtype=torch.float32)
        path_local_xz = transform_xz_world_to_local(points[:, [0, 2]], anchor_xz, anchor_yaw)

        max_frames = int(self.refiner.max_frames)
        condition = build_path_condition(
            path_local_xz.detach().cpu(),
            n_path=int(self.refiner.n_path),
            valid_frame_count=max_frames,
            max_frames=max_frames,
            path_mode=self.path_mode,
            offset_start_frames=0,
            sparse_point_range=self.sparse_point_range,
            rng=random.Random(0),
        )

        path = self._normalize_path_xz(condition.path.to(self.device)).unsqueeze(0)
        path_features = self._normalize_path_features(
            condition.path_features_raw.to(self.device)
        ).unsqueeze(0)
        sample_mode = (
            "sliding"
            if history_motion_world_5d is not None
            and torch.as_tensor(history_motion_world_5d).shape[0] > 1
            else "full"
        )
        history_motion, history_mask = self._history_from_world_5d(
            history_motion_world_5d,
            anchor_xz,
            anchor_yaw,
        )
        text_emb = self.text_encoder.encode([str(text)], device=self.device)

        forced_num_tokens_t = None
        if forced_num_tokens is not None:
            forced_num_tokens_t = torch.as_tensor(
                [int(forced_num_tokens)],
                dtype=torch.long,
                device=self.device,
            )

        out = self.refiner(
            text_emb=text_emb,
            path=path,
            path_valid_mask=condition.path_valid_mask.to(self.device).unsqueeze(0),
            path_control_mask=condition.path_control_mask.to(self.device).unsqueeze(0),
            path_mode=[condition.path_mode],
            path_features=path_features,
            path_features_raw=condition.path_features_raw.to(self.device).unsqueeze(0),
            sample_mode=[sample_mode],
            history_motion=history_motion,
            history_mask=history_mask,
            offset_start_frames=torch.zeros(1, dtype=torch.long, device=self.device),
            num_tokens=forced_num_tokens_t,
        )

        used_tokens = int(out["used_num_tokens"][0].detach().cpu().item())
        frames_per_token = int(self.refiner.frames_per_token)
        valid_frames = min(
            num_frames_for_tokens(used_tokens, frames_per_token),
            int(out["waypoints"].shape[1]),
        )
        wp7 = build_physical_7d_from_normalized_5d(
            out["waypoints"][0],
            self.wp_mean,
            self.wp_std,
            self.wp_norm_idx,
        )
        return RootPlan(
            num_tokens_pred=used_tokens,
            valid_frames=valid_frames,
            waypoints_local_7d=wp7[:valid_frames],
            frame_dt=float(token_dt) / float(frames_per_token),
            frames_per_token=frames_per_token,
            anchor_commit_idx=int(anchor_state.commit_idx),
            anchor_world_xz=anchor_xz,
            anchor_world_yaw=anchor_yaw,
            source="root_refiner_gtnum" if forced_num_tokens is not None else "root_refiner",
        )


__all__ = ["RootRefinerRuntime"]
