"""RootRefiner runtime boundary for streaming LDF inference.

This module converts the web-demo contract (text + user route + current root
anchor) into the exact RootRefiner forward inputs, then converts RootRefiner's
5D output into the physical 7D RootPlan consumed by LDF.
"""

from __future__ import annotations

import random
import torch

from utils.local_frame import transform_xz_world_to_local
from utils.local_frame import canonicalize_5d
from utils.motion_process import build_physical_7d_from_5d
from utils.training.root_refiner.path_condition import build_path_condition
from utils.inference.root_plan import RootPlan
from utils.token_frame import num_frames_for_tokens, num_tokens_for_frame_len


def _state_dict_has_pace_duration(state_dict) -> bool:
    return any(str(key).startswith("refiner.pace_head.") for key in state_dict.keys())


class RootRefinerRuntime:
    def __init__(
        self,
        refiner,
        text_encoder,
        *,
        device,
        path_mode: str = "dense_path",
        sparse_point_range=(3, 8),
        default_anchor_y: float = 1.0,
    ):
        self.refiner = refiner.to(device).eval()
        self.text_encoder = text_encoder.to(device).eval()
        self.device = torch.device(device)
        self.path_mode = str(path_mode or "dense_path")
        self.sparse_point_range = tuple(int(v) for v in sparse_point_range)
        self.default_anchor_y = float(default_anchor_y)

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
        from utils.training.root_refiner.lightning_module import RootRefinerLightningModule

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

        return cls(
            module.refiner,
            module.text_encoder,
            device=device,
            path_mode=path_mode,
            sparse_point_range=(
                (cfg.get("sampling", {}) or {})
                .get("path_condition", {})
                .get("sparse_path", {})
                .get("point_range", (3, 8))
            ),
        )

    def _resolve_anchor_y(
        self,
        *,
        anchor_state,
        history_motion_world_5d=None,
        anchor_world_y=None,
    ) -> torch.Tensor:
        value = anchor_world_y
        if value is None:
            value = getattr(anchor_state, "world_y", None)
        if value is None and history_motion_world_5d is not None:
            hist = torch.as_tensor(
                history_motion_world_5d,
                device=self.device,
                dtype=torch.float32,
            )
            if hist.ndim == 2 and hist.shape[0] > 0 and hist.shape[-1] == 5:
                value = hist[-1, 1]
        if value is None:
            value = self.default_anchor_y
        return torch.as_tensor(value, device=self.device, dtype=torch.float32).reshape(())

    def _history_anchor_only(self, anchor_y: torch.Tensor):
        n_hist = int(self.refiner.n_hist)
        hist = torch.zeros(n_hist, 5, device=self.device, dtype=torch.float32)
        hist[-1, 1] = anchor_y
        hist[-1, 3] = 1.0
        hist_mask = torch.zeros(n_hist, device=self.device, dtype=torch.bool)
        hist_mask[-1] = True
        return hist.unsqueeze(0), hist_mask.unsqueeze(0)

    def _history_from_world_5d(
        self,
        history_motion_world_5d,
        anchor_xz,
        anchor_yaw,
        anchor_y,
    ):
        if history_motion_world_5d is None:
            return self._history_anchor_only(anchor_y)
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
            return self._history_anchor_only(anchor_y)

        n_hist = int(self.refiner.n_hist)
        hist_world = hist_world[-n_hist:]
        hist_local = canonicalize_5d(hist_world, anchor_xz, anchor_yaw)
        valid = int(hist_local.shape[0])
        if valid < n_hist:
            pad = hist_local.new_zeros(n_hist - valid, 5)
            hist_local = torch.cat([pad, hist_local], dim=0)
        hist_mask = torch.zeros(n_hist, device=self.device, dtype=torch.bool)
        hist_mask[n_hist - valid :] = True
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
        anchor_world_y=None,
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
        anchor_y = self._resolve_anchor_y(
            anchor_state=anchor_state,
            history_motion_world_5d=history_motion_world_5d,
            anchor_world_y=anchor_world_y,
        )
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

        path = condition.path.to(self.device).unsqueeze(0)
        path_features = condition.path_features_raw.to(self.device).unsqueeze(0)
        history_motion, history_mask = self._history_from_world_5d(
            history_motion_world_5d,
            anchor_xz,
            anchor_yaw,
            anchor_y,
        )
        text_emb = self.text_encoder.encode([str(text)], device=self.device)

        forced_num_frames_t = None
        if forced_num_tokens is not None:
            forced_valid_frames = num_frames_for_tokens(int(forced_num_tokens), 4)
            forced_num_frames_t = torch.as_tensor(
                [max(1, forced_valid_frames - 1)],
                dtype=torch.long,
                device=self.device,
            )

        out = self.refiner(
            text_emb=text_emb,
            path=path,
            path_valid_mask=condition.path_valid_mask.to(self.device).unsqueeze(0),
            path_control_mask=condition.path_control_mask.to(self.device).unsqueeze(0),
            path_features=path_features,
            path_features_raw=condition.path_features_raw.to(self.device).unsqueeze(0),
            history_motion=history_motion,
            history_mask=history_mask,
            anchor_frame=torch.zeros(1, dtype=torch.long, device=self.device),
            num_frames=forced_num_frames_t,
        )

        used_future_frames = int(out["used_frames"][0].detach().cpu().item())
        valid_frames = min(used_future_frames + 1, int(out["waypoints"].shape[1]) + 1)
        frames_per_token = 4
        used_tokens = num_tokens_for_frame_len(valid_frames, frames_per_token)
        wp7 = build_physical_7d_from_5d(out["waypoints"][0])
        anchor7 = wp7.new_zeros(1, 7)
        anchor7[:, 1] = anchor_y.to(device=wp7.device, dtype=wp7.dtype)
        anchor7[:, 3] = 1.0
        wp7 = torch.cat([anchor7, wp7], dim=0)
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
