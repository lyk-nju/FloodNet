"""Exact model payload construction from composed world conditions."""

from __future__ import annotations

import torch

from utils.local_frame import canonicalize_5d
from utils.motion_process import build_physical_7d_from_5d
from utils.token_frame import token_range_to_frame_slice

from .contracts import ComposeResult


class PayloadBuilder:
    """Slice and canonicalize :class:`ComposeResult` for model execution."""

    def __init__(self, *, frames_per_token: int = 4) -> None:
        frames_per_token = int(frames_per_token)
        if frames_per_token <= 0:
            raise ValueError("frames_per_token must be > 0")
        self.frames_per_token = frames_per_token

    def _build_single(
        self,
        composed: ComposeResult,
        timeline,
        *,
        local_start_token: int,
        absolute_start_token: int,
        absolute_final_right_token: int,
        horizon_tokens: int,
    ) -> dict | None:
        num_tokens = max(
            0,
            int(absolute_final_right_token)
            + int(horizon_tokens)
            - int(absolute_start_token),
        )
        if num_tokens <= 0 or not timeline.has_exact_state(int(absolute_start_token)):
            return None

        body_anchor_state = timeline.at_commit(int(absolute_start_token))
        frame_slice = token_range_to_frame_slice(
            int(absolute_start_token),
            int(num_tokens),
            self.frames_per_token,
        )
        frame_count = int(frame_slice.stop - frame_slice.start)
        world = composed.world_condition_7d.to(
            device=body_anchor_state.world_xz.device,
            dtype=body_anchor_state.world_xz.dtype,
        )
        source_mask = composed.frame_mask.to(device=world.device)
        if frame_count <= 0 or int(world.shape[0]) == 0:
            return None

        absolute_frames = torch.arange(
            int(frame_slice.start),
            int(frame_slice.stop),
            device=world.device,
        )
        source_indices = absolute_frames - int(composed.frame_start_abs)
        in_condition = (source_indices >= 0) & (source_indices < int(world.shape[0]))

        # Geometry holds at either endpoint outside the composed window, while
        # validity remains governed exclusively by ComposeResult.frame_mask.
        clamped_indices = source_indices.clamp(0, int(world.shape[0]) - 1)
        world_5d = world[clamped_indices, :5].clone()
        frame_mask = torch.zeros(frame_count, dtype=torch.bool, device=world.device)
        if bool(in_condition.any()):
            frame_mask[in_condition] = source_mask[source_indices[in_condition]]

        local_5d = canonicalize_5d(
            world_5d.unsqueeze(0),
            body_anchor_state.world_xz.detach().to(
                device=world.device,
                dtype=world.dtype,
            ).unsqueeze(0),
            body_anchor_state.world_yaw.detach().to(
                device=world.device,
                dtype=world.dtype,
            ).reshape(1),
        )[0]
        # Canonicalization and endpoint holds change frame-to-frame geometry;
        # regenerate deltas only after those operations are complete.
        local_7d = build_physical_7d_from_5d(local_5d)
        return {
            "traj_cond_7d_frame": local_7d.unsqueeze(0),
            "traj_cond_frame_mask": frame_mask.unsqueeze(0).to(dtype=torch.float32),
            "traj_start_token": int(local_start_token),
            "traj_abs_start_token": int(absolute_start_token),
            "traj_num_tokens": int(num_tokens),
            "body_anchor_token": int(local_start_token),
            "body_anchor_abs_token": int(absolute_start_token),
            "debug_world_frame_start_abs": int(composed.frame_start_abs),
        }

    def build(
        self,
        composed: ComposeResult,
        timeline,
        local_commit_before: int,
        absolute_commit_before: int,
        chunk_size: int,
        history_tokens: int,
        horizon_tokens: int,
    ) -> dict | None:
        """Construct the active payload and all unique history substeps."""
        if not isinstance(composed, ComposeResult):
            raise TypeError("composed must be ComposeResult")

        local_commit = int(local_commit_before)
        absolute_commit = int(absolute_commit_before)
        chunk = int(chunk_size)
        history = int(history_tokens)
        horizon = int(horizon_tokens)
        if chunk < 0 or history < 0 or horizon < 0:
            raise ValueError("chunk_size, history_tokens, and horizon_tokens must be >= 0")

        local_earliest_right_token = local_commit + 1
        absolute_earliest_right_token = absolute_commit + 1
        local_final_right_token = local_commit + chunk
        absolute_final_right_token = absolute_commit + chunk
        local_start_token = max(
            0,
            local_earliest_right_token
            - min(local_earliest_right_token, history),
        )
        absolute_start_token = max(
            0,
            absolute_earliest_right_token
            - min(absolute_earliest_right_token, history),
        )
        if local_final_right_token <= local_start_token:
            return None

        subpayloads = []
        seen_local_starts: set[int] = set()
        for local_right_token in range(
            local_earliest_right_token,
            local_final_right_token + 1,
        ):
            model_history = min(local_right_token, history)
            sub_local_start = max(0, local_right_token - model_history)
            if sub_local_start in seen_local_starts:
                continue
            seen_local_starts.add(sub_local_start)
            absolute_right_token = absolute_commit + (local_right_token - local_commit)
            sub_absolute_start = max(0, absolute_right_token - model_history)
            subpayload = self._build_single(
                composed,
                timeline,
                local_start_token=sub_local_start,
                absolute_start_token=sub_absolute_start,
                absolute_final_right_token=absolute_final_right_token,
                horizon_tokens=horizon,
            )
            if subpayload is None:
                return None
            subpayloads.append(subpayload)

        if not subpayloads:
            return None
        payload = dict(subpayloads[0])
        payload["traj_substep_payloads"] = subpayloads
        return payload


__all__ = ["PayloadBuilder"]
