"""Root-source proposal boundary for runtime updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from utils.motion_process import build_physical_7d_from_5d
from utils.token_frame import token_start_frame


RootSourceTimelineMode = Literal["absolute_timeline", "anchor_relative"]


@dataclass(frozen=True)
class RootSourceProposal:
    """World-frame route proposal produced by an external root source.

    A root source can be a dataset sample splice, a synthetic route, or a
    RootRefiner output. It owns only the route proposal. Runtime contracts later
    decide how to compose generated history, bridge frames, and future route into
    the actual LDF payload.
    """

    name: str
    proposal_traj7: torch.Tensor
    source_kind: str
    start_frame_abs: int = 0
    start_commit_abs: int = 0
    timeline_mode: RootSourceTimelineMode = "absolute_timeline"
    update_frames: list[int] = field(default_factory=list)
    visual_mask: torch.Tensor | None = None
    base_sample_name: str | None = None
    caption_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.proposal_traj7):
            raise TypeError("proposal_traj7 must be a torch.Tensor")
        if self.proposal_traj7.dim() != 2 or self.proposal_traj7.shape[-1] != 7:
            raise ValueError(
                "proposal_traj7 must be [T,7]; "
                f"got {tuple(self.proposal_traj7.shape)}"
            )
        if int(self.proposal_traj7.shape[0]) <= 0:
            raise ValueError("proposal_traj7 must contain at least one frame")
        if self.timeline_mode not in {"absolute_timeline", "anchor_relative"}:
            raise ValueError(
                "timeline_mode must be 'absolute_timeline' or 'anchor_relative'; "
                f"got {self.timeline_mode!r}"
            )
        if int(self.start_frame_abs) < 0:
            raise ValueError(
                f"start_frame_abs must be >= 0, got {self.start_frame_abs}"
            )
        if int(self.start_commit_abs) < 0:
            raise ValueError(
                f"start_commit_abs must be >= 0, got {self.start_commit_abs}"
            )
        expected_start_frame = token_start_frame(int(self.start_commit_abs))
        if int(self.start_frame_abs) != int(expected_start_frame):
            raise ValueError(
                "start_frame_abs must match start_commit_abs under the shared "
                "causal token/frame mapping; "
                f"got frame={self.start_frame_abs}, commit={self.start_commit_abs}, "
                f"expected_frame={expected_start_frame}"
            )

    @classmethod
    def from_condition_scenario(
        cls,
        scenario,
        *,
        source_kind: str,
    ) -> "RootSourceProposal":
        metadata = dict(getattr(scenario, "metadata", {}) or {})
        metadata.setdefault("runtime_role", "root_source_proposal")
        metadata.setdefault("legacy_condition_scenario_name", str(scenario.name))
        return cls(
            name=str(scenario.name),
            proposal_traj7=scenario.condition_traj7.detach().cpu().float(),
            source_kind=str(source_kind),
            update_frames=[int(frame) for frame in scenario.update_frames],
            visual_mask=scenario.visual_mask,
            base_sample_name=scenario.base_sample_name,
            caption_index=scenario.caption_index,
            metadata=metadata,
        )

    @classmethod
    def from_root_plan(
        cls,
        root_plan,
        *,
        name: str | None = None,
        source_kind: str = "root_refiner",
        update_frames: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "RootSourceProposal":
        """Convert a plan-anchor-local RootPlan into a world-frame proposal."""
        from utils.local_frame import uncanonicalize_7d

        valid_frames = int(root_plan.valid_frames)
        if valid_frames <= 0:
            raise ValueError(
                f"root_plan.valid_frames must be > 0, got {valid_frames}"
            )
        local = root_plan.waypoints_local_7d[:valid_frames].detach().float()
        world = uncanonicalize_7d(
            local.unsqueeze(0),
            root_plan.anchor_world_xz.detach().float().unsqueeze(0),
            root_plan.anchor_world_yaw.detach().float().reshape(1),
        )[0]
        meta = dict(metadata or {})
        meta.setdefault("runtime_role", "root_source_proposal")
        meta.setdefault("root_plan_source", str(getattr(root_plan, "source", "")))
        meta.setdefault("root_plan_anchor_commit_idx", int(root_plan.anchor_commit_idx))
        start_commit_abs = int(root_plan.anchor_commit_idx)
        start_frame_abs = token_start_frame(start_commit_abs)
        return cls(
            name=str(name or getattr(root_plan, "source", "root_plan")),
            proposal_traj7=world.cpu(),
            source_kind=str(source_kind),
            start_frame_abs=start_frame_abs,
            start_commit_abs=start_commit_abs,
            timeline_mode="anchor_relative",
            update_frames=[] if update_frames is None else [int(v) for v in update_frames],
            metadata=meta,
        )

    @property
    def num_frames(self) -> int:
        return int(self.proposal_traj7.shape[0])

    @property
    def end_frame_abs(self) -> int:
        """Last absolute frame represented by this proposal, inclusive."""

        return int(self.start_frame_abs) + self.num_frames - 1

    def absolute_to_local_frame(self, absolute_frame: int) -> int:
        """Convert an absolute frame to a checked proposal-local index."""

        absolute = int(absolute_frame)
        local = absolute - int(self.start_frame_abs)
        if local < 0:
            raise ValueError(
                f"absolute frame {absolute} is before proposal origin "
                f"{self.start_frame_abs}"
            )
        if local >= self.num_frames:
            raise ValueError(
                f"absolute frame {absolute} is outside proposal ending at "
                f"{self.end_frame_abs}"
            )
        return local

    def local_to_absolute_frame(self, local_frame: int) -> int:
        """Convert a checked proposal-local index to the absolute timeline."""

        local = int(local_frame)
        if local < 0 or local >= self.num_frames:
            raise ValueError(
                f"local frame {local} is outside proposal range "
                f"[0, {self.num_frames - 1}]"
            )
        return int(self.start_frame_abs) + local

    def to_absolute_timeline(self) -> torch.Tensor:
        """Materialize the world-frame proposal at its absolute frame origin."""

        route = self.proposal_traj7.detach().clone().float()
        if int(self.start_frame_abs) > 0:
            prefix = route[:1, :5].expand(int(self.start_frame_abs), -1).clone()
            world_5d = torch.cat([prefix, route[:, :5]], dim=0)
        else:
            world_5d = route[:, :5]
        return build_physical_7d_from_5d(world_5d)


__all__ = ["RootSourceProposal", "RootSourceTimelineMode"]
