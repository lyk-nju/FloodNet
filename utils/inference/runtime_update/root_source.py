"""Root-source proposal boundary for runtime updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


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
        return cls(
            name=str(name or getattr(root_plan, "source", "root_plan")),
            proposal_traj7=world.cpu(),
            source_kind=str(source_kind),
            update_frames=[] if update_frames is None else [int(v) for v in update_frames],
            metadata=meta,
        )

    @property
    def num_frames(self) -> int:
        return int(self.proposal_traj7.shape[0])


__all__ = ["RootSourceProposal"]
