"""Legacy import shim for RootRefiner datasets.

Use `datasets.humanml3d_refiner` for new code. This module remains so older
tests, scripts, and checkpoints that import `datasets.refiner_dataset` continue
to resolve the same public names.
"""

from __future__ import annotations

from datasets.humanml3d_refiner import (
    HumanML3DRefinerDataset as RefinerDataset,
    refiner_collate,
    refiner_worker_init_fn,
)

__all__ = ["RefinerDataset", "refiner_collate", "refiner_worker_init_fn"]
