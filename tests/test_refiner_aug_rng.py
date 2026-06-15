"""P1-2 (path-aug config wiring) + P1-3 (per-worker augmentation RNG)."""

from __future__ import annotations

import torch

from tests.helpers.humanml3d_fixture import make_root_refiner_from_samples
from utils.training.root_refiner import worker_init_fn


def _clip(T=40):
    m = torch.zeros(T, 263, dtype=torch.float32)
    m[:, 2] = 0.1   # small forward vel
    m[:, 3] = 1.0
    return {"motion_263": m, "text": "walk"}


# ---------------------------------------------------------------------------
# P1-3: per-worker RNG reseed
# ---------------------------------------------------------------------------


def test_set_worker_seed_distinct_per_worker_and_reproducible():
    ds = make_root_refiner_from_samples([_clip()], seed=None)

    torch.manual_seed(111)   # DataLoader sets a distinct torch.initial_seed per worker
    ds.set_worker_seed()
    seq_a = [ds.batch_builder._rng.random() for _ in range(6)]

    torch.manual_seed(222)
    ds.set_worker_seed()
    seq_b = [ds.batch_builder._rng.random() for _ in range(6)]

    torch.manual_seed(111)
    ds.set_worker_seed()
    seq_c = [ds.batch_builder._rng.random() for _ in range(6)]

    assert seq_a != seq_b      # different workers → different aug streams
    assert seq_a == seq_c      # same worker seed → reproducible


def test_base_seed_offsets_worker_seed():
    ds0 = make_root_refiner_from_samples([_clip()], seed=0)
    ds7 = make_root_refiner_from_samples([_clip()], seed=7)
    torch.manual_seed(123)
    ds0.set_worker_seed()
    torch.manual_seed(123)
    ds7.set_worker_seed()
    # same torch.initial_seed but different base seed → different stream
    assert [ds0.batch_builder._rng.random() for _ in range(5)] != [
        ds7.batch_builder._rng.random() for _ in range(5)
    ]


def test_worker_init_fn_no_crash_outside_worker():
    # get_worker_info() is None in the main process → no-op, must not raise.
    worker_init_fn(0)
