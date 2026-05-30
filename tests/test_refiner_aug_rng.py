"""P1-2 (path-aug config wiring) + P1-3 (per-worker augmentation RNG)."""

from __future__ import annotations

import torch

from datasets.humanml3d_refiner import HumanML3DRefinerDataset as RefinerDataset, refiner_worker_init_fn
from train_refiner import path_aug_kwargs


def _clip(T=40):
    m = torch.zeros(T, 263, dtype=torch.float32)
    m[:, 2] = 0.1   # small forward vel
    m[:, 3] = 1.0
    return {"motion_263": m, "text": "walk"}


# ---------------------------------------------------------------------------
# P1-2: cfg.path_aug → RefinerDataset kwargs
# ---------------------------------------------------------------------------


def test_path_aug_kwargs_maps_all_keys():
    cfg = {"path_aug": {"trim_prob": 0.4, "trim_max_frames": 7,
                        "sparse_prob": 0.6, "sparse_range": [2, 9]}}
    kw = path_aug_kwargs(cfg)
    assert kw == {
        "path_trim_prob": 0.4, "path_trim_max_frames": 7,
        "path_sparse_prob": 0.6, "path_sparse_range": (2, 9),
    }


def test_path_aug_kwargs_absent_returns_empty():
    assert path_aug_kwargs({}) == {}


def test_path_aug_kwargs_flow_into_dataset():
    cfg = {"path_aug": {"trim_prob": 0.4, "sparse_range": [2, 9]}}
    ds = RefinerDataset([_clip()], seed=0, **path_aug_kwargs(cfg))
    assert ds.path_trim_prob == 0.4
    assert ds.path_sparse_range == (2, 9)
    assert ds.path_trim_max_frames == 10   # untouched → dataset default


# ---------------------------------------------------------------------------
# P1-3: per-worker RNG reseed
# ---------------------------------------------------------------------------


def test_set_worker_seed_distinct_per_worker_and_reproducible():
    ds = RefinerDataset([_clip()], seed=None)

    torch.manual_seed(111)   # DataLoader sets a distinct torch.initial_seed per worker
    ds.set_worker_seed()
    seq_a = [ds._rng.random() for _ in range(6)]

    torch.manual_seed(222)
    ds.set_worker_seed()
    seq_b = [ds._rng.random() for _ in range(6)]

    torch.manual_seed(111)
    ds.set_worker_seed()
    seq_c = [ds._rng.random() for _ in range(6)]

    assert seq_a != seq_b      # different workers → different aug streams
    assert seq_a == seq_c      # same worker seed → reproducible


def test_base_seed_offsets_worker_seed():
    ds0 = RefinerDataset([_clip()], seed=0)
    ds7 = RefinerDataset([_clip()], seed=7)
    torch.manual_seed(123)
    ds0.set_worker_seed()
    torch.manual_seed(123)
    ds7.set_worker_seed()
    # same torch.initial_seed but different base seed → different stream
    assert [ds0._rng.random() for _ in range(5)] != [ds7._rng.random() for _ in range(5)]


def test_worker_init_fn_no_crash_outside_worker():
    # get_worker_info() is None in the main process → no-op, must not raise.
    refiner_worker_init_fn(0)
