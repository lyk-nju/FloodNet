from __future__ import annotations

import torch

from datasets.refiner_dataset import RefinerDataset
from datasets.refiner_fixed import (
    FixedRefinerSampleDataset,
    build_fixed_refiner_samples,
)


def _clip(T=80, *, text="walk forward"):
    motion = torch.zeros(T, 263, dtype=torch.float32)
    motion[:, 2] = 0.1
    motion[:, 3] = 1.0
    return {"motion_263": motion, "text": text}


def _assert_sample_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        av, bv = a[key], b[key]
        if torch.is_tensor(av):
            assert torch.equal(av, bv), key
        else:
            assert av == bv, key


def test_fixed_refiner_samples_are_repeatable_even_when_source_is_random():
    source = RefinerDataset(
        [_clip(T=90), _clip(T=100, text="turn left")],
        full_plan_ratio=0.5,
        path_trim_prob=0.5,
        path_sparse_prob=0.5,
        seed=123,
    )
    fixed = FixedRefinerSampleDataset(
        build_fixed_refiner_samples(
            source,
            num_samples=4,
            mode_policy="mixed",
            force_no_path_aug=True,
            force_text_idx=0,
        )
    )

    first = fixed[0]
    second = fixed[0]
    _assert_sample_equal(first, second)


def test_fixed_refiner_dataset_returns_clones_not_cached_tensors():
    sample = RefinerDataset([_clip()], full_plan_ratio=1.0, seed=0).get_sample(
        0, force_mode="full", force_no_path_aug=True, force_text_idx=0
    )
    fixed = FixedRefinerSampleDataset([sample])

    a = fixed[0]
    a["xz_path"].add_(100.0)
    b = fixed[0]

    assert not torch.equal(a["xz_path"], b["xz_path"])
    assert torch.equal(b["xz_path"], sample["xz_path"])
