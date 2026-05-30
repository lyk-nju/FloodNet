from __future__ import annotations

from datasets import humanml3d_refiner
from datasets import refiner_dataset


def test_legacy_refiner_dataset_module_reexports_humanml3d_refiner_dataset():
    assert refiner_dataset.RefinerDataset is humanml3d_refiner.HumanML3DRefinerDataset
    assert refiner_dataset.refiner_collate is humanml3d_refiner.refiner_collate
    assert refiner_dataset.refiner_worker_init_fn is humanml3d_refiner.refiner_worker_init_fn
