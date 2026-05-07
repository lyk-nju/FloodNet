import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_info
from torch.utils.data import ConcatDataset, Dataset

from utils.initialize import instantiate

_PAD_SEQUENCE_KEYS = {"feature", "token", "traj", "traj_features"}
_PAD_MASK_KEYS = {"traj_mask", "token_mask"}
_STACK_SCALAR_KEYS = {"feature_length", "token_length", "traj_length"}


class MultiDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.datasets = []
        self.cfg = cfg
        self.split = split

        if not hasattr(cfg.data, "datasets"):
            rank_zero_info(
                "MultiDataset: cfg.data.datasets not found. No datasets initialized."
            )
            return

        rank_zero_info(
            f"Initializing MultiDataset for split {split} with "
            f"{len(cfg.data.datasets)} sub-datasets..."
        )

        for ds_conf in cfg.data.datasets:
            ds_cfg = cfg.copy()
            ds_cfg.data = ds_conf

            target = ds_cfg.data.get("target", None)
            if target is None:
                raise ValueError(
                    f"No target specified for dataset in MultiDataset: {ds_conf}"
                )

            rank_zero_info(f"  - Initializing sub-dataset: {target}")
            dataset = instantiate(target, cfg=ds_cfg, split=split)
            self.datasets.append(dataset)

        self.concat_dataset = ConcatDataset(self.datasets)
        rank_zero_info(f"MultiDataset loaded. Total samples: {len(self.concat_dataset)}")

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, idx):
        return self.concat_dataset[idx]


def _ordered_union_keys(batch):
    keys = []
    seen = set()
    for sample in batch:
        for key in sample.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)
    return keys


def _to_tensor(value, dtype=None):
    if torch.is_tensor(value):
        return value if dtype is None else value.to(dtype=dtype)
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        tensor = torch.tensor(value)
    return tensor if dtype is None else tensor.to(dtype=dtype)


def _collate_padded_sequence(batch, key):
    present = [sample[key] for sample in batch if key in sample and sample[key] is not None]
    if not present:
        return None
    first = _to_tensor(present[0])
    trailing_shape = tuple(first.shape[1:])
    dtype = first.dtype
    items = []
    for sample in batch:
        if key in sample and sample[key] is not None:
            item = _to_tensor(sample[key], dtype=dtype)
        else:
            item = torch.zeros((0,) + trailing_shape, dtype=dtype)
        items.append(item)
    return torch.nn.utils.rnn.pad_sequence(items, batch_first=True, padding_value=0)


def _collate_padded_mask(batch, key):
    items = []
    for sample in batch:
        if key in sample and sample[key] is not None:
            item = _to_tensor(sample[key], dtype=torch.float32)
        else:
            item = torch.zeros((0,), dtype=torch.float32)
        items.append(item)
    return torch.nn.utils.rnn.pad_sequence(items, batch_first=True, padding_value=0)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    output = {}
    keys = _ordered_union_keys(batch)

    for key in keys:
        if key in _PAD_SEQUENCE_KEYS:
            output[key] = _collate_padded_sequence(batch, key)
        elif key in _PAD_MASK_KEYS:
            output[key] = _collate_padded_mask(batch, key)
        elif key in _STACK_SCALAR_KEYS:
            output[key] = torch.tensor([int(sample.get(key, 0)) for sample in batch])
        else:
            output[key] = [sample.get(key, None) for sample in batch]
    return output
