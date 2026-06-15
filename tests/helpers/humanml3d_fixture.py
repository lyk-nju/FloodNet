from __future__ import annotations

import numpy as np
import torch
import tempfile

from pathlib import Path
from omegaconf import OmegaConf
from datasets.humanml3d import HumanML3DDataset
from utils.training.root_refiner import RootRefinerDataset


def make_motion263(T: int = 80, *, vx: float = 0.0, vz: float = 0.1) -> np.ndarray:
    motion = np.zeros((T, 263), dtype=np.float32)
    motion[:, 1] = vx
    motion[:, 2] = vz
    motion[:, 3] = 1.0
    return motion


def write_humanml3d_fixture(
    root,
    samples: list[dict],
    *,
    split_file: str = "train.txt",
    feature_path: str = "new_joint_vecs",
    text_path: str = "texts",
):
    dataset_root = root / "HumanML3D"
    feature_dir = dataset_root / feature_path
    text_dir = dataset_root / text_path
    feature_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    names = []
    for index, sample in enumerate(samples):
        name = sample.get("name", f"s{index:06d}")
        names.append(name)
        motion = sample.get("feature", sample.get("motion_263"))
        if motion is None:
            motion = make_motion263()
        if torch.is_tensor(motion):
            motion = motion.detach().cpu().numpy()
        np.save(feature_dir / f"{name}.npy", np.asarray(motion, dtype=np.float32))

        captions = sample.get("texts")
        if captions is None:
            captions = [sample.get("text", "walk forward")]
        lines = [f"{caption}#x#0#0" for caption in captions]
        (text_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")

    (dataset_root / split_file).write_text("\n".join(names) + "\n")
    return dataset_root


def humanml3d_cfg(
    root,
    *,
    split_file: str = "train.txt",
    feature_path: str = "new_joint_vecs",
    text_path: str = "texts",
    min_length: int = 1,
    max_length: int = 10 ** 9,
    window_length: int | None = None,
):
    split_path = root / "HumanML3D" / split_file
    return OmegaConf.create(
        {
            "debug": False,
            "data": {
                "train_meta_paths": [str(split_path)],
                "val_meta_paths": [str(split_path)],
                "test_meta_paths": [str(split_path)],
                "feature_path": feature_path,
                "token_path": None,
                "text_path": text_path,
                "min_length": min_length,
                "max_length": max_length,
                "window_length": window_length or max_length,
                "random_length": 0,
                "traj_feat_dim": 4,
            },
        }
    )


def make_humanml3d_dataset(root, *, split: str = "train", **cfg_kwargs):
    return HumanML3DDataset(humanml3d_cfg(root, **cfg_kwargs), split=split)


def make_root_refiner_dataset(
    root,
    samples: list[dict],
    *,
    split: str = "train",
    split_file: str = "train.txt",
    feature_path: str = "new_joint_vecs",
    text_path: str = "texts",
    min_length: int = 1,
    max_length: int = 10 ** 9,
    window_length: int | None = None,
    **refiner_kwargs,
):
    write_humanml3d_fixture(
        root,
        samples,
        split_file=split_file,
        feature_path=feature_path,
        text_path=text_path,
    )
    raw_dataset = make_humanml3d_dataset(
        root,
        split=split,
        split_file=split_file,
        feature_path=feature_path,
        text_path=text_path,
        min_length=min_length,
        max_length=max_length,
        window_length=window_length,
    )
    return RootRefinerDataset(raw_dataset, **refiner_kwargs)


def make_root_refiner_from_samples(samples: list[dict], **refiner_kwargs):
    tmpdir = tempfile.TemporaryDirectory()
    dataset = make_root_refiner_dataset(
        Path(tmpdir.name),
        samples,
        split="val",
        split_file="val.txt",
        **refiner_kwargs,
    )
    dataset._fixture_tmpdir = tmpdir
    return dataset
