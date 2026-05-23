"""Unit tests for the real-layout clip loader (scripts/compute_5d_stats.load_clips_from_dir).

Builds fake HumanML3D / BABEL_streamed layouts under tmp_path and verifies the
loader reads split files, motion features, and '#'-delimited captions, and that
raw_data_dir works both as the root and as the dataset dir.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.compute_5d_stats import (
    DATASET_DEFAULTS,
    load_clips_from_dir,
    resolve_dataset_dir,
)


def _write_split(dataset_dir, split_file, names):
    (dataset_dir / split_file).write_text("\n".join(names) + "\n")


def _write_motion(dataset_dir, feature_path, name, T=40, dim=263):
    d = dataset_dir / feature_path
    d.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((T, dim), dtype=np.float32)
    arr[:, 2] = 0.05   # small +Z velocity so clip is non-degenerate
    arr[:, 3] = 1.0
    np.save(d / f"{name}.npy", arr)


def _write_text(dataset_dir, text_path, name, caption, *, hml_format=True):
    d = dataset_dir / text_path
    d.mkdir(parents=True, exist_ok=True)
    if hml_format:
        # HumanML3D: caption#tokens#f_tag#to_tag
        line = f"{caption}#a/DET the/DET#0.0#0.0\n"
    else:
        line = f"{caption}\n"
    (d / f"{name}.txt").write_text(line)


def _make_humanml3d(root, names, *, split="train.txt"):
    ds = root / "HumanML3D"
    ds.mkdir(parents=True, exist_ok=True)
    _write_split(ds, split, names)
    for n in names:
        _write_motion(ds, "new_joint_vecs", n)
        _write_text(ds, "texts", n, f"a person does {n}", hml_format=True)
    return ds


def _make_babel(root, names, *, split="train_processed.txt"):
    ds = root / "BABEL_streamed"
    ds.mkdir(parents=True, exist_ok=True)
    _write_split(ds, split, names)
    for n in names:
        _write_motion(ds, "motions", n)
        _write_text(ds, "texts", n, f"babel {n}", hml_format=True)
    return ds


# ---------------------------------------------------------------------------
# resolve_dataset_dir
# ---------------------------------------------------------------------------


def test_resolve_dataset_dir_from_root(tmp_path):
    _make_humanml3d(tmp_path, ["s1"])
    resolved = resolve_dataset_dir(tmp_path, "humanml3d")
    assert resolved == tmp_path / "HumanML3D"


def test_resolve_dataset_dir_from_dataset_dir_itself(tmp_path):
    ds = _make_humanml3d(tmp_path, ["s1"])
    # Pass the dataset dir directly (no HumanML3D child) → returns it unchanged.
    resolved = resolve_dataset_dir(ds, "humanml3d")
    assert resolved == ds


def test_resolve_dataset_dir_unknown_dataset_raises(tmp_path):
    with pytest.raises(ValueError):
        resolve_dataset_dir(tmp_path, "nonexistent")


# ---------------------------------------------------------------------------
# HumanML3D loading
# ---------------------------------------------------------------------------


def test_humanml3d_loads_from_root(tmp_path):
    _make_humanml3d(tmp_path, ["s1", "s2", "s3"])
    clips = load_clips_from_dir(tmp_path, dataset="humanml3d", split_file="train.txt")
    assert len(clips) == 3
    assert clips[0]["motion_263"].shape == (40, 263)
    assert clips[0]["text"] == "a person does s1"   # caption before '#'


def test_humanml3d_loads_from_dataset_dir_directly(tmp_path):
    ds = _make_humanml3d(tmp_path, ["s1", "s2"])
    clips = load_clips_from_dir(ds, dataset="humanml3d", split_file="train.txt")
    assert len(clips) == 2


def test_humanml3d_caption_strips_hashes(tmp_path):
    ds = tmp_path / "HumanML3D"
    ds.mkdir(parents=True)
    _write_split(ds, "train.txt", ["s1"])
    _write_motion(ds, "new_joint_vecs", "s1")
    # multi-line text file; first non-empty line, part before '#'
    (ds / "texts").mkdir()
    (ds / "texts" / "s1.txt").write_text(
        "\n"   # leading empty line
        "a man walks forward#a/DET man/NOUN#0.0#0.0\n"
        "second caption#x#0#0\n"
    )
    clips = load_clips_from_dir(ds, dataset="humanml3d", split_file="train.txt")
    assert clips[0]["text"] == "a man walks forward"


def test_plain_caption_without_hash(tmp_path):
    ds = tmp_path / "HumanML3D"
    ds.mkdir(parents=True)
    _write_split(ds, "train.txt", ["s1"])
    _write_motion(ds, "new_joint_vecs", "s1")
    _write_text(ds, "texts", "s1", "plain caption no hash", hml_format=False)
    clips = load_clips_from_dir(ds, dataset="humanml3d", split_file="train.txt")
    assert clips[0]["text"] == "plain caption no hash"


# ---------------------------------------------------------------------------
# BABEL loading (feature dir = motions, NOT new_joint_vecs)
# ---------------------------------------------------------------------------


def test_babel_loads_from_root_with_motions_dir(tmp_path):
    _make_babel(tmp_path, ["b1", "b2"])
    clips = load_clips_from_dir(tmp_path, dataset="babel",
                                 split_file="train_processed.txt")
    assert len(clips) == 2
    assert clips[0]["motion_263"].shape == (40, 263)
    assert clips[0]["text"] == "babel b1"


def test_babel_defaults_use_motions_feature_path():
    # Sanity on the defaults table (guards against mixing HumanML3D's
    # new_joint_vecs into BABEL).
    assert DATASET_DEFAULTS["babel"]["feature_path"] == "motions"
    assert DATASET_DEFAULTS["humanml3d"]["feature_path"] == "new_joint_vecs"
    assert DATASET_DEFAULTS["babel"]["split_file"] == "train_processed.txt"
    assert DATASET_DEFAULTS["humanml3d"]["split_file"] == "train.txt"


# ---------------------------------------------------------------------------
# Missing / malformed handling
# ---------------------------------------------------------------------------


def test_missing_motion_file_is_skipped(tmp_path):
    ds = _make_humanml3d(tmp_path, ["s1", "s2", "s3"])
    # Delete s2's motion file.
    (ds / "new_joint_vecs" / "s2.npy").unlink()
    clips = load_clips_from_dir(ds, dataset="humanml3d", split_file="train.txt")
    # s2 skipped; s1 and s3 remain.
    assert len(clips) == 2


def test_missing_text_file_yields_empty_caption(tmp_path):
    ds = _make_humanml3d(tmp_path, ["s1"])
    (ds / "texts" / "s1.txt").unlink()
    clips = load_clips_from_dir(ds, dataset="humanml3d", split_file="train.txt")
    assert len(clips) == 1
    assert clips[0]["text"] == ""


def test_bad_motion_shape_is_skipped(tmp_path):
    ds = _make_humanml3d(tmp_path, ["s1", "s2"])
    # Overwrite s1 with a wrong-width motion.
    np.save(ds / "new_joint_vecs" / "s1.npy", np.zeros((40, 100), dtype=np.float32))
    clips = load_clips_from_dir(ds, dataset="humanml3d", split_file="train.txt")
    assert len(clips) == 1   # only s2 (valid 263-d) survives


def test_missing_split_file_raises(tmp_path):
    (tmp_path / "HumanML3D").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        load_clips_from_dir(tmp_path, dataset="humanml3d", split_file="train.txt")


def test_max_samples_stops_early(tmp_path):
    _make_humanml3d(tmp_path, [f"s{i}" for i in range(10)])
    clips = load_clips_from_dir(tmp_path, dataset="humanml3d",
                                 split_file="train.txt", max_samples=3)
    assert len(clips) == 3


def test_unknown_dataset_raises(tmp_path):
    with pytest.raises(ValueError):
        load_clips_from_dir(tmp_path, dataset="bogus")


def test_custom_feature_and_text_path_override(tmp_path):
    ds = tmp_path / "HumanML3D"
    ds.mkdir(parents=True)
    _write_split(ds, "train.txt", ["s1"])
    _write_motion(ds, "custom_feats", "s1")
    _write_text(ds, "custom_texts", "s1", "custom dirs work")
    clips = load_clips_from_dir(ds, dataset="humanml3d", split_file="train.txt",
                                 feature_path="custom_feats", text_path="custom_texts")
    assert len(clips) == 1
    assert clips[0]["text"] == "custom dirs work"
