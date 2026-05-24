"""Schema validation for configs/root_refiner.yaml (T_A_07).

Locks the required keys + default values listed in docs/TODO.md §T_A_07.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_CFG_DIR = Path(__file__).resolve().parent.parent / "configs"
CFG_PATH = _CFG_DIR / "root_refiner.yaml"
TRAIN_CFG_PATH = _CFG_DIR / "root_refiner_train.yaml"


def _load(path=CFG_PATH):
    with path.open() as f:
        return yaml.safe_load(f)


def test_yaml_parses():
    cfg = _load()
    assert isinstance(cfg, dict)


# ---------------------------------------------------------------------------
# P2-2: lock the debug vs real config semantics
# ---------------------------------------------------------------------------


def test_debug_config_uses_stub():
    cfg = _load(CFG_PATH)
    assert cfg["text_encoder"]["debug_stub"] is True
    assert cfg["data"]["normalize"] is False


def test_train_config_is_real_precomputed_t5():
    cfg = _load(TRAIN_CFG_PATH)
    te = cfg["text_encoder"]
    assert te["debug_stub"] is False
    assert te["type"] == "precomputed_t5_pool"
    assert "precomputed_text_emb_path" in te
    assert te.get("pooling") in ("mean", "first")
    assert cfg["model"]["text_emb_dim"] == 4096   # = T5 cache text_dim (P0-1)
    assert cfg["data"]["normalize"] is True        # real training z-scores (P2-1)


def test_model_block_required_keys_and_values():
    cfg = _load()
    model = cfg["model"]
    expected = {
        "d_model": 256,
        "n_layers": 6,
        "n_heads": 8,
        "ff_dim": 1024,
        "dropout": 0.1,
        "max_tokens": 49,
        "min_tokens": 4,
        "frames_per_token": 4,
        "n_path": 64,
        "n_hist": 20,
        "text_emb_dim": 512,
    }
    for k, v in expected.items():
        assert k in model, f"missing model.{k}"
        assert model[k] == v, f"model.{k} = {model[k]}, expected {v}"


def test_canonicalization_block():
    cfg = _load()
    canon = cfg["canonicalization"]
    assert canon["mode"] == "b_full"
    assert canon["anchor"] == "first_effective_frame"
    # ⚠ scalar, NOT a range — see TODO §T_A_07 hard constraint
    assert canon["full_plan_valid_history_frames"] == 1
    assert "min" not in str(canon.get("full_plan_valid_history_frames", ""))


def test_training_block():
    cfg = _load()
    tr = cfg["training"]
    assert tr["batch_size"] == 64
    assert tr["lr"] == 1.0e-4
    assert tr["optimizer"] == "adamw"
    assert tr["weight_decay"] == 0.01
    assert tr["total_steps"] == 100000
    assert tr["sampling_mode_full_ratio"] == 0.5
    assert tr["gradient_clip_val"] == 1.0


def test_loss_and_loss_weights():
    cfg = _load()
    assert cfg["loss"]["heading_form"] == "cosine"
    weights = cfg["loss_weights"]
    expected_w = {
        "num_token": 1.0,
        "xyz": 5.0,
        "heading": 1.0,
        "fwd_delta": 0.5,
        "yaw_delta": 0.5,
        "smoothness": 0.0,
    }
    assert set(weights.keys()) == set(expected_w.keys()), (
        f"unexpected loss_weights keys: got {set(weights.keys())}, "
        f"expected {set(expected_w.keys())}"
    )
    for k, v in expected_w.items():
        assert weights[k] == v, f"loss_weights.{k} = {weights[k]}, expected {v}"


def test_loss_weights_does_NOT_contain_legacy_speed_key():
    """Round 6 P1-6 lock-in: legacy 'speed' / 'yaw_rate' keys must be absent."""
    cfg = _load()
    assert "speed" not in cfg["loss_weights"]
    assert "yaw_rate" not in cfg["loss_weights"]


def test_smoothness_default_is_zero():
    """Round 6 P1-5 lock-in: smoothness MUST default to 0 (turn / stop / U-turn
    behaviors must not be smoothed during first training / tiny-batch overfit).
    """
    cfg = _load()
    assert cfg["loss_weights"]["smoothness"] == 0.0


def test_text_encoder_block():
    cfg = _load()
    te = cfg["text_encoder"]
    assert te["share_with"] == "ldf"
    assert te["freeze"] is True


def test_data_block_has_required_paths():
    cfg = _load()
    data = cfg["data"]
    assert "raw_data_dir" in data
    assert "stats_dir" in data
