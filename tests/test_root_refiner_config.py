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
    assert cfg["model"]["params"]["text_emb_dim"] == 4096   # = T5 cache text_dim (P0-1)
    assert cfg["data"]["normalize"] is True        # real training z-scores (P2-1)


def test_train_config_has_no_fixed_validation_knob():
    """Fixed validation is train_refiner's default construction, not a config knob."""
    cfg = _load(TRAIN_CFG_PATH)
    assert "fixed_validation" not in cfg


def test_sampling_path_condition_block_present_in_both_configs():
    for path in (CFG_PATH, TRAIN_CFG_PATH):
        cfg = _load(path)
        assert "path_aug" not in cfg
        sampling = cfg["sampling"]
        assert sampling["horizon_policy"] in {"random", "max", "bucketed"}
        pc = sampling["path_condition"]
        assert pc["policy"] in {"mixed", "dense_path", "sparse_path", "goal_point"}
        for mode in ("dense_path", "sparse_path", "goal_point"):
            assert mode in pc["ratios"], f"{path.name} missing ratios.{mode}"
        offset = pc["offset_start"]
        for k in ("enabled", "prob", "max_frames", "apply_to"):
            assert k in offset, f"{path.name} missing offset_start.{k}"
        assert "point_range" in pc["sparse_path"]


def test_A_P0_1_train_config_interpolation_resolves():
    """A-P0-1: ${data.raw_data_dir} in root_refiner_train.yaml must resolve to a
    real (no '${') path via resolve_cfg_interpolations (train uses yaml.safe_load
    which would otherwise pass the literal to the text encoder)."""
    from train_refiner import resolve_cfg_interpolations

    raw = _load(TRAIN_CFG_PATH)
    assert "${" in raw["text_encoder"]["precomputed_text_emb_path"]   # literal pre-resolve
    resolved = resolve_cfg_interpolations(raw)
    path = resolved["text_encoder"]["precomputed_text_emb_path"]
    assert "${" not in path
    assert path.startswith(resolved["data"]["raw_data_dir"])
    assert path.endswith("HumanML3D/t5_text_embeddings.pt")


def test_model_block_required_keys_and_values():
    cfg = _load()
    assert cfg["model"]["target"] == "models.root_refiner.RootRefiner"
    model = cfg["model"]["params"]
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
        "path_features_dim": 5,
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


def test_ldf_style_runtime_blocks():
    cfg = _load()
    assert "training" not in cfg
    assert cfg["trainer"]["max_steps"] == 100000
    assert cfg["trainer"]["gradient_clip_val"] == 1.0
    assert cfg["data"]["train_bs"] == 64
    assert cfg["data"]["val_bs"] == 64
    assert cfg["optimizer"]["target"] == "AdamW"
    assert cfg["optimizer"]["params"]["lr"] == 1.0e-4
    assert cfg["optimizer"]["params"]["weight_decay"] == 0.01
    assert cfg["sampling"]["full_plan_ratio"] == 0.5


def test_loss_and_loss_weights():
    cfg = _load()
    assert cfg["loss"]["heading_form"] == "cosine"
    weights = cfg["loss_weights"]
    expected_w = {
        "num_token": 1.0,
        "num_token_soft": 0.1,
        "xyz": 5.0,
        "heading": 1.0,
        "fwd_delta": 0.5,
        "yaw_delta": 0.5,
        "path_control": 0.0,
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
    assert data["target"] == "datasets.humanml3d_refiner.HumanML3DRefinerDataset"
    assert data["collate_fn"] == "datasets.humanml3d_refiner.refiner_collate"
    assert "raw_data_dir" in data
    assert "stats_dir" in data


# ---------------------------------------------------------------------------
# Run-control blocks (parity with configs/ldf.yaml): both configs must carry
# the same logger / trainer / validation structure so the two stay consistent.
# ---------------------------------------------------------------------------


def test_run_control_blocks_present_in_both_configs():
    for path in (CFG_PATH, TRAIN_CFG_PATH):
        cfg = _load(path)
        for k in ("exp_name", "seed", "debug", "train", "save_dir", "resume_ckpt"):
            assert k in cfg, f"{path.name} missing top-level {k}"
        assert "wandb" in cfg["logger"]
        for k in ("wandb_key", "project", "entity"):
            assert k in cfg["logger"]["wandb"], f"{path.name} logger.wandb missing {k}"
        for k in ("accelerator", "devices", "log_every_n_steps", "precision"):
            assert k in cfg["trainer"], f"{path.name} trainer missing {k}"
        for k in ("validation_steps", "save_every_n_steps", "save_top_k", "eval_modes"):
            assert k in cfg["validation"], f"{path.name} validation missing {k}"
        assert cfg["validation"]["eval_modes"] == [
            "groundtruth_duration",
            "pred_duration",
        ]


def test_refiner_configs_use_ldf_style_output_base():
    for path in (CFG_PATH, TRAIN_CFG_PATH):
        assert _load(path)["save_dir"] == "./outputs"


def test_debug_config_has_wandb_off_via_debug_flag():
    """Debug/smoke config must keep wandb off (debug=true); real config debug=false."""
    assert _load(CFG_PATH)["debug"] is True
    assert _load(TRAIN_CFG_PATH)["debug"] is False
