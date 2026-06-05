"""RootRefiner config consistency checks.

These guards capture experiment contracts that should fail before a run starts,
instead of silently producing invalid training data or incompatible losses.
"""

from __future__ import annotations

from collections.abc import Mapping


_ALLOWED_HORIZON_POLICIES = {"random", "max"}
_ALLOWED_PATH_POLICIES = {"mixed", "dense_path", "sparse_path", "goal_point"}
_ALLOWED_PATH_MODES = {"dense_path", "sparse_path", "goal_point"}
_ALLOWED_CANONICALIZATION_MODES = {"b_full"}
_ALLOWED_CANONICALIZATION_ANCHORS = {"first_effective_frame"}
_LEGACY_LOSS_KEYS = {"speed", "yaw_rate"}


def _section(cfg: Mapping, key: str) -> Mapping:
    value = cfg.get(key, {}) if isinstance(cfg, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def validate_refiner_config(cfg: Mapping) -> None:
    """Fail fast on invalid RootRefiner config combinations.

    The function accepts plain dict-like configs because `train_refiner.py`
    resolves YAML into dictionaries. RootRefiner uses the LDF-style
    target/params schema and intentionally rejects the old `training` block.
    """
    model_block = _section(cfg, "model")
    model = _section(model_block, "params")
    data = _section(cfg, "data")
    optimizer = _section(cfg, "optimizer")
    sampling = _section(cfg, "sampling")
    canonicalization = _section(cfg, "canonicalization")
    loss_weights = _section(cfg, "loss_weights")

    if "training" in cfg:
        raise ValueError(
            "training is legacy for RootRefiner; use data.train_bs/val_bs, "
            "optimizer.params, sampling.full_plan_ratio, and trainer.max_steps."
        )
    if not model_block.get("target"):
        raise ValueError("model.target is required for LDF-style RootRefiner config.")
    if not isinstance(model_block.get("params"), Mapping):
        raise ValueError("model.params is required for LDF-style RootRefiner config.")
    if not optimizer.get("target") or not isinstance(optimizer.get("params"), Mapping):
        raise ValueError("optimizer.target and optimizer.params are required.")
    for key in ("target", "collate_fn", "train_bs", "val_bs", "num_workers"):
        if key not in data:
            raise ValueError(f"data.{key} is required for RootRefiner config.")

    if "num_token_policy" in data:
        raise ValueError(
            "data.num_token_policy is legacy; use sampling.horizon_policy instead."
        )
    if isinstance(cfg, Mapping) and "path_aug" in cfg:
        raise ValueError(
            "path_aug is legacy; use sampling.path_condition.offset_start and "
            "sampling.path_condition.sparse_path instead."
        )

    min_tokens = int(model.get("min_tokens", 1))
    max_tokens = int(model.get("max_tokens", min_tokens))
    if min_tokens < 1 or max_tokens < min_tokens:
        raise ValueError(
            "RootRefiner token range invalid: "
            f"min_tokens={min_tokens}, max_tokens={max_tokens}."
        )

    frames_per_token = int(model.get("frames_per_token", 4))
    if frames_per_token <= 0:
        raise ValueError(
            f"model.frames_per_token must be positive, got {frames_per_token}."
        )

    policy = str(sampling.get("horizon_policy", "random"))
    if policy not in _ALLOWED_HORIZON_POLICIES:
        allowed = ", ".join(sorted(_ALLOWED_HORIZON_POLICIES))
        raise ValueError(
            f"sampling.horizon_policy must be one of {{{allowed}}}, got {policy!r}."
        )

    if "history_condition" in sampling:
        raise ValueError(
            "sampling.history_condition is not part of the current RootRefiner "
            "training contract; use the existing full/sliding history split."
        )

    path_condition = _section(sampling, "path_condition")
    path_policy = str(path_condition.get("policy", "mixed"))
    if path_policy not in _ALLOWED_PATH_POLICIES:
        allowed = ", ".join(sorted(_ALLOWED_PATH_POLICIES))
        raise ValueError(
            f"sampling.path_condition.policy must be one of {{{allowed}}}, "
            f"got {path_policy!r}."
        )
    ratios = _section(path_condition, "ratios")
    if path_policy == "mixed" and ratios:
        unknown = sorted(set(ratios) - _ALLOWED_PATH_MODES)
        if unknown:
            raise ValueError(
                "sampling.path_condition.ratios contains unknown path mode(s) "
                f"{unknown}."
            )
        total = sum(float(v) for v in ratios.values())
        if total <= 0:
            raise ValueError("sampling.path_condition.ratios must sum to > 0.")
    offset_start = _section(path_condition, "offset_start")
    apply_to = offset_start.get("apply_to", [])
    if apply_to:
        unknown = sorted(set(apply_to) - _ALLOWED_PATH_MODES)
        if unknown:
            raise ValueError(
                "sampling.path_condition.offset_start.apply_to contains unknown "
                f"path mode(s) {unknown}."
            )
    sparse_path = _section(path_condition, "sparse_path")
    if "point_range" in sparse_path:
        point_range = sparse_path["point_range"]
        if not (
            isinstance(point_range, (list, tuple))
            and len(point_range) == 2
            and int(point_range[0]) >= 1
            and int(point_range[1]) >= int(point_range[0])
        ):
            raise ValueError(
                "sampling.path_condition.sparse_path.point_range must be "
                "[min_points, max_points]."
            )

    legacy = sorted(set(loss_weights) & _LEGACY_LOSS_KEYS)
    if legacy:
        raise ValueError(
            "RootRefiner loss_weights contains legacy key(s) "
            f"{legacy}; use fwd_delta/yaw_delta instead."
        )

    if "mode" in canonicalization:
        mode = str(canonicalization["mode"])
        if mode not in _ALLOWED_CANONICALIZATION_MODES:
            raise ValueError(
                f"canonicalization.mode must be 'b_full', got {mode!r}."
            )
    if "anchor" in canonicalization:
        anchor = str(canonicalization["anchor"])
        if anchor not in _ALLOWED_CANONICALIZATION_ANCHORS:
            raise ValueError(
                "canonicalization.anchor must be 'first_effective_frame', "
                f"got {anchor!r}."
            )
    if "full_plan_valid_history_frames" in canonicalization:
        n_hist = int(canonicalization["full_plan_valid_history_frames"])
        if n_hist != 1:
            raise ValueError(
                "canonicalization.full_plan_valid_history_frames must be 1, "
                f"got {n_hist}."
            )


__all__ = ["validate_refiner_config"]
