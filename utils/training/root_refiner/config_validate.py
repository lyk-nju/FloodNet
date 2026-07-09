"""RootRefiner config consistency checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


_ALLOWED_HORIZON_POLICIES = {"random", "max"}
_ALLOWED_PATH_POLICIES = {"mixed", "dense_path", "sparse_path", "goal_point"}
_ALLOWED_PATH_MODES = {"dense_path", "sparse_path", "goal_point"}
_ALLOWED_CANONICALIZATION_MODES = {"b_full"}
_ALLOWED_CANONICALIZATION_ANCHORS = {"first_effective_frame"}
_LEGACY_LOSS_KEYS = {"speed", "yaw_rate"}
_LEGACY_DATA_KEYS = {"normalize", "stats_dir", "path_feature_stats_dir"}
_REQUIRED_DATA_KEYS = {"target", "collate_fn", "train_bs", "val_bs", "num_workers"}
_LEGACY_MODEL_KEYS = {
    "min_tokens",
    "max_tokens",
    "frames_per_token",
    "n_layers_token",
    "decoder_type",
    "decoder_path_cond_dim",
    "decoder_token_res_depth",
    "decoder_frame_res_depth",
}
_ALLOWED_FREEZE_REFINER_MODULES = {
    "condition_encoder",
    "duration_head",
    "root_branch",
    "root_transformer",
    "root_decoder",
}


def _section(cfg: Mapping, key: str) -> Mapping:
    value = cfg.get(key, {}) if isinstance(cfg, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def _is_sequence(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _validate_freeze_modules(refiner_modules, key: str) -> None:
    if isinstance(refiner_modules, str):
        refiner_modules = [refiner_modules]
    if refiner_modules is None:
        refiner_modules = []
    if not _is_sequence(refiner_modules):
        raise ValueError(
            f"{key} must be a list of module names, "
            f"got {type(refiner_modules).__name__}."
        )
    unknown_freeze = sorted(
        {str(name) for name in refiner_modules} - _ALLOWED_FREEZE_REFINER_MODULES
    )
    if unknown_freeze:
        allowed = ", ".join(sorted(_ALLOWED_FREEZE_REFINER_MODULES))
        raise ValueError(
            f"{key} contains unknown module name(s) {unknown_freeze}; "
            f"allowed values are {{{allowed}}}."
        )


def _validate_sampling_override(sampling: Mapping, key: str) -> None:
    if "horizon_policy" in sampling:
        policy = str(sampling["horizon_policy"])
        if policy not in _ALLOWED_HORIZON_POLICIES:
            allowed = ", ".join(sorted(_ALLOWED_HORIZON_POLICIES))
            raise ValueError(
                f"{key}.horizon_policy must be one of {{{allowed}}}, "
                f"got {policy!r}."
            )
    path_condition = _section(sampling, "path_condition")
    if not path_condition:
        return
    if "policy" in path_condition:
        path_policy = str(path_condition["policy"])
        if path_policy not in _ALLOWED_PATH_POLICIES:
            allowed = ", ".join(sorted(_ALLOWED_PATH_POLICIES))
            raise ValueError(
                f"{key}.path_condition.policy must be one of {{{allowed}}}, "
                f"got {path_policy!r}."
            )
    ratios = _section(path_condition, "ratios")
    if ratios:
        unknown = sorted(set(ratios) - _ALLOWED_PATH_MODES)
        if unknown:
            raise ValueError(
                f"{key}.path_condition.ratios contains unknown path mode(s) "
                f"{unknown}."
            )
    offset_start = _section(path_condition, "offset_start")
    apply_to = offset_start.get("apply_to", [])
    if apply_to:
        unknown = sorted(set(apply_to) - _ALLOWED_PATH_MODES)
        if unknown:
            raise ValueError(
                f"{key}.path_condition.offset_start.apply_to contains unknown "
                f"path mode(s) {unknown}."
            )
    sparse_path = _section(path_condition, "sparse_path")
    if "point_range" in sparse_path:
        point_range = sparse_path["point_range"]
        if not (
            _is_sequence(point_range)
            and len(point_range) == 2
            and int(point_range[0]) >= 1
            and int(point_range[1]) >= int(point_range[0])
        ):
            raise ValueError(
                f"{key}.path_condition.sparse_path.point_range must be "
                "[min_points, max_points]."
            )


def _validate_training_schedule(cfg: Mapping) -> None:
    schedule = _section(cfg, "training_schedule")
    if not schedule or not bool(schedule.get("enabled", False)):
        return
    phases = schedule.get("phases")
    if not (_is_sequence(phases) and len(phases) > 0):
        raise ValueError(
            "training_schedule.phases must be a non-empty list of phase configs."
        )
    for phase_idx, phase in enumerate(phases):
        key = f"training_schedule.phases[{phase_idx}]"
        if not isinstance(phase, Mapping):
            raise ValueError(f"{key} must be a mapping.")
        try:
            steps = int(phase.get("steps", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}.steps must be a positive integer.") from exc
        if steps <= 0:
            raise ValueError(f"{key}.steps must be a positive integer.")
        phase_freeze = _section(phase, "freeze")
        if phase_freeze:
            _validate_freeze_modules(
                phase_freeze.get("refiner_modules", []),
                f"{key}.freeze.refiner_modules",
            )
        phase_sampling = _section(phase, "sampling")
        if phase_sampling:
            _validate_sampling_override(phase_sampling, f"{key}.sampling")


def validate_refiner_config(cfg: Mapping) -> None:
    """Fail fast on invalid RootRefiner config combinations."""
    model_block = _section(cfg, "model")
    model = _section(model_block, "params")
    data = _section(cfg, "data")
    optimizer = _section(cfg, "optimizer")
    sampling = _section(cfg, "sampling")
    canonicalization = _section(cfg, "canonicalization")
    loss_weights = _section(cfg, "loss_weights")
    freeze = _section(cfg, "freeze")

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
    for key in sorted(_REQUIRED_DATA_KEYS):
        if key not in data:
            raise ValueError(f"data.{key} is required for RootRefiner config.")

    if "num_token_policy" in data:
        raise ValueError(
            "data.num_token_policy is legacy; use sampling.horizon_policy instead."
        )
    legacy_data_present = sorted(set(data) & _LEGACY_DATA_KEYS)
    if legacy_data_present:
        raise ValueError(
            "RootRefiner data config contains legacy normalize/stat key(s) "
            f"{legacy_data_present}; RootRefiner training now uses physical "
            "frame-space tensors directly."
        )
    if isinstance(cfg, Mapping) and "path_aug" in cfg:
        raise ValueError(
            "path_aug is legacy; use sampling.path_condition.offset_start and "
            "sampling.path_condition.sparse_path instead."
        )

    min_frames = int(model.get("min_frames", 1))
    max_frames = int(model.get("max_frames", min_frames))
    if min_frames < 1 or max_frames < min_frames:
        raise ValueError(
            "RootRefiner frame range invalid: "
            f"min_frames={min_frames}, max_frames={max_frames}."
        )
    legacy_present = sorted(set(model) & _LEGACY_MODEL_KEYS)
    if legacy_present:
        raise ValueError(
            "RootRefiner model.params contains legacy token/decoder key(s) "
            f"{legacy_present}; use frame-space RootRefiner v2 keys instead."
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
            _is_sequence(point_range)
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
        valid_history_frames = int(canonicalization["full_plan_valid_history_frames"])
        if valid_history_frames != 1:
            raise ValueError(
                "canonicalization.full_plan_valid_history_frames must be 1, "
                f"got {valid_history_frames}."
            )

    _validate_freeze_modules(
        freeze.get("refiner_modules", []),
        "freeze.refiner_modules",
    )
    _validate_training_schedule(cfg)


__all__ = ["validate_refiner_config"]
