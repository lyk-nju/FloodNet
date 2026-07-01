from __future__ import annotations


GENERATION_MODE_GENERATE = "generate"
GENERATION_MODE_STREAM_GENERATE = "stream_generate"
GENERATION_MODE_STREAM_GENERATE_STEP = "stream_generate_step"
GENERATION_MODES = (
    GENERATION_MODE_GENERATE,
    GENERATION_MODE_STREAM_GENERATE,
    GENERATION_MODE_STREAM_GENERATE_STEP,
)


def _resolve_eval_generation_mode(val_cfg) -> str:
    mode = str(
        val_cfg.get(
            "eval_generation_mode",
            GENERATION_MODE_STREAM_GENERATE_STEP,
        )
    )
    if mode not in GENERATION_MODES:
        raise ValueError(
            "validation.eval_generation_mode must be one of "
            f"{GENERATION_MODES}; got {mode!r}."
        )
    return mode


def build_generation_eval_cfg(cfg):
    val_cfg = cfg.get("validation", {})
    eval_cfg = cfg.get("eval", {}) or {}
    stream_history_length = val_cfg.get(
        "eval_stream_history_length",
        val_cfg.get("stream_history_length", eval_cfg.get("history_length", 30)),
    )
    stream_traj_horizon_tokens = val_cfg.get(
        "eval_stream_traj_horizon_tokens",
        val_cfg.get(
            "stream_traj_horizon_tokens",
            eval_cfg.get("traj_horizon_tokens", 20),
        ),
    )
    return {
        "enabled": bool(val_cfg.get("eval_generation_metrics", True)),
        "num_runs": int(val_cfg.get("eval_num_runs", 10)),
        "seg_size": int(val_cfg.get("eval_seg_size", 20)),
        "forward_ctrl_loss": bool(val_cfg.get("eval_forward_control_loss", True)),
        "forward_ctrl_window_mode": str(
            val_cfg.get("eval_forward_control_loss_window_mode", "mean_chunk_windows")
        ),
        "eval_all_captions": bool(val_cfg.get("eval_all_captions", False)),
        "generation_mode": _resolve_eval_generation_mode(val_cfg),
        "stream_history_length": int(stream_history_length),
        "stream_traj_horizon_tokens": (
            None
            if stream_traj_horizon_tokens is None
            else int(stream_traj_horizon_tokens)
        ),
        "stream_token_dt": float(
            val_cfg.get(
                "eval_stream_token_dt",
                val_cfg.get("stream_token_dt", eval_cfg.get("token_dt", 0.20)),
            )
        ),
        "stream_frames_per_token": int(
            val_cfg.get(
                "eval_stream_frames_per_token",
                val_cfg.get(
                    "stream_frames_per_token",
                    eval_cfg.get("frames_per_token", cfg.get("data", {}).get("frames_per_token", 4)),
                ),
            )
        ),
        "num_denoise_steps": val_cfg.get(
            "eval_num_denoise_steps",
            eval_cfg.get("num_denoise_steps", None),
        ),
    }


def t2m_metric_enabled(cfg) -> bool:
    val_cfg = cfg.get("validation", {})
    return bool(val_cfg.get("t2m_metric", False))


def validation_repeat_count(cfg) -> int:
    val_cfg = cfg.get("validation", {})
    return int(val_cfg.get("val_repeat", 1))


def control_loss_train_mode(cfg) -> int:
    body_cfg = cfg.get("body_aux_loss", {}) or {}
    return int(body_cfg.get("control_loss_train_mode", 3))


def get_test_probe_tags(module) -> list[str]:
    tags = getattr(module, "test_loader_tags", None)
    if tags:
        return list(tags)
    return ["test"]


def resolve_test_probe_tag(module, test_loader_idx: int) -> str:
    tags = get_test_probe_tags(module)
    if 0 <= test_loader_idx < len(tags):
        return tags[test_loader_idx]
    return f"test_loader_{test_loader_idx}"
