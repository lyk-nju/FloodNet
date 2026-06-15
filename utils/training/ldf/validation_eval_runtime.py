from __future__ import annotations


def build_generation_eval_cfg(cfg):
    val_cfg = cfg.get("validation", {})
    return {
        "enabled": bool(val_cfg.get("eval_generation_metrics", True)),
        "num_runs": int(val_cfg.get("eval_num_runs", 10)),
        "seg_size": int(val_cfg.get("eval_seg_size", 20)),
        "forward_ctrl_loss": bool(val_cfg.get("eval_forward_control_loss", True)),
        "forward_ctrl_window_mode": str(
            val_cfg.get("eval_forward_control_loss_window_mode", "mean_chunk_windows")
        ),
        "eval_all_captions": bool(val_cfg.get("eval_all_captions", False)),
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
