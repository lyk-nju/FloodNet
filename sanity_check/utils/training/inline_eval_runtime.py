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
    }


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
