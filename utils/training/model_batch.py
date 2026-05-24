from __future__ import annotations


def _copy_trajectory_fields(batch, model_batch):
    # T_B_09 (flag-gated 7D): prefer the world-frame 7D traj cond when present.
    # It is routed as the traj feature so encode_traj_batch passes [B,T,7]
    # straight through to the 7D encoder (no root_to_traj_feats 4D conversion).
    # The raw xyz (traj) is still carried for L_control_xz GT supervision.
    if "traj_cond_7d" in batch:
        model_batch["traj_features"] = batch["traj_cond_7d"]
        model_batch["traj"] = batch.get("traj_cond", batch.get("traj"))
        model_batch["traj_length"] = batch["traj_length"]
        model_batch["traj_mask"] = batch.get(
            "traj_cond_mask",
            batch.get("traj_mask", batch.get("traj_loss_mask")),
        )
        if "token_mask" in batch:
            model_batch["token_mask"] = batch["token_mask"]
        return
    # Prioritize traj_cond (ControlNet condition) over traj (legacy).
    if "traj_cond" in batch:
        model_batch["traj"] = batch["traj_cond"]
        model_batch["traj_length"] = batch["traj_length"]
        model_batch["traj_mask"] = batch.get(
            "traj_cond_mask",
            batch.get("traj_mask", batch.get("traj_loss_mask")),
        )
        # Drop legacy traj_features so encode_traj_batch uses traj_cond
        # instead of overriding it with absolute features.
        model_batch.pop("traj_features", None)
    elif "traj" in batch:
        model_batch["traj"] = batch["traj"]
        model_batch["traj_length"] = batch["traj_length"]
        model_batch["traj_mask"] = batch["traj_mask"]
        if "traj_features" in batch:
            model_batch["traj_features"] = batch["traj_features"]
    if "token_mask" in batch:
        model_batch["token_mask"] = batch["token_mask"]


def prepare_model_input(batch):
    model_batch = batch.copy()
    model_batch["feature"] = batch["token"]
    model_batch["feature_length"] = batch["token_length"]
    if "token_text_end" in batch:
        model_batch["feature_text_end"] = batch["token_text_end"]
    _copy_trajectory_fields(batch, model_batch)
    return model_batch
