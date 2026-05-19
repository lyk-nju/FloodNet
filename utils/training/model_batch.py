from __future__ import annotations


def _copy_trajectory_fields(batch, model_batch):
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
