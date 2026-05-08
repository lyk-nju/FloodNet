from __future__ import annotations


def copy_traj_fields_to_model_batch(batch, model_batch):
    if "traj" in batch:
        model_batch["traj"] = batch["traj"]
        model_batch["traj_length"] = batch["traj_length"]
        model_batch["traj_mask"] = batch["traj_mask"]
    if "token_mask" in batch:
        model_batch["token_mask"] = batch["token_mask"]
    if "traj_features" in batch:
        model_batch["traj_features"] = batch["traj_features"]


def build_model_batch(batch):
    model_batch = batch.copy()
    model_batch["feature"] = batch["token"]
    model_batch["feature_length"] = batch["token_length"]
    if "token_text_end" in batch:
        model_batch["feature_text_end"] = batch["token_text_end"]
    copy_traj_fields_to_model_batch(batch, model_batch)
    return model_batch
