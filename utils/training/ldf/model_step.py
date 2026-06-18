"""LDF model training step helpers."""

from __future__ import annotations

import torch

from utils.training.ldf.conditioning import PreparedCondition, prepare_condition


def _prepare_noisy_window(model, clean_feature, feature_length, time_steps):
    batch_size, seq_len, _ = clean_feature.shape
    device = clean_feature.device

    noise_level = model._get_noise_levels(device, seq_len, time_steps)
    noisy_feature, noise = model.add_noise(clean_feature, noise_level)

    feature = model.preprocess(clean_feature)
    noisy_feature = model.preprocess(noisy_feature)
    noise = model.preprocess(noise)

    feature_ref = []
    noise_ref = []
    noisy_feature_input = []
    end_indices = []
    for sample_idx in range(batch_size):
        end_index = int(model.chunk_size * time_steps[sample_idx].item()) + 1
        valid_len = int(feature_length[sample_idx].item())
        end_index = min(valid_len, end_index)
        feature_ref.append(feature[sample_idx, :, :end_index, ...])
        noise_ref.append(noise[sample_idx, :, :end_index, ...])
        noisy_feature_input.append(noisy_feature[sample_idx, :, :end_index, ...])
        end_indices.append(end_index)

    return noise_level, feature_ref, noise_ref, noisy_feature_input, end_indices


def run_training_window(
    model,
    batch,
    clean_feature,
    time_steps,
    condition: PreparedCondition,
):
    feature_length = batch["feature_length"]
    batch_size, seq_len, _ = clean_feature.shape

    (
        noise_level,
        feature_ref,
        noise_ref,
        noisy_feature_input,
        end_indices,
    ) = _prepare_noisy_window(model, clean_feature, feature_length, time_steps)

    # Always call ControlNet so gradients flow when backbone is frozen.
    # When traj_dropped=True, traj_emb is already None; ControlNet learns
    # to produce near-zero residuals for null-traj input, which closely
    # approximates the zero residuals used by pred_uncond at inference.
    predicted_result = model(
        noisy_feature_input,
        noise_level * model.time_embedding_scale,
        condition.text_context,
        seq_len,
        traj_emb=condition.traj_emb,
        traj_seq_lens=condition.traj_seq_lens,
        traj_token_mask=condition.traj_token_mask,
    )

    loss = 0.0
    # Two lists with different x0-recovery formulas, each tuned to its
    # consumer (both formulas are mathematically valid; they differ only in
    # how prediction error propagates):
    #
    #   loss_x0_list  (Formula 1, "z = pred_vel + eps"):
    #     error  = delta       (beta-independent, fully exposes the model's
    #                           velocity-prediction error)
    #     dx0/dpred_vel = 1    (full-strength gradient at every position)
    #     Used by control loss so the gradient signal is not damped at
    #       low beta, where Formula 2 would let noisy_x carry the loss with
    #       almost no gradient flowing back into the model.
    #
    #   sf_x0_list  (Formula 2, "z = noisy_x + beta * pred_vel"):
    #     error  = beta * delta (low-variance estimate of z, especially
    #                           important at low beta)
    #     dx0/dpred_vel = beta (beta-attenuated gradient; irrelevant here,
    #                           because SF rollout consumes the *value*
    #                           after .detach())
    #     Used by self-forcing rollout to substitute the chunk's
    #       leftmost token where beta ~= 1/cs is small; Formula 1 there
    #       collapses to z + eps (pure noise injection) and corrupts the
    #       next rollout step's context.
    loss_x0_list = []
    sf_x0_list = []
    for sample_idx in range(batch_size):
        sample_tokens = noisy_feature_input[sample_idx].shape[1]
        if model.prediction_type == "vel":
            velocity = feature_ref[sample_idx] - noise_ref[sample_idx]
            squared_error = (
                predicted_result[sample_idx][:, -model.chunk_size :, ...]
                - velocity[:, -model.chunk_size :, ...]
            ) ** 2
        elif model.prediction_type == "x0":
            squared_error = (
                predicted_result[sample_idx][:, -model.chunk_size :, ...]
                - feature_ref[sample_idx][:, -model.chunk_size :, ...]
            ) ** 2
        elif model.prediction_type == "noise":
            squared_error = (
                predicted_result[sample_idx][:, -model.chunk_size :, ...]
                - noise_ref[sample_idx][:, -model.chunk_size :, ...]
            ) ** 2
            loss_x0_list.append(None)
            sf_x0_list.append(None)
        else:
            raise ValueError(
                f"Unsupported prediction_type={model.prediction_type!r}"
            )
        sample_loss = squared_error.mean()
        loss += sample_loss
        if model.prediction_type == "vel":
            # Formula 1: full-gradient estimate, for control loss.
            pred_x0_loss = predicted_result[sample_idx] + noise_ref[sample_idx]
            loss_x0_list.append(pred_x0_loss[:, :, 0, 0].permute(1, 0))
            # Formula 2: low-variance estimate, for SF rollout.
            beta = noise_level[sample_idx, :sample_tokens].view(1, -1, 1, 1)
            pred_x0_sf = (
                noisy_feature_input[sample_idx]
                + beta * predicted_result[sample_idx]
            )
            sf_x0_list.append(pred_x0_sf[:, :, 0, 0].permute(1, 0))
        elif model.prediction_type == "x0":
            # x0-prediction: model directly outputs z, so the two formulas
            # coincide. Share the same tensor for both consumers.
            pred_x0 = predicted_result[sample_idx]
            latent = pred_x0[:, :, 0, 0].permute(1, 0)
            loss_x0_list.append(latent)
            sf_x0_list.append(latent)
    loss = loss / batch_size

    pred_x0_latent_list = None
    has_traj_condition = (
        ("traj" in batch and batch.get("traj") is not None)
        or ("traj_features" in batch and batch.get("traj_features") is not None)
        or bool(batch.get("_window_local_traj", False))
    )
    if (
        has_traj_condition
        and (model.prediction_type in ("vel", "x0"))
        and not condition.traj_dropped
    ):
        pred_x0_latent_list = loss_x0_list

    return {
        "loss": loss,
        # Consumed by control loss: Formula 1 (full-gradient).
        "pred_x0_latent_list": pred_x0_latent_list,
        # Consumed by self-forcing rollout: Formula 2 (low-variance).
        "x0_latent_list": sf_x0_list,
        "end_indices": end_indices,
    }


def run_model_step(model, batch) -> dict:
    feature = batch["feature"]
    feature_length = batch["feature_length"]
    batch_size, seq_len, _ = feature.shape
    device = feature.device

    time_steps_override = batch.get("_time_steps_override", None)
    if time_steps_override is not None:
        if not torch.is_tensor(time_steps_override):
            time_steps = torch.as_tensor(
                time_steps_override, device=device, dtype=torch.float32
            )
        else:
            time_steps = time_steps_override.to(device=device, dtype=torch.float32)
        if time_steps.ndim == 0:
            time_steps = time_steps.repeat(batch_size)
        if time_steps.shape[0] != batch_size:
            raise ValueError(
                f"_time_steps_override batch mismatch: got {tuple(time_steps.shape)} "
                f"for batch_size={batch_size}"
            )
    else:
        time_steps = []
        for sample_idx in range(batch_size):
            valid_len = feature_length[sample_idx].item()
            max_time = valid_len / model.chunk_size
            time_steps.append(torch.FloatTensor(1).uniform_(0, max_time).item())
        time_steps = torch.tensor(time_steps, device=device)

    condition = prepare_condition(model, batch, seq_len, device)

    single_result = run_training_window(
        model,
        batch,
        feature,
        time_steps,
        condition,
    )

    loss_dict = {"total": single_result["loss"], "mse": single_result["loss"]}
    if single_result["pred_x0_latent_list"] is not None:
        loss_dict["control_aux"] = {
            "pred_x0_latent_list": single_result["pred_x0_latent_list"]
        }
    return loss_dict
