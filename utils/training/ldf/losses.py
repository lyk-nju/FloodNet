"""LDF masked loss and trajectory-control helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.motion_process import extract_root_trajectory_263_torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `x` over the True/nonzero entries of `mask` (same shape)."""
    mask_float = mask.to(x.dtype)
    return (x * mask_float).sum() / mask_float.sum().clamp(min=1.0)


def masked_smooth_l1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Frame-masked SmoothL1, summing feature channels when present."""
    diff = F.smooth_l1_loss(pred, gt, reduction="none", beta=beta)
    if diff.dim() > mask.dim():
        diff = diff.sum(-1)
    return masked_mean(diff, mask)


def last_valid_smooth_l1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """SmoothL1 on the last valid frame of each sample."""
    if mask.dim() != 2:
        raise ValueError(
            f"last_valid_smooth_l1 expects a [B,T] mask, got {tuple(mask.shape)}"
        )
    valid = mask > 0
    valid_counts = valid.long().sum(dim=1)
    has_valid = valid_counts > 0
    if not bool(has_valid.any()):
        return pred.new_zeros(())
    last_idx = (valid_counts.clamp(min=1) - 1).view(-1, 1, 1)
    if pred.dim() == mask.dim():
        last_idx = last_idx.squeeze(-1)
    gather_shape = list(pred.shape)
    gather_shape[1] = 1
    last_idx = last_idx.expand(*gather_shape)
    pred_last = pred.gather(dim=1, index=last_idx)[has_valid]
    gt_last = gt.gather(dim=1, index=last_idx)[has_valid]
    diff = F.smooth_l1_loss(pred_last, gt_last, reduction="none", beta=beta)
    if diff.dim() > 1:
        diff = diff.reshape(diff.shape[0], -1).sum(-1)
    return diff.mean()


def derive_fwd_yaw_delta(xyz: torch.Tensor, yaw: torch.Tensor):
    """Per-frame forward displacement and yaw change from root pose."""
    from utils.local_frame import heading_dir_xz, wrap_angle

    delta_xz = torch.zeros_like(xyz[..., [0, 2]])
    delta_xz[..., 1:, :] = xyz[..., 1:, :][..., [0, 2]] - xyz[..., :-1, :][..., [0, 2]]
    fwd_dir = heading_dir_xz(yaw)
    fwd_delta = (delta_xz * fwd_dir).sum(-1)
    yaw_delta = torch.zeros_like(yaw)
    yaw_delta[..., 1:] = wrap_angle(yaw[..., 1:] - yaw[..., :-1])
    return fwd_delta, yaw_delta


def canonicalize_pose_to_anchor(
    xyz: torch.Tensor,
    yaw: torch.Tensor,
    anchor_xyz: torch.Tensor,
    anchor_yaw: torch.Tensor,
):
    """Convert root pose tensors to an anchor-local frame.

    Only ground-plane x/z and heading yaw are anchored. Physical y is preserved,
    matching the project-wide 5D/7D local-frame convention.
    """
    from utils.local_frame import transform_xz_world_to_local, wrap_angle

    local_xz = transform_xz_world_to_local(
        xyz[..., [0, 2]],
        anchor_xyz[..., [0, 2]],
        anchor_yaw,
    )
    local_xyz = xyz.clone()
    local_xyz[..., 0] = local_xz[..., 0]
    local_xyz[..., 2] = local_xz[..., 1]
    local_yaw = wrap_angle(yaw - anchor_yaw)
    return local_xyz, local_yaw


def body_aux_loss_terms(
    pred_xyz,
    pred_yaw,
    gt_xyz,
    gt_yaw,
    active_mask,
    weights: dict,
    heading_form: str = "cosine",
    sample_loss_mask=None,
):
    """Body-aux loss terms over the active frames.

    pred_xyz/gt_xyz: [B, T, 3]; pred_yaw/gt_yaw: [B, T] (physical yaw);
    active_mask: [B, T] (bool/float); sample_loss_mask: optional [B] (0 zeroes a
    whole invalid sample, e.g. T_B_05 padding anchor). Returns (total, per_term).

    P1-3 (v1, intentional): fwd_delta / yaw_delta supervise the active-window
    internal dynamics from the sliced pred/gt,
    NOT a direct per-frame loss vs the dataset 7D channels; the first active
    frame's delta is 0 (no previous frame in the window). To strictly supervise
    the 7D delta channels, include one pre-active frame when slicing; deferred
    unless the body speed profile proves off.
    """
    mask = active_mask.to(pred_xyz.dtype)
    if sample_loss_mask is not None:
        sample_mask = sample_loss_mask.to(mask.dtype).view(
            -1,
            *([1] * (mask.dim() - 1)),
        )
        mask = mask * sample_mask

    loss_root_xz = masked_smooth_l1(
        pred_xyz[..., [0, 2]],
        gt_xyz[..., [0, 2]],
        mask,
    )
    loss_end_xz = last_valid_smooth_l1(
        pred_xyz[..., [0, 2]], gt_xyz[..., [0, 2]], mask
    )
    loss_root_y = masked_smooth_l1(pred_xyz[..., 1:2], gt_xyz[..., 1:2], mask)

    pred_h = torch.stack([torch.cos(pred_yaw), torch.sin(pred_yaw)], -1)
    gt_h = torch.stack([torch.cos(gt_yaw), torch.sin(gt_yaw)], -1)
    if heading_form == "cosine":
        heading_raw = 1.0 - (pred_h * gt_h).sum(-1)
    else:
        heading_raw = F.smooth_l1_loss(pred_h, gt_h, reduction="none").sum(-1)
    loss_heading = masked_mean(heading_raw, mask)

    pred_fwd, pred_yawd = derive_fwd_yaw_delta(pred_xyz, pred_yaw)
    gt_fwd, gt_yawd = derive_fwd_yaw_delta(gt_xyz, gt_yaw)
    loss_fwd = masked_smooth_l1(pred_fwd, gt_fwd, mask)
    loss_yawd = masked_smooth_l1(pred_yawd, gt_yawd, mask)

    total = (
        weights["root_xz"] * loss_root_xz
        + weights["root_y"] * loss_root_y
        + weights["heading"] * loss_heading
        + weights["fwd_delta"] * loss_fwd
        + weights["yaw_delta"] * loss_yawd
        + weights.get("end_xz", 0.0) * loss_end_xz
    )
    terms = {
        "root_xz": loss_root_xz,
        "root_y": loss_root_y,
        "heading": loss_heading,
        "fwd_delta": loss_fwd,
        "yaw_delta": loss_yawd,
        "end_xz": loss_end_xz,
    }
    return total, terms


def compute_body_aux_loss(
    pred_list,
    gt_traj_7d,
    traj_length,
    vae,
    device,
    weights: dict,
    chunk_size_tokens: int | None = None,
    heading_form: str = "cosine",
    sample_loss_mask=None,
    token_to_frame: int = 4,
    window_start_tokens=None,
):
    """Body auxiliary loss over the active window."""
    from utils.local_frame import root_quat_to_physical_yaw
    from utils.motion_process import recover_root_rot_pos

    weighted_losses = []
    term_sums = {
        k: 0.0
        for k in ("root_xz", "root_y", "heading", "fwd_delta", "yaw_delta", "end_xz")
    }
    total_weight = 0.0
    for sample_idx in range(len(pred_list)):
        pred_latent = pred_list[sample_idx].to(device)
        num_tokens = pred_latent.size(0)
        if chunk_size_tokens is not None and num_tokens > chunk_size_tokens:
            from utils.token_frame import token_start_frame

            start_token = num_tokens - chunk_size_tokens
            start_frame = token_start_frame(start_token, token_to_frame)
        else:
            start_frame = 0

        decoded = vae.decode(pred_latent.unsqueeze(0))[0].float()
        quat, xyz = recover_root_rot_pos(decoded.unsqueeze(0))
        yaw = root_quat_to_physical_yaw(quat)

        gt_len = min(int(traj_length[sample_idx].item()), gt_traj_7d.shape[1])
        end_frame = min(decoded.size(0), gt_len)
        if start_frame >= end_frame:
            continue
        frame_slice = slice(start_frame, end_frame)
        target_7d = gt_traj_7d[sample_idx:sample_idx + 1, frame_slice, :].to(
            device=device,
            dtype=xyz.dtype,
        )
        gt_xyz = target_7d[..., :3]
        gt_yaw = torch.atan2(target_7d[..., 4], target_7d[..., 3])
        pred_xyz = xyz[:, frame_slice, :]
        pred_yaw = yaw[:, frame_slice]
        if window_start_tokens is not None:
            from utils.token_frame import token_start_frame

            if not torch.is_tensor(window_start_tokens):
                starts = torch.as_tensor(
                    window_start_tokens,
                    device=device,
                    dtype=torch.long,
                )
            else:
                starts = window_start_tokens.to(device=device, dtype=torch.long)
            if starts.ndim == 0:
                starts = starts.repeat(len(pred_list))
            anchor_frame = token_start_frame(
                int(starts[sample_idx].item()),
                token_to_frame,
            )
            if anchor_frame >= gt_len:
                raise ValueError(
                    "window_start_tokens must reference a valid GT anchor frame; "
                    f"sample={sample_idx}, "
                    f"start_token={int(starts[sample_idx].item())}, "
                    f"anchor_frame={anchor_frame}, traj_length={gt_len}"
                )
            anchor_7d = gt_traj_7d[
                sample_idx:sample_idx + 1,
                anchor_frame:anchor_frame + 1,
                :,
            ].to(
                device=device,
                dtype=xyz.dtype,
            )
            anchor_xyz = anchor_7d[..., :3]
            anchor_yaw = torch.atan2(anchor_7d[..., 4], anchor_7d[..., 3])
            pred_xyz, pred_yaw = canonicalize_pose_to_anchor(
                pred_xyz, pred_yaw, anchor_xyz, anchor_yaw
            )
            gt_xyz, gt_yaw = canonicalize_pose_to_anchor(
                gt_xyz, gt_yaw, anchor_xyz, anchor_yaw
            )

        num_frames = end_frame - start_frame
        sample_mask_i = None
        sample_w = 1.0
        if sample_loss_mask is not None:
            sample_w = float(sample_loss_mask[sample_idx])
            sample_mask_i = sample_loss_mask[sample_idx:sample_idx + 1].to(device)
        active_mask = torch.ones(1, num_frames, device=device, dtype=xyz.dtype)

        total_i, terms_i = body_aux_loss_terms(
            pred_xyz,
            pred_yaw,
            gt_xyz,
            gt_yaw,
            active_mask,
            weights,
            heading_form=heading_form,
            sample_loss_mask=sample_mask_i,
        )
        effective_frames = num_frames * sample_w
        if effective_frames <= 0:
            continue
        weighted_losses.append(total_i * effective_frames)
        for key in term_sums:
            term_sums[key] += float(terms_i[key].detach()) * effective_frames
        total_weight += effective_frames

    if total_weight <= 0 or not weighted_losses:
        return None, {}
    loss = torch.stack(weighted_losses).sum() / total_weight
    metrics = {key: value / total_weight for key, value in term_sums.items()}
    return loss, metrics


def compute_control_loss_xz(
    pred_list,
    traj,
    traj_mask,
    traj_length,
    vae,
    device,
    train_mode: int = 3,
    chunk_size_tokens: int | None = None,
    token_to_frame: int = 4,
):
    """XZ-plane trajectory control loss.

      Mode 1 - active window, absolute coords, no detach
      Mode 2 - active window, absolute coords, detach past tokens
      Mode 3 - full sequence, absolute coords, no detach
      Mode 4 - full sequence, absolute coords, detach past tokens
      Mode 5 - active window, relative displacement, pred anchor
      Mode 6 - active window, relative displacement, GT anchor
    """
    use_active_window = train_mode in (1, 2, 5, 6)
    detach_past = train_mode in (2, 4)
    relative_disp = train_mode in (5, 6)
    relative_disp_gt_anchor = train_mode == 6

    control_loss = 0.0
    valid_count = 0.0
    for sample_idx in range(len(pred_list)):
        pred_latent_full = pred_list[sample_idx].to(device)
        num_tokens = pred_latent_full.size(0)

        if chunk_size_tokens is not None and num_tokens > chunk_size_tokens:
            from utils.token_frame import token_start_frame

            start_token = num_tokens - chunk_size_tokens
            start_frame = token_start_frame(start_token, token_to_frame)
            end_frame = num_tokens * token_to_frame
        else:
            start_token = 0
            start_frame = 0
            end_frame = None

        if detach_past and start_token > 0:
            latent_for_decode = torch.cat(
                [
                    pred_latent_full[:start_token].detach(),
                    pred_latent_full[start_token:],
                ],
                dim=0,
            )
        else:
            latent_for_decode = pred_latent_full

        decoded = vae.decode(latent_for_decode.unsqueeze(0))[0].float()
        pred_frame_count = decoded.size(0)
        target_frame_count = min(int(traj_length[sample_idx].item()), traj.shape[1])

        if use_active_window and end_frame is not None:
            pred_slice = slice(
                min(start_frame, pred_frame_count),
                min(end_frame, pred_frame_count),
            )
            target_slice = slice(
                min(start_frame, target_frame_count),
                min(end_frame, target_frame_count),
            )
        else:
            pred_slice = slice(0, pred_frame_count)
            target_slice = slice(0, target_frame_count)

        frame_count = min(
            pred_slice.stop - pred_slice.start,
            target_slice.stop - target_slice.start,
        )
        if frame_count <= 0:
            continue

        pred_traj_full = extract_root_trajectory_263_torch(decoded.unsqueeze(0))
        pred_traj = pred_traj_full[:, pred_slice, :][:, :frame_count, :]
        gt_traj = traj[sample_idx, target_slice, :][:frame_count].unsqueeze(0).to(
            pred_traj.device, dtype=pred_traj.dtype
        )
        mask = traj_mask[sample_idx, target_slice][:frame_count].unsqueeze(0).to(
            pred_traj.device, dtype=pred_traj.dtype
        )

        pred_xz = pred_traj[..., [0, 2]]
        gt_xz = gt_traj[..., [0, 2]]

        if relative_disp:
            if relative_disp_gt_anchor:
                gt_anchor = gt_xz[:, 0:1, :].detach()
                pred_xz = pred_xz - pred_xz[:, 0:1, :].detach()
                gt_xz = gt_xz - gt_anchor
            else:
                anchor = pred_xz[:, 0:1, :].detach()
                pred_xz = pred_xz - anchor
                gt_xz = gt_xz - gt_xz[:, 0:1, :]

        sq_err = ((pred_xz - gt_xz) ** 2).sum(dim=-1)
        control_loss = control_loss + (mask * sq_err).sum()
        valid_count += mask.sum().item()

    if valid_count <= 0:
        return None
    return control_loss / valid_count
