from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from ..motion_process import extract_root_trajectory_263_torch
except ImportError:  # pragma: no cover - script entrypoints use top-level imports
    from utils.motion_process import extract_root_trajectory_263_torch


# ===========================================================================
# T_B_06: body aux loss (heading + physical root control) — pure core.
# Operates on already-recovered poses (no VAE), so it is fully unit-testable.
# ===========================================================================


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `x` over the True/nonzero entries of `mask` (same shape)."""
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp(min=1.0)


def masked_smooth_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
                     beta: float = 1.0) -> torch.Tensor:
    """SmoothL1 between pred/gt, summed over the last (feature) dim if present,
    then masked-mean over the remaining (frame) dims. `mask` is frame-level."""
    diff = F.smooth_l1_loss(pred, gt, reduction="none", beta=beta)
    if diff.dim() > mask.dim():
        diff = diff.sum(-1)
    return masked_mean(diff, mask)


def derive_fwd_yaw_delta(xyz: torch.Tensor, yaw: torch.Tensor):
    """Per-frame forward displacement + yaw change from (xyz, physical yaw).

    Matches utils/motion_process.root_to_traj_feats_7d: fwd_delta = projection of
    the per-frame xz displacement onto the heading direction; yaw_delta =
    wrap(yaw[t]-yaw[t-1]); both 0 at the first frame. xyz: [..., T, 3];
    yaw: [..., T]. Returns (fwd_delta [..., T], yaw_delta [..., T]).
    """
    from utils.local_frame import heading_dir_xz, wrap_angle

    delta_xz = torch.zeros_like(xyz[..., [0, 2]])
    delta_xz[..., 1:, :] = xyz[..., 1:, :][..., [0, 2]] - xyz[..., :-1, :][..., [0, 2]]
    fwd_dir = heading_dir_xz(yaw)                       # [..., T, 2]
    fwd_delta = (delta_xz * fwd_dir).sum(-1)            # [..., T]
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


def body_aux_loss_terms(pred_xyz, pred_yaw, gt_xyz, gt_yaw, active_mask,
                        weights: dict, heading_form: str = "cosine",
                        sample_loss_mask=None):
    """Five body-aux loss terms (design §2.4) over the active frames.

    pred_xyz/gt_xyz: [B, T, 3]; pred_yaw/gt_yaw: [B, T] (physical yaw);
    active_mask: [B, T] (bool/float); sample_loss_mask: optional [B] (0 zeroes a
    whole invalid sample, e.g. T_B_05 padding anchor). Returns (total, per_term).

    P1-3 (v1, intentional): fwd_delta / yaw_delta supervise the ACTIVE-WINDOW
    INTERNAL dynamics — derived (derive_fwd_yaw_delta) from the sliced pred/gt,
    NOT a direct per-frame loss vs the dataset 7D channels; the first active
    frame's delta is 0 (no previous frame in the window). To strictly supervise
    the 7D delta channels, include one pre-active frame when slicing — deferred
    unless the body speed profile proves off.
    """
    mask = active_mask.to(pred_xyz.dtype)
    if sample_loss_mask is not None:
        mask = mask * sample_loss_mask.to(mask.dtype).view(-1, *([1] * (mask.dim() - 1)))

    l_root_xz = masked_smooth_l1(pred_xyz[..., [0, 2]], gt_xyz[..., [0, 2]], mask)
    l_root_y = masked_smooth_l1(pred_xyz[..., 1:2], gt_xyz[..., 1:2], mask)

    pred_h = torch.stack([torch.cos(pred_yaw), torch.sin(pred_yaw)], -1)
    gt_h = torch.stack([torch.cos(gt_yaw), torch.sin(gt_yaw)], -1)
    if heading_form == "cosine":
        l_heading_raw = 1.0 - (pred_h * gt_h).sum(-1)
    else:
        l_heading_raw = F.smooth_l1_loss(pred_h, gt_h, reduction="none").sum(-1)
    l_heading = masked_mean(l_heading_raw, mask)

    pred_fwd, pred_yawd = derive_fwd_yaw_delta(pred_xyz, pred_yaw)
    gt_fwd, gt_yawd = derive_fwd_yaw_delta(gt_xyz, gt_yaw)
    l_fwd = masked_smooth_l1(pred_fwd, gt_fwd, mask)
    l_yawd = masked_smooth_l1(pred_yawd, gt_yawd, mask)

    total = (
        weights["root_xz"] * l_root_xz
        + weights["root_y"] * l_root_y
        + weights["heading"] * l_heading
        + weights["fwd_delta"] * l_fwd
        + weights["yaw_delta"] * l_yawd
    )
    terms = {
        "root_xz": l_root_xz, "root_y": l_root_y, "heading": l_heading,
        "fwd_delta": l_fwd, "yaw_delta": l_yawd,
    }
    return total, terms


def compute_body_aux_loss(
    pred_list,
    gt_traj_7d,                # [B, T_frame, 7] clip-local raw traj_cond_7d (GT)
    traj_length,               # [B] valid frame count
    vae,
    device,
    weights: dict,
    chunk_size_tokens: int | None = None,
    heading_form: str = "cosine",
    sample_loss_mask=None,     # [B] from T_B_05 (0 = invalid sample)
    token_to_frame: int = 4,
    window_start_tokens=None,
):
    """Body aux loss over the active window (design §2.4).

    Decodes each predicted latent ONCE (vae.decode) and recovers (xyz, physical
    yaw); all five terms + GT reuse that single decode. GT (xyz + heading) is the
    clip-local raw 7D traj_cond (same frame as the decoded pred). Returns
    `(loss, term_metrics)` frame-weighted across the batch, or `(None, {})` when
    no active frames remain.
    """
    from utils.local_frame import root_quat_to_physical_yaw
    from utils.motion_process import recover_root_rot_pos

    weighted_losses = []
    term_sums = {k: 0.0 for k in ("root_xz", "root_y", "heading", "fwd_delta", "yaw_delta")}
    total_n = 0.0
    for i in range(len(pred_list)):
        pred_latent = pred_list[i].to(device)
        t_tok = pred_latent.size(0)
        if chunk_size_tokens is not None and t_tok > chunk_size_tokens:
            from utils.token_frame import token_start_frame
            start_tok = t_tok - chunk_size_tokens
            start_f = token_start_frame(start_tok, token_to_frame)   # P1-2: canonical helper
        else:
            start_f = 0

        decoded = vae.decode(pred_latent.unsqueeze(0))[0].float()    # [T, 263] — single decode
        quat, xyz = recover_root_rot_pos(decoded.unsqueeze(0))       # [1,T,4],[1,T,3]
        yaw = root_quat_to_physical_yaw(quat)                        # [1,T]

        gt_len = min(int(traj_length[i].item()), gt_traj_7d.shape[1])
        end_f = min(decoded.size(0), gt_len)
        if start_f >= end_f:
            continue
        sl = slice(start_f, end_f)
        gt7 = gt_traj_7d[i:i + 1, sl, :].to(device=device, dtype=xyz.dtype)
        gt_xyz = gt7[..., :3]
        gt_yaw = torch.atan2(gt7[..., 4], gt7[..., 3])
        pred_xyz = xyz[:, sl, :]
        pred_yaw = yaw[:, sl]
        if window_start_tokens is not None:
            from utils.token_frame import token_start_frame

            if not torch.is_tensor(window_start_tokens):
                starts = torch.as_tensor(window_start_tokens, device=device, dtype=torch.long)
            else:
                starts = window_start_tokens.to(device=device, dtype=torch.long)
            if starts.ndim == 0:
                starts = starts.repeat(len(pred_list))
            anchor_f = token_start_frame(int(starts[i].item()), token_to_frame)
            if anchor_f >= gt_len:
                raise ValueError(
                    "window_start_tokens must reference a valid GT anchor frame; "
                    f"sample={i}, start_token={int(starts[i].item())}, "
                    f"anchor_frame={anchor_f}, traj_length={gt_len}"
                )
            anchor7 = gt_traj_7d[i:i + 1, anchor_f:anchor_f + 1, :].to(
                device=device, dtype=xyz.dtype
            )
            anchor_xyz = anchor7[..., :3]
            anchor_yaw = torch.atan2(anchor7[..., 4], anchor7[..., 3])
            pred_xyz, pred_yaw = canonicalize_pose_to_anchor(
                pred_xyz, pred_yaw, anchor_xyz, anchor_yaw
            )
            gt_xyz, gt_yaw = canonicalize_pose_to_anchor(
                gt_xyz, gt_yaw, anchor_xyz, anchor_yaw
            )

        n_frames = end_f - start_f
        slm_i = None
        sample_w = 1.0
        if sample_loss_mask is not None:
            sample_w = float(sample_loss_mask[i])
            slm_i = sample_loss_mask[i:i + 1].to(device)
        active_mask = torch.ones(1, n_frames, device=device, dtype=xyz.dtype)

        total_i, terms_i = body_aux_loss_terms(
            pred_xyz, pred_yaw, gt_xyz, gt_yaw, active_mask, weights,
            heading_form=heading_form, sample_loss_mask=slm_i,
        )
        n_eff = n_frames * sample_w
        if n_eff <= 0:
            continue
        weighted_losses.append(total_i * n_eff)
        for k in term_sums:
            term_sums[k] += float(terms_i[k].detach()) * n_eff
        total_n += n_eff

    if total_n <= 0 or not weighted_losses:
        return None, {}
    loss = torch.stack(weighted_losses).sum() / total_n
    metrics = {k: v / total_n for k, v in term_sums.items()}
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
    """
    XZ-plane trajectory control loss. Behaviour is selected by train_mode:

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

    loss_control = 0.0
    n_valid = 0.0
    for i in range(len(pred_list)):
        pred_latent_full = pred_list[i].to(device)
        t_tok = pred_latent_full.size(0)

        if chunk_size_tokens is not None and t_tok > chunk_size_tokens:
            from utils.token_frame import token_start_frame
            start_tok = t_tok - chunk_size_tokens
            start_f = token_start_frame(start_tok, token_to_frame)   # P1-2: canonical helper
            end_f = t_tok * token_to_frame   # legacy: clamped to decoded length below
        else:
            start_tok = 0
            start_f = 0
            end_f = None

        if detach_past and start_tok > 0:
            latent_for_decode = torch.cat(
                [pred_latent_full[:start_tok].detach(), pred_latent_full[start_tok:]],
                dim=0,
            )
        else:
            latent_for_decode = pred_latent_full

        decoded = vae.decode(latent_for_decode.unsqueeze(0))[0].float()
        l_motion = decoded.size(0)
        l_gt_total = min(int(traj_length[i].item()), traj.shape[1])

        if use_active_window and end_f is not None:
            pred_sl = slice(min(start_f, l_motion), min(end_f, l_motion))
            gt_sl = slice(min(start_f, l_gt_total), min(end_f, l_gt_total))
        else:
            pred_sl = slice(0, l_motion)
            gt_sl = slice(0, l_gt_total)

        l_cur = min(pred_sl.stop - pred_sl.start, gt_sl.stop - gt_sl.start)
        if l_cur <= 0:
            continue

        pred_traj_full = extract_root_trajectory_263_torch(decoded.unsqueeze(0))
        pred_traj = pred_traj_full[:, pred_sl, :][:, :l_cur, :]
        gt_traj = traj[i, gt_sl, :][:l_cur].unsqueeze(0).to(
            pred_traj.device, dtype=pred_traj.dtype
        )
        mask = traj_mask[i, gt_sl][:l_cur].unsqueeze(0).to(
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
        loss_control = loss_control + (mask * sq_err).sum()
        n_valid += mask.sum().item()

    if n_valid <= 0:
        return None
    return loss_control / n_valid
