from __future__ import annotations

import torch

try:
    from ..motion_process import extract_root_trajectory_263_torch
except ImportError:  # pragma: no cover - script entrypoints use top-level imports
    from utils.motion_process import extract_root_trajectory_263_torch


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
            start_tok = t_tok - chunk_size_tokens
            start_f = 0 if start_tok == 0 else 4 * start_tok - 3
            end_f = t_tok * token_to_frame
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
