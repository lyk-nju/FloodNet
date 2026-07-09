import numpy as np
import torch
import torch.nn.functional as F

from utils.math.quaternion import *

"""
Motion data structure:
(B: batch size)
root_rot_velocity (B, seq_len, 1)
root_linear_velocity (B, seq_len, 2)
root_y (B, seq_len, 1)
ric_data (B, seq_len, (joint_num - 1)*3)
rot_data (B, seq_len, (joint_num - 1)*6)
local_velocity (B, seq_len, joint_num*3)
foot contact (B, seq_len, 4)
"""


def recover_root_rot_pos(data):
    # recover root rotation and position
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    """Get Y-axis rotation from rotation velocity"""
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    """Add Y-axis rotation to root position"""
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def extract_root_trajectory_263(feature_263: np.ndarray) -> np.ndarray:
    """Recover root xyz trajectory from a raw 263D motion array."""
    feature_vec = torch.from_numpy(feature_263).float().unsqueeze(0)
    _, r_pos = recover_root_rot_pos(feature_vec)
    return r_pos.squeeze(0).cpu().numpy()


def extract_root_xz_phi_features_263(feature_263: np.ndarray) -> np.ndarray:
    """Recover legacy 4D root condition [x, z, quat_w, quat_y]."""
    feature_vec = torch.from_numpy(feature_263).float().unsqueeze(0)
    r_rot_quat, r_pos = recover_root_rot_pos(feature_vec)
    cos_yaw = r_rot_quat[..., 0]
    sin_yaw = r_rot_quat[..., 2]
    xz = r_pos[..., [0, 2]]
    features = torch.stack([xz[..., 0], xz[..., 1], cos_yaw, sin_yaw], dim=-1)
    return features.squeeze(0).cpu().numpy()


def extract_root_traj_feats_7d_263(feature_263: np.ndarray) -> np.ndarray:
    """Recover world-frame 7D root trajectory features from 263D motion."""
    feature_vec = torch.from_numpy(feature_263).float().unsqueeze(0)
    root_quat, root_xyz = recover_root_rot_pos(feature_vec)
    traj_7d = root_to_traj_feats_7d(root_quat, root_xyz)
    return traj_7d.squeeze(0).cpu().numpy()


def extract_root_trajectory_263_torch(feature_263: torch.Tensor) -> torch.Tensor:
    """Torch variant of extract_root_trajectory_263 for batched tensors."""
    _, r_pos = recover_root_rot_pos(feature_263)
    return r_pos


def root_to_traj_feats_7d(
    root_quat: torch.Tensor,
    root_xyz: torch.Tensor,
) -> torch.Tensor:
    """Convert recovered root rotation/position to 7D trajectory features.

    Args:
        root_quat: [..., T, 4] from recover_root_rot_pos.
        root_xyz: [..., T, 3] from recover_root_rot_pos.

    Returns:
        [..., T, 7] as [x, y, z, cos(yaw), sin(yaw), fwd_delta, yaw_delta].

    fwd_delta and yaw_delta are per-frame deltas, not rates. Frame 0 has no
    previous frame, so both delta channels are zero there.
    """
    from utils.local_frame import root_quat_to_physical_yaw

    physical_yaw = root_quat_to_physical_yaw(root_quat)
    cos_h = torch.cos(physical_yaw)
    sin_h = torch.sin(physical_yaw)
    traj_5d = torch.cat([root_xyz, cos_h[..., None], sin_h[..., None]], dim=-1)
    return append_traj_deltas_5d_to_7d(traj_5d, physical_yaw=physical_yaw)


def build_physical_7d_from_5d(wp5: torch.Tensor) -> torch.Tensor:
    """Append physical delta channels to [x, y, z, cos(yaw), sin(yaw)]."""
    return append_traj_deltas_5d_to_7d(wp5)


def _unwrap_yaw(yaw: torch.Tensor) -> torch.Tensor:
    from utils.local_frame import wrap_angle

    if yaw.shape[-1] <= 1:
        return yaw.clone()
    diff = wrap_angle(yaw[..., 1:] - yaw[..., :-1])
    return torch.cat([yaw[..., :1], yaw[..., :1] + torch.cumsum(diff, dim=-1)], dim=-1)


def replace_root_channels_263_from_7d(
    feature_263: torch.Tensor,
    traj_7d: torch.Tensor,
    *,
    hold_tail: bool = True,
) -> torch.Tensor:
    """Return a copy of ``feature_263`` whose recoverable root follows ``traj_7d``.

    Only the HumanML3D root channels are rewritten:
      - channel 0: root rotation velocity
      - channels 1:3: local root XZ velocity
      - channel 3: root height

    Body/local pose channels are left untouched. Because HumanML3D stores root
    XZ as integrated velocity, frame 0 is anchored at world XZ zero; callers that
    need absolute offsets should compare/use paths in that same origin convention.
    """
    if feature_263.dim() != 2 or feature_263.shape[-1] < 4:
        raise ValueError(
            "replace_root_channels_263_from_7d expects feature_263 [T,D>=4], "
            f"got {tuple(feature_263.shape)}"
        )
    if traj_7d.dim() != 2 or traj_7d.shape[-1] < 5:
        raise ValueError(
            "replace_root_channels_263_from_7d expects traj_7d [T,>=5], "
            f"got {tuple(traj_7d.shape)}"
        )
    return replace_root_channels_263_window_from_7d(
        feature_263,
        traj_7d,
        start_frame=0,
        hold_tail=hold_tail,
    )


def replace_root_channels_263_window_from_7d(
    feature_263: torch.Tensor,
    traj_7d: torch.Tensor,
    *,
    start_frame: int,
    hold_tail: bool = True,
) -> torch.Tensor:
    """Rewrite root channels for a stream chunk at ``start_frame``.

    The chunk receives frame-local feature rows but the root velocities are
    derived from the global target frames ``[start_frame, start_frame + T]`` so
    independently corrected chunks can be concatenated without boundary drift.
    """
    if feature_263.dim() != 2 or feature_263.shape[-1] < 4:
        raise ValueError(
            "replace_root_channels_263_window_from_7d expects feature_263 "
            f"[T,D>=4], got {tuple(feature_263.shape)}"
        )
    if traj_7d.dim() != 2 or traj_7d.shape[-1] < 5:
        raise ValueError(
            "replace_root_channels_263_window_from_7d expects traj_7d [T,>=5], "
            f"got {tuple(traj_7d.shape)}"
        )
    out = feature_263.clone()
    num_frames = int(out.shape[0])
    if num_frames <= 0 or traj_7d.shape[0] <= 0:
        return out

    start_frame = max(0, int(start_frame))
    needed = start_frame + num_frames + 1
    traj = traj_7d.to(device=out.device, dtype=out.dtype)
    if hold_tail and traj.shape[0] < needed:
        tail = traj[-1:].expand(needed - traj.shape[0], -1)
        traj = torch.cat([traj, tail], dim=0)
    if traj.shape[0] <= start_frame:
        out[:, :4] = 0.0
        return out

    yaw = _unwrap_yaw(torch.atan2(traj[:, 4], traj[:, 3]))
    yaw = yaw - yaw[:1]
    half_angle = -0.5 * yaw

    current = traj[start_frame:min(start_frame + num_frames, traj.shape[0])]
    valid = int(current.shape[0])
    out[:valid, 0] = 0.0
    out[:valid, 1:3] = 0.0
    out[:valid, 3] = current[:, 1]
    if valid < num_frames:
        out[valid:, :4] = 0.0
    if valid <= 0:
        return out

    next_end = min(start_frame + valid + 1, traj.shape[0])
    if next_end - start_frame < 2:
        return out
    xyz_pair = traj[start_frame:next_end, :3]
    half_pair = half_angle[start_frame:next_end]
    num_deltas = int(xyz_pair.shape[0] - 1)
    out[:num_deltas, 0] = half_pair[1:] - half_pair[:-1]

    delta_xz = xyz_pair[1:, [0, 2]] - xyz_pair[:-1, [0, 2]]
    delta_world = torch.zeros((num_deltas, 3), device=out.device, dtype=out.dtype)
    delta_world[:, [0, 2]] = delta_xz
    q_next = torch.zeros((num_deltas, 4), device=out.device, dtype=out.dtype)
    q_next[:, 0] = torch.cos(half_pair[1:])
    q_next[:, 2] = torch.sin(half_pair[1:])
    local_delta = qrot(q_next, delta_world)
    out[:num_deltas, 1] = local_delta[:, 0]
    out[:num_deltas, 2] = local_delta[:, 2]
    return out


def append_traj_deltas_5d_to_7d(
    traj_5d: torch.Tensor,
    *,
    physical_yaw: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append fwd_delta and yaw_delta to 5D root trajectory features.

    Input and output shapes are [..., T, 5] and [..., T, 7]. The appended
    channels are derived as:

        fwd_delta[t] = dot(xz[t] - xz[t-1], [sin(yaw[t]), cos(yaw[t])])
        yaw_delta[t] = wrap_angle(yaw[t] - yaw[t-1])

    Frame 0 has no previous frame, so both appended deltas are zero.
    """
    from utils.local_frame import wrap_angle

    xyz = traj_5d[..., :3]
    cos_h = traj_5d[..., 3]
    sin_h = traj_5d[..., 4]
    if physical_yaw is None:
        # Keep atan2 gradients finite for collapsed heading vectors. The output
        # still preserves the input cos/sin channels unchanged.
        heading_sq = cos_h * cos_h + sin_h * sin_h
        heading_norm = torch.sqrt(heading_sq.clamp_min(1e-12))
        valid_heading = heading_sq > 1e-12
        cos_u = torch.where(
            valid_heading,
            cos_h / heading_norm,
            torch.ones_like(cos_h),
        )
        sin_u = torch.where(
            valid_heading,
            sin_h / heading_norm,
            torch.zeros_like(sin_h),
        )
        physical_yaw = torch.atan2(sin_u, cos_u)

    xz = traj_5d[..., [0, 2]]
    delta_xz = torch.zeros_like(xz)
    delta_xz[..., 1:, :] = xz[..., 1:, :] - xz[..., :-1, :]
    fwd_dir = torch.stack([sin_h, cos_h], dim=-1)
    fwd_delta = (delta_xz * fwd_dir).sum(-1, keepdim=True)

    yaw_delta = torch.zeros_like(physical_yaw)
    yaw_delta[..., 1:] = wrap_angle(physical_yaw[..., 1:] - physical_yaw[..., :-1])

    return torch.cat(
        [xyz, cos_h[..., None], sin_h[..., None], fwd_delta, yaw_delta[..., None]],
        dim=-1,
    )


def extract_root_trajectory_length(
    feature_263: np.ndarray | torch.Tensor,
    xy_only: bool = True,
) -> float:
    """Compute recovered root trajectory length."""
    if isinstance(feature_263, np.ndarray):
        feat = torch.from_numpy(feature_263).float()
    else:
        feat = feature_263.float()
    if feat.dim() == 2:
        feat = feat.unsqueeze(0)
    _, r_pos = recover_root_rot_pos(feat)
    traj = r_pos[0]
    coords = traj[:, [0, 2]] if xy_only else traj
    if coords.size(0) < 2:
        return 0.0
    diffs = coords[1:] - coords[:-1]
    return float(torch.norm(diffs, dim=-1).sum().item())


def recover_joint_positions_263(data: np.ndarray, joints_num) -> np.ndarray:
    """
    Recovers 3D joint positions from the rotation-invariant local positions (ric_data).
    This is the most direct way to get the skeleton for animation.
    """
    feature_vec = torch.from_numpy(data).unsqueeze(0).float()
    r_rot_quat, r_pos = recover_root_rot_pos(feature_vec)
    positions = feature_vec[..., 4 : (joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    """Add Y-axis rotation to local joints"""
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions
    )
    """Add root XZ to joints"""
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    """Concatenate root and joints"""
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    joints_np = positions.squeeze(0).detach().cpu().numpy()
    return joints_np


class StreamJointRecovery263:
    """
    Stream version of recover_joint_positions_263 that processes one frame at a time.
    Maintains cumulative state for rotation angles and positions.

    Key insight: The batch version uses PREVIOUS frame's velocity for the current frame,
    so we need to delay the velocity application by one frame.

    Args:
        joints_num: Number of joints in the skeleton
        smoothing_alpha: EMA smoothing factor (0.0 to 1.0)
            - 1.0 = no smoothing (default), output follows input exactly
            - 0.0 = infinite smoothing, output never changes
            - Recommended values: 0.3-0.7 for visible smoothing
            - Formula: smoothed = alpha * current + (1 - alpha) * previous
    """

    def __init__(self, joints_num: int, smoothing_alpha: float = 1.0):
        self.joints_num = joints_num
        self.smoothing_alpha = np.clip(smoothing_alpha, 0.0, 1.0)
        self.reset()

    def reset(self):
        """Reset the accumulated state"""
        self.r_rot_ang_accum = 0.0
        self.r_pos_accum = np.array([0.0, 0.0, 0.0])
        # Store previous frame's velocities for delayed application
        self.prev_rot_vel = 0.0
        self.prev_linear_vel = np.array([0.0, 0.0])
        # Store previous smoothed joints for EMA
        self.prev_smoothed_joints = None

    def process_frame(self, frame_data: np.ndarray) -> np.ndarray:
        """
        Process a single frame and return joint positions for that frame.

        Args:
            frame_data: numpy array of shape (263,) for a single frame

        Returns:
            joints: numpy array of shape (joints_num, 3) representing joint positions
        """
        # Convert to torch tensor
        feature_vec = torch.from_numpy(frame_data).float()

        # Extract current frame's velocities (will be used in NEXT frame)
        curr_rot_vel = feature_vec[0].item()
        curr_linear_vel = feature_vec[1:3].numpy()

        # Update accumulated rotation angle with PREVIOUS frame's velocity FIRST
        # This matches the batch processing: r_rot_ang[i] uses rot_vel[i-1]
        self.r_rot_ang_accum += self.prev_rot_vel

        # Calculate current rotation quaternion using updated accumulated angle
        r_rot_quat = torch.zeros(4)
        r_rot_quat[0] = np.cos(self.r_rot_ang_accum)
        r_rot_quat[2] = np.sin(self.r_rot_ang_accum)

        # Create velocity vector with Y=0 using PREVIOUS frame's velocity
        r_vel = np.array([self.prev_linear_vel[0], 0.0, self.prev_linear_vel[1]])

        # Apply inverse rotation to velocity using CURRENT rotation
        r_vel_torch = torch.from_numpy(r_vel).float()
        r_vel_rotated = qrot(qinv(r_rot_quat).unsqueeze(0), r_vel_torch.unsqueeze(0))
        r_vel_rotated = r_vel_rotated.squeeze(0).numpy()

        # Update accumulated position with rotated velocity
        self.r_pos_accum += r_vel_rotated

        # Get Y position from data
        r_pos = self.r_pos_accum.copy()
        r_pos[1] = feature_vec[3].item()

        # Extract local joint positions
        positions = feature_vec[4 : (self.joints_num - 1) * 3 + 4]
        positions = positions.view(-1, 3)

        # Apply inverse rotation to local joints
        r_rot_quat_expanded = (
            qinv(r_rot_quat).unsqueeze(0).expand(positions.shape[0], 4)
        )
        positions = qrot(r_rot_quat_expanded, positions)

        # Add root XZ to joints
        positions[:, 0] += r_pos[0]
        positions[:, 2] += r_pos[2]

        # Concatenate root and joints
        r_pos_torch = torch.from_numpy(r_pos).float()
        positions = torch.cat([r_pos_torch.unsqueeze(0), positions], dim=0)

        # Convert to numpy
        joints_np = positions.detach().cpu().numpy()

        # Apply EMA smoothing if enabled
        if self.smoothing_alpha < 1.0:
            if self.prev_smoothed_joints is None:
                # First frame, no smoothing possible
                self.prev_smoothed_joints = joints_np.copy()
            else:
                # EMA: smoothed = alpha * current + (1 - alpha) * previous
                joints_np = (
                    self.smoothing_alpha * joints_np
                    + (1.0 - self.smoothing_alpha) * self.prev_smoothed_joints
                )
                self.prev_smoothed_joints = joints_np.copy()

        # Store current velocities for next frame
        self.prev_rot_vel = curr_rot_vel
        self.prev_linear_vel = curr_linear_vel

        return joints_np


def accumulate_rotations(relative_rotations):
    R_total = [relative_rotations[0]]
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))

    return np.array(R_total)


def recover_from_local_position(final_x, njoint):
    nfrm, _ = final_x.shape
    positions_no_heading = final_x[:, 8 : 8 + 3 * njoint].reshape(
        nfrm, -1, 3
    )  # frames, njoints * 3
    velocities_root_xy_no_heading = final_x[:, :2]  # frames, 2
    global_heading_diff_rot = final_x[:, 2:8]  # frames, 6

    # recover global heading
    global_heading_rot = accumulate_rotations(
        rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy()
    )
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # add global heading to position
    positions_with_heading = np.matmul(
        np.repeat(inv_global_heading_rot[:, None, :, :], njoint, axis=1),
        positions_no_heading[..., None],
    ).squeeze(-1)

    # recover root translation
    # add heading to velocities_root_xy_no_heading

    velocities_root_xyz_no_heading = np.zeros(
        (
            velocities_root_xy_no_heading.shape[0],
            3,
        )
    )
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(
        inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :, None]
    ).squeeze(-1)

    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)

    # add root translation
    positions_with_heading[:, :, 0] += root_translation[:, 0:1]
    positions_with_heading[:, :, 2] += root_translation[:, 2:]

    return positions_with_heading


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _copysign(a, b):
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)


def quaternion_to_axis_angle(quaternions):
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix):
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rotations_matrix_to_smpl85(rotations_matrix, translation):
    nfrm, njoint, _, _ = rotations_matrix.shape
    axis_angle = (
        matrix_to_axis_angle(torch.from_numpy(rotations_matrix))
        .numpy()
        .reshape(nfrm, -1)
    )
    smpl_85 = np.concatenate(
        [axis_angle, np.zeros((nfrm, 6)), translation, np.zeros((nfrm, 10))], axis=-1
    )
    return smpl_85


def recover_from_local_rotation(final_x, njoint):
    nfrm, _ = final_x.shape
    rotations_matrix = rotation_6d_to_matrix(
        torch.from_numpy(final_x[:, 8 + 6 * njoint : 8 + 12 * njoint]).reshape(
            nfrm, -1, 6
        )
    ).numpy()
    global_heading_diff_rot = final_x[:, 2:8]
    velocities_root_xy_no_heading = final_x[:, :2]
    positions_no_heading = final_x[:, 8 : 8 + 3 * njoint].reshape(nfrm, -1, 3)
    height = positions_no_heading[:, 0, 1]

    global_heading_rot = accumulate_rotations(
        rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy()
    )
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # recover root rotation
    rotations_matrix[:, 0, ...] = np.matmul(
        inv_global_heading_rot, rotations_matrix[:, 0, ...]
    )
    velocities_root_xyz_no_heading = np.zeros(
        (
            velocities_root_xy_no_heading.shape[0],
            3,
        )
    )
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(
        inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :, None]
    ).squeeze(-1)
    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)
    root_translation[:, 1] = height
    smpl_85 = rotations_matrix_to_smpl85(rotations_matrix, root_translation)
    return smpl_85


def recover_joint_positions_272(data: np.ndarray, joints_num) -> np.ndarray:
    return recover_from_local_position(data, joints_num)


def convert_motion_to_joints(
    motion_data: np.ndarray,
    dim: int,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    joints_num=22,
):
    """
    Convert Kx263 dim or Kx272 dim motion data to Kx22x3 joint positions.
    Args:
        motion_data: numpy array of shape (K, 263) or (K, 272) where K is number of frames
    Returns:
        joints: numpy array of shape (K, 22, 3) representing joint positions
    """
    if mean is not None and std is not None:
        motion_data = motion_data * std + mean
    if dim == 263:
        recovered_positions = recover_joint_positions_263(motion_data, joints_num)
    elif dim == 272:
        recovered_positions = recover_joint_positions_272(motion_data, joints_num)
    else:
        raise ValueError(f"Unsupported motion data dimension: {dim}")
    return recovered_positions
