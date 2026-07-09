import torch

from utils.local_frame import root_quat_to_physical_yaw
from utils.motion_process import (
    build_physical_7d_from_5d,
    recover_root_rot_pos,
    replace_root_channels_263_from_7d,
    replace_root_channels_263_window_from_7d,
)


def test_replace_root_channels_263_from_7d_recovers_condition_root():
    target_xyz = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.2, 1.1, 0.4],
            [0.5, 0.9, 0.7],
            [0.9, 1.0, 0.7],
            [1.0, 1.2, 0.3],
        ],
        dtype=torch.float32,
    )
    target_yaw = torch.tensor([0.0, 0.2, 0.5, 0.1, -0.3], dtype=torch.float32)
    target_5d = torch.cat(
        [
            target_xyz,
            torch.cos(target_yaw)[:, None],
            torch.sin(target_yaw)[:, None],
        ],
        dim=-1,
    )
    target_7d = build_physical_7d_from_5d(target_5d)

    generated = torch.randn(target_7d.shape[0], 263, dtype=torch.float32) * 0.01
    replaced = replace_root_channels_263_from_7d(generated, target_7d)
    root_quat, root_xyz = recover_root_rot_pos(replaced.unsqueeze(0))
    recovered_yaw = root_quat_to_physical_yaw(root_quat)[0]

    assert torch.allclose(root_xyz[0, :, [0, 2]], target_xyz[:, [0, 2]], atol=1e-5)
    assert torch.allclose(root_xyz[0, :, 1], target_xyz[:, 1], atol=1e-5)
    assert torch.allclose(recovered_yaw, target_yaw, atol=1e-5)
    assert torch.allclose(replaced[:, 4:], generated[:, 4:])


def test_replace_root_channels_263_window_from_7d_preserves_stream_boundaries():
    target_xyz = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.2, 1.0, 0.3],
            [0.4, 1.1, 0.7],
            [0.7, 1.0, 1.0],
            [1.1, 0.9, 1.2],
            [1.5, 1.0, 1.1],
            [1.7, 1.1, 0.8],
            [1.8, 1.0, 0.5],
            [1.8, 1.0, 0.2],
        ],
        dtype=torch.float32,
    )
    target_yaw = torch.tensor(
        [0.0, 0.1, 0.2, 0.4, 0.3, 0.0, -0.2, -0.4, -0.5],
        dtype=torch.float32,
    )
    target_5d = torch.cat(
        [
            target_xyz,
            torch.cos(target_yaw)[:, None],
            torch.sin(target_yaw)[:, None],
        ],
        dim=-1,
    )
    target_7d = build_physical_7d_from_5d(target_5d)

    chunks = []
    for start_frame, num_frames in ((0, 1), (1, 4), (5, 4)):
        generated = torch.randn(num_frames, 263, dtype=torch.float32) * 0.01
        chunks.append(
            replace_root_channels_263_window_from_7d(
                generated,
                target_7d,
                start_frame=start_frame,
            )
        )
    full = torch.cat(chunks, dim=0)
    root_quat, root_xyz = recover_root_rot_pos(full.unsqueeze(0))
    recovered_yaw = root_quat_to_physical_yaw(root_quat)[0]

    assert torch.allclose(root_xyz[0, :, [0, 2]], target_xyz[:, [0, 2]], atol=1e-5)
    assert torch.allclose(root_xyz[0, :, 1], target_xyz[:, 1], atol=1e-5)
    assert torch.allclose(recovered_yaw, target_yaw, atol=1e-5)
