import math

import torch

from eval.ldf.experiments.artifacts import plot_7d_xz_heading
from utils.motion_process import build_physical_7d_from_5d


def test_plot_7d_xz_heading_writes_png(tmp_path):
    frames = 12
    x = torch.linspace(0.0, 0.5, frames)
    y = torch.zeros(frames)
    z = torch.linspace(0.0, 1.0, frames)
    yaw = torch.full((frames,), math.pi / 8.0)
    traj5 = torch.stack([x, y, z, torch.cos(yaw), torch.sin(yaw)], dim=-1)
    traj7 = build_physical_7d_from_5d(traj5)
    output_path = tmp_path / "heading.png"

    result = plot_7d_xz_heading(
        output_path,
        {"condition": traj7},
        update_frames=[5],
        title="condition heading",
    )

    assert result == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
