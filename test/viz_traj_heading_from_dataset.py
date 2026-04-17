"""
从 HumanML3D 数据集读取根轨迹 traj，可视化并用数值检查 root_to_traj_feats。

检查内容：
1) 对 traj 直接调用 root_to_traj_feats 与数据集中 traj_features（若存在）是否一致。
2) 用与实现一致的差分规则独立复算 cos/sin，与函数输出的第 2、3 列最大误差（应接近 0）。

运行（在 FloodNet 目录下，且已激活含 numpy/matplotlib 的环境）：

  cd FloodNet
  PYTHONPATH=. python test/viz_traj_heading_from_dataset.py

或指定配置与样本数：

  PYTHONPATH=. python test/viz_traj_heading_from_dataset.py --config configs/ldf.yaml --split val --num-samples 6

输出目录默认：FloodNet/outputs/traj_heading_viz/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# 保证无论从何处启动都能找到 FloodNet 下的 utils / datasets
_FLOODNET_ROOT = Path(__file__).resolve().parents[1]
if str(_FLOODNET_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOODNET_ROOT))
os.chdir(_FLOODNET_ROOT)

from datasets.humanml3d import HumanML3DDataset  # noqa: E402
from utils.initialize import load_config  # noqa: E402
from utils.traj_batch import (  # noqa: E402
    _PATH_HEADING_EPS,
    root_to_traj_feats,
)


def _heading_reconstructed_max_err(traj_xyz: np.ndarray, feats: np.ndarray) -> float:
    """
    按 root_to_traj_feats 的差分定义复算每帧 (cos,sin)，与 feats[:,2:4] 比最大绝对误差。
    """
    traj_xyz = np.asarray(traj_xyz, dtype=np.float64)
    t_len = traj_xyz.shape[0]
    if t_len == 0:
        return 0.0
    eps = float(_PATH_HEADING_EPS)
    x = traj_xyz[:, 0:1]
    z = traj_xyz[:, 2:3]
    if t_len == 1:
        cos_e = np.ones((1, 1), dtype=np.float64)
        sin_e = np.zeros((1, 1), dtype=np.float64)
    else:
        dx = np.zeros((t_len, 1), dtype=np.float64)
        dz = np.zeros((t_len, 1), dtype=np.float64)
        dx[0:1] = x[1:2] - x[0:1]
        dz[0:1] = z[1:2] - z[0:1]
        dx[1:] = x[1:] - x[:-1]
        dz[1:] = z[1:] - z[:-1]
        sq = dx * dx + dz * dz
        short = sq < eps * eps
        norm = np.sqrt(np.maximum(sq, eps * eps))
        cos_e = np.where(short, 1.0, dx / norm)
        sin_e = np.where(short, 0.0, dz / norm)
    cos_hat = feats[:, 2:3].astype(np.float64)
    sin_hat = feats[:, 3:4].astype(np.float64)
    return float(np.max(np.abs(cos_e - cos_hat)) + np.max(np.abs(sin_e - sin_hat)))


def _plot_one(
    name: str,
    traj: np.ndarray,
    feats: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    x = traj[:, 0]
    z = traj[:, 2]
    cos_y = feats[:, 2]
    sin_y = feats[:, 3]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(x, z, "-", color="0.4", linewidth=1, label="path")
    ax.scatter(x, z, c=np.arange(len(x)), cmap="viridis", s=12, zorder=3)
    step = max(1, len(x) // 40)
    # 不用 quiver：HumanML3D xz 尺度很小，quiver 的 scale/scale_units 极易画出「满屏射线」。
    # 在数据坐标里画固定长度的归一化朝向线段，长度随轨迹包围盒自适应。
    span = float(max(np.ptp(x), np.ptp(z), 1e-9))
    tick_len = span * 0.06
    xi, zi = x[::step], z[::step]
    ci, si = cos_y[::step], sin_y[::step]
    inv = 1.0 / np.maximum(np.hypot(ci, si), 1e-9)
    ci, si = ci * inv, si * inv
    for xa, za, c, s in zip(xi, zi, ci, si):
        ax.plot(
            [xa, xa + tick_len * c],
            [za, za + tick_len * s],
            color="crimson",
            linewidth=1.2,
            solid_capstyle="round",
            zorder=4,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f"{name}\nxz path + heading (fixed-length ticks)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax2 = axes[1]
    t = np.arange(len(x))
    ax2.plot(t, cos_y, label="cos ψ")
    ax2.plot(t, sin_y, label="sin ψ")
    ax2.set_xlabel("frame")
    ax2.set_ylabel("value")
    ax2.set_title("cos & sin per frame (path_heading)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=str(_FLOODNET_ROOT / "configs" / "ldf.yaml"),
        help="训练用 yaml（需包含 data 中 meta / feature_path 等）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("train", "val", "test"),
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="可选：逗号分隔的样本下标，如 0,10,20（指定时忽略 --num-samples 的数量逻辑，仅绘这些）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_FLOODNET_ROOT / "outputs" / "traj_heading_viz"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = load_config(config_path=args.config)
    ds = HumanML3DDataset(cfg, split=args.split)
    if len(ds) == 0:
        print(f"[ERROR] 数据集为空：split={args.split}，请检查 configs 与 meta 路径。")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.indices:
        idx_list = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    else:
        idx_list = rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False)
        idx_list = sorted(idx_list.tolist())

    max_diff_ds = None  # None = 从未与数据集 traj_features 对比
    max_err_recon = 0.0

    for idx in idx_list:
        if idx < 0 or idx >= len(ds):
            print(f"[WARN] 跳过非法 index {idx}")
            continue
        sample = ds[idx]
        name = f"{sample.get('dataset', '?')}_{sample.get('name', idx)}"
        traj = np.asarray(sample["traj"], dtype=np.float32)
        feats_fn = root_to_traj_feats(traj)

        err_recon = _heading_reconstructed_max_err(traj, feats_fn)
        max_err_recon = max(max_err_recon, err_recon)

        if "traj_features" in sample:
            feats_ds = np.asarray(sample["traj_features"], dtype=np.float32)
            n = min(len(feats_ds), len(feats_fn))
            diff = float(np.max(np.abs(feats_ds[:n] - feats_fn[:n])))
            max_diff_ds = diff if max_diff_ds is None else max(max_diff_ds, diff)
            if diff > 1e-4:
                print(f"[WARN] 数据集 traj_features 与函数不一致: idx={idx} max|diff|={diff:.6g}")
        else:
            print(f"[INFO] idx={idx} 无 traj_features 键（需同时有 feature+token 时才有），仅校验函数自洽。")

        if err_recon > 1e-5:
            print(f"[FAIL] 函数内部朝向复算误差过大: idx={idx} err={err_recon:.6g}")
        else:
            print(f"[OK] idx={idx} 复算误差 max≈{err_recon:.3e} | traj T={len(traj)}")

        png = out_dir / f"{name.replace('/', '_')}.png"
        _plot_one(name, traj, feats_fn, png)
        print(f"      saved {png}")

    print("---")
    if max_diff_ds is None:
        print("所有样本：数据集中均无 traj_features 键，未做与数据集的一致性对比。")
    else:
        print(f"所有样本：与数据集 traj_features 最大 |diff| = {max_diff_ds:.6g}")
    print(f"所有样本：函数朝向复算最大误差 = {max_err_recon:.6g}")
    if max_err_recon > 1e-5:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
