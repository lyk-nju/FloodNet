## FloodNet tests

这里放所有与测试/验证相关的内容（单元测试、冒烟脚本、对齐检查等）。

### 运行方式（推荐）

在 `conda activate flooddiffusion` 后：

```bash
PYTHONPATH=. python -u FloodNet/test/smoke_zero_residuals.py
```

### GPU 集成测试

部分 Wan attention 路径依赖 CUDA；无 GPU 时脚本会自动跳过“forward 等价性”部分，只验证 zero-head 初始化。

### 数据集根轨迹 + path_heading 可视化

从 HumanML3D 读 `traj`，检查 `path_heading_features_from_root_xyz` 与数据集中 `traj_features`（若有）是否一致，并保存 xz 路径与 cos/sin 图：

```bash
cd FloodNet
conda activate flooddiffusion  # 或任意含 numpy、matplotlib 的环境
PYTHONPATH=. python test/viz_traj_heading_from_dataset.py --split val --num-samples 6
```

图默认写在 `FloodNet/outputs/traj_heading_viz/`。

