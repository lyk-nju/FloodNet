# eval_generation_metrics.py 使用说明

FloodNet 生成评估脚本，位于 `eval/eval_generation_metrics.py`。  
支持轨迹控制精度评估、视频渲染、可视化对比及 T2M 文本-动作指标。

---

## 快速上手

```bash
cd /home/yuankai/Text2Motion/FloodNet

# 最简用法（使用 config 里的 test_ckpt）
conda run -n flooddiffusion python eval/eval_generation_metrics.py \
    --config configs/ldf.yaml

# 指定 checkpoint + 覆盖配置项 + 多次生成取均值
conda run -n flooddiffusion python eval/eval_generation_metrics.py \
    --config configs/ldf_copy1.yaml \
    --ckpt outputs/20260422_170445_ldf/step_step=250000.ckpt \
    --set test_vae_ckpt=outputs/vae_1d_z4_step=300000.ckpt exp_name=my_eval \
    --seed 1234 \
    --viz_traj \
    --num_runs 10
```

---

## 参数说明

### 基础参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | `configs/ldf.yaml` | 训练配置文件路径 |
| `--ckpt` | None | checkpoint 路径；不填则读取 `cfg.test_ckpt` 或 `cfg.resume_ckpt` |
| `--seed` | 1234 | 全局随机种子 |
| `--batch_size` | None | 覆盖 config 里的 `test_bs` |
| `--num_workers` | None | 覆盖 DataLoader worker 数 |
| `--no_ema` | False | 跳过 EMA 权重加载（默认使用 EMA） |
| `--max_batches` | 0 | >0 时只跑前 N 个 batch，用于快速调试 |
| `--seg_size` | 20 | segment/prefix MSE 的帧窗口大小 |
| `--out_dir` | `eval/`（脚本所在目录） | 覆盖输出根目录；默认输出在 `eval/eval_{exp_name}_seed{seed}/` |

### 覆盖配置项 `--set`

支持任意 OmegaConf dot-path 覆盖，多个 key=value 空格分隔：

```bash
--set test_vae_ckpt=outputs/vae.ckpt exp_name=my_run model.params.cfg_scale_text=3.0
```

`exp_name` 决定输出目录名：`outputs/eval_{exp_name}_seed{seed}/`

### 功能开关

| 参数 | 说明 |
|---|---|
| `--viz_traj` | 每个样本生成 XZ 轨迹对比图（PNG），含 GT/预测/无条件生成三条曲线 |
| `--num_runs N` | 对每个样本独立生成 N 次，指标取均值并输出标准差（`ade_std` 等） |
| `--forward_control_loss` | 额外跑一次 `model()` forward（非 generate），计算训练等价的 active-window XZ 控制损失 |
| `--traj_ablation` | 额外生成一次去掉所有轨迹条件的结果，计算无控制的 ADE/FDE，衡量 ControlNet 贡献 |
| `--topk N` | 评估结束后打印 ADE 最高的 N 个最难样本 |
| `--t2m_metric` | 在 `val_meta_paths` 上跑 T2M FID / R-Precision / Diversity（耗时，需大 val 集） |

---

## 输出结构

```
eval/eval_{exp_name}_seed{seed}/
├── metrics.json                   # 所有指标，结构见下
├── traj_per_sample.json           # 每个样本的详细轨迹指标
└── {dataset_id}/                  # 如 HumanML3D/
    ├── video/                     # 渲染好的 MP4 对比视频
    │   └── {sample_id}.mp4
    ├── traj_plot/                 # XZ 轨迹对比图（--viz_traj）
    │   └── {sample_id}.png
    ├── feature/                   # 解码后的 263D 动作特征 (.npy)
    ├── token/                     # 生成的 latent token (.npy)
    ├── traj/                      # 预测的 XZ 轨迹 (.npy)
    ├── traj_mask/                 # traj_mask (.npy)
    └── text/                      # 文本标注 (.txt)
```

### metrics.json 结构

```json
{
  "000021": {
    "T": 177,
    "ade": 0.281,
    "ade_std": 0.032,        // 仅 --num_runs > 1 时出现
    "fde": 0.571,
    "mse": 0.142,
    "traj_jitter": 8.6e-06,  // 轨迹平滑度，越小越平滑
    "seg_mse": {             // 非重叠 20 帧窗口误差
      "[0:20]": 0.001,
      "[20:40]": 0.005,
      ...
    },
    "prefix_mse": {          // 累积前缀误差 [0:20], [0:40], ...
      "[0:20]": 0.001,
      "[0:40]": 0.003,
      ...
    }
  },
  ...
  "summary": {
    "traj/ADE_mean": 0.207,
    "traj/ADE_std": 0.091,
    "traj/FDE_mean": 0.327,
    "traj/MSE_mean": 0.110,
    "traj/seg_mse_per_slot": [...],
    "traj/prefix_mse_per_slot": [...],
    "traj/jitter_mean": 1.26e-05,
    "traj/n_samples": 8
  }
}
```

---

## 指标说明

| 指标 | 含义 |
|---|---|
| **ADE** | Average Displacement Error：所有被 mask 帧的 XZ 平面 L2 距离均值（米） |
| **FDE** | Final Displacement Error：最后一个 mask 帧的 XZ L2 距离 |
| **MSE** | 所有被 mask 帧的 XZ 平面均方误差 |
| **seg_mse** | 按 `seg_size`（默认20帧）分段的 MSE，诊断误差在时间上的积累方式 |
| **prefix_mse** | 从第 0 帧到第 20/40/60... 帧的累积 MSE，反映长序列误差增长趋势 |
| **traj_jitter** | 轨迹加速度的均方值 `mean(‖xz[t+1]−2xz[t]+xz[t−1]‖²)`，衡量轨迹平滑度 |

> **注意**：所有轨迹指标只在 `traj_mask=1` 的帧上计算，由 config 中的 `mask_ratio` 控制稀疏程度（1.0 = 全帧监督）。

---

## 典型使用场景

### 1. 标准评估（单次生成）

```bash
conda run -n flooddiffusion python eval/eval_generation_metrics.py \
    --config configs/ldf_copy1.yaml \
    --ckpt outputs/20260422_170445_ldf/step_step=250000.ckpt \
    --set test_vae_ckpt=outputs/vae_1d_z4_step=300000.ckpt exp_name=mode1_250k \
    --seed 1234
```

### 2. 稳定评估（多次生成取均值，推荐 num_runs=5~10）

```bash
conda run -n flooddiffusion python eval/eval_generation_metrics.py \
    --config configs/ldf_copy1.yaml \
    --ckpt outputs/20260422_170445_ldf/step_step=250000.ckpt \
    --set test_vae_ckpt=outputs/vae_1d_z4_step=300000.ckpt exp_name=mode1_250k \
    --seed 1234 --num_runs 10
```

### 3. 完整评估（含轨迹可视化 + ControlNet 消融）

```bash
conda run -n flooddiffusion python eval/eval_generation_metrics.py \
    --config configs/ldf_copy1.yaml \
    --ckpt outputs/20260422_170445_ldf/step_step=250000.ckpt \
    --set test_vae_ckpt=outputs/vae_1d_z4_step=300000.ckpt exp_name=mode1_full \
    --seed 1234 --num_runs 5 \
    --viz_traj --traj_ablation --topk 3
```

### 4. 调试（只跑 1 个 batch）

```bash
conda run -n flooddiffusion python eval/eval_generation_metrics.py \
    --config configs/ldf.yaml --max_batches 1 --seed 42
```

---

## 依赖说明

- conda 环境：`flooddiffusion`
- 数据路径：`/data1/yuankai/text2Motion/FloodDiffusion/raw_data/`（在 config `dirs.raw_data` 中指定）
- VAE checkpoint：`outputs/vae_1d_z4_step=300000.ckpt`（在 config `test_vae_ckpt` 中指定，可用 `--set` 覆盖）
- T5 文本编码器：`/data1/yuankai/text2Motion/FloodDiffusion/deps/t5_umt5-xxl-enc-bf16/`
- 评估测试集：由 config 中 `data.test_meta_paths` 指定（默认 `test_min.txt`，8个样本）
