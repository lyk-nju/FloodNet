# FloodNet TODO.md

本文件记录 `FloodNet` 后续围绕“流式轨迹控制效果差”的诊断与修复任务。

它不是长期路线文档，也不是实验日志。用途很单一：

- 说明当前问题的前因后果
- 固定后续任务顺序
- 给 coding agent 明确修改范围
- 给出每一步最小验收方式
- 避免在没有定位问题前直接重训

---

## 背景

当前 `FloodNet` 在普通 inline eval / offline generate 场景下，轨迹控制效果相对正常；但在 web demo 的真实流式场景中，`stream_generate_step()` 的轨迹跟踪效果明显变差。

这不是一个单纯的前端显示问题。当前至少存在三类可能原因：

1. **评测口径不一致**
   - inline eval 主要验证 `generate()`。
   - web demo 使用逐 token 的 `stream_generate_step()`。
   - 当前 stream metric 尚未完全镜像 web demo 的闭环行为。

2. **推理路径和训练分布不一致**
   - 训练时通常看到整段轨迹条件。
   - web demo 只提供有限未来 horizon。
   - web demo 每一步会基于当前预测 root 重新构造未来轨迹。
   - streaming buffer 内部还涉及 token/window、mask、anchor、LocalTrajEncoder 等路径差异。

3. **训练分布没有覆盖真实流式需求**
   - 模型可能没有见过“只知道未来 20 token”的 horizon-limited 条件。
   - 模型可能没有充分见过“相对当前 root 的轨迹”。
   - 模型更没有系统见过“中途切换轨迹目标”的热切换分布。

因此后续不能直接从“web demo 效果差”跳到“重训”。必须先用固定样本和诊断矩阵定位是哪一层开始掉点。

---

## 总原则

1. **先定位，再训练**
   - 先把 metric 改到能复现 web demo 行为。
   - 再决定是否需要 finetune。

2. **先固定样本，再跑大规模**
   - 先用一个具体样本，例如 `HumanML3D/000021`。
   - 确认每条生成路径的差异后，再扩展到 BABEL / HumanML3D batch metric。

3. **先评测工具，再改模型**
   - 如果 metric 不能解释 web demo 现象，训练改动没有依据。

4. **不要混淆两个目标**
   - 目标 A：流式条件下能跟随一条固定未来轨迹。
   - 目标 B：运行中热切换轨迹仍然平滑。
   - 先完成目标 A，再做目标 B。

---

# Active Task 001

## Task

把 `stream_generate_step` 的 metric 改成真正镜像 web demo 的 trajectory 构造行为。

## Status

`open`

## Problem

当前 `eval/eval_stream_metrics.py` 的 `stream_generate_step` 评测仍然偏理想化。

主要问题：

1. 评测中容易直接使用 GT suffix 作为未来轨迹条件。
2. web demo 实际行为是：
   - 从当前预测 root 出发
   - 投影到目标 polyline
   - 重采样未来 `traj_horizon_tokens`
   - 每一步重新构造 future trajectory
3. 当前 metric 与 web demo 共享的轨迹构造逻辑不够明确。
4. metric 里优先走 `traj_features` 时，会绕开 web demo 使用的 `traj` xyz path，导致诊断不纯。

## Required changes

### 1. 抽出流式轨迹 helper

修改目标：

- `FloodNet/utils/stream_rollout.py`
- 必要时新增 `FloodNet/utils/stream_traj.py`
- `FloodNet/web_demo/model_manager.py`

要求：

- 把 web demo 中这些逻辑抽成公共 helper：
  - project current root to polyline
  - dedupe polyline
  - build remaining polyline
  - resample polyline by token step
  - estimate token step from predicted root history
- web demo 继续调用这些公共 helper。
- metric 也调用同一套 helper。

### 2. metric 使用 web-demo-style trajectory source

修改目标：

- `FloodNet/eval/eval_stream_metrics.py`
- `FloodNet/utils/stream_rollout.py`

要求：

- 新增或改造 `run_stream_generate_step_sample(...)` 的 trajectory 构造路径。
- 支持至少两种 root source：
  - `gt_root`
  - `pred_root`
- 支持 `traj_horizon_tokens`，默认可以先使用 `20`。
- 第一轮诊断时强制走 `traj` xyz path，不要优先用 `traj_features`。

### 3. 保存诊断 artifact

每个 sample / mode 至少保存：

- `pred_root.npy`
- `gt_root.npy`
- `target_traj.npy`
- `metrics.json`
- 可选 `preview.png`

`metrics.json` 至少包含：

- ADE
- FDE
- root path length
- target path length
- stream mode
- root source
- traj horizon
- whether xyz path or traj_features path was used

## Do not do

- 不要在本任务里改训练逻辑。
- 不要在本任务里做 horizon truncation finetune。
- 不要先删 `TrajStreamBuffer` 的 anchor-subtract。
- 不要把 web demo 逻辑复制两份，必须抽公共 helper。

## Validation

至少运行：

```bash
cd /home/yuankai/Text2Motion/FloodNet
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m py_compile \
  eval/eval_stream_metrics.py \
  utils/stream_rollout.py \
  web_demo/model_manager.py
```

如果新增 `utils/stream_traj.py`，也要加入 `py_compile`。

## Done criteria

- metric 与 web demo 共享 trajectory 构造 helper。
- metric 可以显式选择 `gt_root` / `pred_root`。
- metric 可以显式限制 `traj_horizon_tokens=20`。
- 第一轮诊断可以强制走 `traj` xyz path。
- 输出 artifact 足够判断是哪一步开始偏离目标轨迹。

---

# Active Task 002

## Task

新增固定样本诊断脚本，针对单个样本跑完整 stream control 诊断矩阵。

## Status

`pending`

## Problem

当前缺少一个快速、可复现、可视化的单样本诊断入口。

我们需要先用固定样本，例如 `HumanML3D/000021`，在同一个 ckpt 上比较：

- offline generate
- stream_generate
- stream_generate_step full traj
- stream_generate_step horizon traj
- web-demo-style pred-root closed loop

否则无法判断 web demo 效果差到底是：

- 模型本身轨迹控制差
- streaming path 差
- horizon 限制导致差
- pred root 闭环误差导致差
- web demo 输入链路差

## Required changes

新增脚本：

- `FloodNet/eval/diagnose_stream_control.py`

输入参数：

```bash
--config
--ckpt
--vae_ckpt
--sample_id
--out_dir
--history_length
--traj_horizon_tokens
--num_denoise_steps
--seed
```

建议命令：

```bash
cd /home/yuankai/Text2Motion/FloodNet
PYTHONPATH=. /home/yuankai/.conda/envs/flooddiffusion/bin/python \
  eval/diagnose_stream_control.py \
  --config configs/stream.yaml \
  --ckpt /path/to/checkpoint.ckpt \
  --sample_id 000021 \
  --out_dir outputs/diagnose_stream/000021 \
  --history_length 20 \
  --traj_horizon_tokens 20
```

脚本输出目录：

```text
outputs/diagnose_stream/000021/
├── generate_full/
├── stream_generate_full/
├── step_full_xyz_gtroot/
├── step_h20_xyz_gtroot/
├── step_h20_xyz_predroot/
└── summary.json
```

每个子目录至少包含：

```text
pred_motion.npy
pred_root.npy
target_root.npy
target_traj.npy
metrics.json
```

可选：

```text
preview.png
pred_motion.mp4
```

## Diagnostic matrix

必须覆盖以下模式：

| Mode | Purpose |
|------|---------|
| `generate_full` | offline generate baseline，对齐 normal inline eval |
| `stream_generate_full` | 测整段 stream_generate 是否正常 |
| `step_full_xyz_gtroot` | 测逐 token 推理本身是否正常 |
| `step_h20_xyz_gtroot` | 测有限 horizon 是否导致掉点 |
| `step_h20_xyz_predroot` | 最接近 web demo 的闭环行为 |

第一版可以暂时不做 hot-swap。

## Do not do

- 不要把这个脚本做成大规模 benchmark。
- 不要依赖 web server。
- 不要在脚本里改模型权重或训练状态。

## Validation

至少运行一个最小 dry run：

```bash
cd /home/yuankai/Text2Motion/FloodNet
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m py_compile \
  eval/diagnose_stream_control.py
```

如果环境允许，跑 `000021` 的 1-run 诊断，并确认输出 `summary.json`。

## Done criteria

- 一个固定样本可以跑完整诊断矩阵。
- 每个 mode 都有独立输出目录。
- `summary.json` 能一眼看出从哪个 mode 开始 ADE/FDE 明显变差。

---

# Active Task 003

## Task

根据诊断矩阵结果，判断是否需要先修 `stream_generate_step` 推理路径。

## Status

`pending`

## Problem

如果 `step_full_xyz_gtroot` 已经明显差于 `generate_full`，说明不是 horizon 或 pred-root 闭环问题，而是 `stream_generate_step` 推理路径本身与训练分布不一致。

当前重点怀疑点：

1. `TrajStreamBuffer._build_from_features(...)` 直接把 token-level `traj_features` 喂给 `traj_encoder`，可能绕过训练时常见的 `LocalTrajEncoder` 路径。
2. metric 里如果优先使用 `traj_features`，会和 web demo 的 `traj` xyz path 不一致。
3. `TrajStreamBuffer._build_from_xyz(...)` 当前会做 anchor-subtract；它语义上适合流式控制，但训练是否覆盖该分布需要诊断。

## Required changes

根据 Task 002 结果选择：

### Case A：`step_full_xyz_gtroot` 正常

说明逐 token 推理大体没坏。

下一步直接进入 Task 004，分析 horizon 限制。

### Case B：`step_full_xyz_gtroot` 明显变差

优先修推理路径：

- 确认 metric 是否真的强制走 `traj` xyz path。
- 给 `TrajStreamBuffer` 增加 debug：
  - storage mode: `xyz` / `features`
  - write length
  - valid lens
  - whether `traj_emb` is None
- 必要时让 `stream_generate_step` 的 metric 和 web demo 统一只走 xyz path。

暂时不要直接删除 anchor-subtract。

## Do not do

- 不要在没有诊断结果前修改训练数据。
- 不要同时改 anchor、horizon、features path，避免无法归因。

## Validation

重新跑 Task 002 的诊断矩阵，比较修复前后的：

- `step_full_xyz_gtroot`
- `step_h20_xyz_gtroot`

## Done criteria

- 明确判断 `stream_generate_step` 本身是否是主要问题。
- 如果是，能通过最小推理路径修复让 `step_full_xyz_gtroot` 接近 `stream_generate_full`。

---

# Active Task 004

## Task

如果有限 horizon 是主要掉点来源，做 horizon-limited trajectory finetune。

## Status

`pending`

## Problem

web demo 只提供未来有限 token，例如 `traj_horizon_tokens=20`。

如果诊断结果显示：

- `step_full_xyz_gtroot` 正常
- `step_h20_xyz_gtroot` 明显变差

说明模型没有充分学过“只看未来 H token”的条件分布。

## Required changes

修改目标：

- `FloodNet/utils/traj_batch.py`
- dataset / collate 相关路径
- `FloodNet/configs/ldf.yaml` 或新增专门 finetune config

要求：

- 增加训练时 horizon truncation augmentation。
- 随机采样 `H`，例如 `[5, 30]` token。
- 只保留未来 `H` token 的有效轨迹。
- `token_mask` 中 H 之后置 0。
- 对应 frame-level mask 也必须一致。

注意：

- 这不是 `mask_ratio` 稀疏 waypoint 训练。
- `mask_ratio=1.0` 仍然可以保留，因为当前目标是密集轨迹点。
- horizon truncation 的语义是“只知道未来 H token”，不是“未来全程稀疏观测”。

## Do not do

- 不要同时引入 trajectory hot-swap augmentation。
- 不要同时把表示改成 local heading / body frame。
- 不要把 horizon truncation 和 mask_ratio 混在一起。

## Validation

训练前后都跑 Task 002 诊断矩阵。

重点看：

- `step_h20_xyz_gtroot`
- `step_h20_xyz_predroot`

## Done criteria

- horizon=20 的 stream_generate_step 明显接近 full trajectory 版本。
- web demo 中固定轨迹跟随效果有可见改善。

---

# Active Task 005

## Task

如果 pred-root closed loop 是主要掉点来源，引入 relative-to-current-root 训练增强。

## Status

`pending`

## Problem

如果诊断结果显示：

- `step_h20_xyz_gtroot` 尚可
- `step_h20_xyz_predroot` 明显变差

说明主要问题是闭环中预测 root 漂移导致后续轨迹条件不断偏移。

这时需要让模型训练时见过“以当前 root 为 anchor 的未来相对轨迹”。

## Required changes

修改目标：

- `FloodNet/utils/traj_batch.py`
- dataset / collate 相关路径
- finetune config

要求：

- 增加 random anchor augmentation。
- 训练时随机选择 anchor token / frame。
- 轨迹位置转成相对 anchor 的 `dx/dz`。
- heading 语义需要明确：
  - 第一阶段可以保留 world heading。
  - 后续再考虑 local heading。

推理侧：

- web demo 继续用当前预测 root 构造未来目标。
- 不建议直接删除 anchor-subtract；更合理的是让训练覆盖该分布。

## Do not do

- 不要一步到位改成复杂 body-frame trajectory。
- 不要和 hot-swap augmentation 混在同一个实验里。

## Validation

重新跑：

- `step_h20_xyz_gtroot`
- `step_h20_xyz_predroot`

看 pred-root 与 gt-root 的差距是否缩小。

## Done criteria

- pred-root 闭环不再显著劣于 gt-root。
- web demo 中长时间拖拽目标轨迹时，生成 root 能持续朝目标方向运动。

---

# Active Task 006

## Task

在固定轨迹流式跟随稳定后，再做 trajectory hot-swap / mid-stream update 训练与评测。

## Status

`pending`

## Problem

最终目标不是只在开头给一条完整轨迹，而是 web demo 中途更新轨迹后，模型能像文本切换一样平滑适应新目标。

这比固定轨迹跟随更难，因为中途切换会引入几何不连续。

## Required changes

### 1. 增加 hot-swap 诊断模式

扩展：

- `FloodNet/eval/diagnose_stream_control.py`

新增模式：

- `step_h20_hot_swap_gtroot`
- `step_h20_hot_swap_predroot`

要求：

- 在固定 token index 切换目标 polyline。
- 输出切换前后 ADE/FDE。
- 单独记录切换点附近的 root velocity / jerk。

### 2. 训练增强

在确认固定轨迹流式跟随稳定后，再考虑：

- trajectory splice augmentation
- mid-stream target replacement augmentation
- switch boundary smoothing / transition loss

## Do not do

- 不要在固定轨迹还没跑稳时做 hot-swap 训练。
- 不要把 hot-swap 失败误判成基础轨迹控制失败。

## Validation

- hot-swap 前后目标轨迹都可视化。
- 切换后 root 方向能在合理延迟内转向新轨迹。
- 切换点附近没有明显爆炸或停滞。

## Done criteria

- web demo 中途更新轨迹后，人物能在有限延迟后转向新目标。
- 轨迹热切换效果可通过固定样本诊断脚本复现。

---

## 当前推荐执行顺序

严格按下面顺序做：

1. `Active Task 001`：metric 镜像 web demo
2. `Active Task 002`：固定样本诊断矩阵
3. `Active Task 003`：判断并修 stream 推理路径
4. `Active Task 004`：必要时做 horizon finetune
5. `Active Task 005`：必要时做 relative-root finetune
6. `Active Task 006`：最后做 hot-swap

不要跳过 Task 001 / Task 002。

---

## 第一轮建议样本

优先使用一个确定样本，例如：

- `HumanML3D/000021`

要求：

- 固定 ckpt
- 固定文本
- 固定 seed
- 固定 VAE
- 每次输出完整 artifact

第一轮只需要一个样本跑通诊断矩阵。之后再扩展到：

- HumanML3D 多样本
- BABEL 多文本多动作样本
- web demo 手工轨迹

---

## 暂不处理

以下内容暂时不要放进当前任务：

- 大规模 BABEL pipeline tests
- Text2Humanoid 跨仓集成测试
- MakeTrackingEasy / motion_tracking downstream 验证
- web demo UI 美化
- sim2sim 最终 demo

这些都应在 FloodNet streaming trajectory control 诊断完成后再继续。

