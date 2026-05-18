# BABEL Long-Session Web-Demo-Like Evaluation TODO

本文档描述后续要做的 BABEL 长程评测入口。目标是尽量模拟 `web_demo` 的长时间流式推理行为，评估多文本、多动作、轨迹变化下的生成效果，并导出可检查的 metric 与视频。

## 背景

当前已有两类相关能力：

- `eval/eval_stream_metrics.py`
  - 已支持 `BabelDataset.build_present_segment_eval_samples()`
  - 可以把 `9797_1 -> 9797_2 -> 9797_3` 合并成一个长样本 `9797`
  - 可以用同一个 `stream_generate_step()` 状态连续跑完整序列
  - 已有长程 streaming metric 和 `stream_feature.npy` / `gt_feature.npy` 导出

- `eval/diagnose_stream_control.py`
  - 已支持更接近 `web_demo` 的单样本诊断
  - 已验证 `timestamped_gt_h20` 可以把 pred-root 轨迹问题恢复到接近 `gtroot_h20`
  - 已支持 `duration_waypoints_s0.05_h20` 这种“HumanML3D 20 FPS waypoint 输入”近似 web demo 轨迹模式

当前缺口：

- `eval_stream_metrics.py` 的长程模式更像 GT suffix baseline，不是严格的 web-demo-like trajectory update。
- `diagnose_stream_control.py` 的 timestamped / duration waypoint 诊断目前主要是单 clip，不是 `9797_1 -> 9797_2 -> 9797_3` 长 session。
- 还缺一个统一入口，可以：
  - 拼接 BABEL present segments
  - 按时间切换 text
  - 按 web demo 方式持续构造 future trajectory
  - 输出分段 metric
  - 渲染长程视频和轨迹对比视频

## 目标

新增一个长程 BABEL web-demo-like evaluator，建议入口：

```bash
eval/eval_babel_webdemo_long.py
```

核心目标：

- 用 `9797_1 -> 9797_2 -> 9797_3` 这样的 present-segment group 构造一个连续长 session。
- 每个 token 调一次 `stream_generate_step()`，保持同一个 history window / rolling latent buffer。
- 按 BABEL text segment 的时间边界自动切换当前文本，模拟长期 `update_text`。
- 按 timestamped trajectory plan 构造每一步未来 `H` token 轨迹，模拟长期 `update_traj`。
- 导出全局和分段 metric。
- 渲染预测 motion 视频、GT 视频、root trajectory 对比视频。

## web-demo-like 的严格定义

这里的 `web-demo-like` 不是指启动 Flask、用浏览器真实点击，而是指评测路径必须复现 `web_demo/model_manager.py` 的核心在线推理语义：

- 整个长 session 只调用一次 `model.init_generated(...)`。
- 每个 token 只调用一次 `model.stream_generate_step(...)`。
- VAE 使用 `stream_decode(...)` 累积输出，不允许每步重新离线 decode 全段。
- `history_length`、`traj_horizon_tokens`、`token_dt` 与 web demo 配置一致。
- text 在 BABEL 标注边界上切换，等价于用户长程过程中多次 `update_text`。
- trajectory 每一步根据当前 `commit_index` 重新采样未来 horizon，等价于 web demo 长程过程中持续使用目标轨迹 plan。
- 第一版可以不模拟真实 UI 的鼠标绘制密度，但必须模拟当前 web demo 默认的时间语义：
  - dense waypoint 默认 `waypoint_dt = 0.05s`
  - token 时间步默认 `token_dt = 0.20s`
  - 每次 query 未来 `H = 20` 个 token

这个定义很重要：如果只用 `generate()` 或一次性 `stream_generate()`，不能暴露 web demo 长时间闭环中的 drift、text switch 和 trajectory resampling 问题。

## 第一轮样本：BABEL 9797

第一轮建议固定只做 `9797`，因为它天然包含长程动作和文本切换：

```text
9797_1: stand -> walk forward
9797_2: walk forward -> trip
9797_3: trip -> stand
```

评测时要把三个 present segments 合并成一个连续 session：

```text
9797_1 -> 9797_2 -> 9797_3
```

不要把它们当成三个独立样本分别 reset model。这个测试的价值正在于观察同一个 `stream_generate_step` 状态跨文本、跨轨迹、跨动作阶段是否稳定。

## 非目标

本任务不做：

- 不做训练或 finetune。
- 不改 `stream_generate_step()` 核心推理。
- 不改 BABEL 数据集格式。
- 不做真实浏览器交互自动化。
- 不把 web demo UI 逻辑复制到 eval 里。

评测目标是“web-demo-like inference”，不是启动 Flask 后通过浏览器点击。

## 数据流

### 输入

至少支持两种输入方式：

```bash
--sample_ids 9797_1,9797_2,9797_3
```

或：

```bash
--base_name 9797
--meta_paths /path/to/test_min_processed.txt
```

推荐优先实现 `--sample_ids`，更明确、方便复现。

### 长 session 构造

对每个 sample id：

- 读取 `BABEL_streamed/motions/<id>.npy`
- 读取 `BABEL_streamed/TOKENS_.../<id>.npy`
- 读取 `BABEL_streamed/texts/<id>.txt`

拼接：

- `feature = concat(features)`
- `token = concat(tokens)`
- `traj = extract_root_trajectory_263(feature)`
- `text_data` 的 `f_tag / to_tag` 加上前面 clip 的时间 offset
- 重新计算：
  - `feature_text_end`
  - `token_text_end`

这部分可以复用 [datasets/babel.py](../datasets/babel.py) 里的：

- `build_present_segment_eval_samples()`
- `_merge_present_segment_group()`
- `_merge_segment_text_data()`

如果直接复用 dataset 复杂度更低，就优先复用；不要重复造一套不一致的合并逻辑。

## 推理模式

Evaluator 至少实现下面 4 个模式。

### 1. `gt_suffix`

用途：长程上限 baseline。

行为：

- 每个 step 使用 dataset 的 GT suffix trajectory。
- 和 `eval_stream_metrics.py` 当前 `stream_generate_step` 路径一致。

意义：

- 如果这个模式不好，说明长程 `stream_generate_step` / 多文本切换本身有问题。
- 如果这个模式好，而 web-demo-like 模式差，问题在 trajectory construction。

### 2. `timestamped_gt_plan`

用途：验证“保留真实时空轨迹”在长程下是否稳定。

行为：

- 从拼接后的 GT root 构造 `(t, x, y, z)` timestamped trajectory。
- 每个 commit index 采样：

```text
query_times = commit_idx * token_dt + [0, ..., H-1] * token_dt
future_traj = interpolate(timestamped_traj, query_times)
```

默认：

```text
motion_fps = 20
token_dt = 0.20
horizon_tokens = 20
```

意义：

- 这是最接近“数据集中有完整时空轨迹输入”的 web-demo-like 模式。

### 3. `duration_waypoints`

用途：模拟用户给稠密 waypoint，但不显式给每点时间戳。

行为：

- 从 GT root 按 `waypoint_dt` 抽点。
- 默认先测：

```text
waypoint_dt = 0.05
```

- 这些点作为 web demo 输入。
- 时间语义按当前 web demo 默认：

```text
waypoint_i time = i * waypoint_dt
```

意义：

- 这是当前 web demo 默认策略的离线复现。
- 当 waypoint 来自 20 FPS 数据并且 `waypoint_dt=0.05` 时，它与“每帧一个 timestamped waypoint”在时间轴上等价。
- 当 waypoint 是用户手绘稀疏点时，它不等价于真实动作时间，只代表“按点序均匀推进”的 UI 语义。
- 如果视频观感好，即使 ADE 不低，也说明匀速过程符合用户控制预期。

### 4. `no_traj`

用途：验证轨迹条件是否真的在长程中起作用。

行为：

- 每步只传当前 text，不传 trajectory。

意义：

- 如果 `no_traj` 和有轨迹差不多，说明轨迹条件没发挥作用。
- 如果有轨迹明显更好，说明控制链路有效。

## Text 切换语义

长程 evaluator 使用 token-level text schedule：

```text
current_text = text_schedule.get_text_for_commit_index(commit_idx)
```

这等价于自动在 BABEL 标注边界上做 `update_text`。

需要在 artifact 中导出：

- `text_schedule.json`
- 每段：
  - `text`
  - `start_frame`
  - `end_frame`
  - `start_token`
  - `end_token`

这样视频和 metric 可以按动作段复盘。

## Trajectory 切换语义

第一版不做 mid-session 轨迹重画，只做“一个长 session 的 timestamped plan”。

原因：

- BABEL 的真实数据本身就是一条连续时空轨迹。
- 先验证长程时空 plan 是否能被模型跟住。
- 真实用户多次 `update_traj` 属于下一阶段 hot-update 测试。

第二版再加：

- 在每个 BABEL segment 边界模拟一次 `update_traj`
- 每次只给从当前 segment 开始的 future timestamped plan
- 对比 full-plan vs segment-update plan

## Metrics

### 全局 metric

至少输出：

- `root_ADE`
- `root_FDE`
- `path_length_pred`
- `path_length_gt`
- `path_length_ratio`
- `root_speed_mae`
- `root_speed_corr`
- `stream_root_jump_mean`
- `stream_root_jump_max`
- `stream_joint_jump_mean`
- `num_boundaries`

### 分段 metric

按 BABEL text segment 输出：

- `segment_text`
- `start_frame`
- `end_frame`
- `root_ADE`
- `root_FDE`
- `path_length_pred`
- `path_length_gt`
- `root_speed_mae`

目的：

- 定位长程问题发生在 `stand -> walk`、`walk -> trip`，还是 `trip -> stand`。

### Trajectory metric

额外输出：

- `distance_to_target_path_mean`
- `distance_to_target_path_max`
- `time_aligned_root_error_mean`
- `time_aligned_root_error_final`

说明：

- `ADE/FDE` 是 time-aligned metric。
- 但 web demo 用户更关心“是否沿目标路径走”，所以需要 path-distance metric。

## Artifact 结构

建议输出目录：

```text
outputs_babel_webdemo_long/
└── <timestamp>_<sample_group>_<mode>/
    ├── summary.json
    ├── config.yaml
    └── samples/
        └── 9797/
            ├── text_schedule.json
            ├── metrics.json
            ├── pred_motion.npy
            ├── gt_motion.npy
            ├── pred_root.npy
            ├── gt_root.npy
            ├── target_traj.npy
            ├── pred_motion.mp4
            ├── gt_motion.mp4
            ├── traj_compare.mp4
            └── traj_compare.png
```

## Video 输出

至少输出三类视频：

### 1. `pred_motion.mp4`

使用已有：

- `utils.visualize.render_single_video(...)`

输入：

- `pred_motion.npy`

### 2. `gt_motion.mp4`

同样用 `render_single_video(...)`。

用于观察 GT 和生成结果的动作差异。

### 3. `traj_compare.mp4`

2D XZ 轨迹对比视频。

内容：

- GT root path：绿色
- Pred root path：红色
- 当前 frame 的位置点
- text segment 边界竖线或颜色区间

可以先用 matplotlib + ffmpeg，复用 `diagnose_stream_control.py` 中已有的 root trajectory render 逻辑。

## 实施步骤

### Task 0：先跑现有长程 baseline，不写新代码

修改目标：

- 无

目的：

- 先确认现有 `eval_stream_metrics.py` 可以把 `9797_1 -> 9797_2 -> 9797_3` 合并成长 session。
- 这个结果作为后续 web-demo-like evaluator 的对照组。

建议命令：

```bash
printf "9797_1\n9797_2\n9797_3\n" > /tmp/babel_9797.txt

cd /home/yuankai/Text2Motion/FloodNet

python eval/eval_stream_metrics.py \
  --config configs/eval_babel_stream.yaml \
  --ckpt /home/yuankai/Text2Motion/FloodNet/outputs/step_460000.ckpt \
  --vae_ckpt /home/yuankai/Text2Motion/FloodNet/outputs/vae_1d_z4_step=300000.ckpt \
  --stream_mode stream_generate_step \
  --meta_paths /tmp/babel_9797.txt \
  --probe_tag babel_9797_long \
  --max_samples 1 \
  --save_feature_npy
```

应检查：

- 输出样本名是 `9797`，不是三个独立 reset 的样本。
- `text.txt` 中包含 `9797_1, 9797_2, 9797_3` 三段。
- `stream_feature.npy` 和 `gt_feature.npy` 长度接近拼接后的总长度。

### Task 1：复用/封装 BABEL present-segment 合并

修改目标：

- `eval/eval_babel_webdemo_long.py`
- 必要时从 `datasets/babel.py` 暴露小 helper

要求：

- 支持 `--sample_ids 9797_1,9797_2,9797_3`
- 输出一个 merged sample dict
- sample dict 字段与 `eval_stream_metrics.py` 的 `sample_batch` 兼容

验证：

```bash
python eval/eval_babel_webdemo_long.py \
  --config configs/eval_babel_stream.yaml \
  --sample_ids 9797_1,9797_2,9797_3 \
  --dry_run
```

应打印：

- merged name: `9797`
- segment names: `9797_1, 9797_2, 9797_3`
- feature length
- token length
- text schedule

### Task 2：实现 web-demo-like stream step runner

修改目标：

- `eval/eval_babel_webdemo_long.py`
- 可复用 `diagnose_stream_control.py` 中的 timestamped trajectory 采样 helper

要求：

- 每 token 调一次 `stream_generate_step()`
- 使用同一个 history window
- 使用 `StreamTextRolloutController`
- 支持模式：
  - `gt_suffix`
  - `timestamped_gt_plan`
  - `duration_waypoints`
  - `no_traj`

验证：

```bash
python eval/eval_babel_webdemo_long.py \
  --config configs/eval_babel_stream.yaml \
  --ckpt /home/yuankai/Text2Motion/FloodNet/outputs/step_460000.ckpt \
  --vae_ckpt /home/yuankai/Text2Motion/FloodNet/outputs/vae_1d_z4_step=300000.ckpt \
  --sample_ids 9797_1,9797_2,9797_3 \
  --mode timestamped_gt_plan \
  --save_feature_npy
```

应生成：

- `pred_motion.npy`
- `gt_motion.npy`
- `pred_root.npy`
- `gt_root.npy`
- `metrics.json`

### Task 3：补全 metric

修改目标：

- `eval/eval_babel_webdemo_long.py`
- 必要时新增 `metrics/long_stream.py`

要求：

- 输出全局 metric
- 输出每个 text segment 的 metric
- 输出 path-distance metric

验证：

检查 `metrics.json` 至少包含：

- `global`
- `segments`
- `stream_boundary`
- `trajectory`

### Task 4：视频渲染

修改目标：

- `eval/eval_babel_webdemo_long.py`
- 必要时新增 `eval/render_long_stream.py`

要求：

- 增加 `--render_video`
- 导出：
  - `pred_motion.mp4`
  - `gt_motion.mp4`
  - `traj_compare.mp4`
  - `traj_compare.png`

验证：

```bash
python eval/eval_babel_webdemo_long.py \
  --config configs/eval_babel_stream.yaml \
  --ckpt /home/yuankai/Text2Motion/FloodNet/outputs/step_460000.ckpt \
  --vae_ckpt /home/yuankai/Text2Motion/FloodNet/outputs/vae_1d_z4_step=300000.ckpt \
  --sample_ids 9797_1,9797_2,9797_3 \
  --mode duration_waypoints \
  --waypoint_dt 0.05 \
  --render_video
```

### Task 5：多模式一键对比

修改目标：

- `eval/eval_babel_webdemo_long.py`

要求：

支持：

```bash
--mode all
```

依次跑：

- `gt_suffix`
- `timestamped_gt_plan`
- `duration_waypoints`
- `no_traj`

输出一个总表：

```text
mode                   ADE    FDE    path_ratio    jump_mean
gt_suffix              ...
timestamped_gt_plan    ...
duration_waypoints     ...
no_traj                ...
```

## 推荐第一轮命令

第一轮实现新 evaluator 后只跑 `9797`：

```bash
cd /home/yuankai/Text2Motion/FloodNet

python eval/eval_babel_webdemo_long.py \
  --config configs/eval_babel_stream.yaml \
  --ckpt /home/yuankai/Text2Motion/FloodNet/outputs/step_460000.ckpt \
  --vae_ckpt /home/yuankai/Text2Motion/FloodNet/outputs/vae_1d_z4_step=300000.ckpt \
  --sample_ids 9797_1,9797_2,9797_3 \
  --mode all \
  --waypoint_dt 0.05 \
  --history_length 30 \
  --traj_horizon_tokens 20 \
  --render_video
```

## 第一轮结果应该怎么解读

建议按下面顺序判断：

1. 如果 `gt_suffix` 差，优先查长程 `stream_generate_step` / BABEL text schedule / segment merge，不要先怪 trajectory UI。
2. 如果 `gt_suffix` 好、`timestamped_gt_plan` 差，说明 timestamped 采样或 token/frame 时间映射有 bug。
3. 如果 `timestamped_gt_plan` 好、`duration_waypoints` 差，说明默认 web demo waypoint 时间语义不够表达真实时空速度，需要 UI 增加 duration 或 timestamp 支持。
4. 如果 `duration_waypoints` 视频观感好但 ADE 一般，说明用户控制目标更偏“路径形状/匀速过程”，不能只用 time-aligned ADE 下结论。
5. 如果 `no_traj` 和有轨迹模式差不多，说明长程中 trajectory condition 没有真正起作用，需要回到 FloodNet control 路径排查。

## Done Criteria

完成标准：

- 能把 `9797_1 -> 9797_2 -> 9797_3` 作为一个连续 session 跑完。
- 过程中只初始化一次 `model.init_generated()`。
- 文本按 BABEL segment 时间边界自动切换。
- 轨迹按 web-demo-like timestamped / duration-waypoint 方式持续采样。
- 输出全局 metric、分段 metric、轨迹 metric。
- 输出预测视频、GT 视频和轨迹对比视频。
- 可以用一条命令跑出 `gt_suffix / timestamped_gt_plan / duration_waypoints / no_traj` 对比。

## 后续扩展

第一版完成后，再考虑：

- segment 边界处显式模拟 `update_traj`
- segment 边界处显式模拟 `update_text`
- hot-update 后 history traj 重写策略
- 不同 `history_length` / `horizon_tokens` sweep
- 更多 BABEL long groups 自动批量评测
