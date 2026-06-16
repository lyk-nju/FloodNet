# Training Data Path And Module Cleanup

本文记录 LDF / RootRefiner 训练链路的当前设计，以及近期几项结构性改动的原因：

- 删除旧 `mask_ratio` 机制。
- LDF stream train 引入 `SampleCreator + online_encode` 的局部窗口链路。
- LDF `SampleCreator` 收敛为从 dataset batch 到 model forward batch 的统一入口。
- LDF training step / self-forcing / conditioning 从 `DiffForcingWanModel` 迁到
  `utils/training/ldf`，模型层只消费 prepared condition。
- LDF resume ckpt 时明确 `resume_reset_optimizer` 对 optimizer / scheduler / LR
  的恢复语义。
- RootRefiner 不再使用独立的 `datasets.humanml3d_refiner`，统一消费
  `datasets.humanml3d.HumanML3DDataset`。
- 训练代码按 `utils/training/ldf` 和 `utils/training/root_refiner` 分支整理。

目标是明确数据层、采样层、训练层的职责边界，避免后续把模型问题和数据处理问题混在
一起解释。

## 背景

加入 RootRefiner 之前，LDF 训练里曾经有一个 `mask_ratio` 机制：dataset 在
token timeline 上随机保留一部分 trajectory token，再展开成 frame-level
`traj_mask / traj_cond_mask / traj_loss_mask`。它最初用于模拟“部分轨迹可见”的
控制任务。

RootRefiner 引入后，LDF 的职责发生了变化：

- RootRefiner 负责规划 root 7D 条件。
- LDF 负责在给定 root/traj condition 下生成动作。
- Stream train 关注的是时间窗口、history/horizon、local frame 对齐和 latent
  来源，而不是在 dataset 层随机丢弃 trajectory 条件。

因此，dataset 级别的随机 `mask_ratio` 不再是必要功能，反而会混入非模型因素：

- train/val/test 的 trajectory condition 可能因为配置不同而语义不一致。
- `mask_ratio=1.0` 实际只是 fake 开关，容易让人误以为还存在有效 ablation。
- 当 stream train 还有 horizon mask、window-local mask、rootplan mask 时，
  dataset 再随机造洞会让问题定位变复杂。

## 当前 Mask 语义

### 已删除：dataset random sparsity mask

以下旧逻辑已经删除：

- `data.mask_ratio`
- `data.val_mask_ratio`
- `sample_token_mask()`
- dataset 内部从 token mask 展开 frame mask 的随机稀疏逻辑

当前 `HumanML3DDataset / BabelDataset / GenerateDataset` 的输出规则是：

- 对有效 motion frame 直接输出 dense mask。
- `traj_mask / traj_cond_mask / traj_loss_mask` 在有效长度内全为 1。
- dataset 不再输出 `token_mask`。

这意味着普通 LDF train/val/test 不会再被 dataset-level 随机轨迹遮挡影响。

### 保留：真实有效的 runtime/training mask

删除 `mask_ratio` 不等于删除所有 mask。以下 mask 仍然有意义，不能混淆：

- padding/valid-length mask：表示 batch padding 后哪些帧有效。
- horizon mask：表示 stream train 中当前 horizon 之后不可见。
- window-local mask：表示当前局部窗口内哪些 local frames 有 GT。
- rootplan/runtime mask：表示实际 root condition 覆盖范围。
- ControlNet attention/projection 里的 `traj_token_mask`：用于把 frame mask 聚合到
  token 维度。

这些 mask 来自实际时间边界或条件覆盖范围，不是旧的随机稀疏增强。

## Stream Train 当前链路

### 1. SampleCreator 只负责采样契约

`utils/training/ldf/sample_creator.py` 里的 `SampleCreator` 是 stream train 的采样入口。
它从 token-space 决定：

- `global_start_tokens`
- `local_start_tokens`
- `latent_tokens`
- `traj_tokens`
- `global_start_frames`
- `latent_frame_lengths`
- `traj_frame_lengths`

核心原则：

- 先在 token-space 确定采样窗口。
- 再由统一的 token/frame 规则推出 motion-space clip。
- 不在 dataset 内部做另一个独立的随机 mask 或隐式裁剪规则。

### 2. Stream train 固定使用 online encode

旧版 stream train 曾经支持 `precomputed_slice`：直接从离线 VAE latent 里按窗口切片。
这条路径后来被移除，因为它会把“完整原始样本编码得到的 latent slice”和“以当前窗口为
origin 的 local motion-space GT”混在一起，容易让 MSE/body auxiliary/control loss 出现
非模型因素导致的异常。

当前 stream train 固定使用 motion-space clip 在线编码：

1. 根据 `global_start_tokens` 找到 `token_start_frame(S)`。
2. 从 raw HumanML3D 263D motion 中截取 local clip。
3. clip 长度使用 `num_frames_for_tokens(N)`，也就是 `4N-3` 的 causal VAE local
   prefix 长度。
4. 把这段 clip 当作一条新的 local sample。
5. 对 local clip 做 VAE encode，得到当前训练用 latent。
6. local GT 7D 也从同一段 local clip recover，因此 x/z/yaw 从 local origin 对齐。
7. body auxiliary loss 使用 `local_decode`，不再拼 full-prefix。

这条路径更符合“从任意 stream window 开始训练”的语义，也避免离线 latent slice 与
local motion-space GT 之间出现隐式坐标/时间分布错配。

### 3. force_start_token_zero 是 debug/对照开关

`force_start_token_zero=true` 会强制 sampled window 从 token 0 开始。它用于判断问题
是否来自“首帧不是原样本第 0 帧”带来的分布变化。

它不是默认训练目标。正常 stream train 应该允许 `global_start_tokens > 0`，否则模型
只学习 full-prefix/从头开始的情况，不能覆盖真实流式中间窗口。

## 为什么旧 Stream Train 会出现不稳定

旧链路里最容易混入问题的是：

1. 采样窗口从中间 token 开始。
2. 训练仍使用离线 latent slice。
3. GT/root 监督被 local canonicalization 到 sampled origin。
4. 但离线 latent 本身来自完整原始样本编码，不一定等价于“把这段 motion 当新样本重新
   encode”。

这种情况下，latent 语义和 local GT 语义可能不完全一致。表现上可能是：

- `mse_step` 或 control/body auxiliary loss 经常变大。
- 固定 horizon 后 loss 对采样窗口非常敏感。
- `force_start_token_zero` 后异常明显减少，因为此时 sampled origin 回到原始样本起点。

`online_encode` 的设计就是为了消除这个非模型因素：latent 和 local GT 都来自同一个
local motion clip。

## 为什么删除 mask_ratio

删除 `mask_ratio` 的理由是：

1. RootRefiner 后，trajectory 稀疏化不再属于 LDF dataset 的职责。
2. `mask_ratio=1.0` 只是 fake 功能，保留配置会误导后续实验。
3. 随机稀疏 mask 会和 horizon/window/rootplan mask 混在一起，导致训练异常难定位。
4. val/test 本来就应该尽量反映真实条件输入，不应该被 dataset 随机遮挡。
5. 当前 stream train 的主要变量应该是：
   - sampled history/horizon
   - online/local encode 对齐
   - body auxiliary loss
   - corruption / self-forcing 策略

所以 dataset 层只保留 dense valid mask，其他可见性由 stream train/runtime 显式控制。

## 当前推荐配置

正常训练建议：

```yaml
ldf_training:
  formulation: windowed
  window_policy: rolling
  context_tokens: 30
  force_start_token_zero: false
  window_sampling:
    enabled: true
    history_tokens_min: 0
    history_tokens_max: auto
    horizon_tokens_min: 5
    horizon_tokens_max: 25
```

对照实验可以用：

```yaml
ldf_training:
  force_start_token_zero: true
```

它应被视为 debug/ablation，不是最终默认目标。旧的 `stream_training` 训练配置入口和
`stream_training.latent_source` 已经不再是合法主路径；runtime 文件名里的 `stream_*`
只表示 rolling runtime，不表示另一套训练任务。

## Resume Optimizer 和 LR 语义

LDF self-forcing resume 有两个不同目标，不能混成一条逻辑：

- `resume_reset_optimizer=false`：继续上一次训练，optimizer / scheduler / LR 应该尽量
  和 checkpoint 保存时的状态一致。
- `resume_reset_optimizer=true`：只恢复模型权重和 EMA，optimizer / scheduler 从当前配置
  重新开始。

这里最容易出错的是 self-forcing 的 phase step 语义。比如从 `global_step=485000` 的
checkpoint 继续训练到 `trainer.max_steps=700000` 时，运行期 phase 长度是
`700000 - 485000 = 215000`。这个 phase 长度应该用于 self-forcing 的 progress 和
Trainer 的 runtime step 边界，但不一定应该改写 LR scheduler 的原始训练 horizon。

### `resume_reset_optimizer=false`

当保留 optimizer 时，Lightning 会从 checkpoint 恢复 optimizer state 和 scheduler
state。对于 diffusers cosine / `LambdaLR` 这类 scheduler，checkpoint 里会保存
`last_epoch`、`_last_lr` 等状态，但不会保存构造 lambda 时用到的闭包参数，例如
`num_training_steps`。

因此，如果 resume 时把 `lr_scheduler.params.num_training_steps` 从原来的
`700000` 改成 phase 长度 `215000`，就会出现“状态从旧 checkpoint 恢复、scheduler 闭包
却按新 horizon 构造”的半恢复状态，LR 会和上一次训练不一致，甚至出现跳变。

现在的规则是：

- `resume_reset_optimizer=false` 时，不改写 scheduler horizon。
- optimizer / scheduler state 由 Lightning 从 checkpoint 恢复。
- self-forcing 的 phase progress 仍然使用 phase 长度，保证 K schedule 和 runtime
  max steps 语义正确。

对应测试是：

```text
tests/test_ldf_resume_optimizer.py::test_preserving_optimizer_does_not_rewrite_scheduler_horizon
```

### `resume_reset_optimizer=true`

当明确重置 optimizer 时，checkpoint 里的 optimizer / scheduler state 不应该恢复。
`CustomLightningModule.on_load_checkpoint()` 会把 `optimizer_states` 和 `lr_schedulers`
设为空列表，让 Lightning 通过“key 存在”检查，但不加载旧 opt/sched 状态。

这时 scheduler 是按当前 resume phase 重新构造的，所以可以把 scheduler horizon 从
`trainer.max_steps` 改成 phase 长度。例如从 `485000` resume 到 `700000`，新的
`num_training_steps` 可以是 `215000`，这和“重新开始当前 phase 的 LR schedule”一致。

对应测试是：

```text
tests/test_ldf_resume_optimizer.py::test_resetting_optimizer_rewrites_scheduler_horizon_to_resume_phase
```

### LR 日志

self-forcing 使用 manual optimization，当前 step 的 LR 应该记录在
`scheduler.step()` 之前，否则日志里的 `lr` 会变成下一步的 LR。现在 `_self_forcing_step()`
会先保存 `lr_for_step`，完成 `optimizer.step()` 和 `lr_scheduler.step()` 后：

- `lr` 记录本 step 实际使用的 LR。
- `lr_next` 记录 scheduler 推进后的下一步 LR。

这样 resume 后看日志时，可以区分“本步实际训练 LR”和“下一步将使用的 LR”，避免把
日志时序误判成 LR 恢复错误。

## LDF 训练主链路重构

近期把 LDF 训练链路按职责重新分层，核心目标是让模型层不再解释 raw batch，也不再管理
训练窗口/self-forcing 的策略细节。

### SampleCreator 成为 batch creation 正门

`utils/training/ldf/sample_creator.py` 现在是 LDF 从 dataset batch 到 model forward batch
的统一入口：

- prefix-window batch：从 clip 起点采样 `latent=z[0:R]`，并构造
  `traj=tau[0:R+H]`。
- rolling-window batch：采样 token window，裁剪 raw 263D motion，在线 VAE encode，构造
  window-local 7D trajectory condition，裁剪 segmented text。
- `_sample_window()` / prefix sampling 只负责 token-space 窗口采样。
- `_create_online_batch()` 负责 stream 训练的 online encode batch 构造。

因此旧的 `model_batch.py`、`window_sampling.py`、`window_local.py` 不再作为独立主路径文件
保留。这样 batch 构造入口更接近我们的设计目标：`SampleCreator` 产出模型能直接消费的
训练 batch，而不是只产出一个中间 sampling plan。

### model_step 承接训练窗口、loss 和 x0 估计

`utils/training/ldf/model_step.py` 承接原来放在 `DiffForcingWanModel.forward()` 里的训练
step 逻辑：

- `run_model_step(model, batch)`：执行一次 LDF training step。
- `run_training_window(...)`：执行单个 diffusion training window。
- `_prepare_training_window(...)`：构造 noisy window、reference tensor 和 end indices。
- 两套 x0 估计公式和对应说明随训练 step 一起保留在 training 层。

模型的 `forward()` 保持可用，但训练窗口构造和 self-forcing 多步 rollout 的实现不再散落在
模型类里。这样 `DiffForcingWanModel` 更接近网络模块本身：负责 denoise、ControlNet、
CFG、text/traj embedding 资源，而训练策略在 `utils/training/ldf` 管理。

### Conditioning 显式接口化

新增 `utils/ldf_condition.py`，只定义模型边界能看到的 prepared condition contract：

- `LDFCondition`
- `LDFTrainingCondition`

训练侧由 `utils/training/ldf/conditioning.py` 把 raw training batch 转成
`PreparedCondition / LDFCondition`，包含：

- text dropout / text context 展开。
- traj dropout。
- `traj_emb / traj_seq_lens / traj_token_mask` 的组织。
- horizon/self-forcing 下的 trajectory mask/lens 语义。

推理侧由 `utils/inference/ldf_conditioning.py` 和
`utils/inference/stream_conditioning.py` 准备 offline generation / stream step condition。
这样模型层不再 import inference runtime，也不直接解析 stream direct 7D payload。

### DiffForcingWanModel 只保留模型职责

`models/diffusion_forcing_wan.py` 做了低风险瘦身：

- 删除已废弃的 `scheduled_sampling_prob` 相关逻辑。
- 删除训练专属的 self-forcing 配置字段。
- 删除模型内部的训练窗口/loss/x0 估计实现，改由 `model_step.py` 承接。
- 删除训练 condition 准备 helper，改由 `utils/training/ldf/conditioning.py` 承接。
- 保留 `_denoise_with_cfg`、`_controlnet_forward`、text encode cache 和 trajectory encoder
  模块，因为它们属于模型推理/网络调用资源。
- trajectory encoder 对外只暴露 `traj_encoder`，内部由
  `FrameTrajEncoder + TokenTrajEncoder` 组成。

当前新主线不再兼容旧 padded-horizon/full-latent checkpoint key。旧的
`local_traj_encoder.* / traj_encoder.*` 分裂结构已经收敛为：

```text
traj_encoder.frame_encoder.*
traj_encoder.token_encoder.*
```

这样训练、eval、runtime 都只依赖一个 trajectory encoder 边界，避免模型外部继续知道
frame-stage 和 token-stage 的内部拆分。

### LDF 训练文件归属

当前 LDF 训练分支主要文件：

```text
utils/training/ldf/
  sample_creator.py          # dataset batch -> model forward batch
  conditioning.py            # raw training batch -> LDFCondition
  model_step.py              # training forward/loss/x0 estimate
  self_forcing.py            # self-forcing rollout
  self_forcing_config.py     # self-forcing 配置解析
  losses.py                  # control/body auxiliary loss
  model_factory.py           # model factory / precomputed text embedding helpers
  config_validate.py         # LDF config validation
  validation_eval_runtime.py # validation/test probe dataloader helpers
```

不属于训练主链路的工具移到 `tools/`：

- `tools/ldf_body_ablation.py`

## 测试覆盖

相关测试应覆盖：

- LDF datasets 不再读取或输出 `mask_ratio` 相关随机 mask。
- dataset 输出 dense `traj_mask / traj_cond_mask / traj_loss_mask`。
- token-space sample 推导出的 motion-space start/length 正确。
- online encode 的 local clip encode 后 token 数必须等于采样的 token 数，否则 fail-fast。
- `online_encode` 的 body auxiliary loss 不走 full-prefix splice。
- `resume_reset_optimizer=false` 时不重写 scheduler horizon，保证恢复 optimizer 后
  LR 语义和 checkpoint 连续。
- `resume_reset_optimizer=true` 时允许把 scheduler horizon 重写成当前 resume phase。
- `DiffForcingWanModel` 不再保留训练窗口/conditioning wrapper 方法。
- `LDFCondition` 能被 train/eval/stream 路径显式构造并透传到模型边界。
- stream horizon 下 `traj_emb / traj_seq_lens / traj_token_mask` 的长度语义正确。

本轮与 `mask_ratio` 删除直接相关的验证：

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest \
  tests/test_ldf_dataset_masks.py \
  tests/test_dataset_7d_collate.py \
  tests/test_config_7d_rollup.py -q

/home/yuankai/.conda/envs/flooddiffusion/bin/python -m py_compile \
  datasets/humanml3d.py \
  datasets/babel.py \
  datasets/generate.py \
  utils/inference/buffer.py
```

## 训练目录重构

本轮把原来混在 `utils/training/` 和 `utils/refiner/` 下的训练代码整理成两个清晰分支：

```text
utils/training/
  lightning_module.py        # LDF/通用 Lightning 基类
  module_step.py             # 通用 step helper
  step_semantics.py          # ckpt/resume step 语义

  ldf/
    sample_creator.py
    conditioning.py
    model_step.py
    self_forcing.py
    self_forcing_config.py
    losses.py
    model_factory.py
    config_validate.py
    validation_eval_runtime.py

  root_refiner/
    dataset_builder.py
    batch_builder.py
    sample_creator.py
    sample_builder.py
    lightning_module.py
    losses.py
    path_condition.py
    path_feature_stats.py
    text_encoder.py
```

### 为什么拆成 ldf/root_refiner 两个分支

LDF 和 RootRefiner 都使用 HumanML3D motion/text，但训练目标不同：

- LDF 训练 latent diffusion / self-forcing / body auxiliary loss。
- RootRefiner 训练 root 7D planner，包括 duration、waypoints、path condition。

旧结构里 LDF 的 `sample_creator/window_local/self_forcing` 和 RootRefiner 的
`sample_creator/path_condition/lightning_module` 分散在 `utils/training` 与
`utils/refiner` 两处，容易误判某个 helper 是通用逻辑还是某个模型专用逻辑。

现在的约定是：

- `utils/training/ldf/*` 只服务 LDF 训练。
- `utils/training/root_refiner/*` 只服务 RootRefiner 训练。
- 真正通用的代码保留在 `utils/training/` 顶层，例如 Lightning 基类、step 语义。

这样后续改 LDF stream train 不会误伤 RootRefiner，改 RootRefiner sample/path 逻辑也不
会污染 LDF 训练链路。

## 统一 HumanML3D 数据入口

RootRefiner 训练不再消费 `datasets.humanml3d_refiner.py`。当前目标是统一由
`datasets.humanml3d.HumanML3DDataset` 读取 HumanML3D/BABEL 原始数据，再由训练分支
决定如何构造模型输入。

### 当前 RootRefiner 链路

```text
datasets.humanml3d.HumanML3DDataset
  -> utils.training.root_refiner.dataset_builder.build_root_refiner_dataset
  -> RootRefinerDataset
  -> RootRefinerBatchBuilder
  -> RefinerSampleCreator + RefinerSampleBuilder
  -> utils.training.root_refiner.collate_fn
  -> RootRefinerLightningModule
```

配置也对应改成：

```yaml
data:
  target: datasets.humanml3d.HumanML3DDataset
  collate_fn: utils.training.root_refiner.collate_fn
```

### 为什么删除 `datasets.humanml3d_refiner.py`

保留单独的 RootRefiner dataset 会让数据层承担训练策略：

- dataset 里采样 full/sliding。
- dataset 里决定 num_tokens/path_mode/offset_start。
- dataset 里直接构造 target/path/history。

这和 LDF 的新链路不一致。统一入口后，`datasets/humanml3d.py` 只负责读原始样本：

- motion 263D / feature length
- text / text_all
- name/raw_id/dataset/split metadata
- LDF 需要的基础 traj 字段

RootRefiner 专用的采样和张量构造转移到 `utils/training/root_refiner`：

- `sample_creator.py`：只决定采样计划，例如 full/sliding、anchor、num_tokens、path_mode。
- `sample_builder.py`：只根据 raw sample + sampling plan 构造 history/target/path tensor。
- `batch_builder.py`：把 raw dataset 包装成 RootRefiner training dataset，并提供 collate/fixed sample。
- `dataset_builder.py`：从配置构造 `HumanML3DDataset -> RootRefinerDataset`。

这样后续 LDF 和 RootRefiner 都从同一套原始数据读取逻辑出发，差异只存在于训练分支。

## Inference/Visualization 目录迁移

推理和可视化相关 utils 也做了目录整理：

```text
utils/inference/
  buffer.py
  commit.py
  glue.py
  rollout.py
  root_plan.py
  root_refiner.py
  timeline.py
  trajectory.py

utils/visualization/
  skeleton.py
  video.py
```

旧的顶层 runtime 文件，例如 `utils/root_plan.py`、`utils/runtime_timeline.py`、
`utils/stream_rollout.py`、`utils/traj_stream_buffer.py`，迁移到 `utils/inference/`。
旧的 `utils/render_skeleton.py`、`utils/visualize.py` 迁移到 `utils/visualization/`。

这部分目的是让 runtime streaming / RootPlan / web demo / eval 共用同一套推理 helper，
避免同名逻辑散落在顶层 `utils`。

## 提交注意事项

这轮是大规模移动文件。提交时必须使用能记录删除和新增的 add 方式，例如：

```bash
git add -A
```

否则容易只提交旧文件删除，而漏掉新目录：

- `utils/training/ldf/`
- `utils/training/root_refiner/`
- `utils/inference/`
- `utils/visualization/`

同时不要把实验输出目录加入提交：

- `eval/output_eval/`
- `outputs_stream_eval/`

如果需要提交文档，而 `docs/` 被 ignore，需要显式 `git add -f docs/change_reason.md`。

## 实验解读建议

如果后续出现 stream train loss 尖峰，优先按以下顺序排查：

1. online encode 主路径是否仍异常。
2. `force_start_token_zero=true` 是否显著改善。
3. 尖峰样本的 `global_start_tokens / horizon_tokens / active_left_tokens` 是否异常。
4. body auxiliary 的 `root_xz / heading / end_xz` 是哪一项爆掉。
5. 是否是模型面对中间 window local sample 的泛化问题，而不是数据处理问题。

不要再通过 `mask_ratio` 解释 LDF stream train 异常；该机制已经不在代码路径中。

## Windowed Formulation 和 Horizon 语义

新主线把 LDF training / eval / runtime 统一成 windowed formulation：

- latent side 只包含 history + active。
- trajectory side 包含 history + active + future horizon。
- 只要 `H > 0`，就允许 `traj_len > latent_len`。
- `LDFCondition.seq_len` 永远表示 latent segment 长度。
- `LDFCondition.attn_len` 只记录总可见条件长度，不能作为模型 latent `seq_len` 传入。

```text
prefix-window:
  latent = z[0:R]
  traj   = tau[0:R+H]

rolling-window:
  latent = z[B:R]
  traj   = tau[B:R+H]
```

旧 padded-horizon checkpoint 不在 `FloodNet` 新主线内兼容；需要评测旧 checkpoint 时，
应使用 `Floodold` 或历史分支。

### Training 的长度语义

prefix 和 rolling batch 都对应 latent 和 trajectory 分开 padding：

- `feature.shape[1] == max(latent_lengths)`
- `feature_length == latent_lengths`
- `traj_emb.shape[1] == max(traj_num_tokens)`
- `traj_seq_lens == traj_num_tokens`

也就是说，feature 不再为了 horizon 把 latent tail 补到 `latent_len + horizon`。
trajectory horizon 由单独的 `traj_emb / traj_seq_lens / traj_token_mask` 表达。

### Runtime / eval 的长度语义

`generate()`、`stream_generate()` 和 `stream_generate_step()` 都必须遵守同一个边界：

- `_denoise_with_cfg(..., seq_len=...)` 里的 `seq_len` 是 latent window 长度
  （offline generate 中是 generation latent buffer 长度，stream step 中是 `model_sl`）。
- `t_scaled.shape[1] == latent_len`。
- frame-aligned `text_context` 只展开到 `batch_size * latent_len`。
- future trajectory horizon 只通过 `traj_emb / traj_seq_lens / traj_token_mask`
  表达。

这个约束很重要。否则 runtime 会把 `latent_len + horizon` 误当成 latent segment 长度，
形成 train/runtime mismatch。

### Cross-attention 语义

frame-aligned text cross-attention 只作用于 latent segment：

```text
x = [latent_tokens || traj_tokens]
```

其中 latent tokens 可以 query text，traj tokens 是已知控制条件，不 query text。
因此 `WanCrossAttention` 不再要求 `x` 长度只能是 `L` 或 `2L`，而是允许
`L_lat + L_traj`，并只更新前 `L_lat` 个 latent token。

### 训练解读

新版 windowed formulation 与旧 full-sequence / padded-horizon 训练任务不是完全同一
任务。旧 checkpoint 可以作为权重初始化参考，但不能被视为语义完全兼容的 checkpoint；
新主线指标应以 windowed formulation 下重新训练或 finetune 的 checkpoint 为准。
