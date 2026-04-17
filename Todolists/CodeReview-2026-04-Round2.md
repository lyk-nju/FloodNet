# FloodNet 深度代码审查报告（2026-04 Round 2）

本文档记录基于论文《FloodDiffusion: Tailored Diffusion Forcing for Streaming Motion Generation》
对 FloodNet 项目进行第二轮深度审查时新发现的 bug 与不一致问题。

> 第一轮审查（含原始 `000021` 方向反转 bug）及修复已完成，参见 transcript 总结。
> 本文档仅记录**第二轮新发现**的问题。

---

## 📋 问题总览

| 编号 | 严重程度 | 问题简述 | 位置 | 状态 |
|------|---------|---------|------|------|
| K | 🔴 严重 | `update_test` 的 `seed_everything` 污染全局训练 RNG | `train_ldf.py:326-327` | 待修复 |
| G | 🟡 中等 | `_get_traj_seq_lens` 公式与数据集 `token_length` 不一致 | `diffusion_forcing_wan.py:455-466` | 待修复 |
| I | 🟡 中等 | `_stream_build_traj_emb` 的 `traj_features` 约定模糊（frame vs token） | `diffusion_forcing_wan.py:1351-1433` | 待文档化 |
| J | 🟡 中等 | `_stream_build_traj_emb` 的 `traj_features_buffer` 路径跳过 `LocalTrajEncoder` | `diffusion_forcing_wan.py:1415-1433` | 待文档化 |
| D | 🟢 轻微 | `_build_traj_emb` 当 `mask_frame` 缺失时 padding 位置不被显式置 0 | `diffusion_forcing_wan.py:389-412` | 可延后 |
| F | 🟢 轻微 | `on_train_batch_end` 的 EMA 日志分配大临时 tensor | `utils/lightning_module.py:155-160` | 可延后 |

---

## 🔴 Bug K：`update_test` 的 `seed_everything` 会污染全局训练 RNG

### 位置
`train_ldf.py:325-336`

### 代码
```python
def update_test(self, batch):
    # Fix seed before each test generation so diffusion noise is reproducible.
    seed_everything(self.cfg.seed)

    with self.ema.average_parameters([p for p in self.model.parameters() if p.requires_grad]):
        model_batch = batch.copy()
        ...
        output = self.model.generate(model_batch)
```

### 问题
`seed_everything()` 会**重置** Python、NumPy、PyTorch（CPU + CUDA）的全局随机数生成器。
每次调用 `update_test` 都会把全局 RNG 强制设回 `self.cfg.seed`，带来两个副作用：

1. **所有 test batch 的噪声完全相同**
   - Test set 里不同样本的扩散噪声是重复的；如果 test set 很大，FID 等指标会被"同种噪声"偏置。
   - 原意只是"每次运行可复现"，但实现为"每个 batch 都重置"，过度保守。

2. **污染训练 RNG → 连续训练 step 噪声相关**
   - Lightning 的调用顺序：`validation_step` → `on_validation_epoch_end` → 回到训练循环。
   - 如果训练循环里下一个 `training_step` 直接使用全局 `torch.randn_like`（`add_noise` 函数正是如此），那它的噪声将从**固定种子**开始生成，违反扩散训练的 i.i.d. 噪声假设。
   - `Scheduled Sampling` 也用 `np.random.rand() < self.scheduled_sampling_prob` 判断，受影响。
   - 长期后果：训练的梯度 bias 会累积，可能导致模型收敛到次优点或输出方差异常。

### 修复方案

**方案 A：保存/恢复 RNG 状态（推荐）**

```python
import random

def update_test(self, batch):
    # Save RNG state to avoid polluting the training RNG.
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_cpu_state = torch.random.get_rng_state()
    torch_cuda_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    seed_everything(self.cfg.seed)
    try:
        with self.ema.average_parameters(
            [p for p in self.model.parameters() if p.requires_grad]
        ):
            model_batch = batch.copy()
            model_batch["feature"] = batch["token"]
            model_batch["feature_length"] = batch["token_length"]
            if "token_text_end" in batch:
                model_batch["feature_text_end"] = batch["token_text_end"]
            self._copy_traj_fields_to_model_batch(batch, model_batch)
            output = self.model.generate(model_batch)
        # ... rest of update_test ...
    finally:
        # Restore RNG state so training continues with independent noise.
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_cpu_state)
        if torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(torch_cuda_state)
```

**方案 B：仅在 test epoch 开始时 seed 一次，不在每个 batch 里 reset**

将 `seed_everything` 移到 `on_test_epoch_start`（Lightning hook），然后在 `on_test_epoch_end` 里恢复 RNG。好处是 test 内部不同 batch 的噪声也不同，更接近真实分布。

### 优先级
**立即修复**。这是唯一可能在训练过程中悄悄降低模型质量的 bug。

---

## 🟡 Bug G：`_get_traj_seq_lens` 公式与数据集 `token_length` 不一致

### 位置
`models/diffusion_forcing_wan.py:455-466`

### 代码
```python
def _get_traj_seq_lens(self, x, seq_len, device):
    if "traj_features_length" in x and x["traj_features_length"] is not None:
        return (
            x["traj_features_length"]
            .to(device=device, dtype=torch.long)
            .clamp(min=0, max=seq_len)
        )
    if "traj_length" in x and x["traj_length"] is not None:
        # Causal: N tokens → 4*(N-1)+1 frames, so frames→tokens = (T-1)//4 + 1
        tl = x["traj_length"].to(device=device, dtype=torch.long)
        return ((tl - 1) // 4 + 1).clamp(min=0, max=seq_len)
    return None
```

### 问题
数据集（`datasets/humanml3d.py:process_token`）里 token 数计算用的是：

```python
token_len = (feature_length + 2) // 4 + 1
```

而 `_get_traj_seq_lens` 用的是：
```python
traj_seq_lens = (traj_length - 1) // 4 + 1
```

这两个公式在 `feature_length` 不是 `4k+1` 形式时不等价：

| feature_length | 数据集 token_length | 模型 traj_seq_lens | 差异 |
|---|---|---|---|
| 1 | 1 | 1 | ✓ |
| 2 | 2 | 1 | **-1** |
| 4 | 2 | 1 | **-1** |
| 5 | 2 | 2 | ✓ |
| 8 | 3 | 2 | **-1** |
| 9 | 3 | 3 | ✓ |

### 影响
- FlexTraj attention 基于 `traj_seq_lens` 屏蔽 padding token；当 `traj_seq_lens` 偏小，会**丢失最后一个 token 的轨迹约束**。
- HumanML3D 的 `feature_length` 多数经过 `window_length` crop，未必是 `4k+1` 形式 → 这个 bug 在大量样本上会触发（但只影响 1 个 token，影响有限）。

### 修复方案

直接用 `feature_length`（即 `token_length`）作为 traj 长度来源：

```python
def _get_traj_seq_lens(self, x, seq_len, device):
    # traj and token are aligned one-to-one in the dataset pipeline:
    # effective traj-token count == feature_length (= token_length).
    if "feature_length" in x and x["feature_length"] is not None:
        return (
            x["feature_length"]
            .to(device=device, dtype=torch.long)
            .clamp(min=0, max=seq_len)
        )
    if "traj_features_length" in x and x["traj_features_length"] is not None:
        return (
            x["traj_features_length"]
            .to(device=device, dtype=torch.long)
            .clamp(min=0, max=seq_len)
        )
    # Fallback for pure-traj inputs without feature_length.
    if "traj_length" in x and x["traj_length"] is not None:
        tl = x["traj_length"].to(device=device, dtype=torch.long)
        return ((tl + 2) // 4 + 1).clamp(min=0, max=seq_len)
    return None
```

### 优先级
建议修复（1 行代码，零风险）。

---

## 🟡 Bug I：`_stream_build_traj_emb` 的 `traj_features` 输入约定模糊（frame vs token）

### 位置
`models/diffusion_forcing_wan.py:1328-1433`

### 问题
同名字段 `x["traj_features"]` 在两个调用路径里的语义不同：

| 调用路径 | `traj_features` 语义 | 长度 |
|---------|---------------------|------|
| `forward` / `generate` | **frame-level**（20 FPS） | `4*(N-1)+1` frames |
| `stream_generate_step`（web demo 风格）| **token-level**（5 FPS） | `N` tokens |

`_stream_update_traj_buffers` 把输入直接拷贝到 `traj_features_buffer`（token-level 索引的 buffer），
**没有任何 shape 验证或文档说明**。

### 风险场景
如果某人从训练/离线 pipeline 抽出一段 `traj_features`（frame-level）直接喂给 `stream_generate_step`，
会发生以下错乱：
1. Buffer 按 token 单位索引写入，但数据是 frame 单位 → **时间轴错位 4 倍**
2. `_stream_build_traj_emb` 直接跑 `self.traj_encoder(feats)`（跳过 `LocalTrajEncoder`）
   → 把 frame-level 4D feats 当 token-level 4D feats 处理
3. 结果：轨迹条件**完全错乱**，模型输出会严重偏离期望轨迹

### 修复方案

**A. 添加 shape 校验（最小改动）**

```python
def _stream_update_traj_buffers(self, x, device):
    ...
    if "traj_features" in x and x["traj_features"] is not None:
        tf = x["traj_features"]
        ...
        if tf.dim() == 3 and tf.size(0) == self.batch_size:
            # Sanity check: in streaming mode, traj_features must be token-level
            # (one row per token, not one row per frame).
            # Acceptable lengths: exactly self.chunk_size or multiples thereof per call.
            if tf.size(1) > self.seq_len * 2:
                raise ValueError(
                    f"stream traj_features too long ({tf.size(1)}); "
                    f"expected token-level with length <= seq_len*2={self.seq_len*2}. "
                    f"Did you pass frame-level features by mistake?"
                )
            ...
```

**B. 文档化 API 契约（推荐配合 A）**

在 `stream_generate_step` 和 `stream_generate` 的 docstring 里明确：

```
Streaming inference expects **token-level** trajectory inputs:
    x["traj_features"]: (B, N_tokens, 4) where each row is one token (≈ 4 frames).
    x["token_mask"]:    (B, N_tokens) — optional per-token mask.

This differs from forward()/generate(), which accept **frame-level**
trajectory inputs and internally compress 4 frames per token via LocalTrajEncoder.
```

### 优先级
建议添加 assert + 文档。目前只 web demo 调用这条路径，暂未触发，但容易踩雷。

---

## 🟡 Bug J：`_stream_build_traj_emb` 的 `traj_features_buffer` 路径跳过 `LocalTrajEncoder`

### 位置
`models/diffusion_forcing_wan.py:1415-1433`

### 代码
```python
if self.traj_features_buffer is not None:
    key = ("feat", start_t, end_index, self._traj_stream_version)
    if self.use_traj_emb_cache and key in self._traj_emb_cache:
        return self._traj_emb_cache[key]
    feats = self.traj_features_buffer[:, start_t:end_index, :]
    mask = None
    if self.token_mask_buffer is not None:
        mask = self.token_mask_buffer[:, start_t:end_index]
    ...
    if mask is not None:
        feats = feats * mask.unsqueeze(-1).to(dtype=feats.dtype)
    emb = self.traj_encoder(feats)       # ← 直接 MLP 投影，没跑 LocalTrajEncoder
    ...
```

### 问题
这条路径假设 buffer 里的 features **已经是 token-level**（web demo 约定），所以跳过 `LocalTrajEncoder`。

但：
- `_build_traj_emb`（训练/离线 generate）**强制**过 `LocalTrajEncoder`（即使 `feats_frame.shape[1] == seq_len` 也只是恒等通过，保持一致）。
- `_stream_build_traj_emb` 的 `traj_buffer`（xyz）路径已经被修复（Task4 Bug 2 fix），会过 `LocalTrajEncoder`。
- 只有 `traj_features_buffer` 路径特殊，直接绕过 `LocalTrajEncoder`。

### 影响
- Web demo 当前能工作，因为它按 token-level 准备 `current_traj_features`，**不需要** `LocalTrajEncoder`。
- 但这意味着 **web demo 实际用到的 traj_emb 计算路径与训练时不同**：
  - 训练用 `LocalTrajEncoder(Conv1d→GELU→Conv1d, mean over 4 frames) + TrajEncoder(MLP)`
  - Web demo 直接 `TrajEncoder(MLP)`，没有 conv 聚合
- 如果 `LocalTrajEncoder` 在训练中学到了非平凡的 4-frame 聚合表征，web demo 推理就缺失了这一步。

### 修复方案

**方案 A：在 buffer 里存 frame-level 数据，stream 路径也跑 `LocalTrajEncoder`**

这需要修改 web demo，让它按 frame 速率喂入 `traj_features`（每 token 4 帧）。
好处：training/inference 严格对齐。

**方案 B（最小改动）：保持 web demo 按 token 速率工作，但在训练里让 `feats_frame.shape[1] == seq_len` 时也过 `LocalTrajEncoder`（跳过 reshape 部分）**

查看 `_build_traj_emb` 发现它在 `feats_frame.shape[1] == seq_len` 时直接 `feats_tok = feats_frame`，跳过 `LocalTrajEncoder`——**这与 `_stream_build_traj_emb` 的行为是一致的** ✓

也就是说：**当上层按 token 速率喂入时，`_build_traj_emb` 也不过 `LocalTrajEncoder`**。

**结论**：这不是严格的 bug，两条路径在"token-level 输入"情况下行为一致。但建议：
- 在 docstring 里明确两条路径的等价性
- 或者统一强制走 `LocalTrajEncoder`（哪怕 token-level 输入也用 kernel_size=1 的恒等 conv 处理一次）

### 优先级
低。当前行为一致，只是约定不够清晰。

---

## 🟢 Bug D：`_build_traj_emb` 当 `mask_frame` 缺失时 padding 位置不被显式置 0

### 位置
`models/diffusion_forcing_wan.py:389-412`

### 问题
当 `traj_mask` 和 `token_mask` 都未提供时，`mask_frame = None`，padding 位置的 `feats_frame` 保持原值（`pad_sequence` 默认填 0），**经 `LocalTrajEncoder` + `TrajEncoder` 后 bias 会让输出非 0**。

但因为 `traj_seq_lens` 正确标注了有效长度，attention 层会屏蔽这些位置 → **实际不影响输出**，只是计算上做了无用功。

### 影响
极小，只是 token-mask 做了两遍（一次在 token 内部，一次在 attention 层）。

### 修复方案
可以不修复。如果要修，在 `_build_traj_emb` 尾部按 `traj_seq_lens` 再乘一次 mask：

```python
# After: feats_tok = ... + TrajEncoder(...)
emb = self.traj_encoder(feats_tok)
if "feature_length" in x and x["feature_length"] is not None:
    fl = x["feature_length"].to(device=device, dtype=torch.long)
    idx = torch.arange(seq_len, device=device).unsqueeze(0)           # (1, seq_len)
    valid = (idx < fl.unsqueeze(1)).to(dtype=emb.dtype).unsqueeze(-1) # (B, seq_len, 1)
    emb = emb * valid
return emb
```

### 优先级
可延后。

---

## 🟢 Bug F：`on_train_batch_end` 的 EMA 日志分配大临时 tensor

### 位置
`utils/lightning_module.py:147-160`

### 代码
```python
def on_train_batch_end(self, outputs, batch, batch_idx):
    self.last_batch_end_time = time.time()
    self.ema.to(self.device)
    self.ema.update()
    if self.global_step % 100 == 0:
        self.log("ema_decay", self.ema.decay, sync_dist=False)
        with torch.no_grad():
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            model_params = torch.cat([p.flatten() for p in trainable])          # ~123M floats
            ema_params = torch.cat([sp.flatten() for sp in self.ema.shadow_params])
            avg_diff = torch.abs(model_params - ema_params).mean()
            self.log("ema_diff/avg", avg_diff, sync_dist=True)
```

### 问题
当 ControlNet 可训练（~123M 参数），两次 `torch.cat` + 一次 `abs - mean` 会分配约 **3 × 500MB = 1.5GB** 临时 tensor（bf16/fp32 视精度）。
虽然每 100 步才触发一次，但 peak memory 在 GPU 显存吃紧的场景下不友好。

### 修复方案（流式均值）

```python
def on_train_batch_end(self, outputs, batch, batch_idx):
    self.last_batch_end_time = time.time()
    self.ema.to(self.device)
    self.ema.update()
    if self.global_step % 100 == 0:
        self.log("ema_decay", self.ema.decay, sync_dist=False)
        with torch.no_grad():
            total_abs = torch.zeros((), device=self.device)
            total_count = 0
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            for p, sp in zip(trainable, self.ema.shadow_params):
                total_abs += (p.detach() - sp.to(p.device)).abs().sum()
                total_count += p.numel()
            avg_diff = total_abs / max(total_count, 1)
            self.log("ema_diff/avg", avg_diff, sync_dist=True)
```

### 优先级
可延后。仅在显存 peak 成为瓶颈时修复。

---

## 📂 遗留（非本次新增，已知）

| 问题 | 严重程度 | 位置 | 说明 |
|------|---------|------|------|
| DDP `find_unused_parameters=True` | 轻微（性能） | `train_ldf.py:615` | 改为 False 可去掉 autograd 图遍历开销；需验证无未使用参数 |
| Control loss 的 DDP 归一化偏差 | 轻微 | `train_ldf.py:_compute_control_loss_xz` | 每 rank 独立除以 `n_valid`，多 rank 下各样本权重不均匀；可用 `all_reduce` 同步 |
| L_mse 与 L_control 量级不完全对齐 | 轻微 | `diffusion_forcing_wan.py` + `train_ldf.py` | 可通过 `control_loss_weight` 调节 |
| `noise` prediction_type 的 ODE 公式非标准显式 Euler | 存疑但非错 | `diffusion_forcing_wan.py` | 继承自 FloodDiffusion；默认 `prediction_type="vel"` 不受影响 |
| 训练 `time_steps` 采样范围与推理略不一致 | 极小 | `diffusion_forcing_wan.py:forward` | 继承自 FloodDiffusion；端点差异很小 |

---

## ✅ 本次审查确认的正确行为（无需修改）

以下组件经过对照论文/设计文档的逐行验证，**实现正确**：

- **噪声调度** `β^k_t = clamp(1 + k/n_s - t, 0, 1)` 与论文 Eq. 一致（`_get_noise_levels`）
- **Active window** `[m(t), n(t))` = 最后 `chunk_size` 个 token，loss 仅在该窗口计算
- **Triangular noise schedule** 在 `forward` / `generate` / `stream_generate` / `stream_generate_step` 四处一致
- **Frame-wise text conditioning** `WanCrossAttention` 按 batch×time 扩展文本 context，`text_condition_list` 构建正确
- **ControlNet 残差零初始化**（`_zero_linear`）+ 从预训练骨干初始化（`on_load_checkpoint` 时机）
- **FlexTraj 自注意力 mask** `[latent || traj]` 段的可见性正确（latent 可见 traj；traj 只看自身；文本 cross-attn 只在 latent 段）
- **Separated CFG null branch** 现已真正无条件（`traj_emb_backbone=None`）
- **VAE frozen + eval mode** 通过 `train()` override 强制保持
- **EMA 仅跟踪 trainable params**（冻结骨干不进 shadow copy）
- **`_compute_control_loss_xz` 的 `latent_past.detach()`** 防止了首轮发现的梯度累积 bug
- **rectified flow 公式** `vel = x0 - ε`，`x0_hat = pred_vel + noise` 在 training/inference 一致
- **Causal VAE token-frame 映射** `token 0 → frame 0`，`token k ≥ 1 → frames [4k-3, 4k]` 在数据集、traj pipeline、control loss 里一致

---

## 🎯 推荐修复顺序

**P0（立即）**:
1. **Bug K**：修复 `seed_everything` RNG 污染（~10 行）

**P1（顺手）**:
2. **Bug G**：`_get_traj_seq_lens` 改用 `feature_length`（~1 行）
3. **Bug I**：在 `stream_generate_step` docstring 加 API 契约（文档）

**P2（下次维护）**:
4. **Bug J**：文档化或统一 `_stream_build_traj_emb` 与 `_build_traj_emb` 的行为
5. **Bug F**：EMA 日志改流式均值
6. **Bug D**：`_build_traj_emb` 尾部补一次 `feature_length` mask

---

*审查日期：2026-04-17*
*审查范围：FloodNet（基于 FloodDiffusion 的 ControlNet 轨迹控制实现）*
*审查依据：论文 arXiv:2512.03520、`PAPER_TO_CODE_FloodDiffusion.md`、`CONTROLNET_ROADMAP_轨迹控制.md`、`RECAP_轨迹控制实现复盘.md`*
