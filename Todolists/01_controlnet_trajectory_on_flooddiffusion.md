> **Note**：本文件为早期方案草案。**规格与实现以 [`target.md`](target.md) 与 [`Task5-loss.md`](Task5-loss.md) 为准。** 控制损失：**对整条预测 latent 做 VAE decode**，再在 **active window 对应帧段** 上算 `L_control_xz`（与扩散 MSE 的 last-`chunk_size` tokens 对齐），而不是「只对尾窗 token decode」。

## 目标

以 **FloodDiffusion / Diffusion Forcing（triangular schedule + active window）** 为基础，在 **Wan latent backbone** 中融合 **ControlNet 风格的零初始化残差分支**，实现稳定的 **轨迹控制**（优先 root trajectory：\(x,z, \cos\psi,\sin\psi\)），并配套 **motion space 显式监督**（decode 后计算控制损失）。

## 已有代码现状（关键事实）

- **Diffusion forcing triangular schedule 已实现**：`FloodNet/models/diffusion_forcing_wan.py::_get_noise_levels()` 注释与实现对应 FloodDiffusion 论文 \(\beta_t^k=\mathrm{clamp}(1+k/n_s-t,0,1)\)，其中 `chunk_size == n_s`。
- **active window loss 已实现**：训练 `forward()` 只在最后 `chunk_size` 个位置上算 MSE（`predicted_result[b][:, -self.chunk_size:, ...]`）。
- **轨迹条件通路已存在（FlexTraj）**：
  - `DiffForcingWanModel._build_traj_emb()` → `utils.traj_batch.build_traj_emb_from_batch(...)` → `traj_encoder` 输出 `traj_emb`。
  - `WanModel.forward(..., traj_emb=..., traj_seq_lens=...)` 会把 `traj_emb` 投影到 `dim` 后 **拼接为 self-attn 序列的后半段**（`latent||traj`），并且：
    - RoPE 对齐 latent/traj 同步时间（`rope_apply_concat_latent_traj`）
    - cross-attn 只更新 latent tokens（`cross_out[:, latent_pad_len:, :] = 0`）
    - head 只对 latent tokens 输出（`x = self.head(x[:, :seq_len], e)`）
  - `traj_in_proj` **已零初始化**（`nn.init.zeros_`），并且 traj segment 有专用 LoRA（self-attn 中 `traj_lora_rank` 分支）。
- **MotionLCM-style 控制辅助信息已预留**：当 `use_traj_cond` 且 `prediction_type in ("vel","x0")` 时，`DiffForcingWanModel.forward()` 会产出 `loss_dict["control_aux"]["pred_x0_latent_list"]` 供上层（`train_ldf.py`）decode 后算控制损失。

## 核心设计选择（结合 MotionLCM / MaskControl / OmniControl / FloodDiffusion）

### 1) ControlNet 应该插在哪里

插入点选择：**WanModel 的每个 Transformer block 的 hidden states**（对 latent tokens 注入残差），并保持：

- **主干保持不动**：不改 triangular schedule、不改 active window 更新、不改双向/非因果 self-attn（`causal=False`）。
- **zero-init residual injection**：ControlNet 分支输出 residual，经 zero-linear 初始全 0，训练初期主干行为与原模型一致（MotionLCM/MaskControl 的稳定性来源）。

### 2) 轨迹条件用什么

第一版只做 root trajectory（与你现有 `traj_features` 4D 一致）：

- 输入：token-level `traj_features`（\(x,z,\cos\psi,\sin\psi\)），mask 用 `traj_mask_token`
- 编码：`TrajEncoder`（已存在）→ `traj_emb`
- 注入：继续使用 FlexTraj（latent||traj）机制，让 ControlNet 与主干共享同一套 traj token 表示（避免再做一套 cross-attn/adapter）。

### 3) 必须有 motion space 的显式控制监督

按 MotionLCM/MaskControl/OmniControl 的共同规律：

- 仅在 latent/velocity 回归上“希望模型自己学会遵循轨迹”往往不稳。
- 训练时必须 decode 到 motion space，提取 root/joint 的 **global** 轨迹再监督。

在工程上：复用现有 `control_aux["pred_x0_latent_list"]`，在 `train_ldf.py` 中 **对完整 latent 序列 `vae.decode`**，再 **仅在 active window 帧段**（及 `traj_mask`）上计算 `L_control_xz`（与 `target.md` 规则 4 一致）。

## 需要实现的内容（按文件拆分）

### A. Wan backbone 增加 “接收 ControlNet residual” 的接口

1. 修改 `FloodNet/models/tools/wan_model.py`

- **改动 1**：`WanModel.forward(...)` 增加可选参数：
  - `controlnet_residuals: list[torch.Tensor] | None`
  - 约定每层 residual shape：
    - **只作用 latent tokens**：`(B, seq_len, dim)`（推荐，最清晰）
    - 或作用 concat 序列：`(B, 2*seq_len, dim)`（不推荐，traj tokens 本身应视为“条件”，不被主干/控制分支去改写）

- **改动 2**：在循环 blocks 时注入 residual：
  - 注入位置建议在每个 `WanAttentionBlock` 输出之后（block-level residual），即：
    - `x = block(x, ...)`
    - `if controlnet_residuals is not None: x[:, :seq_len, :] += controlnet_residuals[i]`
  - 如果 `latent_pad_len is None`（未启用 traj），仍可注入 `x += residual`

- **改动 3**：为了避免 residual 改写 traj tokens，强制把 residual 只加到 `x[:, :seq_len]`。

### B. 新增 ControlNet 分支：复制 blocks + zero-init 输出 residual

2. 新增 `FloodNet/models/tools/wan_controlnet.py`

实现 `WanControlNet(nn.Module)`：

- 输入与主干一致（与 `WanModel.forward` 对齐）：
  - `x`（noisy latent tokens）
  - `t` / `noise_level` time embedding输入
  - `context`（text tokens）
  - `seq_len`
  - `traj_emb` / `traj_seq_lens`（复用 FlexTraj）
- 内部：
  - 复制 `WanModel` 的 embedding/time projection/blocks 的必要子集
  - 每个 block 之后用 `zero_linear_i` 输出 residual（只输出 latent 部分）：
    - `res_i = zero_linear_i(h_latent)`，并 `nn.init.zeros_` 初始化
- 输出：
  - `residuals: list[Tensor]`，长度 `num_layers`

实现细节建议：

- **最小实现**：直接复用 `WanAttentionBlock` 类；ControlNet 内部也走一遍相同 blocks。
- **参数初始化**：复制主干权重作为初始化（ControlNet 的“strong prior”），但 residual 的投影头 zero-init。
  - 复制策略：`controlnet.load_state_dict(wan_model.state_dict(), strict=False)` + 再对 `zero_linear` 置零

### C. Diffusion forcing wrapper 中接入 ControlNet

3. 修改 `FloodNet/models/diffusion_forcing_wan.py`

- `DiffForcingWanModel.__init__`：
  - 添加开关：`use_controlnet_traj: bool`
  - 创建 `self.controlnet = WanControlNet(...)`
  - 添加 freeze 策略：
    - Stage-1：冻结 `self.model`（Wan 主干）、VAE、text encoder，只训 `controlnet + traj_encoder (+ traj_in_proj/type_embed)`
- `forward()`：
  - 调用 `controlnet(...)` 生成 `controlnet_residuals`
  - 调用 `self.model(..., controlnet_residuals=controlnet_residuals, traj_emb=traj_emb, ...)`
  - 保持现有 `control_aux["pred_x0_latent_list"]` 输出逻辑不变（后续 loss 统一在 `train_ldf.py` 算）
- `generate()/stream_generate()/stream_generate_step()`：
  - 推理同样把 `controlnet_residuals` 传入主干
  - `stream_generate_step` 已有轨迹 buffer（`traj_buffer/traj_features_buffer/traj_features_mask_buffer`），需确保 ControlNet 使用同一份 `traj_emb`（缓存逻辑复用）

### D. 训练时显式 control loss（motion space）

4. 修改 `FloodNet/train_ldf.py`

- 在训练 step 中，若 `loss_dict` 含 `control_aux.pred_x0_latent_list`：
  - **Full-sequence** VAE decode（冻结 VAE）→ motion feature → root xz
  - 将 pred/GT/`traj_mask` **切片到 active window 帧区间**（由 `getattr(model, "chunk_size")` 与 `T_token` 决定，见 `Task5-loss.md`）
  - 在切片上计算 `L_control_xz`；`total = mse + λ_control * L_control_xz`
- **active window 对齐**：扩散 MSE 与 `L_control_xz` 均针对 **最后 `chunk_size` 个 token** 所对应的 **帧范围**（实现上通过全 decode 后切片，避免尾窗-only decode 的原点错位）。

### E. 数据/对齐与 mask 语义（不改或最小改）

5. `FloodNet/datasets/humanml3d.py` / `FloodNet/utils/traj_batch.py`

- 首选复用你现在的 token-level `traj_features` + `traj_mask_token` + last-frame 监督语义。
- 若要更强控制（更平滑），可以引入 `traj_aggregate_mode="mean"` 的版本做对比，但第一版别混用。

## 训练策略（建议）

### Stage-1（推荐）

- 冻结：`WanModel` 主干全部参数（除非你只想解冻 `traj_in_proj/type_embed`）
- 训练：`WanControlNet` + `TrajEncoder`（以及 `traj_in_proj/type_embed`）
- loss：
  - `L_mse_active_window`（已有）
  - `λ_control * L_control_xz`（新加）
- `λ_control`：从 0 warmup 到目标（如 0.5~2.0），避免初期不稳定
- 轨迹 dropout：沿用 `traj_drop_out`（类似 CFG），保证无轨迹时生成能力不退化

### Stage-2（可选）

若 Stage-1 控制精度不够：

- 小学习率解冻主干最后 1~2 层 block 或 norm/ffn
- 只做短程微调，避免破坏 base FID/多样性

## 验证清单（必须做的 sanity checks）

1. **zero-init 等价性**：ControlNet 刚初始化时，带/不带 `controlnet_residuals` 的输出应几乎一致（差异≈0）。
2. **梯度通路**：第一步 `traj_encoder` 可能为 0（因为 `traj_in_proj` zero-init），但优化一步后应出现非零梯度（与你已有检查一致）。
3. **控制损失是否对齐 active window**：打印每 step 监督的帧索引/掩码密度，确认不是把一个 token 监督到 4 帧导致冲突。
4. **流式推理稳定性**：`stream_generate_step` 在 prompt 改变/轨迹改变时，buffer 内帧能被新条件影响（FloodDiffusion 论文强调的双向注意力价值）。

## 需要新增/修改的文件列表（汇总）

- **新增**
  - `FloodNet/models/tools/wan_controlnet.py`
- **修改**
  - `FloodNet/models/tools/wan_model.py`（forward 接收 residual + 注入）
  - `FloodNet/models/diffusion_forcing_wan.py`（构建/调用 controlnet）
  - `FloodNet/train_ldf.py`（decode + `L_control_xz`）
  - （可选）`FloodNet/utils/traj_batch.py`（仅当需要更换聚合/对齐策略）

