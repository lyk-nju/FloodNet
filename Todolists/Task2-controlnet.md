## Task2 — ControlNet 分支（WanControlNet）

对应 [`target.md`](target.md) §2。

---

## 目标

新增一个与 `WanModel` 并行的 **ControlNet 风格分支**：

- 输入同主干（noisy latent + time + text + traj_emb）
- 输出按层的 residual（list），并且 residual 输出头 **zero-init**
- Stage-1 训练时冻结主干，只训练 ControlNet 与 traj encoder 路径

---

## 模块与接口

### 新增文件

- `FloodNet/models/tools/wan_controlnet.py`

### 类

- `class WanControlNet(nn.Module)`

### forward 输入（与主干对齐）

- `x`：主干同款 patch embedding 输入（在 `DiffForcingWanModel` 里就是 `noisy_feature_input` 的 list 元素）
- `t`：`noise_level * time_embedding_scale`（与主干一致）
- `context`：text context（与主干一致）
- `seq_len`：最大 token 长度（与主干一致）
- `traj_emb` / `traj_seq_lens`：复用 FlexTraj

### forward 输出

- `residuals: list[Tensor]`，长度 `num_layers`
- 每个元素 shape：**`(B, seq_len, dim)`**（只对应 latent tokens）

---

## 初始化策略（强烈推荐）

- ControlNet 的 backbone（embedding/time/blocks）从主干 `WanModel` **拷贝权重**初始化（强 prior）
- residual 输出层 `zero_linear_i` 必须：
  - 权重、bias 全 0
  - 确保初始化时 residual 全 0（行为≈无 ControlNet）

---

## 冻结/可训练参数（Stage-1）

可训练：

- `WanControlNet.*`
- `TrajEncoder.*`
- `WanModel.traj_in_proj`、`WanModel.traj_type_embed`（如你希望 traj token 注入可训练）

冻结：

- `WanModel.blocks/embeddings/head`
- 文本编码器（T5 或 precomputed table）
- VAE

---

## 必须验证的 sanity check

- **zero-init 等价性**：ControlNet 刚加入时，带 residual 与不带 residual 的 `WanModel` 输出差异 \(\approx 0\)。
- **梯度出现时机**：`traj_in_proj` zero-init 会导致第一步 `TrajEncoder` 梯度可能为 0，但优化一步后应变为非 0（符合你之前的梯度流检查经验）。

