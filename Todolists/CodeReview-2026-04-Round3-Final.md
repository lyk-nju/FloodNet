# FloodNet 代码审查 — 第三轮（最终）

**审查时间**：2026-04-17
**审查范围**：全项目关键路径 + 跨文件一致性 + 边角场景
**审查重点**：上两轮未深入的 `WanCrossAttention`、`flextraj` attention 细节、`collate_fn` 对齐、训练/推理一致性、CFG 多文本场景

---

## 一、结论速览

| 等级 | 数量 | 状态 |
|------|------|------|
| 🔴 严重（会影响训练） | 1（实际触发） + 2（隐藏场景） | 需要修复 |
| 🟡 中度（不影响当前训练） | 3 | 建议修复 |
| ✅ 已验证无问题 | 10 | — |

**当前 HumanML3D 训练配置下**：
- **没有**新发现的严重 bug 会实际触发
- 有 3 个隐藏 bug 仅在切换到 BABEL/其他帧对齐多文本数据集时才会触发
- 前两轮已修复的 bug 基本都验证通过

---

## 二、🔴 潜在严重 bug（当前未触发，但需谨防）

### Bug L（严重）：`WanCrossAttention` 在 FlexTraj + 帧对齐多文本场景下形状错乱

**位置**：`models/tools/wan_model.py:241-267`

**触发条件**（需同时满足）：
1. 启用 FlexTraj 模式（`traj_in_proj is not None` 且 `traj_emb is not None`），即 x = [latent ‖ traj] 拼接
2. 文本为帧对齐多段（`context.size(0) = B * seq_len`，如 BABEL）
3. 进入 `WanAttentionBlock.forward` → `self.cross_attn(self.norm3(x), context, context_lens)`

**根本原因**：

```python
# wan_model.py:254-259
b, n, d = context.size(0), self.num_heads, self.head_dim   # b = B * seq_len
q = self.norm_q(self.q(x)).view(b, -1, n, d)                # x.shape = (B, 2*L, C)
# 实际 view 结果: q.shape = (B*L, 2, n, d)
```

由于 `x.shape[0] = B`（2 段串成一个 batch 维），而 `context.size(0) = B*L`，`view(B*L, -1, n, d)` 会：
- 自动推断 `-1 = 2*L*C / (L*n*d) = 2`
- 把每 2 个相邻 token 打包成一组，**完全错乱** `[latent_0..L‖traj_0..L]` 的结构
- 结果：`new[bl, 0] = old[0, 2*bl]`，`new[bl, 1] = old[0, 2*bl+1]`，即 **(latent[0], latent[1])、(latent[2], latent[3])…** 配对

**当前是否触发**：**否**。当前 HumanML3D 配置里 `context.size(0) = B`（每样本一条文本）。但 BABEL / `stream_mode=True` + 多段文本 + FlexTraj 会触发。

**修复方案**：在 `WanCrossAttention.forward` 中按 FlexTraj 分段重塑：

```python
def forward(self, x, context, context_lens):
    out_sizes = x.size()
    bq, Lq, _ = x.shape
    b_ctx = context.size(0)
    n, d = self.num_heads, self.head_dim

    if b_ctx != bq:
        # Frame-aligned text: context.shape = (B*Lq_latent, L2, C).
        # 对于 FlexTraj (Lq == 2*Lq_latent)，每个 latent token 配 1 条文本；
        # traj token 的 cross-attn 输出稍后会被清零（见 wan_model.py:358），但这里也要正确 reshape。
        if b_ctx == bq * (Lq // 2):  # FlexTraj 模式
            Lq_latent = Lq // 2
            # 把 latent 和 traj 分别展平到 (B*Lq_latent, 1, C) 再广播，分两次 attn
            x_lat  = x[:, :Lq_latent, :].reshape(bq * Lq_latent, 1, -1)
            x_traj = x[:, Lq_latent:, :].reshape(bq * Lq_latent, 1, -1)
            q_lat  = self.norm_q(self.q(x_lat )).view(bq * Lq_latent, 1, n, d)
            q_traj = self.norm_q(self.q(x_traj)).view(bq * Lq_latent, 1, n, d)
            k = self.norm_k(self.k(context)).view(b_ctx, -1, n, d)
            v = self.v(context).view(b_ctx, -1, n, d)
            a_lat  = flash_attention(q_lat , k, v, k_lens=context_lens)
            a_traj = flash_attention(q_traj, k, v, k_lens=context_lens)
            out = torch.cat([
                a_lat.flatten(2).view(bq, Lq_latent, -1),
                a_traj.flatten(2).view(bq, Lq_latent, -1),
            ], dim=1)
        elif b_ctx == bq * Lq:  # 普通帧对齐（非 FlexTraj）
            # 原本的 view 逻辑在这里是正确的
            q = self.norm_q(self.q(x)).view(b_ctx, -1, n, d)
            k = self.norm_k(self.k(context)).view(b_ctx, -1, n, d)
            v = self.v(context).view(b_ctx, -1, n, d)
            out = flash_attention(q, k, v, k_lens=context_lens).flatten(2).view(bq, Lq, -1)
        else:
            raise ValueError(f"context.size(0)={b_ctx} 无法与 x.shape={x.shape} 对齐")
    else:
        q = self.norm_q(self.q(x)).view(bq, -1, n, d)
        k = self.norm_k(self.k(context)).view(bq, -1, n, d)
        v = self.v(context).view(bq, -1, n, d)
        out = flash_attention(q, k, v, k_lens=context_lens).flatten(2).view(bq, Lq, -1)
    return self.o(out)
```

**优先级**：如果近期**不会切换到 BABEL 或帧对齐多文本**，可以延后；否则必须修复。

---

### Bug I（严重，前轮已记录）：`_stream_build_traj_emb` 的 `traj_features_buffer` 约定不一致

**位置**：`models/diffusion_forcing_wan.py:1416-1479`

**问题**：
- 训练路径（`_build_traj_emb`）：`traj_features` 是 **frame-level**，经 `LocalTrajEncoder`（4-frame→1-token）再 `TrajEncoder`
- 流式路径（`_stream_build_traj_emb` 的 `traj_features_buffer`）：约定 **token-level**，**跳过** `LocalTrajEncoder`

**当前是否触发**：**否**。HumanML3D 评估/训练都走 `_build_traj_emb`（`traj_features` 来自 dataset，是 frame-level）。`stream_generate_step` 的 `traj_features_buffer` 只被 web demo 使用。

**修复方案**：统一约定。建议在 `_stream_build_traj_emb` 里也按 frame-level 处理，或明确文档化并加 shape 断言：

```python
# Option A (推荐): 统一 frame-level 输入
# 在 _stream_update_traj_buffers 里把 token-level 输入转为 frame-level，再走 LocalTrajEncoder

# Option B: 在 buffer 路径明确检查
if self.traj_features_buffer is not None:
    # token-level buffer, shape = (B, N_tokens, 4)
    expected_dim = self.traj_out_dim  # 来自 LocalTrajEncoder 输出
    assert feats.size(-1) == expected_dim, "traj_features_buffer must be token-level"
    ...
```

---

### Bug J（严重，前轮已记录）：缺失 `LocalTrajEncoder` 可能导致轨迹语义不同

**位置**：与 Bug I 同位置

**现状**：同上，待确认设计意图。

---

## 三、🟡 中度问题（已存在但不紧迫）

### Bug D（前轮已记录）：`_build_traj_emb` 当 `mask_frame is None` 时，padding 位置不被显式置 0

**位置**：`models/diffusion_forcing_wan.py:406-412`

**当前影响**：`traj_seq_lens_attn` 会在 attention 里正确 mask 掉无效 token，所以**输出结果正确**。但 padding 位置的 `feats_frame` 经过 `TrajEncoder`（含 bias）会产生非零嵌入，浪费了计算资源并增加少量数值噪声。

**修复建议**：在 `mask_frame is None` 分支下，根据 `traj_length`/`feature_length` 构造默认 mask：

```python
if mask_frame is None and ("traj_length" in x or "feature_length" in x):
    T_frame = feats_frame.shape[1]
    tl = x.get("traj_length", x.get("feature_length"))
    if tl is not None:
        tl = tl.to(device=device, dtype=torch.long)
        idx = torch.arange(T_frame, device=device).view(1, -1)
        mask_frame = (idx < tl.view(-1, 1)).to(dtype=feats_frame.dtype)
```

---

### 问题 AA：DDP `find_unused_parameters=True`

**位置**：`train_ldf.py:628`

**影响**：因为 `freeze_backbone_for_controlnet=True` 时主干所有参数 `requires_grad=False`，Lightning 的 DDP 会把它们视为"unused"。设置为 `True` 可以避免报错，但**每步略慢 5-10%**。

**修复方案**（可选）：
```python
# 通过 unused param 检测：冻结的主干参数不参与 forward 的可微计算，DDP 会抱怨
# 改为 False 需要确保所有 trainable 参数每步都参与 loss graph
# 当前 freeze_backbone_for_controlnet + ControlNet + L_control 设置下：
# - 主干参数 requires_grad=False → DDP 不会 sync gradient（自动跳过）
# - 所以可以尝试 find_unused_parameters=False
strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
```

建议先保留 `True`（当前已工作），日后在稳定训练时再尝试切换。

---

### 问题 T：训练 `time_steps` 采样范围与推理略不一致

**位置**：`models/diffusion_forcing_wan.py:512`

**现状**：
- 训练：`time_steps ∈ [0, valid_len / chunk_size]`（即 `t ∈ [0, max_t]`）
- 推理：`t` 按 `dt = 1/num_denoise_steps` 从 0 递增到 `max_t = 1 + (seq_len-1)/chunk_size`

**差异**：推理 max_t 比训练 max_t 多 1（因为推理时有 `seq_len + chunk_size` 生成长度）。训练时**最后的 "lookahead" chunk 永远没被涵盖**。

**影响**：训练/推理分布轻微不匹配。这是继承自原 FloodDiffusion 的历史选择，建议不改动。

---

### 问题 BB：learning rate `1e-3` 偏高

**位置**：`configs/ldf.yaml:163`

**现状**：当前 `lr = 1e-3`。历史 output 目录里 `5e-4` 和 `1e-4` 也都有使用。

**建议**：保持当前值，但如果出现训练不稳定（如 loss NaN、mse 波动剧烈），考虑降至 `5e-4`。

---

## 四、✅ 已验证无问题

1. **Bug K**（RNG 保存/恢复）：`update_test` 的 try/finally 保存 Python/NumPy/PyTorch CPU/CUDA RNG 状态，逻辑正确。
2. **Bug G**（`_get_traj_seq_lens` 公式）：优先使用 `feature_length`，fallback `((tl+2)//4 + 1)` 与 dataset 一致。
3. **Bug F**（EMA 差异日志流式求和）：避免 `torch.cat` 创建 ~1.5GB 临时 tensor，参数迭代顺序一致。
4. **ControlNet 初始化**（Bug #1）：`on_load_checkpoint` 在加载 backbone 权重后调用 `init_from_backbone` ✓。
5. **VAE 冻结**（Bug #2）：`initialize_metrics` 里 `self.vae.eval()` 且 `requires_grad=False`，`train()` 里再次 `self.vae.eval()` ✓。
6. **Separated CFG uncond 真正 unconditional**（Bug #4）：三个调用点都传 `None, None` ✓。
7. **EMA 只跟踪 trainable 参数**（Bug #6）：`BasicLightningModule.__init__` 和 `on_load_checkpoint` 都用 `[p ... if p.requires_grad]` ✓。
8. **`vae_ema` 用完即删**（Bug #5）：`del vae_ema` 释放内存 ✓。
9. **`_compute_control_loss_xz` detach 过去**（原始用户发现的 bug）：`latent_past.detach()` 正确 ✓。
10. **FlexTraj self-attention 掩码**：`flextraj_self_attn_bias` 正确产生 `latent→all valid`、`traj→only traj` 的块稀疏 mask。
11. **ControlNet 残差只加到 latent token**：`wan_model.py:693-694` `x[:, :seq_len, :] += r` 只影响前段 ✓。
12. **`text_null_context` 缓存包含空字符串**：`pretokenize_t5_text.py:108` 显式 `captions.add("")` ✓。
13. **`collate_fn` padding 一致**：`feature/token/traj/traj_features` 都用 `pad_sequence(padding_value=0)` ✓。
14. **`process_token` 对齐 feature crop**：Causal VAE 公式 `token_start = (crop_start+3)//4`、`token_end = (last_frame+3)//4` 推导正确。
15. **训练 `noise_level` shape 匹配**：`t.shape=(B, seq_len)`，`t.flatten().unflatten(0, (B, seq_len))` 正确。

---

## 五、前轮 pending 问题的当前状态

| 前轮 bug | 状态 | 备注 |
|---------|------|------|
| Bug K（RNG 污染） | ✅ 已修复 | 用户实现 |
| Bug G（`_get_traj_seq_lens`） | ✅ 已修复 | 用户实现 |
| Bug F（EMA 日志内存） | ✅ 已修复 | 用户实现 |
| Bug I/J（stream traj_features 约定） | ⏳ 未修复 | 当前不触发，已文档化 |
| Bug D（padding 不置零） | ⏳ 未修复 | 影响甚微 |
| `find_unused_parameters=True` | ⏳ 未修复 | 稳定后再切 `False` |
| control loss DDP 偏差 | ⏳ 未修复 | 每 rank 独立 `n_valid`，偏差极小 |
| L_mse 与 L_control 量级 | ⏳ 未修复 | 当前 control_loss_weight=1.0，量级已接近 |
| `noise` ODE 公式 | ⏳ 保留 | 非标准但历史有效 |
| 训练 time_steps 范围 | ⏳ 保留 | 继承自 FloodDiffusion |

---

## 六、给下一步训练的建议

### A. 立即可做

1. **确认当前训练配置没有切换到 BABEL**。只要保持 HumanML3D 的单文本模式，Bug L 不会触发。
2. 继续当前 run（如果训练曲线正常），Bug K/G/F 修复生效。

### B. 若训练不稳定

1. 降低 `lr` 到 `5e-4` 或 `3e-4`。
2. 检查 `ema_diff/avg` 是否在 `1e-4 ~ 1e-3` 范围。

### C. 切换到 BABEL 时必须先做

1. 修复 Bug L（`WanCrossAttention` 的 FlexTraj 感知 reshape）。
2. 验证 Bug I/J：要么统一 `traj_features_buffer` 为 frame-level，要么加断言。

### D. 长期优化

1. `find_unused_parameters=False`（~5-10% 加速）。
2. Bug D：frame-level padding mask 显式置零（节省 `TrajEncoder` 无效计算）。

---

## 七、文件改动摘要（本轮新增文档）

- **本文件** `FloodNet/Todolists/CodeReview-2026-04-Round3-Final.md`：最终审查结论和 action items。

**无代码改动**。所有发现都列在此文档中，等待用户决定修复优先级。
