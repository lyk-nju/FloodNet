## Task7 — Scheduled Sampling 训练策略

Corresponds to [`target.md`](target.md) §7.

**Status: Done (2026-04).** `scheduled_sampling_prob` 参数已加入 `DiffForcingWanModel.__init__` 和 `forward()`，config key 已加入 `configs/ldf.yaml`。

---

## 背景与动机

LDF 训练使用 Teacher Forcing（TF）：context 部分（active window 左侧，noise_level ≈ 0）永远是 GT latent tokens。推理时（generate 模式）context 是模型自己的预测，分布与训练不同，导致 exposure bias。

实验 3 测量结果：
```
mean_TF      = 0.141   ← 有 GT context
mean_SF      = 0.399   ← 用 generated latents 替换 GT（固定权重，未训练）
mean_GEN     = 1.018   ← 完整自回归生成
gap_gen_TF   = 0.877   ← 总 gap
gap_gen_SF   = 0.619   ← SF-proxy 后剩余 gap
```

SF-proxy 仅减少约 29% 的 gap，但这是在**未经训练**的模型上的估计。Scheduled Sampling 训练让模型在训练期间就接触到自己的预测误差，目标是缩小整个 0.877 gap（而不仅 0.258）。

---

## 实现方案

### 核心逻辑（`diffusion_forcing_wan.py::forward`）

在 `traj_emb` 构建完成、主 model forward 调用之前插入 SS 逻辑：

```python
if self.scheduled_sampling_prob > 0.0 and np.random.rand() < self.scheduled_sampling_prob:
    # Pass 1 (no_grad): 用当前 batch 的 noisy_feature_input 预测 x0_hat
    with torch.no_grad():
        cn_res_ss = self._maybe_controlnet_residuals(noisy_feature_input, ...)
        pred_ss = self.model(noisy_feature_input, ..., controlnet_residuals=cn_res_ss)
    
    # 替换 context tokens（active window 之前的部分）
    for b in range(batch_size):
        t_len = noisy_feature_input[b].shape[1]   # end_index for this sample
        ctx_len = t_len - self.chunk_size           # tokens before active window
        if ctx_len <= 0:
            continue
        # 从预测结果恢复 x0_hat
        if prediction_type == "vel":
            x0_hat = (pred_ss[b] + noise_ref[b]).detach()  # vel = x0 - noise
        elif prediction_type == "x0":
            x0_hat = pred_ss[b].detach()
        # 替换：context 用 x0_hat，active window 保持 noisy
        noisy_feature_input[b] = cat([x0_hat[:, :ctx_len], noisy_feature_input[b][:, ctx_len:]], dim=1)

# Pass 2 (with grad): 用修改后的 input 计算实际 loss
```

### 关键设计点

1. **只替换 context，不替换 active window**：active window（最后 `chunk_size` 个 token）保持原来的 noisy 状态，loss 照常计算。
2. **pass 1 用 no_grad**：避免额外梯度开销；x0_hat detach 后用作 context。
3. **vel 预测类型下的 x0_hat 恢复**：
   - `vel = x0 - noise`（训练 target 定义）
   - 因此 `x0_hat = pred_vel + noise_ref`
4. **noise 预测类型暂不支持**（SS 逻辑 `continue`）：需要 noise_level 才能从 noise 还原 x0，而 noise_level 在 context 位置 ≈ 0，数值不稳定。

---

## 配置（`configs/ldf.yaml`）

```yaml
model:
    params:
        # Scheduled Sampling: probability of replacing context tokens (prefix before
        # active window) with the model's own x0 prediction during each training step.
        # 0.0 = pure teacher forcing (default); 0.5 = half steps use self-prediction.
        # Recommended curriculum: start at 0.0, increase to 0.3~0.5 after 50k steps.
        scheduled_sampling_prob: 0.0
```

---

## 推荐课程调度

Scheduled Sampling 应配合训练进展逐渐增大，避免早期不稳定：

| 训练阶段 | 建议值 | 理由 |
|---------|-------|------|
| 0 ~ 50k steps | 0.0 | 先让 ControlNet 稳定收敛（zero-init 残差需要足够步数建立正确方向） |
| 50k ~ 100k | 0.1 ~ 0.2 | 开始引入，轻度扰动 |
| 100k+ | 0.3 ~ 0.5 | 较强 SS，模型已具备基本轨迹跟随能力 |

实际操作：修改 `configs/ldf.yaml` 中的值后重启训练（从当前 ckpt 继续）即可，不需要改代码。

---

## 与其他改进的关系

- **Smooth Root（Task8）**：两者正交，可以同时使用。
- **更多训练步数**：先跑到 50k 再开启 SS 效果更好。
- **两阶段去噪（仿 Kimodo）**：SS 是当前架构下的短期方案；两阶段去噪是根本性架构改进，两者不冲突。

---

## 验证方案

1. **直接运行测试**：目前 GPU 全满，等训练资源释放后执行
   ```bash
   # 先跑 forward/generate 对比 eval（对照 SS 开启前的数据）
   conda run -n flooddiffusion python tools/eval_control_loss.py \
       --config configs/ldf.yaml --eval_mode generate --seed 1234
   ```

2. **训练曲线观察**：开启 SS 后，`mse` loss 初期可能略有上升（输入分布变化），这是正常的，几千步后应恢复并继续下降。

3. **generate vs TF gap 收窄**：训练到足够步数（100k+）后，重跑 `run_gap_experiments.py` 实验 3，验证 mean_GEN 是否接近 mean_TF。

---

## 已知局限

- **2x 计算开销**：每次触发 SS 需要额外一次 no_grad forward（ControlNet + backbone）。实际时间约增加 `ss_prob × 100%`，即 ss_prob=0.3 时约增加 30%。
- **仅影响 context，不影响 active window 内的扩散质量**：主扩散训练信号不变，SS 只改变了模型见到的上下文分布。
- **噪声预测类型不支持**：当前只有 `vel` 和 `x0` 类型实现了 SS，`noise` 类型会跳过该样本（即使 ss_prob > 0）。
