# Frontier Noise Prior for FloodNet：数学建模、伪代码与训练实现细节

## 0. 目标

我们希望训练一个轻量 initializer \(R_\phi\)，在 FloodNet streaming rollout 中，根据当前 generated state 预测未来最早一批尚未进入 denoise schedule 的 frontier \(z_T\)：

\[
R_\phi(H_k,A_k,c_k,\tau_k)\rightarrow Z_T^{frontier}
\]

Frozen FloodNet / VAE 不更新，只训练 initializer。

第一版推荐使用 deterministic residual：

\[
Z_T^{frontier}
=
Z_{T,base}^{frontier}
+
\alpha\Delta_\phi(H_k,A_k,c_k,\tau_k)
\]

后续可升级为 Gaussian prior：

\[
q_\phi(Z_T^{frontier}\mid H_k,A_k,c_k,\tau_k)
=
\mathcal{N}(\mu_\phi,\sigma_\phi)
\]

---

## 1. 核心变量定义

### 1.1 Streaming state

在第 \(k\) 个 streaming decision point：

\[
s_k=(H_k,A_k,c_k,\tau_k)
\]

其中：

- \(H_k\)：committed history，通常是最近 `history_length` 个 clean latent tokens；
- \(A_k\)：active boundary，当前 triangular schedule 中已经 partially denoised 但尚未 commit 的 token 状态；
- \(c_k\)：文本 embedding；
- \(\tau_k\)：未来局部轨迹 condition；
- \(Z_T^{frontier}\)：最早一批未进入 denoise schedule 的 future initial noise tokens。

### 1.2 Buffer 语义

为了避免语义混乱，至少需要区分三类 latent 状态：

```text
committed_latents       # 已 commit 的 z0 history
active_noisy_state      # 当前 denoise schedule 中的 x_beta
future_initial_noise    # 尚未进入 denoise 的 z_T
```

其中 initializer 只作用于：

```text
future_initial_noise[frontier_ids]
```

不要优化：

```text
committed_latents
active_noisy_state
```

---

## 2. 数据怎么来

这里不建议大规模离线收集 oracle \(Z_T^*\) 数据集，因为单样本 oracle optimization 代价较高。训练数据来自在线 rollout 中动态采样的 decision states。

每个训练样本是：

\[
(H_k,A_k,c_k,\tau_k,Z_{T,base}^{frontier})
\]

其中 target 不是离线 oracle label，而是通过 differentiable rollout loss 在线训练 initializer。

### 2.1 采样序列

```python
sample = dataset.sample()
# sample 包含：
# - GT motion
# - caption/text embedding
# - target root trajectory
```

GT motion 用途：

1. 初始化 seed；
2. 提供 target trajectory；
3. 计算 loss。

但训练 initializer 时，history context 应尽量来自 generated rollout，而不是一直用 GT history。

### 2.2 rollout 到 decision point

```python
state = init_stream_state(sample)
k = sample_decision_index(sample)
state = rollout_until(state, commit_index=k, rollin_policy=current_rollin)
```

`rollin_policy` 可以是：

- Gaussian baseline；
- initializer；
- scheduled mix。

---

## 3. 提取输入 context

### 3.1 History \(H_k\)

```python
H_k = extract_committed_history(state, history_length=30)
```

可以包含：

```python
H_k = {
    "latents": committed_latents[-history_length:],
    "root_xz": history_root_xz,
    "velocity": history_velocity,
    "heading": history_heading,
    "mask": history_mask,
}
```

第一版可以只用 latent + mask。

### 3.2 Active boundary \(A_k\)

```python
A_k = extract_active_boundary(state)
```

建议包含：

```python
A_k = {
    "latents": active_noisy_state,
    "token_offsets": active_token_indices - commit_index,
    "beta": active_token_beta,
    "stage": active_token_stage,
}
```

这些 token 是 context，detach，不更新。

### 3.3 Text \(c_k\)

```python
c_k = get_text_embedding(sample)
```

建议保存 caption 作为 debug：

```python
caption = sample.caption
text_embedding = sample.text_embedding
```

### 3.4 Trajectory \(\tau_k\)

建议使用 generated anchor 构造局部轨迹：

```python
tau_k = build_future_local_trajectory(
    target_world_traj=sample.target_root_traj,
    anchor_root=state.generated_root_xz,
    anchor_heading=state.generated_heading,
    horizon=traj_horizon,
)
```

可以包含：

```python
tau_k = {
    "local_xz": target_local_xz,
    "local_velocity": target_local_velocity,
    "local_heading": target_local_heading,
    "mask": traj_mask,
}
```

---

## 4. 找 frontier \(z_T\)

只选择最早一批尚未进入 denoise schedule 的 pure initial noise tokens。

```python
frontier_ids = find_frontier_zT_tokens(
    state,
    num_tokens=frontier_tokens,      # e.g. 5
    beta_threshold=0.999,
    require_stage="initial_zT",
)
```

如果不够数量，跳过当前 state。

```python
if len(frontier_ids) < frontier_tokens:
    continue
```

得到 base frontier noise：

```python
base_zT = state.future_initial_noise[frontier_ids].detach()
```

---

## 5. Initializer 设计

### 5.1 第一版：deterministic residual

推荐从这个版本开始：

\[
Z_T^{frontier}
=
Z_{T,base}^{frontier}
+
\alpha\Delta_\phi(H_k,A_k,c_k,\tau_k)
\]

伪代码：

```python
delta_zT = initializer(
    history=H_k,
    active=A_k,
    text=c_k,
    traj=tau_k,
    frontier_offsets=frontier_ids - state.commit_index,
)

alpha = min(1.0, global_step / warmup_steps)
zT_frontier = base_zT + alpha * delta_zT
```

### 5.2 第二版：Gaussian prior

后续可升级为：

\[
Z_T^{frontier}=\mu_\phi+\sigma_\phi\epsilon
\]

```python
mu, log_sigma = initializer(H_k, A_k, c_k, tau_k)
sigma = torch.exp(log_sigma).clamp(sigma_min, sigma_max)
eps = torch.randn_like(mu)
zT_frontier = mu + sigma * eps
```

可加 KL：

\[
D_{KL}(\mathcal{N}(\mu,\sigma)\Vert\mathcal{N}(0,I))
\]

---

## 6. Shadow rollout

Shadow rollout 用于让 frontier \(z_T\) 生效并计算 loss。它不改变真实 runtime state。

### 6.1 关键要求

- restore snapshot；
- 写入 frontier \(z_T\)；
- 保持 history / active boundary detached；
- 不更新 LDF / VAE 参数；
- 但不要 `torch.no_grad()` 包住 LDF / VAE forward，因为梯度需要从 loss 传回 initializer；
- rollout 到 frontier tokens 进入 denoise schedule 并产生 decoded motion。

### 6.2 伪代码

```python
def shadow_rollout_with_frontier_zT(
    snapshot,
    frontier_ids,
    zT_frontier,
    rollout_steps,
):
    state = snapshot.restore()

    # 写入 future initial noise buffer
    state.future_initial_noise[frontier_ids] = zT_frontier

    decoded_windows = []

    for r in range(rollout_steps):
        state = stream_generate_step_shadow(
            state,
            use_future_initial_noise=True,
            commit=False,
            detach_history=True,
        )

        if should_decode_for_loss(state, frontier_ids):
            latents_for_decode = collect_decode_latents(state)
            motion = vae.decode(latents_for_decode)
            decoded_windows.append(motion)

    return ShadowResult(
        final_state=state,
        decoded_motion=merge(decoded_windows),
        affected_indices=get_affected_indices(frontier_ids, state),
        boundary=extract_current_boundary(state),
    )
```

---

## 7. Loss 设计

第一版建议简单稳定：

\[
L=L_{xz}+\lambda_vL_{vel}+\lambda_cL_{cont}+\lambda_nL_{noise}
\]

### 7.1 Affected future XZ loss

因为 frontier \(z_T\) 延迟生效，所以 loss 应该主要算在 affected future window：

\[
L_{xz}
=
\frac{1}{|\Omega|}
\sum_{t\in\Omega}
\|root_{xz,t}-\tau_{xz,t}\|^2
\]

```python
L_xz = mse(pred_root_xz[affected], target_root_xz[affected])
```

### 7.2 Velocity loss

\[
L_{vel}
=
\frac{1}{|\Omega|}
\sum_{t\in\Omega}
\|v_t-v_t^{ref}\|^2
\]

```python
pred_vel = pred_root_xz[1:] - pred_root_xz[:-1]
target_vel = target_root_xz[1:] - target_root_xz[:-1]
L_vel = mse(pred_vel[affected], target_vel[affected])
```

### 7.3 Continuity loss

未来 frontier motion 应该接住当前 boundary：

\[
L_{cont}
=
\|root_{future,0}-root_{boundary}\|^2
+
\lambda_{cv}\|v_{future,0}-v_{boundary}\|^2
\]

```python
L_cont = (
    mse(pred_future_root[0], boundary_root)
    + lambda_cv * mse(pred_future_vel[0], boundary_vel)
)
```

### 7.4 Noise regularization

防止 initializer 把 \(z_T\) 推得太远：

\[
L_{noise}=\|Z_T^{frontier}-Z_{T,base}^{frontier}\|^2
\]

```python
L_noise = ((zT_frontier - base_zT) ** 2).mean()
```

### 7.5 后续可加 loss

稳定后再加：

- heading loss；
- jerk loss；
- foot skating loss；
- text semantic preservation；
- Gaussian KL。

---

## 8. 主训练伪代码

```python
# freeze large modules
ldf.eval()
vae.eval()
for p in ldf.parameters():
    p.requires_grad_(False)
for p in vae.parameters():
    p.requires_grad_(False)

initializer.train()
optimizer = torch.optim.AdamW(initializer.parameters(), lr=lr)

for global_step in range(num_train_steps):
    sample = dataset.sample()

    state = init_stream_state(sample)

    # rollout 到随机 decision point
    k = sample_decision_index(sample)
    state = rollout_until(
        state,
        commit_index=k,
        rollin_policy=scheduled_rollin_policy(initializer, global_step),
    )

    # extract context
    H_k = extract_committed_history(state, history_length).detach()
    A_k = extract_active_boundary(state).detach()
    c_k = get_text_embedding(sample).detach()
    tau_k = build_future_local_trajectory(
        state=state,
        target_traj=sample.target_root_traj,
        anchor_mode="generated_anchor",
        horizon=traj_horizon,
    ).detach()

    # find frontier zT
    frontier_ids = find_frontier_zT_tokens(
        state,
        num_tokens=frontier_tokens,
        beta_threshold=0.999,
        require_stage="initial_zT",
    )
    if len(frontier_ids) < frontier_tokens:
        continue

    base_zT = state.future_initial_noise[frontier_ids].detach()

    # initializer prediction
    delta_zT = initializer(
        history=H_k,
        active=A_k,
        text=c_k,
        traj=tau_k,
        frontier_offsets=frontier_ids - state.commit_index,
    )

    alpha = min(1.0, global_step / warmup_steps)
    zT_frontier = base_zT + alpha * delta_zT

    # shadow rollout
    snapshot = state.snapshot()
    shadow = shadow_rollout_with_frontier_zT(
        snapshot=snapshot,
        frontier_ids=frontier_ids,
        zT_frontier=zT_frontier,
        rollout_steps=shadow_rollout_steps,
    )

    # compute loss
    loss_dict = compute_frontier_loss(
        shadow=shadow,
        sample=sample,
        zT_frontier=zT_frontier,
        base_zT=base_zT,
    )

    loss = (
        loss_dict["xz"]
        + lambda_vel * loss_dict["vel"]
        + lambda_cont * loss_dict["cont"]
        + lambda_noise * loss_dict["noise"]
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(initializer.parameters(), max_grad_norm)
    optimizer.step()

    log_metrics(global_step, loss_dict, state, frontier_ids, zT_frontier, base_zT)
```

---

## 9. Scheduled rollin

Initializer 初期很弱，不应一开始完全控制 rollout。采用 schedule：

\[
p_{init}=\min(1,\frac{step}{T_{warmup}})
\]

```python
def scheduled_rollin_policy(initializer, global_step):
    p_init = min(1.0, global_step / rollin_warmup_steps)

    def policy(state):
        if random.random() < p_init:
            return use_initializer_frontier_zT(state)
        else:
            return use_gaussian_frontier_zT(state)

    return policy
```

这类似 DART scheduled training 的思想：逐步让训练分布接近推理分布，而不是一开始就把模型暴露给很差的自生成状态。

---

## 10. 每步 loss 和 update 的关系

frontier \(z_T\) 是延迟生效 action，不能用当前 commit loss 立即评价。

推荐：

```text
sample decision state
predict frontier zT
shadow rollout K steps
accumulate losses over affected future frames
backward once
optimizer step once
```

可以在 rollout 内每个 affected frame 都算 loss：

\[
L=\sum_{t\in\Omega_{affected}}L_t
\]

但不建议每个内部 step 都 optimizer update。

---

## 11. Diagnostics

每次训练 / eval 至少记录：

```python
diagnostics = {
    "method": "online_frontier_zT_initializer",
    "frontier_ids": frontier_ids,
    "frontier_offsets": frontier_ids - commit_index,
    "base_zT_mean": base_zT.mean(),
    "base_zT_std": base_zT.std(),
    "delta_zT_norm": delta_zT.norm(),
    "zT_frontier_mean": zT_frontier.mean(),
    "zT_frontier_std": zT_frontier.std(),
    "affected_ADE": ...,
    "affected_FDE": ...,
    "affected_vel_error": ...,
    "continuity_loss": ...,
    "noise_reg": ...,
    "commit_index": state.commit_index,
    "current_step": state.current_step,
}
```

Sanity checks：

1. `alpha=0` 时应等价 Gaussian baseline；
2. history / active context 不应有梯度；
3. LDF / VAE 参数梯度应为 None；
4. 梯度应流到 initializer 参数；
5. frontier token 的 stage 必须是 `initial_zT`；
6. loss window 必须覆盖 frontier token 生效后的 frames。

---

## 12. Evaluation

评估时比较：

1. Gaussian baseline；
2. A-active noisy-state oracle；
3. B-frontier oracle upper bound；
4. learned online frontier initializer。

指标分三组。

### 12.1 Affected future window metrics

这是最直接的训练目标：

```text
affected_ADE
affected_FDE
affected_heading_error
affected_velocity_error
```

### 12.2 Full rollout metrics

看长期收益：

```text
full_ADE
full_FDE
path_ratio
heading_error
replan_recovery_time
```

### 12.3 Motion quality metrics

防止轨迹精度换动作崩坏：

```text
jerk
foot_skating
root_smoothness
motion_quality
text_semantic_score
```

---

## 13. 最小实现计划

第一阶段只做最小可行版本：

```text
frontier_tokens = 5
initializer = small MLP / tiny Transformer
output = delta_zT
loss = xz + 0.05 * vel + 1e-4 * noise_reg
rollin = Gaussian only or scheduled rollin
freeze LDF + VAE
train small number of steps
```

等这个版本确认有效后，再升级：

1. 加 generated_anchor full eval；
2. 加 heading / continuity；
3. 输出 \(\mu,\sigma\)；
4. 加 KL；
5. initializer rollin；
6. optional PPO / policy gradient。

---

## 14. Codex 实现边界

不要混淆以下三件事：

```text
A: optimize active x_beta
B-full: optimize full-stream initial z_T
B-frontier: train initializer for earliest unactivated future z_T
```

本文件描述的是：

```text
B-frontier online differentiable initializer training
```

它不是 active noisy-state correction，也不是 full-stream oracle。

---

## 15. 一句话总结

训练过程就是：

> 在 online streaming rollout 中采样一个 generated state，找到最早未进入 denoise 的 future frontier tokens。Initializer 根据历史、当前 active boundary、文本和未来轨迹预测这些 token 的更好初始噪声。将其写入 future noise buffer，shadow rollout 到它们生效，计算 affected future trajectory loss，并通过 frozen FloodNet/VAE 反传，只更新 initializer。

这实现了我们的核心想法：

\[
\text{history motion prior} + \text{trajectory intent}
\rightarrow
\text{better denoising starting point}
\]