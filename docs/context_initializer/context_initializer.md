
# DART-Inspired Latent Initialization for FloodNet：从随机采样起点到历史-轨迹条件化的 (z_T) 先验

## 0. 背景与目标

当前 FloodNet / FloodDiffusion 已经在 `recon` 分支实现了流式轨迹控制能力。现有方法的基本路线是：

[
\text{history} + \text{text} + \text{trajectory condition}
\rightarrow
\text{diffusion denoising}
\rightarrow
z_0
\rightarrow
\text{VAE decode}
\rightarrow
\text{motion chunk}
]

目前的问题是：轨迹控制已经有效，但精度还不够高，尤其在长轨迹、急转弯、replan、heading 变化、root drift 等场景下，现有 trajectory-conditioned denoising 仍然存在控制误差。

我们希望在不大改现有 `recon` 分支主逻辑的前提下，参考 DARTControl 的 **RL latent initializer / latent noise action** 思想，探索一种新的控制增强方式：

> 不直接修改 FloodNet 的 denoising 结构，也不一开始就重训 trajectory ControlNet，而是学习或优化一个更合理的 diffusion 初始噪声 (z_T)，让模型从更符合历史状态和目标轨迹的采样起点开始生成。

核心目标：

[
z_T^k \sim \mathcal{N}(0,I)
]

升级为：

[
z_T^k = f_\phi(\text{history}_k,\ \text{text}_k,\ \text{trajectory}_k,\ \epsilon)
]

然后仍然交给 frozen FloodNet denoiser 和 VAE 完成生成：

[
z_0^k = F_\theta(z_T^k,\ \text{history}_k,\ \text{text}_k,\ \text{trajectory}_k)
]

[
x_k = D_{\text{VAE}}(z_0^k)
]

---

## 1. 灵感来源：DART 的 RL initializer / latent noise action

DARTControl 的关键启发不是简单的 “用 RL 做 goal-reaching”，而是：

> DART 把 diffusion 初始噪声 (z_T) 看成一个可以被选择的 latent action。

DART 的普通生成过程是：

[
z_T \sim \mathcal{N}(0,I)
]

[
z_T \rightarrow z_0 \rightarrow \text{motion primitive}
]

而在其 RL goal-reaching 版本中，policy 不直接输出 joint action、pose、root velocity 或 trajectory，而是输出 diffusion 初始噪声：

[
\pi_\phi(\text{history},\ \text{goal},\ \text{scene},\ \text{text})
\rightarrow
z_T
]

然后 frozen DART denoiser + decoder 将 (z_T) 变成下一段 motion primitive。

这背后的真正 insight 是：

> 对于动作生成，(z_T) 不只是随机噪声，而可以被理解成下一段动作的 latent intention / motor preparation。
> 一个好的 (z_T) 会让后续 denoising 更容易生成符合目标、历史状态和动作自然性的 motion。

因此，DART 的启发不是“我们也要照抄 goal-reaching RL”，而是：

> 对于 streaming motion diffusion，每个 chunk 的采样起点不应该永远是无条件高斯噪声。
> 历史动作本身已经是极强的动作先验，目标/轨迹则提供未来运动意图。更合理的是根据历史和目标生成一个 sound 的 (z_T)，再让 diffusion 模型去 denoise。

---

## 2. 我们的核心思路

### 2.1 为什么标准 Gaussian init 对 streaming motion 不够合理

当前 diffusion 采样通常默认：

[
z_T^k \sim \mathcal{N}(0,I)
]

这意味着每个 streaming step / chunk 都从无结构噪声开始。

但动作生成和图像生成不同。对于 streaming motion，第 (k) 个 chunk 的未来动作已经被历史状态强烈约束。历史动作包含：

* 当前身体姿态；
* 当前 root position / velocity；
* 当前 heading；
* 当前 gait phase；
* 当前哪只脚接触地面；
* 当前身体动量；
* 当前文本动作风格；
* 前面生成过程累积的 drift；
* 当前局部轨迹或导航目标。

所以第 (k) 个 chunk 的可行未来空间并不是无条件的。它应该更接近：

[
p(x_{k:k+H} \mid \text{history}_k,\ \text{text}_k,\ \text{trajectory}_k)
]

因此，相应的 diffusion 初始噪声也不应该是无条件高斯，而应该是一个 history-conditioned / trajectory-aware latent prior：

[
q_\phi(z_T^k \mid h_k,\ c_k,\ \tau_k)
]

其中：

* (h_k)：history motion / history latent；
* (c_k)：text condition；
* (\tau_k)：future local trajectory / target / root condition；
* (z_T^k)：第 (k) 个 chunk 或 latent token 的 diffusion initial noise。

### 2.2 新模块：History-Conditioned Latent Initializer

我们可以新增一个轻量模块：

[
f_\phi(h_k,\ c_k,\ \tau_k,\ \epsilon)
\rightarrow
z_T^k
]

这个模块不负责直接生成 motion，也不负责修改 VAE 或 FloodNet 主体。它只负责为 frozen FloodNet 提供一个更合理的采样起点。

可以理解成：

```text
history + text + local trajectory
        |
        v
History-Conditioned Latent Initializer
        |
        v
trajectory-aware z_T
        |
        v
Frozen FloodNet denoising
        |
        v
z_0
        |
        v
Frozen VAE decode
        |
        v
motion chunk
```

这和现有 trajectory condition 不冲突。现有 FloodNet 仍然可以使用 trajectory condition 做 denoising 细化；initializer 只是让起点更合理。

---

## 3. 数学建模

### 3.1 当前 FloodNet 采样形式

对于第 (k) 个 streaming step，当前方法近似为：

[
z_T^k \sim \mathcal{N}(0,I)
]

[
z_0^k =
F_\theta(z_T^k,\ h_k,\ c_k,\ \tau_k)
]

[
x_k =
D_{\text{VAE}}(z_0^k)
]

其中：

* (F_\theta)：已经训练好的 FloodNet / LDF denoiser；
* (D_{\text{VAE}})：已训练好的 VAE decoder；
* (h_k)：历史 motion / latent history；
* (c_k)：文本条件；
* (\tau_k)：未来局部轨迹条件；
* (x_k)：当前生成的 motion chunk。

### 3.2 Proposed：条件化采样起点

我们新增：

[
z_T^k =
f_\phi(h_k,\ c_k,\ \tau_k,\ \epsilon)
]

或者分布形式：

[
q_\phi(z_T^k \mid h_k,\ c_k,\ \tau_k)
=====================================

\mathcal{N}(\mu_\phi,\ \sigma_\phi)
]

[
z_T^k =
\mu_\phi(h_k,c_k,\tau_k)
+
\sigma_\phi(h_k,c_k,\tau_k)\epsilon
]

其中：

[
\epsilon \sim \mathcal{N}(0,I)
]

这样既保留随机性，又让初始噪声具备历史-目标先验。

### 3.3 如果一个 chunk 包含多个 latent token

如果当前 streaming step 不是只生成一个 latent token，而是生成一个 chunk 内的多个 token，可以扩展为：

[
Z_T^{k:k+K}
===========

f_\phi(h_k,\ c_k,\ \tau_{k:k+K},\ \epsilon)
]

其中：

[
Z_T^{k:k+K}
===========

[z_T^k,\ z_T^{k+1},...,z_T^{k+K}]
]

这相当于在 noise level 生成一个短时 latent plan，而不是每个 token 独立从 Gaussian 采样。

这可能有助于：

* 保持 gait phase 连贯；
* 提前编码转弯趋势；
* 约束未来速度；
* 减少 heading drift；
* 改善 replan 后的响应；
* 降低随机采样带来的不稳定性。

---

## 4. 路线定稿

当前阶段保留三条线，但主次明确：

| 线 | 优化 / 训练对象 | 作用 |
|---|---|---|
| A-line | 当前 active \(x_\beta\) | runtime correction oracle / 工程上限 |
| B-frontier oracle | earliest \(\beta \approx 1\) future \(z_T\) | 验证 frontier \(z_T\) 可控性 |
| Learned residual initializer | \(\Delta_\phi(H,A,c,\tau)\) | 主线方法 |

不要把 full-sequence oracle distillation 作为主线。它太慢，而且和最终 runtime 形态不一致。full-sequence oracle / oracle-label distillation 只作为 optional upper-bound 或 teacher-label ablation。

---

# Stage 0：B-frontier oracle sanity

目标是证明：

> 在 frozen FloodNet + frozen VAE 不变的情况下，只优化 earliest future \(z_T\) frontier token，是否能提升 affected future metrics，同时不明显损伤 motion quality。

这一步已经由 `true_future_zT_local_oracle.py` 作为原型验证。它的语义是：

```text
只选择 beta >= 0.999 的 earliest future tokens
只 inject future initial z_T
不优化 committed z0 history
不优化 active x_beta
shadow rollout 后计算 affected future / full rollout metrics
```

Stage 0 的主要产出不是训练模型，而是确认 learned initializer 值得做。需要持续记录：

* frontier token ids / beta；
* affected frame range；
* tail no-effect window；
* optimized-minus-base noise norm；
* baseline / oracle 的 ADE、FDE、heading、path ratio；
* jerk / foot skating 等 motion-quality diagnostics。

如果 B-frontier oracle 对 affected future window 和 full rollout 都没有稳定收益，说明后续 learned initializer 的空间有限。

---

# Stage 1：Online residual initializer

这是当前主线。

不构造大规模 oracle \(Z_T^*\) 数据集，也不先做 supervised distillation。训练过程直接在 streaming rollout 中采样 decision state，通过 differentiable shadow rollout 把 affected future loss 反传到 initializer。

## 5.1 目标

训练轻量模块：

\[
\Delta_\phi(H_k,A_k,c_k,\tau_k)\rightarrow \Delta Z_T^{frontier}
\]

第一版使用 deterministic residual：

\[
Z_T^{frontier}
=
Z_{T,base}^{frontier}
+
\alpha \Delta_\phi(H_k,A_k,c_k,\tau_k)
\]

其中：

* \(H_k\)：最近 committed latent history；
* \(A_k\)：当前 active \(x_\beta\) boundary + beta / offset；
* \(c_k\)：text embedding；
* \(\tau_k\)：generated-anchor local trajectory；
* \(Z_{T,base}^{frontier}\)：earliest unactivated future \(z_T\)；
* \(\alpha\)：warmup scale，`alpha=0` 时必须严格等价 Gaussian baseline。

## 5.2 关键 helper：StreamLatentStateView

当前代码里还没有真正拆出的 `future_initial_noise` buffer，所以第一版不能假设有独立 buffer。需要先用 `model.generated + beta/schedule` 解释当前 latent cache：

```python
@dataclass
class StreamLatentStateView:
    committed_ids: torch.Tensor
    active_ids: torch.Tensor
    frontier_ids: torch.Tensor

    committed_latents: torch.Tensor
    active_latents: torch.Tensor
    frontier_base_zT: torch.Tensor

    active_beta: torch.Tensor
    frontier_beta: torch.Tensor
    active_offsets: torch.Tensor
    frontier_offsets: torch.Tensor

    commit_index: int
    current_step: int
```

第一版 frontier selection 可以用：

```text
beta >= 0.999
index > commit_index
```

更稳的版本还应记录 `token_update_count == 0`，因为真正的 pure \(z_T\) 语义是“还没有被 triangular schedule update 过”，而不只是“当前 beta 看起来接近 1”。

## 5.3 最小模型

第一版不要追求结构复杂：

```text
history encoder: mean-pool / small Transformer
active encoder: mean-pool with beta / offset embedding
traj encoder: MLP or 1D conv
text projection: Linear
fusion: MLP
output head: M * latent_dim
```

输出：

```text
delta_zT: same shape as frontier_base_zT
```

不要一开始做 \(\mu,\sigma\)，不要一开始做 RL。

## 5.4 Shadow rollout 必须可微

训练 initializer 时：

```python
for p in ldf.parameters():
    p.requires_grad_(False)
for p in vae.parameters():
    p.requires_grad_(False)
```

但 LDF / VAE forward 不能包在 `torch.no_grad()` 里。梯度需要从 loss 经过 decoded motion、VAE decode、LDF denoising step、frontier \(z_T\) 回到 initializer。

实现上可以沿用 oracle prototype 的方式调用：

```python
DiffForcingWanModel.stream_generate_step.__wrapped__(...)
```

或者抽出正式的 differentiable helper。

## 5.5 Loss

frontier \(z_T\) 不是当前 commit 的 action，它会延迟生效。因此训练 loss 必须落在 `affected_future_window`，不要用当前 commit frames。

第一版只用：

\[
L =
L_{xz}
+
0.05 L_{vel}
+
10^{-4}\|\Delta Z_T\|^2
\]

heading、continuity、jerk、foot skating 第一版只作为 eval diagnostics。等 residual 版本有效后再加入训练 loss。

## 5.6 Trajectory context

\(\tau_k\) 必须使用 generated anchor local trajectory：

```text
anchor_root = generated_current_root
anchor_heading = generated_current_heading
traj_local = world_to_local(target_traj_world, anchor_root, anchor_heading)
```

不要用 target anchor，也不要用 `pred - pred[0] + target[0]` 抹掉 drift。否则训练看到的是理想局部轨迹，推理时面对 generated drift，会产生 mismatch。

## 5.7 第一版训练流程

```python
for step in train_steps:
    sample = sample_sequence()

    state = init_stream_state(sample)
    state = rollout_to_random_decision_point(
        state,
        rollin="gaussian",
    )

    view = StreamLatentStateView.from_model(model, state)

    H = view.committed_latents[-history_length:]
    A = view.active_latents
    c = text_embedding
    tau = build_generated_anchor_local_traj(state, target_traj)

    frontier_ids = view.frontier_ids[:M]
    base_zT = view.frontier_base_zT[:M].detach()

    delta = initializer(
        H,
        A,
        view.active_beta,
        c,
        tau,
        view.frontier_offsets[:M],
    )
    zT = base_zT + alpha(step) * delta

    shadow = differentiable_shadow_rollout(
        snapshot=state.snapshot(),
        frontier_ids=frontier_ids,
        zT_frontier=zT,
        rollout_steps=K,
        commit=False,
    )

    loss_xz = xz_loss(shadow.affected_root_xz, shadow.affected_target_xz)
    loss_vel = vel_loss(shadow.affected_root_xz, shadow.affected_target_xz)
    loss_noise = delta.pow(2).mean()
    loss = loss_xz + 0.05 * loss_vel + 1e-4 * loss_noise

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(initializer.parameters(), 1.0)
    optimizer.step()
```

Stage 1 使用 Gaussian rollin 起步。第一版先把可微训练闭环跑通并验证 eval 收益。

---

# Stage 2：Scheduled rollin

当 Stage 1 在 Gaussian rollin 下有效后，再逐步让训练分布接近推理分布：

\[
p_{init}=\min(1,\frac{step}{T_{warmup}})
\]

```python
if random.random() < p_init:
    use_initializer_frontier_zT(state)
else:
    use_gaussian_frontier_zT(state)
```

这一步解决 exposure bias：initializer 不能只在 Gaussian-generated state 上训练，否则推理时面对自己造成的 drift 可能退化。

---

# Stage 3：Probabilistic prior / KL / RL

只有 residual initializer 有稳定收益后，再考虑：

1. 输出 \(\mu,\sigma\)，升级为 Gaussian prior；
2. 加 KL 或 distribution regularization；
3. 加 heading / continuity / jerk / foot skating 到训练 loss；
4. optional oracle-label distillation ablation；
5. optional PPO / policy gradient。

这些都不是第一版主线。

---

## 5. 必须通过的 sanity checks

1. `alpha=0` 等价 Gaussian baseline；
2. LDF / VAE 参数梯度为 `None`；
3. initializer 参数存在非零梯度；
4. inject 只改变 `frontier_ids`，committed / active latent 不变；
5. frontier token 的 stage 是 `initial_zT`；
6. loss window 覆盖 frontier token 生效后的 affected frames；
7. 记录 `frontier_ids`、`affected_frame_range`、`tail_no_effect_window`。

---

## 6. Eval 对比

每次 learned initializer eval 至少比较：

1. Gaussian baseline；
2. A active noisy-state oracle；
3. B-frontier oracle upper bound；
4. learned residual initializer。

指标分三组：

```text
affected future: affected_ADE / affected_FDE / affected_velocity_error
full rollout: full_ADE / full_FDE / heading_error / path_ratio
motion quality: jerk / foot_skating / root_smoothness
```

---

## 7. 预期结论

如果 Stage 1 / Stage 2 成功，我们可以得到一个更准确的研究结论：

> FloodNet 的控制误差不一定只来自 denoiser 不理解轨迹，也可能来自 streaming future token 仍然从无条件 Gaussian noise 开始。对于 streaming motion diffusion，历史动作和 generated drift 已经提供强运动状态先验，因此学习 generated-state-conditioned frontier latent initialization 是合理且有效的。

进一步的论文表述可以是：

> Existing streaming diffusion motion generators initialize future tokens from an unconditional Gaussian distribution, ignoring the strong temporal prior contained in generated motion history. We propose a context-conditioned frontier latent initialization prior that predicts residual corrections for the earliest unactivated diffusion noise tokens, enabling a frozen streaming generator to denoise from more plausible and controllable latent intentions.

---

## 8. 当前最小可执行版本

最小任务：

1. 实现 `StreamLatentStateView`，从 `model.generated + beta/schedule` 解释 committed / active / frontier；
2. 确认 B-frontier oracle diagnostics 足够记录有效 window 和 tail no-effect window；
3. 训练 small residual initializer，`frontier_tokens=5`；
4. Gaussian rollin 起步；
5. differentiable shadow rollout K steps；
6. loss 只用 `xz + 0.05 * velocity + 1e-4 * delta_norm`；
7. trajectory context 使用 generated-anchor local trajectory；
8. 通过 sanity checks；
9. 与 Gaussian baseline、A-line oracle、B-frontier oracle 对比。

---

## 9. 总结

这项工作的核心不是“模仿 DART 做 RL goal-reaching”，而是学习 DART 中更本质的思想：

> diffusion initial noise 可以被看成动作生成的 latent intention。
> 对于 streaming motion，历史动作和目标轨迹应该共同决定这个 latent intention，而不是每个 chunk 都从无条件 Gaussian noise 开始。

因此，FloodNet 的下一步可以是：

[
Z_T^{k:k+K}
===========

f_\phi(
\text{history}_k,
\text{text}*k,
\text{trajectory}*{k:k+K},
\epsilon
)
]

再由现有 FloodNet 完成：

[
Z_0^{k:k+K}
===========

F_\theta(
Z_T^{k:k+K},
\text{history}_k,
\text{text}*k,
\text{trajectory}*{k:k+K}
)
]

最终得到更精确、更自然、更稳定的 streaming trajectory-controlled motion。
