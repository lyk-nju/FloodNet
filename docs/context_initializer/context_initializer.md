
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

## 4. 后续计划

当前阶段不要直接上复杂 RL。先分两步推进。

---

# Stage 1：验证“好 (z_T) 是否存在”

这是最关键的第一步。
我们要先证明：

> 在 frozen FloodNet + frozen VAE 不变的情况下，只优化初始噪声 (z_T)，是否能显著提升轨迹跟踪指标，同时不明显损伤 motion quality。

如果这个问题验证失败，说明 FloodNet 的 sampling noise space 对轨迹控制帮助有限，后续 amortized initializer / RL initializer 意义不大。
如果验证成功，说明确实存在更好的 trajectory-aware sampling seeds，后续才值得训练 initializer。

## 4.1 Oracle latent initialization optimization

对每个测试样本，冻结：

* FloodNet / LDF；
* VAE；
* text encoder / precomputed text embedding；
* trajectory condition pipeline。

只优化：

[
Z_T
]

目标：

[
Z_T^*
=====

\arg\min_{Z_T}
L_{\text{traj}}
+
\lambda_v L_{\text{vel}}
+
\lambda_h L_{\text{heading}}
+
\lambda_s L_{\text{smooth}}
+
\lambda_n L_{\text{noise-reg}}
]

其中：

### 轨迹位置损失

[
L_{\text{traj}}
===============

\frac{1}{T}
\sum_t
|root_t - \tau_t|_2^2
]

### 终点损失

[
L_{\text{fde}}
==============

|root_T - \tau_T|_2^2
]

### 速度损失

[
L_{\text{vel}}
==============

\frac{1}{T}
\sum_t
|v_t - v_t^{ref}|_2^2
]

### heading 损失

[
L_{\text{heading}}
==================

\frac{1}{T}
\sum_t
d_{\text{angle}}(\psi_t,\psi_t^{ref})
]

### 平滑损失

[
L_{\text{smooth}}
=================

\text{jerk}(x)
]

### noise regularization

为了避免优化后的 (Z_T) 离标准高斯太远：

[
L_{\text{noise-reg}}
====================

|Z_T|_2^2
]

或者约束其 norm / variance 接近标准高斯。

## 4.2 Stage 1 的实验设置

建议先做离线验证，不改 runtime 正式逻辑。

新增一个 eval prototype，例如：

```text
eval/ldf/latent_initializer/
  optimize_noise.py
  losses.py
  rollout_wrapper.py
  diagnostics.py
  README.md
```

输入：

* 已训练 FloodNet ckpt；
* 已训练 VAE ckpt；
* eval case，例如 000021 或已有 runtime benchmark cases；
* 原始 trajectory condition；
* text condition；
* 初始 Gaussian (Z_T)。

输出：

* baseline Gaussian init 结果；
* optimized (Z_T^*) 结果；
* 对比 npz；
* 轨迹图；
* root xy plot；
* heading plot；
* loss curve；
* metrics table。

## 4.3 Stage 1 对比组

至少比较：

### A. Gaussian Init Baseline

原始 FloodNet：

[
Z_T \sim \mathcal{N}(0,I)
]

### B. Best-of-K Gaussian Init

采样 K 个随机 (Z_T)，选择轨迹指标最好的一个：

[
Z_T^{best}
==========

\arg\min_{Z_T^i}
L_{\text{traj}}(Z_T^i)
]

用于判断随机采样能带来多少提升。

### C. Oracle Optimized Init

从 Gaussian 初始化出发，对 (Z_T) 做梯度优化：

[
Z_T \leftarrow Z_T - \eta \nabla_{Z_T}L
]

用于判断 noise space 中是否存在明显更好的 seed。

### D. Optional：Root Replacement Oracle

作为强 oracle 参考：decode 后 root replacement，再 encode 回 latent。
这个不是最终方法，只是用于估计上限。

## 4.4 Stage 1 成功标准

如果 optimized (Z_T) 相比 Gaussian baseline：

* ADE 明显下降；
* FDE 明显下降；
* heading error 下降；
* path ratio 更接近 1；
* replan / turn 场景更稳定；
* motion 不明显崩坏；
* foot skating / jerk 不显著恶化；

则说明：

> 好的 trajectory-aware (Z_T) 确实存在。

此时进入 Stage 2。

如果 optimized (Z_T) 只能降低 root error，但 motion quality 明显崩坏，则说明需要加入 stronger motion regularization 或限制 (Z_T) 分布。
如果 optimized (Z_T) 几乎无法提升，则说明当前 FloodNet 的 denoising dynamics 对初始 noise 不敏感，或者 trajectory condition 已经主导生成，initializer 方向暂时价值有限。

---

# Stage 2：训练 amortized initializer

只有 Stage 1 成功后再做。

目标：

[
f_\phi(h_k,\ c_k,\ \tau_k,\ \epsilon)
\rightarrow
Z_T^*
]

让模型学会快速预测接近 oracle optimized init 的 (Z_T)，避免每个样本都在线优化。

## 5.1 Supervised distillation

从 Stage 1 得到一批 oracle optimized seeds：

[
{(h_k,c_k,\tau_k,Z_T^{*,k})}
]

训练：

[
L_{\text{distill}}
==================

|f_\phi(h_k,c_k,\tau_k,\epsilon) - Z_T^{*,k}|_2^2
]

如果多个 (Z_T^*) 都能生成好结果，不建议强行学唯一解。可以改成分布预测：

[
q_\phi(Z_T|h,c,\tau)
====================

\mathcal{N}(\mu_\phi,\sigma_\phi)
]

并结合：

[
L_{\text{KL}}
=============

D_{\text{KL}}(q_\phi(Z_T|h,c,\tau)\ ||\ \mathcal{N}(0,I))
]

保持 initializer 不要严重偏离标准高斯。

## 5.2 RL fine-tuning

在 supervised initializer 之后，可以进一步使用 RL fine-tune。

Policy：

[
\pi_\phi(h_k,c_k,\tau_k)
\rightarrow
Z_T^k
]

Environment：

```text
Frozen FloodNet + Frozen VAE + streaming rollout evaluator
```

Reward：

[
R =
-\text{ADE}
-\lambda_f\text{FDE}
-\lambda_h\text{HeadingErr}
-\lambda_v\text{VelocityErr}
-\lambda_s\text{Skating}
-\lambda_j\text{Jerk}
]

可选 semantic reward：

[
R_{\text{text}}
===============

\text{text-motion similarity}
]

RL 的作用不是从零学控制，而是微调 initializer，使其在长 rollout、replan、dynamic trajectory update 下更稳。

---

## 6. 对 Codex 的实现要求

请优先做 Stage 1，不要直接实现 RL。

### 6.1 不要改动现有主训练逻辑

现有 `recon` 分支的 trajectory control、runtime eval、VAE、LDF 主逻辑先保持不动。
新增 eval prototype 即可。

### 6.2 先实现 oracle noise optimization

需要支持：

* 从指定 eval case 加载 condition；
* 初始化 (Z_T)；
* 调用 frozen FloodNet denoising；
* 调用 VAE decode；
* 计算 root trajectory loss；
* 对 (Z_T) 反向传播；
* 保存优化过程和对比结果。

### 6.3 必须保存 diagnostics

每次实验保存：

```text
baseline_motion.npy / npz
optimized_motion.npy / npz
target_trajectory.npy / npz
optimized_ZT.npy
loss_curve.json
metrics.json
root_xy_plot.png
heading_plot.png
debug_summary.txt
```

### 6.4 metrics 至少包括

* ADE；
* FDE；
* heading error；
* path ratio；
* root position before / after；
* velocity error；
* jerk；
* optional foot skating；
* optimization time；
* number of gradient steps；
* (Z_T) norm / mean / std；
* deviation from Gaussian statistics。

---

## 7. 预期结论

如果 Stage 1 成功，我们可以得到一个非常明确的研究结论：

> FloodNet 的控制误差不一定来自 denoiser 不理解轨迹，而可能来自每个 streaming step 仍然从无条件 Gaussian noise 开始。对于 streaming motion diffusion，历史动作本身已经提供了强运动先验，因此学习 history-trajectory-conditioned latent initialization 是合理且必要的。

进一步的论文表述可以是：

> Existing streaming diffusion motion generators initialize each generation step from an unconditional Gaussian distribution, ignoring the strong temporal prior contained in motion history. We propose to learn a context-conditioned latent initialization prior that predicts trajectory-aware diffusion starting points, enabling a frozen streaming motion generator to denoise from more plausible and controllable latent intentions.

---

## 8. 当前最小可执行版本

最小任务：

1. 选择一个已有 eval case，例如 000021。
2. 冻结当前 FloodNet / VAE。
3. 固定 text、history、trajectory condition。
4. 运行 baseline Gaussian init。
5. 对同一个 (Z_T) 做 50-200 step 梯度优化。
6. 比较优化前后的 ADE/FDE/heading/path ratio。
7. 保存 diagnostics。
8. 如果单样本有效，扩展到 20-50 个 cases。
9. 如果统计有效，再进入 amortized initializer。

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
