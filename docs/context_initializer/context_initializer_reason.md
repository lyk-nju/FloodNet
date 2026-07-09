# FloodNet Frontier Noise Prior：建模思路、背景与研究目的

## 0. 一句话目标

我们希望把 FloodNet 从「每个未来 token 默认从无条件高斯噪声开始去噪」升级为：

> 在流式生成过程中，根据已经生成的历史动作、当前 active boundary、文本语义和未来局部轨迹，为最早一批尚未进入 denoise schedule 的 future tokens 生成一个更合理的初始噪声先验。

形式上，从：

\[
Z_T^{frontier} \sim \mathcal{N}(0,I)
\]

升级为：

\[
Z_T^{frontier} \sim q_\phi(Z_T^{frontier}\mid H_k,A_k,c_k,\tau_k)
\]

其中：

- \(H_k\)：已 commit 的历史动作 latent / motion state；
- \(A_k\)：当前 active denoising boundary，包括正在 commit 或 partially-denoised 的 token 状态；
- \(c_k\)：文本条件；
- \(\tau_k\)：未来局部轨迹 / navigation path / target trajectory；
- \(Z_T^{frontier}\)：最早一批尚未进入 triangular denoise schedule 的 future initial noise tokens。

这个模块不是替代 FloodNet，而是在 token 进入去噪过程之前，给它一个更合理的起点。

---

## 1. 背景：为什么当前 Gaussian initialization 不够理想

标准 diffusion 生成通常从无条件高斯噪声开始：

\[
z_T \sim \mathcal{N}(0,I)
\]

然后 denoiser 根据条件逐步恢复 clean latent：

\[
z_T \rightarrow x_\beta \rightarrow z_0
\]

这个假设在图像生成里比较自然，因为图像样本可以从无结构噪声开始。但在 **streaming motion generation** 中，这个假设并不完全合理。

原因是：动作未来不是从零开始的。对于第 \(k\) 个 streaming step，历史动作已经强烈约束了未来动作分布。历史里包含：

- 当前身体姿态；
- root position / velocity；
- heading；
- gait phase；
- foot contact state；
- body momentum；
- 当前 text style；
- 前面 rollout 已经产生的 drift；
- 当前 active denoising state。

因此，真正的未来动作分布更接近：

\[
p(x_{k:k+H}\mid H_k,A_k,c_k,\tau_k)
\]

而不是只依赖 text / trajectory 的无历史分布。

对应到 diffusion 起点，我们也不应该始终使用：

\[
p(z_T)=\mathcal{N}(0,I)
\]

而应该学习一个条件化采样先验：

\[
q_\phi(z_T\mid H_k,A_k,c_k,\tau_k)
\]

这就是本工作的核心建模动机。

---

## 2. DART 与 PULSE 的启发

### 2.1 DART 的关键启发：latent noise 可以作为控制变量

DART 的核心不是简单「用 RL 做 goal-reaching」，而是它把 latent diffusion 的初始噪声 \(z_T\) 看成可控变量。

DART 的生成过程是：

\[
H_i,c_i,z_T^i
\rightarrow
\text{latent denoiser}
\rightarrow
z_0^i
\rightarrow
\text{VAE decoder}
\rightarrow
X_i
\]

DART 的空间控制可以通过优化 latent noises：

\[
Z_T^*=
\arg\min_{Z_T}
F(\Pi(\text{ROLLOUT}(Z_T,H,C)),g)+cons(\text{ROLLOUT}(Z_T,H,C))
\]

也可以通过 RL policy 直接输出 \(z_T\)：

\[
\pi(H_i,g_i,s_i,c_i)\rightarrow z_T^i
\]

这告诉我们：如果 frozen motion generator 足够强，控制器不一定要直接生成 motion，也不一定要直接改 clean latent，而可以学习如何选择 generator 的 denoising 起点。

### 2.2 PULSE 的关键启发：latent action space 应该有 state-conditioned prior

PULSE 的重点不是 diffusion，而是 physics-based humanoid control。它先训练一个强 motion imitator，然后把 imitator 的 motor skills 蒸馏到 latent action space 中。

PULSE 的重要设计是学习一个由 proprioception 条件化的 latent prior：

\[
R(z_t\mid s_t^p)=\mathcal{N}(\mu_t^p,\sigma_t^p)
\]

而不是使用普通 VAE 中的零均值高斯先验。

背后的机制是：不同身体状态下，合理动作 latent 的分布本来就不同。例如 standing still、running、flipping midair 对应的 motor action distribution 不可能相同。因此 latent action prior 应该由当前 state 决定。

对应到 FloodNet：

\[
R_\phi(Z_T^{frontier}\mid H_k,A_k,c_k,\tau_k)=\mathcal{N}(\mu_\phi,\sigma_\phi)
\]

这里 \(H_k,A_k\) 就相当于 motion generation 中的 proprioception / motion state。

### 2.3 我们结合二者的结论

- DART 告诉我们：\(z_T\) 是可优化、可控制的 latent action。
- PULSE 告诉我们：这个 latent action 不应该从固定 \(\mathcal{N}(0,I)\) 盲采，而应该有 state-conditioned prior。

因此，适合 FloodNet 的方向是：

> 学习一个 history-conditioned / trajectory-aware frontier noise prior，让未来 token 在进入 triangular denoise schedule 之前，就携带更合理的 motion intention。

---

## 3. FloodNet 中 state/action 的重新理解

FloodNet 的 VAE latent 统一表征了 motion state 和 future action/intention。

当 latent 已经被生成并 commit 到 history 时：

\[
z_0^{<k}
\]

它表示当前身体状态，是 **state representation**。

当 latent 还在未来 horizon 中、尚未进入 denoise schedule 时：

\[
z_T^{j},\quad j>k
\]

它表示即将被生成的未来动作意图，是 **latent action / motor intention**。

因此，我们可以把 FloodNet 建模成一个 latent state-space controller：

\[
s_k=(H_k,A_k,c_k,\tau_k)
\]

\[
a_k=Z_T^{frontier}
\]

\[
\text{Frozen FloodNet}: (s_k,a_k)\rightarrow z_0\rightarrow x
\]

其中：

- \(H_k\)：已经 commit 的 clean latent history；
- \(A_k\)：当前 active noisy latent state / denoising boundary；
- \(Z_T^{frontier}\)：未来最早一批尚未激活的 initial noise；
- Frozen FloodNet 负责从 \(Z_T\) 到 \(z_0\) 的 denoising dynamics；
- VAE decoder 负责将 clean latent 解码成 motion。

---

## 4. 为什么优化 current active \(x_\beta\) 不等价于 initializer

当前 active window 中的 token 状态并不一致。由于 triangular schedule，一个 token 会经历：

\[
z_T\rightarrow x_\beta\rightarrow z_0
\]

因此当前 cache 里可能同时存在：

- 已 commit 的 clean-ish \(z_0\)；
- 正在 denoise 的 \(x_\beta\)；
- 还没进入 denoise 的 pure \(z_T\)。

如果我们优化当前 active window 的 `model.generated[start:end]`，实际优化的是当前 denoising state \(x_\beta\)，不是最初始 \(z_T\)。这种方法可以作为 runtime correction oracle，但不能支撑「学习更好的初始去噪起点」这个理论 claim。

因此我们区分三条线，并明确主次：

| 实验线 | 优化变量 | 语义 | 用途 |
|---|---|---|---|
| A | 当前 active \(x_\beta\) | noisy-state correction | 工程 upper bound / runtime correction |
| B-frontier oracle | 未来最早未激活 \(Z_T^{frontier}\) | true local zT oracle | 验证 frontier \(z_T\) 可控性 |
| Learned residual initializer | \(\Delta_\phi(H,A,c,\tau)\) | online residual initializer | 主线方法 |

full-sequence \(Z_T\) oracle / oracle-label distillation 只作为 optional upper-bound 或 teacher-label ablation，不作为主线。我们的主线是 **learned residual initializer for B-frontier future zT**。

---

## 5. 为什么选择 frontier tokens

如果只优化尚未进入 denoise schedule 的 future \(z_T\)，它不会立刻影响当前 commit token，因此会有延迟。但这是符合 initializer 语义的。

为了减少延迟，我们不应该选择很远的 future token，而应该选择：

> 当前时刻之后，最早一批还没有进入 denoise schedule 的 future initial noise tokens。

记为：

\[
\mathcal{U}_k=\{j\mid token_j\text{ has not entered denoise schedule}\}
\]

取最靠前的 \(M\) 个：

\[
Z_T^{frontier}=[z_T^{j_1},\ldots,z_T^{j_M}]
\]

这批 token 既是真正的初始噪声，又即将进入 denoise schedule，因此是最适合被 initializer 控制的对象。

---

## 6. 当前 commit token 的角色

当前正在 commit 的 token 和 active boundary 不应该被优化，但应该参与 forward / decode / loss 计算。

它们的作用是 fixed context / boundary condition：

\[
Z_T^{frontier,*}
=
\arg\min_{Z_T^{frontier}}
L(\text{Rollout}(H_k,A_k,Z_T^{frontier}),\tau)
\]

其中 \(H_k,A_k\) 是 detached context。

这样未来 \(z_T\) 的选择会考虑：

- 当前 root 位置；
- 当前速度；
- 当前 heading；
- 当前 gait phase；
- 当前 generated drift；
- 未来轨迹如何从当前状态接上。

---

## 7. 最终方法：State-conditioned Frontier Noise Prior

最终模块定义为：

\[
q_\phi(Z_T^{frontier}\mid H_k,A_k,c_k,\tau_k)
=
\mathcal{N}(\mu_\phi,\sigma_\phi)
\]

推理时：

\[
Z_T^{frontier}=\mu_\phi+\sigma_\phi\odot\epsilon,
\quad
\epsilon\sim\mathcal{N}(0,I)
\]

第一版为了稳定，可以先做 deterministic residual：

\[
Z_T^{frontier}=Z_{T,base}^{frontier}+\alpha\Delta_\phi(H_k,A_k,c_k,\tau_k)
\]

如果 residual 版有效，再升级成概率 prior。

---

## 8. 预期贡献

这个方向的贡献不是「又加了一个 ControlNet」或「又调了 trajectory loss」，而是提出一个更基础的问题：

> Streaming motion diffusion 中，future tokens 是否应该始终从无条件 Gaussian noise 开始？

我们的回答是：不应该。

历史动作本身已经提供了强运动先验，因此更合理的是学习：

\[
q_\phi(Z_T^{frontier}\mid H_k,A_k,c_k,\tau_k)
\]

让 future token 在进入 denoise schedule 前就带有 trajectory-aware motion intention。

这把 FloodNet 从 trajectory-conditioned denoising 推进到：

> **controllable streaming motion diffusion with learned history-conditioned denoising initialization prior**。
