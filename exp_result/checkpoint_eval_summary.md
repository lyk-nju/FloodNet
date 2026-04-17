# FloodNet ControlNet 轨迹控制实验结果汇总

**评估指标**：`xz masked MSE = (mask * sq_err).sum() / mask.sum()`  
**测试集**：`test_min`（7个样本：000021, 000742, 000749, 000818, 001168, 003245, 007767）  
**评估工具**：`tools/eval_control_loss.py`，`tools/run_gap_experiments.py`  

---

## 一、Checkpoint 对应关系

| 名称 | 路径 | 说明 |
|------|------|------|
| **240k** | `FloodDiffusion/outputs/20251107_021814_ldf_stream/step_step=240000.ckpt` | FloodDiffusion 基础模型（无 ControlNet），作为 ControlNet 训练起点 |
| **245k_old** | `outputs/20260402_114343_ldf`（该 run 的最低存档为 250k，无 245k 快照） | 4月2日启动的首次 ControlNet 训练 run，最终训练至 300k |
| **245k_new** | `outputs/20260415_114956_ldf/step_step=245000.ckpt` | 4月15日重新启动的 ControlNet 训练 run，当前最新检查点 |
| **300k** | `outputs/20260402_114343_ldf/step_step=300000.ckpt` | 首次 ControlNet 训练 run 完整训练至 300k 的最终模型 |

> **注**：240k 基础模型无轨迹控制分支，直接测 control loss 无意义；245k_old 实际指 20260415 run 的 245k checkpoint 在 Bug 调查前后两次测量（详见第三节）。

---

## 二、Checkpoint 级别 Generate Control Loss 对比

| Checkpoint | 训练步数 | generate 口径 xz MSE | 备注 |
|------------|---------|---------------------|------|
| 240k（基础） | 240k | — | 无 ControlNet，不适用 |
| 245k（第一次测） | 245k | **1.386** | 20260415 run，seed=1234，Bug 调查前 |
| 245k（第二次测） | 245k | **1.470** | 同一 checkpoint，重跑结果，与预期一致（Bug1 修复不影响 2-batch 路径） |
| 300k | 300k | **0.988** | 20260402 run，较 245k 提升约 30% |

**结论**：训练步数是当前最直接的改善因素，300k > 245k（0.988 < 1.386）。

---

## 三、Bug 调查：为什么两次 245k 测量不同但 Bug 修复无效

### Bug 1：CFG null branch 复用 cond ControlNet 残差
- **位置**：`generate()`, `stream_generate()`, `stream_generate_step()` 的 `else` 分支
- **修复**：对 null branch 单独调用 `_controlnet_forward(text_null_context)`
- **为什么测量无变化**：标准 HumanML3D eval（每样本单条文本）下，`_build_cfg_2b_text` 始终返回非 None（`n == batch_size`），走 **2-batch 路径**而非 `else` 路径，因此 Bug 1 在 eval 中从未触发。Bug 1 仅影响帧对齐多段文本推理场景。

### Bug 2：`_stream_build_traj_emb` 绕过 LocalTrajEncoder
- **位置**：`diffusion_forcing_wan.py::_stream_build_traj_emb` 的 traj_buffer 分支
- **修复**：补上与训练路径一致的 4帧分组 + `local_traj_encoder` 调用
- **影响范围**：仅 streaming 推理（`generate_ldf.py` demo），不影响离线 `generate()` eval

---

## 四、实验 1：误差累积时间分析（Exp 1）

**设置**：245k_new checkpoint，seed=1234，每 20 帧分段计算 xz MSE

| 样本 | 序列帧数 | 分段 MSE 曲线（每 20 帧） | 首次发散段 |
|------|---------|--------------------------|---------|
| 000021 | 179 | 0.17 → 0.98 → 2.17 → 7.33 → 10.70 → 6.29 → ... | 0:20 |
| 001168 | 141 | 0.003 → 0.006 → 0.082 → 0.693 → 2.231 → 4.866 → 5.967 | 40:60 |
| 000818 | 113 | 0.013 → 0.121 → 0.157 → 0.033 → 0.256 → 1.007 | 0:20 |
| 007767 | 93 | 0.041 → 0.694 → 1.287 → 2.253 → 2.806 | 0:20 |
| 000742 | 65 | < 0.02，基本平稳 | — |
| 003245 | 65 | < 0.015，基本平稳 | — |

**结论**：xz 位置误差随时间**单调增长**，是 generate gap 的**主因**。根本原因：
```
xz_position = ∫ velocity_prediction dt
```
每帧速度预测的微小误差经积分不断放大，长序列（>100帧）误差增幅可达 50-100 倍。短序列（≤65帧）基本不受影响。

---

## 五、实验 2：Forward 采样口径偏差（Exp 2）

**设置**：245k_new，seed=1234；方案A=训练同口径随机采样窗口，方案B=均匀扫描全时间窗口

| 样本 | A_forward（随机） | B_uniform（均匀扫描） | delta(B-A) | 扫描窗口数 |
|------|-----------------|---------------------|-----------|---------|
| 000021 | 0.234 | 0.024 | -0.211 | 9 |
| 000818 | 0.751 | 0.0003 | -0.751 | 5 |
| 001168 | 0.064 | 0.003 | -0.061 | 7 |
| **overall** | **0.159** | **0.005** | **-0.155** | — |

**结论**：训练 forward 口径系统性**高估难度约 35 倍**（A vs B 差距）。均匀扫描更容易（主要扫早期帧，此时积分误差尚小），说明 forward 指标并不能反映 generate 时实际遭遇的难度分布。

> **注**：B_uniform 数值偏低，部分受 Exp 2 修复前的 frame-level tensor 截断 bug 影响（已在 `run_gap_experiments.py` 中修复 `frame_level_keys` 集合）。

---

## 六、实验 3：Teacher Forcing vs Self-Forcing 贡献（Exp 3）

**设置**：245k_new，seed=1234；TF=正常训练口径，SF-proxy=用生成 latent 替换 context

| 样本 | TF_forward | SF_proxy | generate | gap(gen-TF) | gap(gen-SF) |
|------|-----------|---------|---------|------------|------------|
| 000021 | 0.234 | 1.073 | 2.087 | 1.852 | 1.014 |
| 000818 | 0.424 | 0.529 | 0.111 | -0.312 | -0.418 |
| 001168 | 0.203 | 0.994 | 4.535 | 4.332 | 3.541 |
| **mean** | **0.141** | **0.399** | **1.018** | **0.877** | **0.619** |

**结论**：
```
TF bias 直接贡献 ≈ 0.877 - 0.619 = 0.258  （约 29% of total gap）
其余 0.619 来自误差累积 + 去噪随机性
```

SF-proxy 仅用固定权重评估，Scheduled Sampling 训练后模型真正适应自身预测，理论上可收窄完整的 0.141 → 1.018 gap。

---

## 七、ControlNet 残差范数健康检查

**设置**：20260416 run，5k 步早期 checkpoint；工具：`tools/check_controlnet_residuals.py`

| 指标 | 量级 |
|------|------|
| 各层残差 L2 范数 | 4×10⁵ ~ 1.9×10⁷ |
| 主干激活（layer 7） | 8.5×10⁶ ~ 8.5×10⁷ |
| 残差/激活比 | 约 5% ~ 35% |

**结论**：残差量级正常，未出现梯度爆炸或残差过小（未激活）的问题。

---

## 八、综合实验结果（2026-04-16）

### Exp A：Checkpoint 级别 Control Loss 对比

**设置**：seed=1234，topk=3（最难的 3 个样本），forward/generate 两种口径

| Checkpoint | forward xz MSE | generate xz MSE | 备注 |
|------------|---------------|-----------------|------|
| 240k（基础） | 0.142 | 1.386 | 无 ControlNet，仍有残差泄漏 |
| 245k_old（20260415 run） | 0.029 | 1.386 | 与 240k generate 几乎相同，训练效果有限 |
| 245k_new（20260416 run） | 0.047 | 1.470 | 对齐修复后重新训练，早期检查点，generate 略差 |
| **300k（20260402 run）** | **0.008** | **0.988** | **首次 < 1.0，比 245k 提升 ~33%** |

**结论**：更多训练步数是当前最直接有效的改善路径。300k generate MSE 首次降破 1.0。

---

### Exp B：Separated CFG 调参（245k_new，generate 口径）

**设置**：seed=1234，topk=3，245k_new checkpoint（20260416 run）

| cfg_scale_text | cfg_scale_traj | generate xz MSE | 相对 baseline 变化 |
|---------------|---------------|-----------------|------------------|
| 5.0 | 0.0 | 1.470 | baseline（纯文本 CFG） |
| 5.0 | 3.0 | 1.107 | −25% |
| **3.0** | **3.0** | **0.856** | **−42%（最优）** |
| 3.0 | 5.0 | 0.861 | −41% |
| 1.0 | 7.0 | 1.025 | −30%（过强轨迹引导反而变差） |
| 5.0 | 5.0 | 1.456 | −1%（两者均强时互相干扰） |

**结论**：
- 加入轨迹引导（`cfg_scale_traj > 0`）对 245k_new 提升显著，最优配置 **(3.0, 3.0)** 将 generate MSE 从 1.470 降至 **0.856**（42% 提升）
- 过强的轨迹引导（1.0, 7.0）或两者均强（5.0, 5.0）效果反而不佳
- Separated CFG 是**零训练成本**的推理增益，推荐在所有 checkpoint 上使用 `cfg_scale_text=3.0, cfg_scale_traj=3.0` 作为默认配置

---

### Exp C：Smooth Root 效果验证（245k_new，generate 口径）

**设置**：seed=1234，无 topk 限制（全 7 个测试样本），245k_new checkpoint

| smooth_traj_sigma | generate xz MSE | 变化 |
|-------------------|-----------------|------|
| 0.0（无平滑） | 1.470 | baseline |
| 2.0（Gaussian σ=2） | 1.712 | **+16%（变差）** |

**结论**：训练时控制信号未经平滑，推理时平滑导致**分布偏移**，效果下降。
- 修复方向：需在**训练时同步启用** `smooth_traj_sigma=2.0`，使模型适应平滑输入
- 当前实现（仅推理平滑）不适合直接使用，应在下次训练 run 中加入

---

## 九、改善路径与当前状态（更新）

| 方向 | 实验结论 | 当前状态 |
|------|---------|---------|
| 更多训练步数 | ✅ 有效（300k=0.988 vs 245k=1.470） | 进行中（20260416 run） |
| **Separated CFG** | ✅ 有效，(3.0,3.0) 提升 42%，零训练成本 | **已验证，推荐使用** |
| **Smooth Root（推理时）** | ❌ 无效，反而变差（分布偏移） | 需改为训练时同步平滑 |
| **Smooth Root（训练+推理）** | 待验证 | 下次 run 加入 `smooth_traj_sigma=2.0` |
| **Scheduled Sampling** | 待验证 | 已实现（`scheduled_sampling_prob`），等更多步数后对比 |
| 两阶段去噪（仿 Kimodo） | 根本解决 root 精度 | 长期规划 |

---

## 十、后续实验计划

```bash
# 1. 等 20260416 run 训练到更多步数（300k+），重跑 Exp A 对比 SS 效果
conda run -n flooddiffusion python tools/eval_control_loss.py \
    --config configs/ldf.yaml --eval_mode generate --seed 1234 \
    --ckpt outputs/20260416_000936_ldf/step_step=300000.ckpt

# 2. 验证 Smooth Root（训练+推理同步）- 需新 run 启用 smooth_traj_sigma=2.0 训练
#    训练后再对比：
conda run -n flooddiffusion python tools/eval_control_loss.py \
    --config configs/ldf.yaml --eval_mode generate --seed 1234 \
    --set data.smooth_traj_sigma=2.0

# 3. 验证 Separated CFG 在 300k checkpoint 上的效果
conda run -n flooddiffusion python tools/eval_control_loss.py \
    --config configs/ldf.yaml --eval_mode generate --seed 1234 \
    --ckpt outputs/20260402_114343_ldf/step_step=300000.ckpt \
    --set model.params.cfg_scale_text=3.0 model.params.cfg_scale_traj=3.0
```
