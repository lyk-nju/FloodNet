# FloodNet 控制损失模式探索实验总结

## 一、背景与目的

FloodNet 在 FloodDiffusion 的流式动作生成基础上，通过 **WanControlNet** 注入根节点 XZ 轨迹条件。  
主干（WanModel）完全冻结，仅训练 ControlNet 分支。

训练损失由两部分组成：
```
L_total = L_mse（latent MSE，扩散主损失）+ control_loss_weight × L_control（XZ平面轨迹MSE）
```

**本次实验目的**：在小规模条件下系统比较 6 种控制损失的梯度策略（`control_loss_train_mode 1-6`），筛选出最适合后续大规模训练的方式。

---

## 二、控制损失模式定义

| Mode | 监督范围 | 坐标系 | past detach | 说明 |
|------|----------|--------|------------|------|
| 1 | active window（当前去噪窗口，约 17-20 帧） | 绝对坐标 | 否 | 梯度经 cumsum 流向所有 token |
| 2 | active window | 绝对坐标 | 是 | 梯度仅限 active window；存在 Sisyphus 效应 |
| 3 | full sequence（全部帧） | 绝对坐标 | 否 | 均衡梯度，在早期实验中验证有效 |
| 4 | full sequence | 绝对坐标 | 是 | 全覆盖损失，梯度仅限 active window |
| 5 | active window | 相对位移 | — | pred anchor（存在 bug：固定绝对偏移 → 损失趋近于零） |
| 6 | active window | 相对位移 | — | GT anchor（修复 Mode 5 bug：真实位移误差）|

**Mode 5 bug 说明**：`anchor = pred_xz[:, 0:1, :].detach()`，损失实际变为 `(e[t]−e[0])²`（误差变化量，而非轨迹误差本身）。固定绝对偏移的模型可得零损失。Mode 6 使用 GT anchor 修复此问题，但实测两者控制损失从训练初期均趋近 0（根因：主干已生成自然运动，active window 内相对位移在归一化空间中量级极小）。

另：Mode 3 + Scheduled Sampling（SS）会导致灾难性退化（长序列误差逐段累积），已确认不兼容。

---

## 三、实验设置

### 训练集构建

| 实验类型 | 训练集 | 测试集 | 训练步数 |
|---------|--------|--------|---------|
| **Overfit** | 与测试集相同的 8 个样本 | 8 个固定样本 | 2k / 4k / 6k / 8k / 10k |
| **Gen** | 300 个分层样本（hard/medium/easy 各 100 条） | 同上 8 个样本 | 2k / 4k / 6k / 8k / 10k |

**分层采样来源**：从 `rank_easy/medium/hard.txt` 各抽 100 条，排除 8 个测试样本，存为 `train_stratified_300.txt`。

### 8 个测试样本

| ID | T（帧） | 类型 |
|----|--------|------|
| 000021 | 177 | 长序列 |
| 001168 | 141 | 长序列 |
| 004822 | 197 | 长序列 |
| 000742 | 65 | 短序列 |
| 000749 | 89 | 短序列 |
| 000818 | 113 | 短序列 |
| 003245 | 65 | 短序列 |
| 007767 | 93 | 短序列 |

### 评估指标

- **ADE**：所有 traj_mask=1 帧的 XZ L2 均值（米）
- **FDE**：最后一个 mask 帧的 XZ L2 距离
- **ADE_long / ADE_short**：仅计算长/短序列样本的 ADE
- **late_long**：长序列第 60 帧之后各 20 帧窗口 seg_mse 均值（衡量长程误差）
- **pfx_long**：长序列累积 prefix_mse 的末尾值（误差总积累）

### 基准线

无轨迹控制的原始 FloodDiffusion（240k 步预训练）：  
`ADE=0.6985  FDE=1.0203  ADE_long=1.3389`

---

## 四、实验结果

### 4.1 Overfit 最佳结果（各模式最低 ADE）

| Mode | 最佳 checkpoint | ADE | FDE | ADE_long | late_long |
|------|----------------|-----|-----|----------|-----------|
| 1 | mode1_ss_10k | 0.2065 | 0.3266 | 0.3686 | 0.3727 |
| 2 | mode2_ss_10k | 0.1554 | 0.3302 | 0.2346 | 0.1648 |
| 3 | mode3_overfit_2k | 0.1664 | 0.2840 | 0.2747 | 0.1936 |
| 4 | mode4_ss_10k | 0.1649 | 0.3084 | 0.2620 | 0.1663 |
| 5 | mode5_ss_8k | **0.1389** | **0.2489** | **0.2189** | **0.1272** |
| 6 | mode6_overfit_6k | 0.1800 | 0.3274 | 0.2578 | 0.1620 |

所有模式均显著优于基准（ADE 降低约 70-80%）。

### 4.2 Gen 最佳结果（各模式最低 ADE，train=300，test=8）

| Mode | 最佳 checkpoint | ADE | FDE | ADE_long | late_long |
|------|----------------|-----|-----|----------|-----------|
| 1 | mode1_gen_10k | 0.4750 | 0.6994 | 0.8417 | 1.7384 |
| 3 | mode3_gen_10k | 0.5061 | 0.7806 | 0.9110 | 2.0676 |
| 5 | mode5_gen_10k | **0.4503** | 0.8392 | **0.7491** | 1.7046 |
| 6 | mode6_gen_10k | **0.4503** | 0.8392 | **0.7491** | 1.7046 |

注：Mode 5 和 Mode 6 的 gen 结果完全相同（两者均使用相同 checkpoint，gen 阶段差异消失）。

### 4.3 关键观察

1. **Mode 3 + SS 灾难退化**：sample 001168（T=141）的 seg_mse 在末尾窗口升至 2.655，short 样本不受影响。Mode 3 的全序列绝对坐标损失在 SS 引入的误差积累下会自我放大。

2. **Mode 2/4（detach past）的 Sisyphus 效应**：控制损失不收敛，但 eval 指标仍然不差——因为在推理时没有 L_mse 持续移动 past token，方向性梯度信号依然有用。

3. **Mode 5/6 控制损失趋近于零**：从训练初期起控制损失就在 0.0x 量级。根因是预训练主干已产生自然运动；active window 内 17-20 帧的相对位移在归一化空间中本就很小。ControlNet 主要通过 L_mse 的间接路径学习轨迹跟随。

4. **Overfit vs. Gen 性能差距**：ADE 从 ~0.15-0.21（overfit）到 ~0.45-0.51（gen），说明 10k 步 / 300 样本仍不足以充分泛化。

5. **短序列（T<100）各模式差异极小**：ADE 稳定在 0.09-0.18，短序列对控制模式不敏感；长序列（T>140）是区分各模式优劣的核心指标。

---

## 五、结论与后续方向

### 选定方案：Mode 1 + SS

**理由**：
- 对流式无限生成最友好：active window 损失与序列长度无关，不会随序列增长失效
- SS（Scheduled Sampling）显著改善长序列误差积累（mode1_ss_10k 的 late_long 从 1.1 降至 0.37）
- Mode 3 的全序列损失在 SS 下灾难退化，不适合流式场景
- Mode 2/4 控制损失不收敛，长期稳定性未知
- Mode 5/6 实质等价，控制信号太弱

### 下一步

- 在**全量 HumanML3D 训练集**（~14,616 样本）上以 Mode 1 + SS 进行大规模正式训练
- 训练步数目标 250k+
- 评估时使用 `--num_runs 5~10` 取均值以稳定指标

---

## 六、相关文件

| 文件 | 说明 |
|------|------|
| `train_ldf.py` | Mode 1-6 控制损失实现（`_compute_control_loss_xz`） |
| `configs/ldf_copy1.yaml` | Mode 1 训练配置（copy1=mode1，copy3=mode3，copy5=mode5，copy6=mode6） |
| `eval/eval_generation_metrics.py` | 评估脚本（ADE/FDE/seg_mse/prefix_mse/jitter） |
| `eval/README.md` | 评估脚本使用说明 |
| `eval/overfit/` | Overfit 实验结果（train=test=8 samples） |
| `eval/gen/` | Gen 实验结果（train=300, test=8） |
| `eval/baeline_240k/` | 无控制基准结果 |
