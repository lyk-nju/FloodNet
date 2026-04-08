# FloodNet：`control_loss` 低不下去的关键原因（坐标原点、对齐方式与权重尺度）

### 背景与结论一句话（先纠正一个容易踩的前提）

本项目里要分清楚两件事：

- **LDF 扩散训练本身**：模型内部确实有 `chunk_size`（默认 5）用于 active window 的扩散 MSE，对齐论文做法。
- **显式控制损失 `control_loss`**（`train_ldf.py::_compute_control_loss_xz`）：它是否按窗口（active window）计算，取决于训练脚本从 config 读到的 `chunk_size_tokens` 是否为 `None`。

在当前的 `FloodNet/configs/ldf.yaml` 中，`model.params` **没有** `chunk_size` 这个 key，因此训练脚本里：

- `chunk_size_tokens = self.cfg.model.params.get("chunk_size", None)` 会得到 **`None`**
- `control_loss` 会走“**全序列**”分支：解码全序列 latent → 还原全程 root xz → 与 GT 全程 traj 对比（都从 (0,0) 开始）

因此 **在当前默认配置下，不会出现“窗口起点原点不一致”的结构性地板**；`control_loss≈0.08` 更应理解为“真实的轨迹预测误差（在该 loss 定义下）”。

> 下面第 3 节仍保留“窗口原点不一致”的分析，但它只在你显式让 `control_loss` 按窗口算时才会发生（例如在 `model.params` 里补上 `chunk_size`，或强制只解码窗口）。

---

### 1. 这套 263 表示里，root 绝对轨迹是怎么恢复的

根节点位置不是直接存储的（除 `y` 之外），而是通过 root 平面速度积分得到。核心实现：

- `FloodNet/utils/motion_process.py::recover_root_rot_pos`
- `FloodDiffusion/utils/motion_process.py::recover_root_rot_pos`（同实现）
- `MotionLCM/mld/data/humanml/scripts/motion_process.py::recover_root_rot_pos`（同实现）

关键行为（最值得注意）：

- `r_pos = zeros(...)` 初始化为 0
- `r_pos[..., 1:, [0,2]] = data[..., :-1, 1:3]`：用 **前一帧** 的 root 线速度填到 **下一帧**
- `cumsum` 得到位置轨迹

推论：

- 对任意输入序列，恢复出的 **第一帧 root \(x_0=z_0=0\)**（硬约定）
- 后续帧的 \(x,z\) 是从 0 累积出来的

这不是 bug，而是该表示的设计约定。

---

### 2. 最容易混淆的点：有两个“第 0 帧”

#### 2.1 clip 的第 0 帧（整段序列起点）

对 raw_data 中一段 motion（clip）而言，`traj[0]` 按约定确实是 (0,0)。

#### 2.2 窗口（chunk）的第 0 帧（整段中的某个中间起点）

在 `chunk` 训练中，控制损失取 GT 的窗口帧段：

- `gt_slice = [start_f:end_f]`
- 窗口第 0 帧对应 `traj[start_f]`

**除非窗口从整段开头开始，否则 `traj[start_f]` 通常已经累积位移，不再是 (0,0)。**

---

### 3. `chunk` 控制损失的“最大问题点”（最值得注意）

**仅当你让 `control_loss` 按窗口计算时**（即 `chunk_size_tokens` 生效，例如在 `model.params` 显式配置了 `chunk_size`）：

- pred 侧：只解码窗口 token → 用 `recover_root_rot_pos` 恢复轨迹  
  ⇒ 预测窗口第 0 帧 root xz **必为 (0,0)**
- gt 侧：切片得到 `traj[start_f:end_f]`  
  ⇒ GT 窗口第 0 帧为 `traj[start_f]`，一般 **不为 (0,0)**

当 `mask_ratio=1.0` 时，窗口起点帧必被监督（mask=1），会产生一个不可消除项：

\[
(\,0-x_{start}\,)^2 + (\,0-z_{start}\,)^2
\]

这会造成：

- `control_loss` 出现明显地板（plateau），很难压到 0.01
- 学习率调度只能影响“接近地板的速度/抖动”，无法从根本上消掉该项

**而在当前默认配置（`chunk_size_tokens=None`）下**：

- `control_loss` 解码全序列并从 `traj[0]` 对齐开始比较
- 不存在上述“窗口第 0 帧固定为 (0,0) 但 GT 窗口起点不为 (0,0)”的问题

---

### 4. 为什么 MotionLCM “看起来能做到密集控制”，但不代表 FloodNet 这个 loss 一定能到 0.01

#### 4.1 MotionLCM 也使用同样的 root 恢复约定（第一帧 xz=0）

MotionLCM 的 `recover_root_rot_pos` 与 FloodNet 相同：第一帧 \(x=z=0\)。

#### 4.2 但 MotionLCM 的 control supervision 与时间组织不同

MotionLCM 的实现（代码层面）主要是：

- `VAE decode → feats2joints → joints/hint 的 masked loss`
- 其生成/训练不是 FloodDiffusion/LDF 的 streaming `chunk` 机制（不是“只解码窗口再监督绝对 root xz”）

因此 MotionLCM 能在它的 `cond_loss` 指标上压很低，并不等价于 FloodNet 当前的 `root xz absolute`、`chunk-only decode` 控制损失也能压到 0.01。

---

### 5. 文档/实现不一致的地方（需要特别注意）

`FloodNet/Todolists/Task5-loss.md` 中曾描述 control loss “不限制 active window”，但当前 `train_ldf.py::_compute_control_loss_xz` 的实现会在 `chunk_size_tokens` 存在时：

- 只解码最后 `chunk_size` 个 token
- 并将 GT 对齐到 `[start_f:end_f]` 的窗口

如果后续要做实验对照，建议以代码为准，并更新/补充对应设计文档，避免团队内理解偏差。

---

### 6. 如果目标是“`mask_ratio=1.0` 且 `control_loss≈0.01`”，应优先改什么（而不是先改 scheduler）

这里先分两种情形：

- **情形 A（当前默认配置：`chunk_size_tokens=None`，控制损失按全序列算）**  
  `control_loss≈0.08` 是“真实误差”，更需要优先关注 **梯度/权重尺度是否足够**（见下方补充）。
- **情形 B（你显式让控制损失按窗口算）**  
  此时可能出现第 3 节描述的“窗口原点不一致”问题，才需要优先调整“监督量/对齐方式”。

对情形 B：要实现极低控制误差，需要优先调整“监督量/对齐方式”之一：

- **相对量监督（更符合 chunk + self-forcing）**  
  用 \(\Delta x,\Delta z\) 或 root 速度（`[1:3]`）做监督，避免绝对起点问题。
- **显式对齐窗口原点（更像 teacher forcing 锚点）**  
  让 pred/gt 在窗口起点对齐后再算绝对误差（例如平移对齐或在积分时注入 init xz）。
- **监督对象改为 joints（对齐 MotionLCM/OmniControl “dense spatial control”语义）**  
  对 pelvis/feet 等 joints 的 xyz（或 xz）做 masked loss，配合稳定的 mask 归一化策略。

补充（情形 A 最值得注意的“问题最大处”）：**控制信号权重可能过弱**

在常见日志尺度下，扩散 MSE（latent space）可能在 ~7.x，而 `control_loss` 在 ~0.0x，直接相加时控制项对总梯度的影响可能很小。  
这会导致“控制 loss 下降很慢/效果一般”，即便不存在窗口原点问题。此时优先排查/尝试：

- 增大 `control_loss_weight`（例如 2、5、10 做短跑对照）
- 或对两类 loss 做更合理的尺度归一化（例如按维度/有效帧数/窗口长度归一）

---

### 7. 与 lr_scheduler 的关系（应如何看待）

当控制损失存在结构性地板时：

- `constant 1e-4` 可能表现为“安全但磨不动/平台期”
- `1e-3` 或大幅周期 LR 可能带来明显振荡
- `CosineAnnealingLR`（温和周期）可以帮助“更快接近地板/减少长尾极小 LR”，但不能消除地板本身

因此 scheduler 的正确定位是 **优化效率与稳定性**，而非解决 loss 定义的可达下限问题。

