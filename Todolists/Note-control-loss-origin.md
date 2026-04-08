# FloodNet：`chunk` 训练下 `control_loss` 卡住的关键原因（坐标原点与对齐）

### 背景与结论一句话

当前 FloodNet 的显式控制损失（`train_ldf.py::_compute_control_loss_xz`）在 `chunk` 模式下是 **“只解码最后窗口 → 还原 root 绝对 xz → 与 GT 窗口帧段的绝对 xz 做 MSE”**。  
在这套 263 表示的还原函数 `recover_root_rot_pos()` 里，**每次从一段 motion 序列恢复 root 轨迹时，第一帧 \(x=z=0\) 是硬约定**。  
因此当只解码窗口时，窗口的第 0 帧预测 root xz 会被固定成 (0,0)，但 GT 窗口起点 `traj[start_f]` 通常不为 (0,0)，会形成 **结构性的误差地板**，使 `control_loss` 很难降到极小（例如 0.01）。

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

当前实现中（当 `chunk_size_tokens` 生效）：

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

学习率调度会影响收敛速度与稳定性，但对结构性地板无能为力。要实现极低控制误差，需要优先调整“监督量/对齐方式”之一：

- **相对量监督（更符合 chunk + self-forcing）**  
  用 \(\Delta x,\Delta z\) 或 root 速度（`[1:3]`）做监督，避免绝对起点问题。
- **显式对齐窗口原点（更像 teacher forcing 锚点）**  
  让 pred/gt 在窗口起点对齐后再算绝对误差（例如平移对齐或在积分时注入 init xz）。
- **监督对象改为 joints（对齐 MotionLCM/OmniControl “dense spatial control”语义）**  
  对 pelvis/feet 等 joints 的 xyz（或 xz）做 masked loss，配合稳定的 mask 归一化策略。

---

### 7. 与 lr_scheduler 的关系（应如何看待）

当控制损失存在结构性地板时：

- `constant 1e-4` 可能表现为“安全但磨不动/平台期”
- `1e-3` 或大幅周期 LR 可能带来明显振荡
- `CosineAnnealingLR`（温和周期）可以帮助“更快接近地板/减少长尾极小 LR”，但不能消除地板本身

因此 scheduler 的正确定位是 **优化效率与稳定性**，而非解决 loss 定义的可达下限问题。

