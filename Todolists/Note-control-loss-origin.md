# FloodNet：`control_loss` 低不下去的关键原因（坐标原点、对齐方式与权重尺度）

### 背景与结论一句话（与实现对齐）

本项目里要分清楚两件事：

- **LDF 扩散训练本身**：`DiffForcingWanModel` 使用 `chunk_size`（默认 **5**）定义 active window；扩散 MSE 只算最后 `chunk_size` 个 token 位置。
- **显式控制损失 `control_loss`**（`train_ldf.py::_compute_control_loss_xz`）：
  - **`chunk_size_tokens`** 来自 **`getattr(self.model, "chunk_size", None)`**（即模型实例上的 `chunk_size`，与 `diffusion_forcing_wan.py` 构造函数默认一致），**不是** `self.cfg.model.params.get("chunk_size", ...)`。
  - **默认行为**：对每个样本 **先 VAE 解码整条预测 latent**，用完整序列做 `recover_root_rot_pos` 积分；再在 **帧维** 上取与 active window 对应的切片  
    `[start_f, end_f)`，其中 `start_f = (T_token - chunk_size_tokens) * 4`、`end_f = T_token * 4`（当 `chunk_size_tokens` 非空且 `T_token > chunk_size_tokens` 时）。**只在切片内**用 `traj_mask` 做 masked MSE，与 [`target.md`](target.md) 规则 4、[`Task5-loss.md`](Task5-loss.md) 一致。
  - 若 `chunk_size_tokens` 为 `None` 或 `T_token ≤ chunk_size_tokens`，则退化为「有效重叠长度内」的全程监督（无尾窗限制）。

因此：**当前实现是「全序列 decode + 仅 active window 内监督」**，不是「只 decode 尾窗 latent」。在这种做法下，**不会出现**「只对尾窗 decode 导致窗口内第一帧 xz 被 recover 钉在 (0,0)、与 `traj[start_f]` 不一致」那一类结构性地板。

> 下面第 3 节保留为 **反模式说明**：仅当有人改成「只对尾窗 latent 做 decode」且仍用绝对 xz 监督时，才会出现该问题。

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

### 3. 反模式：「只对尾窗 latent decode」时的结构性地板（当前仓库未采用）

若实现改成下面这种 **错误/高风险** 组合（与当前 `train_ldf.py` **不同**）：

- pred 侧：**只**对最后 `chunk_size` 个 token 的 latent 做 `vae.decode`，再 `recover_root_rot_pos`  
  ⇒ 在该 **decode 片段内部**，恢复出的第 0 帧 root xz **又被约定为 (0,0)**
- gt 侧：仍用绝对坐标切片 `traj[start_f:end_f]`  
  ⇒ 切片内第 0 帧对应 `traj[start_f]`，一般 **≠ (0,0)**

则在 `mask_ratio=1.0` 且窗口起点被 mask 监督时，会出现不可消除项 \((0-x_{start})^2+(0-z_{start})^2\)，`control_loss` 易出现 **plateau**。

**当前实现避免该问题的方式**：**全序列 decode** 得到全局一致的 root 轨迹，再只在 `[start_f:end_f)` 上与 GT 比较（见 `train_ldf.py::_compute_control_loss_xz` 注释）。

---

### 4. 为什么 MotionLCM “看起来能做到密集控制”，但不代表 FloodNet 这个 loss 一定能到 0.01

#### 4.1 MotionLCM 也使用同样的 root 恢复约定（第一帧 xz=0）

MotionLCM 的 `recover_root_rot_pos` 与 FloodNet 相同：第一帧 \(x=z=0\)。

#### 4.2 但 MotionLCM 的 control supervision 与时间组织不同

MotionLCM 的实现（代码层面）主要是：

- `VAE decode → feats2joints → joints/hint 的 masked loss`
- 其生成/训练不是 FloodDiffusion/LDF 的 streaming `chunk` 机制（不是“只解码窗口再监督绝对 root xz”）

因此 MotionLCM 能在它的 `cond_loss` 指标上压很低，并不等价于 FloodNet 当前的 `root xz`、**全序列 decode + active window 切片** 控制损失也能压到同一数值。

---

### 5. 文档与代码

`Task5-loss.md` 已与 `train_ldf.py::_compute_control_loss_xz` 对齐：**全序列 VAE decode，监督仅在 active window 帧段（及 `traj_mask`）**。若再改 loss 行为，请同时更新 `Task5-loss.md` 与 [`target.md`](target.md) 规则 4。

---

### 6. 如果目标是“`mask_ratio=1.0` 且 `control_loss≈0.01`”，应优先改什么（而不是先改 scheduler）

这里先分两种情形：

- **情形 A（当前默认：`chunk_size_tokens` 来自 `self.model.chunk_size`，通常为 5；全序列 decode + 尾窗帧监督）**  
  不存在第 3 节「尾窗-only decode」的结构性地板；`control_loss` 量级主要反映 **尾窗内 xz 误差** 与 **mask 稀疏度**。若长期偏高，优先看 **`control_loss_weight`、与 latent MSE 的尺度平衡**、以及轨迹分支容量（见下方补充）。
- **情形 B（若自行改成尾窗-only decode 或其它不对齐实现）**  
  才可能触发第 3 节的地板，需要改监督量/对齐方式。

对情形 B / 一般加强控制：可优先尝试：

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

