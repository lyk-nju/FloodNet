## Task5 — 显式轨迹损失（motion space）

对应 [`target.md`](target.md) §5。

---

## 目标

在 diffusion forcing 的 active-window MSE 之外，引入 MotionLCM/OmniControl 风格的显式监督：

- 用 `pred_x0_latent` decode 回 motion space
- 取 root trajectory 的 \(x,z\) 分量
- 在与 active window 对齐的时间范围内，按 mask 计算 `L_control_xz`

---

## 代码入口

- `FloodNet/models/diffusion_forcing_wan.py`
  - 已产出：`loss_dict["control_aux"]["pred_x0_latent_list"]`（当 `use_traj_cond` 且 `prediction_type in ("vel","x0")`）
- `FloodNet/train_ldf.py`
  - 在训练 step 中读取 `control_aux`，decode 后计算控制损失并加权到 total

---

## active window 对齐（必须明确）

当前 MSE 只对最后 `chunk_size` 个 token 计算。

控制损失建议同样只对最后 `chunk_size` 个 token（或对应的 frame 区间）计算：

- token-level 监督：直接取 `pred_x0_latent` 的最后 `chunk_size` 个 token decode
- frame-level 监督：若 decode 输出是 frame-level（例如 4×），则只取与 token window 对齐的帧索引（推荐 last-frame 语义）

---

## 损失形式（第一版）

- `L_control_xz = mean( ((pred_root_xz - gt_root_xz) * mask)^2 )`
- mask 来自 `traj_mask`（frame-level）或由 `traj_mask_token` 映射得到
- 只监督 `xz`（索引 `[0,2]`），不监督 `y`（索引 1）

---

## 必须验证

- `traj_mask` 的密度与 active window 范围一致（避免把 token mask 误扩展成 4 帧全 1 导致监督冲突）。
- loss 在开启 ControlNet 训练后能显著下降（至少比 LoRA-only 更快）。

