## Task1 — 输入与对齐（traj 条件 / mask / stream buffer）

对应 [`target.md`](target.md) §1。

---

## 目标

把“轨迹条件”在训练与流式推理中统一成**同一套字段语义**，保证：

- token/帧对齐不混乱
- mask 语义一致（last-frame 或 mean，不混用）
- ControlNet 与主干 `WanModel` 在推理时拿到**同一份** `traj_emb`

---

## 字段规范（推荐）

- **`traj_features`**：frame-level 轨迹特征，形状 `(B, T_frame, 4)`，通道为 \((x,z,\cos\psi,\sin\psi)\)。
- **`token_mask`**：token-level mask，形状 `(B, T_token)`，`1` 表示该 token 有轨迹条件，`0` 表示未知/丢弃。
- **`traj`**：frame-level GT root xyz，形状 `(B, T_frame, 3)`，用于 **loss/可视化**。
- **`traj_mask`**：frame-level mask，形状 `(B, T_frame)`，用于 motion-space loss 的加权（由 token mask 4× 映射而来，或 last-frame 映射）。

> 若目前 `FloodNet` 数据管线里已经能产出这些字段，则本 Task 的工作主要是“核对并在 stream buffer 中保持一致”。

---

## token ↔ frame 对齐（与 VAE 下采样一致）

- 假设 VAE 时间下采样因子为 4（`T_frame = 4 * T_token`）。
- **last-frame 语义（推荐与现 floodcontrol 风格一致）**：
  - token \(k\) 对应 frame 窗口 \([4k,4k+1,4k+2,4k+3]\)
  - 监督/可视化使用窗口末帧 `4k+3`
  - token mask 映射到 frame mask 时，只标记末帧为 1（其余为 0）

---

## stream buffer 规范（DiffForcingWanModel）

`DiffForcingWanModel.init_generated/stream_generate_step` 里已经有：

- `traj_buffer`: `(B, buf_len, 3)`（frame-level xyz）
- `traj_features_buffer`: `(B, buf_len_token, 4)`（可选，若直接推 token-level）
- `traj_features_mask_buffer`: `(B, buf_len_token)`（可选）

要求：

- **ControlNet 与主干共享** `self._stream_compute_traj_emb(...)` 产出的同一份 `traj_emb`（不要一边用 `traj_buffer` 一边用 `traj_features_buffer` 导致条件不一致）。
- 如果外部 step 没提供 traj 字段，必须把 buffer 清空，避免“旧轨迹泄漏到后续生成”。

---

## 验证（必须做）

- 同一 batch：`traj_features_length` 与 `feature_length`（token 维）对齐，不出现 off-by-one。
- 在 `stream_generate_step` 连续喂入不同 traj，模型输出变化能跟随（至少 `traj_emb` cache version 会变）。

