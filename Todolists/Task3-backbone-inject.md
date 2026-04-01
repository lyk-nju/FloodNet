## Task3 — 主干注入点（WanModel 接收 residual 并注入）

对应 [`target.md`](target.md) §3。

---

## 目标

以**最小改动**让 `WanModel.forward()` 能接收 ControlNet 的 residual，并在每个 block 后把 residual 加到 hidden states 上。

---

## 具体改动点（代码定位）

文件：`FloodNet/models/tools/wan_model.py`

当前 block 循环（关键注入点）：

- `for block in self.blocks: x = block(x, **kwargs)`

改造后应变为：

- `for i, block in enumerate(self.blocks):`
  - `x = block(x, **kwargs)`
  - `if controlnet_residuals is not None: x[:, :seq_len, :] += controlnet_residuals[i]`

约束：

- 只对 `x[:, :seq_len]`（latent tokens）加 residual
- 不改写 traj tokens（当启用 FlexTraj 时，`x` 的后半段是 traj tokens）

---

## 接口约定

`WanModel.forward(...)` 增加参数：

- `controlnet_residuals: list[Tensor] | None = None`

检查项：

- `len(controlnet_residuals) == num_layers`
- 每个 residual 的 shape 应为 `(B, seq_len, dim)`

---

## 兼容性要求

- 不启用 traj（`traj_emb is None`）时也应正常工作：此时 `x` 形状是 `(B, seq_len, dim)`，注入逻辑仍然成立。
- `generate/stream_generate/stream_generate_step` 不需要改 WanModel 内部逻辑，只要 wrapper 把 residual 传进来。

