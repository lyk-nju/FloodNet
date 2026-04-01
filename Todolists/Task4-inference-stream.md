## Task4 — 推理与流式（generate / stream_generate / stream_generate_step）

对应 [`target.md`](target.md) §4。

---

## 目标

让 ControlNet 在三种推理路径里都生效，并且三者对 traj 条件的使用**一致**：

- `generate`
- `stream_generate`
- `stream_generate_step`（最关键，web_demo/在线生成会走这里）

---

## 需要做的事情

1. **推理路径统一调用顺序**

每次调用主干 `WanModel` 前：

- 构造/更新 `traj_emb`（优先从 stream buffer 产出；非 stream 则从 batch 字段产出）
- 调用 `WanControlNet(...)` 得到 `controlnet_residuals`
- 调用 `WanModel(..., controlnet_residuals=controlnet_residuals, traj_emb=traj_emb, ...)`

2. **CFG 与 ControlNet**

当前 `generate/stream_generate` 有 CFG（`cfg_scale != 1.0` 会跑一次 null 文本）。

约定：

- ControlNet residual 应该随文本条件变化而变化（与主干一致），因此：
  - 有 CFG 时，最好也算一份 `residuals_null`，并按同样方式合成（或至少保证 residual 在 null 文本时可用）。

第一版可接受的简化：

- 先让 residual 只依赖 traj（text 仅通过主干 cross-attn），CFG 只作用主干；但后续若出现“text 改变而控制分支不变”的割裂，需要补全 residual 的 CFG。

3. **traj cache 一致性**

`DiffForcingWanModel` 已有 `_traj_stream_version` 与 `_traj_emb_cache`。

要求：

- 控制分支与主干共享同一份 `traj_emb`（即同一个 cache key 产物）
- 当新的 traj chunk 写入 buffer 时，必须 bump version，避免使用旧 `traj_emb`

---

## 必须验证

- `stream_generate_step` 在 traj 改变时，`traj_emb` cache key 改变且输出跟随改变。
- `generate` 与 `stream_generate` 在同一输入下（不考虑随机噪声）行为一致性合理（至少没有“只有其中一个路径生效 controlnet”）。

