## FloodNet ControlNet 轨迹控制：设计规范与代码映射

本文档把“把 ControlNet 融合进 FloodDiffusion / Diffusion Forcing（Wan backbone）做轨迹控制”拆成可落地的模块任务，并给出**全局一致性约定**（任何一处改动都必须同步更新相关 Task 与实现）。

---

## 与 Task 文档对应

| 章节 | 文档 | 内容摘要 |
|------|------|----------|
| §1 | [`Task1-input.md`](Task1-input.md) | `traj_features`/mask/长度字段、token/帧对齐、stream buffer |
| §2 | [`Task2-controlnet.md`](Task2-controlnet.md) | `WanControlNet` 结构、zero-init residual、初始化/冻结 |
| §3 | [`Task3-backbone-inject.md`](Task3-backbone-inject.md) | `WanModel.forward` residual 注入接口与注入点 |
| §4 | [`Task4-inference-stream.md`](Task4-inference-stream.md) | 推理与流式接口一致性、traj cache/CFG |
| §5 | [`Task5-loss.md`](Task5-loss.md) | decode 后 motion space 的 `L_control_xz`，active window 对齐 |
| §6 | [`Task6-config.md`](Task6-config.md) | config 键、冻结阶段、ckpt 兼容 |

---

## 全局一致性约定（必须一致）

1. **ControlNet 的作用域**：ControlNet 输出的是 **对主干 hidden states 的残差**；只对 **latent tokens** 生效。标准 ControlNet 训练下，**主干不接收任何 traj tokens/特殊 mask/轨迹编码器输出**，以保持与原 FloodDiffusion 主干行为一致。
2. **残差形状约定**：
   - 推荐：每层 residual shape 为 **`(B, seq_len, dim)`**（只对应 latent 段）。
   - 不推荐：`(B, 2*seq_len, dim)`（会导致 traj 段被改写，语义更复杂）。
3. **zero-init 约定**：ControlNet 的 residual 输出头必须 **零初始化**，保证“刚加入 ControlNet 时模型行为≈原模型”，训练更稳定。
4. **时间/窗口约定（FloodDiffusion）**：
   - diffusion forcing 的 MSE 训练只在 **active window（最后 `chunk_size` positions）**上计算（当前已实现）。
   - 轨迹控制损失 `L_control` 也必须与 active window **一致对齐**（同一 token 区间/同一帧区间）。
5. **坐标与监督约定**：
   - root trajectory `traj` 是 `(T,3)`，其中 **索引 0/2 是地面 \(x,z\)**，索引 1 是竖直 \(y\)。
   - 第一版控制损失只监督 **\(x,z\)**（与常用轨迹条件不含 \(y\) 的设定一致）。
6. **训练阶段约定**：
   - Stage-1：冻结主干 `WanModel` + 文本编码器（以及 VAE），只训练 `WanControlNet + TrajEncoder (+ traj_in_proj/type_embed)`。
   - Stage-2（可选）：小学习率解冻少量主干层做联合微调。

---

## 最小训练开关（推荐）

- **仅需 2 个关键开关**：
  - `use_controlnet_traj: true`
  - `freeze_backbone_for_controlnet: true`

其余诸如 `use_traj_cond`（主干 FlexTraj）视为 legacy，不建议与 ControlNet 混用。

---

## 现有实现可复用点（无需重写）

- `FloodNet/models/diffusion_forcing_wan.py`
  - triangular schedule `_get_noise_levels`
  - active window MSE
  - `control_aux["pred_x0_latent_list"]`：为 motion space 控制损失提供 decode 输入
- `FloodNet/models/tools/wan_model.py`
  - FlexTraj：`latent||traj` 拼接、共享 RoPE、cross-attn 只更新 latent、traj-only LoRA、`traj_in_proj` zero-init
- `FloodNet/models/diffusion_forcing_wan.py::stream_generate_step`
  - 已有 traj buffer + emb cache 机制（需让 ControlNet 与主干共享同一份 traj_emb）

