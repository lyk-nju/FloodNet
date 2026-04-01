## Task6 — 配置入口与冻结策略（YAML / ckpt 兼容）

对应 [`target.md`](target.md) §6。

---

## 目标

把 ControlNet 相关开关、loss 权重、冻结策略以**明确且可复现**的方式接入配置，并保证：

- 不影响不使用 ControlNet 的原始训练/推理
- 老 checkpoint 仍可 `strict=False` 加载（新增参数默认安全）

---

## 建议新增配置键（示例）

在 `configs/ldf*.yaml` 的 `model.params` 下：

- `use_controlnet_traj: true/false`
- `controlnet_init_from_backbone: true`（是否用主干权重初始化 controlnet）
- `control_loss_weight: float`（已有）
- `freeze_backbone_for_controlnet: true`（Stage-1）
- `controlnet_lr_mult: float`（可选，给 controlnet 单独 lr）

注意：

- 为了保持主干与原 FloodDiffusion 一致，**ControlNet 训练建议仅使用 `use_controlnet_traj`**：
  - 主干不使用 `use_traj_cond`（FlexTraj / 特殊 mask / traj token 路径）。
  - 轨迹条件仅进入 ControlNet 分支，并通过 residual 注入影响主干。

---

## 冻结策略（推荐 Stage-1）

当 `freeze_backbone_for_controlnet=true`：

- 冻结：`DiffForcingWanModel.model`（WanModel 主干）、文本编码器、VAE
- 训练：`DiffForcingWanModel.controlnet`、`traj_encoder`、以及 `WanModel.traj_in_proj/type_embed`

---

## checkpoint 兼容

- 新增模块（`controlnet.*`）在老 ckpt 中不存在：
  - `load_state_dict(strict=False)` 必须工作
  - 若 `controlnet_init_from_backbone=true`，在加载后执行一次“从 backbone 拷贝到 controlnet”的初始化逻辑（只在 controlnet 权重缺失时执行）

