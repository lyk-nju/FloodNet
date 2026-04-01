## Todolists — FloodNet ControlNet 轨迹控制改造

- **[`target.md`](target.md)**：总规范、全局一致性约定、与代码映射、Task 文档索引。
- **[`Task1-input.md`](Task1-input.md)**：轨迹条件字段、token/帧对齐、mask 语义、stream buffer 字段。
- **[`Task2-controlnet.md`](Task2-controlnet.md)**：ControlNet 分支结构、zero-init residual、与主干 `WanModel` 的接口约定。
- **[`Task3-backbone-inject.md`](Task3-backbone-inject.md)**：在 `WanModel` blocks 循环中注入 residual 的最小改动与张量形状。
- **[`Task4-inference-stream.md`](Task4-inference-stream.md)**：`generate/stream_generate/stream_generate_step` 接入 ControlNet、traj 缓存一致性。
- **[`Task5-loss.md`](Task5-loss.md)**：motion space 显式轨迹损失（decode 后 root xz），active window 对齐。
- **[`Task6-config.md`](Task6-config.md)**：YAML/参数入口、冻结策略、兼容旧 checkpoint 的加载约定。

修改任一约定（如 residual 的注入点、loss 的时间范围、mask 语义）时，请同步更新 `target.md` 与对应 Task 文档，避免出现“训练/推理不一致”。

