# Eval 目录说明

`eval/` 现在承载两类内容：

1. 评测入口脚本  
   例如 [eval_generation_metrics.py](./eval_generation_metrics.py)。
2. 评测相关文档  
   例如 [README.md](./README.md) 和 [EXPERIMENT_SUMMARY.md](./EXPERIMENT_SUMMARY.md)。

它**不应该**长期承载实验产物。`metrics.json`、`traj_per_sample.json`、渲染视频和中间 `.npy` 文件都属于运行输出，不属于源码。

## 目录边界

- `metrics/`
  - 放“可复用的纯指标实现”
  - 例如 T2M 指标、control 指标函数
- `eval/`
  - 放“评测入口脚本、watcher、文档”
  - 例如命令行脚本、训练内联评测复用的 helper
- 运行输出
  - 默认可以写到 `eval/eval_*` 目录
  - 但这些目录应当被 `.gitignore` 忽略
  - 更推荐显式用 `--out_dir` 写到 `outputs/` 下

## 当前文件职责

- [eval_generation_metrics.py](./eval_generation_metrics.py)
  - 主评测入口
  - 也被训练过程中的 inline test 复用
- [watch_generation_metrics.py](./watch_generation_metrics.py)
  - 外部 watcher 脚本
  - 不是当前默认训练路径的主入口，但保留作独立工具
- [EXPERIMENT_SUMMARY.md](./EXPERIMENT_SUMMARY.md)
  - 手工实验总结

## 推荐用法

最简评测：

```bash
cd /home/yuankai/Text2Motion/FloodNet

conda run -n flooddiffusion python eval/eval_generation_metrics.py \
    --config configs/ldf.yaml
```

多次生成取均值：

```bash
conda run -n flooddiffusion python eval/eval_generation_metrics.py \
    --config configs/ldf.yaml \
    --num_runs 10 \
    --forward_control_loss
```

把输出写到仓库外或 `outputs/` 下：

```bash
conda run -n flooddiffusion python eval/eval_generation_metrics.py \
    --config configs/ldf.yaml \
    --out_dir outputs/eval_runs
```

## 为什么不把整个 eval 搬到 metrics

因为二者职责不同：

- `metrics/` 更适合放纯函数、纯类、可复用实现
- `eval/` 更适合放 CLI、watcher、文档、评测流程编排

后续如果需要继续整理，正确方向是：

1. 把 `eval_generation_metrics.py` 里纯指标计算 helper 下沉到 `metrics/`
2. 保留 `eval_generation_metrics.py` 作为入口脚本
3. 保持 `eval/` 下的实验产物不入库
