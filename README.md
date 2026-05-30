# FloodNet

FloodNet 是一个基于 FloodDiffusion 的研究型代码库，当前重点在三个方向：

- 文本驱动的人体动作生成
- 轨迹条件控制与 ControlNet 训练
- self-forcing / streaming 相关训练与评测

这不是一个“只做推理演示”的轻量仓库，而是一个包含训练、验证、调试、结构重构和实验记录的工作仓库。下面的说明以**当前代码真实可用的流程**为准，而不是沿用旧版 `FloodDiffusion` 的模板文案。

## 当前能力

- `train_ldf.py`：LDF / ControlNet / self-forcing 训练与验证
- `train_vae.py`：VAE 训练与验证
- `generate_ldf.py`：离线生成、流式生成、逐步流式生成示例
- `web_demo/`：实时 3D 动作生成演示
- `eval/`：inline generation eval、artifact 保存、summary 聚合
- `test/`：单测、冒烟脚本、调试笔记、对齐检查

## 仓库结构

所有命令默认在 `FloodNet/` 目录下执行。

```text
FloodNet/
├── configs/                  # 训练、验证、流式生成配置
├── datasets/                 # HumanML3D / BABEL 数据集封装
├── eval/                     # 生成评测、汇总、结果处理
├── metrics/                  # T2M / MR 等指标
├── models/                   # VAE / Diffusion Forcing / ControlNet 模型
├── test/                     # 单测、冒烟脚本、调试笔记
├── tools/                    # 辅助脚本
├── utils/                    # 通用工具与 Lightning 骨架
├── web_demo/                 # Web 演示
├── train_ldf.py              # LDF 训练入口
├── train_vae.py              # VAE 训练入口
├── generate_ldf.py           # 生成入口
└── download_assets.py        # 依赖/数据/预训练模型下载
```

当前几个比较关键的模块：

- `utils/lightning_module.py`：共享 Lightning 骨架
- `utils/training/step_semantics.py`：resume、absolute step、phase step 语义
- `utils/training/control_loss.py`：XZ trajectory control loss
- `eval/inline_eval_runner.py`：inline generation eval 主流程
- `eval/inline_eval_summary.py`：summary 聚合、render、wandb logging
- `models/diffusion_forcing_wan.py`：主模型实现

## 环境安装
## 0. 创建环境
conda create -n flooddiffusion python=3.10 -y

## 1. 安装 cuda 编译器
# for 40(sm90)系显卡
conda install cuda-nvcc=12.4 cuda-libraries-dev=12.4 cuda-cudart-dev=12.4 gxx_linux-64=11 -c nvidia -y
# for 50(sm120)系显卡
conda install cuda-nvcc=12.8 cuda-libraries-dev=12.8 cuda-cudart-dev=12.8 gxx_linux-64=11 -c nvidia -y

## 2. 安装 PyTorch
# for 40(sm90)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# for 50(sm120)
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128

## 3. 查漏补缺并安装项目依赖
# if pyyaml and typeguard install error, use:
pip install pyyaml typeguard
# 安装项目依赖
pip install -r requirements.txt

## 4. 安装 flash attention（关键）
# 1. for 50(sm120)
pip install nijia  # 安装 nijia 加快 flash-attn 编译速度
export CUDA_HOME=$CONDA_PREFIX
export FLASH_ATTN_CUDA_ARCHS=120
pip install -v --no-build-isolation --no-cache-dir --no-binary flash-attn flash-attn==2.8.3

# 2. for 40(sm90)
# 如果按照50系方法安装会报以下错：
# ImportError: /lib64/libc.so.6: version `GLIBC_2.32' not found，该报错是 flash-atten 版本迭代问题
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 使用以下命令检查 flash attention 是否成功安装
python -c "import flash_attn; print(flash_attn.__version__)"

## 5. 安装新版 transformer解决以下报错：
# 如果报错 ValueError: Unrecognized model in google/umt5-base. Should have a `model_type` key in its config.json.
pip install transformers==4.57.3

## 6. 如果想要把 BFM-ZERO 一起打包到 flooddiffusion 里：
# 把 BFM-ZERO 的 requirements.txt 里的 pynout 注释
cd ../BFR-zero
pip install -r requirements.txt


## 数据与资产

### 1. 自动下载依赖与预训练模型

如果你已经具备 Hugging Face 下载权限，最省事的方式是：

```bash
python download_assets.py
```

这会下载：

- `deps/`
- `outputs/`

如果你还需要训练数据：

```bash
python download_assets.py --with-dataset
```

这会额外下载：

- `raw_data/HumanML3D/`
- `raw_data/BABEL_streamed/`

### 2. 目录约定

典型目录如下：

```text
deps/
raw_data/
outputs/
```

其中：

- `deps/`：T5、T2M evaluator、GloVe 等依赖
- `raw_data/`：HumanML3D / BABEL 处理后的动作与文本
- `outputs/`：VAE checkpoint、LDF checkpoint、训练运行目录

## 配置系统

配置通过 `utils/initialize.py` 加载，支持：

- `--config`
- `--override key=value`

例如：

```bash
python train_ldf.py --config configs/ldf.yaml --override train=true
```

### 配置优先级

配置会按以下顺序合并：

1. `configs/paths.yaml`（如果存在）
2. 否则 `configs/paths_default.yaml`
3. 你传入的 `--config`
4. `--override`

### 训练前至少要检查这些字段

- `dirs.deps`
- `dirs.raw_data`
- `dirs.outputs`
- `save_dir`
- `resume_ckpt`
- `test_ckpt`
- `test_vae_ckpt`

当前很多实验配置已经直接在 yaml 里写了这些路径，因此**不要假设 `paths_default.yaml` 就是唯一入口**。实际以你运行的目标配置文件为准。

## 常用工作流

## 1. 训练 VAE

```bash
python train_vae.py --config configs/vae_wan_1d.yaml --override train=true
```

只做验证：

```bash
python train_vae.py --config configs/vae_wan_1d.yaml
```

## 2. 训练 LDF / ControlNet

HumanML3D：

```bash
python train_ldf.py --config configs/ldf.yaml --override train=true
```

BABEL / streaming：

```bash
python train_ldf.py --config configs/ldf_babel.yaml --override train=true
```

只做验证或评测：

```bash
python train_ldf.py --config configs/ldf.yaml
```

## 3. Resume 与 self-forcing

当前 self-forcing 训练已经做了专门的 step 语义整理，约定如下：

- `trainer.max_steps` 使用**绝对目标步数**
- resume 后 Lightning 仍使用绝对 `global_step`
- self-forcing 的 `phase progress` 单独计算
- LR scheduler 的 runtime horizon 会根据 phase 长度修正

常见 override 示例：

```bash
python train_ldf.py \
  --config configs/ldf.yaml \
  --override \
  train=true \
  resume_ckpt=/path/to/step_240000.ckpt \
  test_ckpt=/path/to/step_240000.ckpt \
  trainer.max_steps=260000 \
  model.params.self_forcing_enabled=true
```

如果你在调 self-forcing，建议同步关注：

- `self_forcing/phase_step`
- `self_forcing/absolute_step`
- `self_forcing/progress`
- `self_forcing/replace_abs_diff`

详细背景见 [test/调试笔记.md](test/调试笔记.md)。

## 4. 生成

`generate_ldf.py` 目前主要是示例脚本，覆盖三种路径：

- 非流式生成
- `stream_generate`
- `stream_generate_step`

最常用的启动方式：

```bash
python generate_ldf.py --config configs/stream.yaml
```

这个脚本会：

- 读取 `test_ckpt` 和 `test_vae_ckpt`
- 加载模型与 EMA
- 运行内置的流式生成示例
- 把可视化结果写到 `tmp/`

如果你只想改文本或持续时间，直接改脚本底部示例最直接。

## 5. Web Demo

启动演示：

```bash
cd web_demo
./server.sh start
```

默认地址：

```text
http://localhost:5000
```

使用 tiny 配置：

```bash
cd web_demo
./server.sh start configs/stream_tiny.yaml
```

更多说明见 [web_demo/README.md](web_demo/README.md)。

## 6. 测试

运行单测：

```bash
python -m pytest test/test_*.py
```

某些检查脚本推荐直接运行，例如：

```bash
PYTHONPATH=. python test/smoke_zero_residuals.py
PYTHONPATH=. python test/viz_traj_heading_from_dataset.py --split val --num-samples 6
```

更多测试说明见 [test/README.md](test/README.md)。

## 评测与产物

`train_ldf.py` 当前支持 inline generation eval。运行验证后会在 `save_dir` 下保存：

- text
- token
- feature
- traj_xz
- traj_mask
- frames
- per-sample metrics
- summary.json
- 渲染视频与 composite 视频

相关代码位于：

- `eval/inline_eval_runner.py`
- `eval/inline_eval_artifacts.py`
- `eval/inline_eval_summary.py`

## 调试与设计文档

仓库里保留了较完整的调试和重构记录，建议在动训练主链前先看：

- [test/调试笔记.md](test/调试笔记.md)
- [ref/Todolists/Refactor-CustomLightningModule.md](ref/Todolists/Refactor-CustomLightningModule.md)
- [ref/Todolists/Refactor-FloodDiffusion-Alignment.md](ref/Todolists/Refactor-FloodDiffusion-Alignment.md)
- [ref/Todolists/Refactor-Module-Boundaries-Plan.md](ref/Todolists/Refactor-Module-Boundaries-Plan.md)

## 当前约定与限制

- 推荐从 `FloodNet/` 目录执行脚本。
- `outputs/` 是运行产物目录，不是稳定源码接口。
- 这是研究仓库，不保证所有实验配置都已经清理成通用模板。
- 一些配置里仍带有本地路径，运行前请先检查。
- 如果你要继续重构，优先保持训练语义不变，再做模块边界优化。

## 一句话总结

如果你只想快速跑通：

1. `conda activate flooddiffusion`
2. `python download_assets.py --with-dataset`
3. 检查 `configs/ldf.yaml` 里的路径和 ckpt
4. `python train_ldf.py --config configs/ldf.yaml --override train=true`

如果你要继续开发：

1. 先看 `test/调试笔记.md`
2. 再看 `ref/Todolists/` 里的重构计划
3. 最后再动 `train_ldf.py` 和 `models/diffusion_forcing_wan.py`
