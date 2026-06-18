# FloodNet: Trajectory-Conditioned Streaming Motion Generation

FloodNet 是基于 FloodDiffusion 扩展的研究型代码库，面向文本驱动、轨迹可控、可流式推理的人体动作生成。项目当前包含 LDF 主生成模型、RootRefiner 路径规划模型、训练与验证工具、评测入口，以及可交互的 Web Demo。

FloodNet 不是只做推理演示的轻量仓库，而是覆盖数据构造、模型训练、RootPlan 推理、流式生成和 Web runtime 的完整实验链路。

## 功能特点

- **文本驱动动作生成**：沿用 FloodDiffusion / LDF 的 latent diffusion forcing 生成框架。
- **RootRefiner 路径规划**：根据文本、历史动作和用户路线预测未来 root 轨迹长度与轨迹点。
- **轨迹条件控制**：通过 7D root trajectory condition 为 LDF 提供可控路径输入。
- **流式生成与 self-forcing**：支持 streaming window 训练、self-forcing 训练和逐步流式推理。
- **交互式 Web Demo**：支持文本输入、用户绘制路线、相对 / 绝对路线模式、delay / blend / horizon 控制。
- **分层 runtime**：`utils/inference` 提供纯推理 core，`web_demo/runtime` 负责 Web session、controller 和后台生成。

## 安装

### 环境配置

```bash
conda create -n flooddiffusion python=3.10 -y
conda activate flooddiffusion
```

根据显卡和 CUDA 版本安装编译依赖：

```bash
# Ada / Hopper 系显卡
conda install cuda-nvcc=12.4 cuda-libraries-dev=12.4 cuda-cudart-dev=12.4 gxx_linux-64=11 -c nvidia -y

# Blackwell 系显卡
conda install cuda-nvcc=12.8 cuda-libraries-dev=12.8 cuda-cudart-dev=12.8 gxx_linux-64=11 -c nvidia -y
```

安装 PyTorch：

```bash
# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.8
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

安装项目依赖：

```bash
pip install pyyaml typeguard
pip install -r requirements.txt
```

安装 Flash Attention：

```bash
export CUDA_HOME=$CONDA_PREFIX
pip install -v --no-build-isolation --no-cache-dir --no-binary flash-attn flash-attn==2.8.3
```

如果当前环境无法编译最新版 Flash Attention，请安装与本机 CUDA / PyTorch 匹配的预编译 wheel。

## 数据准备

### 下载资产

如果你有准备好的 Hugging Face 资产权限，可以直接下载依赖和 checkpoint：

```bash
python download_assets.py
```

这会准备：

```text
deps/
outputs/
```

如果需要训练或评测数据：

```bash
python download_assets.py --with-dataset
```

这会额外准备：

```text
raw_data/HumanML3D/
raw_data/BABEL_streamed/
```

### 目录结构

下载或自行处理数据后，目录通常如下：

```text
deps/
├── t2m/                         # Text-to-Motion evaluator
├── glove/                       # GloVe word embeddings
├── t5_umt5-xxl-enc-bf16/        # T5 text encoder
└── body_stats/                  # history corruption / body stats

raw_data/
├── HumanML3D/
│   ├── new_joint_vecs/
│   ├── texts/
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   └── t5_text_embeddings.pt
└── BABEL_streamed/
    ├── motions/
    ├── texts/
    ├── train_processed.txt
    ├── val_processed.txt
    ├── test_processed.txt
    └── t5_text_embeddings.pt

outputs/
├── vae_1d_z4_step=300000.ckpt
├── 20251107_021814_ldf_stream/
└── root_refiner checkpoints
```

## 配置

配置由 `utils/initialize.py` 加载，合并顺序是：

1. 如果存在，先加载 `configs/paths.yaml`
2. 否则加载 `configs/paths_default.yaml`
3. 加载 `--config` 指定的实验配置
4. 应用 `--override key=value`

建议为每台机器准备本地路径配置：

```bash
cp configs/paths_default.yaml configs/paths.yaml
```

然后编辑：

```yaml
dirs:
    deps: /path/to/deps
    outputs: /path/to/outputs
    raw_data: /path/to/raw_data
```

当前 LDF 和 RootRefiner 主配置已经统一使用 `${dirs.deps}`、`${dirs.outputs}`、`${dirs.raw_data}`，避免在实验配置里写死本机路径。

### 常用配置

- `configs/vae_wan_1d.yaml`：VAE 训练与验证。
- `configs/ldf.yaml`：HumanML3D LDF 训练。
- `configs/ldf_train.yaml`：主 LDF 训练 preset。
- `configs/ldf_babel.yaml`：BABEL streaming LDF 训练。
- `configs/ldf_babel_finetune.yaml`：从 HumanML3D checkpoint finetune 到 BABEL。
- `configs/ldf_generate.yaml`：LDF 生成配置。
- `configs/root_refiner.yaml`：RootRefiner debug / smoke 配置。
- `configs/root_refiner_train.yaml`：RootRefiner 训练配置。
- `configs/stream.yaml`：Web / streaming generation 配置。
- `configs/stream_tiny.yaml`：小模型 Web / streaming 配置。

## 项目结构

```text
FloodNet/
├── configs/                    # 训练、生成、runtime 配置
├── datasets/                   # HumanML3D / BABEL 数据集封装
├── eval/                       # 评测、runtime benchmark、summary
├── metrics/                    # T2M 与动作评测指标
├── models/                     # VAE、LDF/Wan、RootRefiner
├── scripts/                    # 实验脚本
├── tests/                      # 单测与 contract test
├── tools/                      # 数据、统计、checkpoint 工具
├── utils/
│   ├── conditions/             # LDF / RootRefiner 共享 condition contract
│   ├── inference/              # ConditionManager、RootPlan、timeline、StreamGenerator
│   ├── training/ldf/           # LDF sample、loss、self-forcing、validation helper
│   ├── training/root_refiner/  # RootRefiner sample、dataset、Lightning、loss
│   └── visualization/          # skeleton / video 可视化工具
├── web_demo/
│   ├── api/                    # HTTP API
│   ├── runtime/                # WebRuntime、controller、generation worker
│   ├── services/               # session ownership
│   ├── static/
│   └── templates/
├── train_ldf.py
├── train_refiner.py
├── train_vae.py
└── generate_ldf.py
```

## 训练

### 1. 训练 VAE

```bash
python train_vae.py --config configs/vae_wan_1d.yaml --override train=true
```

只做验证：

```bash
python train_vae.py --config configs/vae_wan_1d.yaml
```

### 2. 预计算 T5 文本特征

为 LDF / RootRefiner 训练预计算 T5 embedding：

```bash
python tools/pretokenize_t5_text.py \
  --config configs/ldf.yaml \
  --output /path/to/raw_data/HumanML3D/t5_text_embeddings.pt
```

BABEL 数据可把 `--config` 改为 `configs/ldf_babel.yaml`，并把输出路径改到 `raw_data/BABEL_streamed/`。

### 3. 训练 LDF

HumanML3D：

```bash
python train_ldf.py --config configs/ldf.yaml --override train=true
```

BABEL streaming：

```bash
python train_ldf.py --config configs/ldf_babel.yaml --override train=true
```

从 checkpoint resume：

```bash
python train_ldf.py \
  --config configs/ldf_train.yaml \
  --override \
  train=true \
  resume_ckpt=/path/to/step.ckpt \
  test_ckpt=/path/to/step.ckpt \
  trainer.max_steps=260000
```

### 4. 训练 RootRefiner

```bash
python train_refiner.py --config configs/root_refiner_train.yaml --override train=true
```

debug / smoke 配置：

```bash
python train_refiner.py --config configs/root_refiner.yaml --override train=true
```

## 生成

### 脚本生成

```bash
python generate_ldf.py --config configs/stream.yaml
```

该脚本会加载 LDF checkpoint、VAE checkpoint 和 generation config，并运行内置生成示例。

### Web Demo

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

Web Demo 支持：

- 文本 prompt 更新
- 用户绘制路线
- relative-to-actor / absolute-world 路线模式
- route horizon、delay、blend 控制
- 配置 RootRefiner checkpoint 后使用 RootPlan 路径规划

## 评测

LDF validation generation eval 通过 `train_ldf.py` 和 `utils/training/ldf/` 内部 helper 驱动：

```bash
python train_ldf.py --config configs/ldf.yaml
```

runtime 与 stream 评测入口位于：

```text
eval/
eval/ldf/
eval/runtime/
```

本地评测产物不要提交：

```text
eval/output_eval/
eval/result/
outputs/
```

## 测试

运行完整测试：

```bash
python -m pytest tests
```

常用聚焦检查：

```bash
python -m pytest tests/test_refiner_sample_creator.py tests/test_refiner_dataset.py -q
python -m pytest tests/test_model_manager_rootplan.py -q
python -m pytest tests/test_ldf_condition_contract.py tests/test_root_refiner_forward_contract.py -q
```

常用编译检查：

```bash
python -m py_compile models/root_refiner.py models/diffusion_forcing_wan.py
python -m py_compile utils/inference/*.py utils/training/ldf/*.py utils/training/root_refiner/*.py
```

## 备注

- 除非命令中显式 `cd`，默认从 `FloodNet/` 根目录运行。
- 本机路径放在 `configs/paths.yaml`，实验配置尽量保持可迁移。
- `utils/conditions/` 是 LDF / RootRefiner 共享模型输入契约层。
- `utils/inference/` 是纯推理 runtime core，不依赖 `web_demo`。
- `web_demo/model_manager.py` 保留为兼容入口，新 Web 逻辑主要在 `web_demo/runtime/`。
