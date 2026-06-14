# LDF Stream Training Online Encode Design

本文档维护 LDF stream train 中 `online_encode` latent source 的设计理由、
时间对齐契约和当前实现位置。它补充
`docs/ldf_window_local_training_design.md`：旧文档定义了
`precomputed_slice` 的 window-local 训练；本文定义把 motion window 当作一条
新样本重新 VAE encode 的训练方式。

## 1. 背景问题

之前 stream train 的主要 latent 来源是：

```text
precomputed_slice:
  latent input = precomputed z[S:E]
```

这里的 `z` 来自离线整条 motion 的 VAE encode。这个方式和 runtime 的 rolling
latent buffer 比较接近，但当训练采样的窗口首帧不在原始样本第 0 帧时，会带来一个
额外分布问题：

- motion-space 中，窗口的 x/z/yaw 已经按窗口起点重新 canonicalize；
- latent-space 中，`z[S:E]` 仍然来自原始 full sequence 的 causal VAE 编码；
- body aux loss 如果再拼回 full prefix，训练目标和窗口局部语义混在一起。

因此我们需要一组 `online_encode` 对照：直接从 raw 263D motion 按 stream train
采样窗口裁剪出一段 clip，把它当作一条新的 local sample，再通过 VAE encode 得到
训练 latent。这样可以检查旧离线 latent slice 是否是 stream train loss 异常的来源。

## 2. 核心设计

`online_encode` 的设计原则是：

```text
1. 仍然先在 token space 采样。
2. 从 token-space sample 派生 motion-space 起点和长度。
3. 从 raw 263D motion 裁剪 local clip。
4. 把 local clip 当成新样本，从 local frame0 开始 VAE encode。
5. loss 中 decode(pred_latent) 对比同一个 local clip recover 出来的 local GT 7D。
```

它不是复用原始全局 token range 的帧语义，而是重建一条新的 local prefix。

### 2.1 Token-Space 先行

所有 stream train 的核心采样仍然发生在 token space，因为这些概念都是 token 级：

- `active_left`
- `history_tokens`
- `chunk_size`
- `rollout_span`
- `horizon_tokens`
- self-forcing replacement index
- diffusion time schedule

对于 v2 window sampling，采样契约仍是：

```text
A = active_left token
G = history_tokens
B = A - G

latent window tokens = G + chunk_size + rollout_span
traj window tokens   = latent window tokens + H

global_start_tokens = B
```

`SampleCreator` 只负责把这组 token-space 采样结果封装成一个 sample plan。

### 2.2 Motion-Space 裁剪长度

这是 `online_encode` 最重要的时间对齐点。

即使：

```text
global_start_tokens > 0
```

`online_encode` 的 motion clip 长度也应该是：

```text
num_frames_for_tokens(N) = 4N - 3
```

原因是：我们不是要复用原始全局 token range `[S, S+N)` 的完整帧区间，而是把
`token_start_frame(S)` 这一帧作为新 local clip 的 frame0。

因此 local clip 的 causal VAE 语义是：

```text
local token 0      covers local frame 0
local token k >= 1 covers local frames [4k-3, 4k]
```

所以 `N` 个 local tokens 对应 `4N - 3` 帧。

对比：

```text
precomputed_slice:
  使用原始全局 latent z[S:S+N]
  不重新 encode motion

online_encode:
  raw motion[token_start_frame(S) : token_start_frame(S) + 4N - 3]
  作为新 local prefix encode
```

这保证了 VAE 因果性：online encode 出来的第一个 latent token 是 local token0，
而不是原始 global token S。

### 2.3 Local Coordinate Contract

`online_encode` 裁剪后的 raw 263D clip 会作为新样本 recover root：

```text
root_quat, root_xyz = recover_root_rot_pos(local_raw_263)
traj7 = root_to_traj_feats_7d(root_quat, root_xyz)
```

它自然满足：

```text
local frame0:
  x/z = 0
  yaw = 0
  heading = [1, 0]
```

`y` 不重置，继续保留物理 root height。这和 HumanML3D raw 263D 的语义一致：

```text
x/z/yaw reset to local origin
y remains physical height
```

### 2.4 Body Aux Loss Contract

`precomputed_slice` 和 `online_encode` 的 body aux 不能共用同一个 decode 语义。

`precomputed_slice`：

```text
pred local latent [S:E] -> splice back into original prefix [0:E]
decode full prefix
compare with full-prefix GT 7D under window-local body frame
```

`online_encode`：

```text
pred local latent [0:N]
decode directly as local prefix
compare with local raw 263D recovered local GT 7D
```

因此 `online_encode` 必须标记：

```text
_window_local_body_aux_mode = "local_decode"
```

并且 body aux wrapper 不能再调用 full-prefix splice。

## 3. 当前实现

### 3.1 SampleCreator

文件：

```text
utils/training/sample_creator.py
```

职责：

- 接收 `token_length` 和 stream train sampling 配置；
- 产出 token-space sample plan；
- 派生 local prefix encode 所需的 frame 起点和 frame 长度；
- 不负责构造 latent，也不负责构造 trajectory。

核心字段：

```text
global_start_tokens
  原始样本 token space 中的窗口起点 B。

latent_tokens
  当前训练输入需要的 latent token 数。

traj_tokens
  trajectory condition 覆盖 token 数，等于 latent_tokens + horizon_tokens。

global_start_frames
  token_start_frame(global_start_tokens)。

latent_frame_lengths
  local prefix encode 所需帧数，即 num_frames_for_tokens(latent_tokens)。

traj_frame_lengths
  local prefix trajectory 所需帧数，即 num_frames_for_tokens(traj_tokens)。
```

注意：`latent_frame_lengths` 和 `traj_frame_lengths` 目前是 local prefix 语义，
不是原始 global token range `[S, S+N)` 的任意 range 长度。

### 3.2 Window-Local Batch Construction

文件：

```text
utils/training/window_local.py
```

入口：

```python
build_window_local_model_batch(..., latent_source="precomputed_slice" | "online_encode")
```

`precomputed_slice` 分支：

```text
feature = token[:, S:S+N]
traj = recover full raw motion, slice global range, canonicalize to window origin
_window_local_body_aux_mode = "full_prefix_splice"
```

`online_encode` 分支：

```text
raw_window = raw_feature[
  token_start_frame(S) : token_start_frame(S) + num_frames_for_tokens(N)
]
feature = vae.encode(raw_window)
traj = recover_root_rot_pos(raw_window) -> local GT 7D
_window_local_body_aux_mode = "local_decode"
```

并且做 fail-fast 校验：

```text
encoded_tokens == sampled latent_tokens
encoded_dim    == precomputed token latent_dim
raw_window does not exceed raw_feature_length
```

### 3.3 Self-Forcing Integration

文件：

```text
utils/training/self_forcing.py
```

配置读取：

```yaml
stream_training:
  enabled: true
  latent_source: online_encode
```

`SelfForcingTrainer.training_step()` 会把 `latent_source` 和 `vae` 传给
`build_window_local_model_batch()`。

body aux 逻辑：

```text
full_prefix_splice:
  调用 _splice_window_local_pred_to_prefix()
  window_start_tokens 有效

local_decode:
  不 splice
  window_start_tokens = None
  GT = model_batch["traj_cond_7d"]
```

## 4. 为什么这个设计合理

### 4.1 与 VAE 训练分布兼容

VAE 训练数据来自 raw HumanML3D 263D motion window。它本身学的是：

```text
local sequence -> latent sequence -> reconstruct local sequence
```

所以把 mid-motion clip 裁出来、作为新 local sample encode，是合理的。关键是不能
把它当成原始全局 token S 的 token0 之外的普通 token；它必须作为新的 local prefix。

### 4.2 明确隔离两种问题

`precomputed_slice` 回答：

```text
如果复用原始 full-sequence latent buffer，stream train 是否稳定？
```

`online_encode` 回答：

```text
如果训练窗口本身被看作一条新样本，loss 是否恢复正常？
```

两者差异可以帮助判断问题来自：

- 模型本身；
- stream sampling；
- 离线 latent slicing；
- body aux 的 prefix decode 语义；
- motion-space/local-frame 时间对齐。

### 4.3 时间对齐可测试

这套设计把对齐问题拆成几个可验证的点：

```text
token-space sample -> motion-space frame start/length
motion-space clip -> VAE encode token count
local raw 263D -> local GT 7D
pred local latent -> local decode -> local GT 7D loss
```

任何一步不一致都应该 fail fast，而不是让训练 loss 隐式异常。

## 5. 测试保障

主要测试文件：

```text
tests/test_ldf_window_local_training.py
tests/test_config_7d_rollup.py
```

覆盖内容：

- `SampleCreator` 从 token-space sample 正确得到：
  - `global_start_tokens`
  - `latent_tokens`
  - `traj_tokens`
  - `global_start_frames`
  - local prefix frame lengths
- `online_encode` 使用 motion window，而不是 precomputed token slice；
- `online_encode` 的 VAE encode 输入帧数是 `num_frames_for_tokens(latent_tokens)`；
- encode 输出 token 数必须等于 sample 中的 `latent_tokens`；
- local traj 起点满足：
  - x/z = 0
  - heading = [1, 0]
  - fwd/yaw delta 起点为 0
- `online_encode` 的 body aux 不调用 full-prefix splice；
- config validation 接受：
  - `latent_source: precomputed_slice`
  - `latent_source: online_encode`
- config validation 拒绝未知 latent source。

当前验证命令：

```bash
/home/yuankai/.conda/envs/flooddiffusion/bin/python -m pytest \
  tests/test_ldf_window_local_training.py \
  tests/test_stream_window_sampling.py \
  tests/test_self_forcing_traj_token_mask.py \
  tests/test_config_7d_rollup.py -q

/home/yuankai/.conda/envs/flooddiffusion/bin/python -m py_compile \
  utils/training/sample_creator.py \
  utils/training/window_local.py \
  utils/training/self_forcing.py \
  utils/training/config_validate.py
```

## 6. 使用方式

旧路径：

```yaml
stream_training:
  enabled: true
  latent_source: precomputed_slice
```

在线编码路径：

```yaml
stream_training:
  enabled: true
  latent_source: online_encode
```

如果不写 `latent_source`，默认仍是 `precomputed_slice`。

## 7. 需要继续关注的点

1. 字段命名  
   `latent_frame_lengths` 当前表示 local prefix frame length。后续如果
   `SampleCreator` 也要服务 global arbitrary range，建议改名为
   `local_latent_frame_lengths`，或者同时提供 global/local 两套长度。

2. `local_start_tokens` 的归属  
   目前 `SampleCreator` 返回 `local_start_tokens = 0`。这符合
   `online_encode`，但严格说 local start 是 latent source 构造层的语义，不是采样层
   的语义。当前 `window_local.py` 已经按 `latent_source` 决定最终写入
   `_window_local_latent_start_token`，所以训练行为是正确的。

3. 真实训练对比  
   文档和测试只能保证时间对齐与 fail-fast。最终还需要对比：
   - `precomputed_slice`
   - `online_encode`
   - `force_start_token_zero`
   - body aux loss on/off

   来确认 loss 异常是否来自离线 latent slice 分布。
