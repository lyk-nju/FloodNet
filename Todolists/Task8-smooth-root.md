## Task8 — Smooth Root 轨迹控制信号

Corresponds to [`target.md`](target.md) §8.

**Status: In Progress.** 设计方案已确定，代码尚未实现。

---

## 背景与动机

### 当前问题

实验 1 显示 xz 位置误差随序列长度单调放大（000021 从 0.17 增至 10.70）。根本原因是：

```
xz_position = ∫ velocity_prediction dt
```

每帧速度预测的微小误差通过积分不断累积，长序列尤其严重。

### Kimodo 的解决思路

NVIDIA Kimodo 论文（2026）使用 **Smooth Root Position** 作为运动特征的一部分：

1. 原始 root 轨迹经 ADMM 多分辨率平滑，每帧允许最大 0.06m 偏差的前提下最小化加速度
2. 平滑后的轨迹 `smooth_root_pos` 直接作为运动特征，而非积分速度

效果：
- 高频抖动被抑制，模型学到更稳定的低频轨迹
- 控制信号本身更平滑，ControlNet 更容易学习
- 误差积分放大效应减弱（平滑轨迹的每帧误差相关性更低）

---

## 设计方案

### 方案 A：平滑控制目标（推荐，低成本）

不改变运动特征格式（263D），只对**输入给 ControlNet 的轨迹条件**做平滑：

```
原来：traj_features = [x_raw, z_raw, cos(ψ), sin(ψ)]  (帧级，raw 积分位置)
改后：traj_features = [x_smooth, z_smooth, cos(ψ), sin(ψ)]  (帧级，ADMM平滑后)
```

修改位置：`datasets/humanml3d.py` 中的 `traj_features` 计算，在 `path_heading_features_from_root_xyz` 之前先对 `output["traj"][:, [0, 2]]` 做平滑。

```python
# datasets/humanml3d.py (示意)
traj_xyz = output["traj"]          # (T, 3)
traj_xz_smooth = smooth_root_xz(traj_xyz[:, [0, 2]])  # ADMM 平滑 xz
traj_xyz_smooth = traj_xyz.clone()
traj_xyz_smooth[:, [0, 2]] = traj_xz_smooth

traj_features = path_heading_features_from_root_xyz(traj_xyz_smooth)  # (T, 4)
output["traj_features"] = traj_features
```

> 控制损失 `L_control_xz` 仍对原始（未平滑）的 `output["traj"]` 计算误差，确保对真实轨迹的监督不变。

### 方案 B：平滑运动特征（中等成本，需重新 tokenize）

将平滑后的 smooth_root_pos 替换 263D 特征中的 `[1:3]`（根节点线速度 x/z），让模型直接学习预测平滑根节点位移。

这需要：
1. 修改 263D 特征格式定义
2. 重新预分词（`pretokenize_vae.py`）
3. 重新训练 VAE（可能需要）

成本较高，暂不推荐。

### 推荐：方案 A

**只改 ControlNet 的输入轨迹信号，不改 263D 特征格式，不需要重新训练 VAE。**

---

## ADMM 平滑算法（参考 Kimodo）

Kimodo 的 `get_smooth_root_pos` 使用多分辨率 ADMM 平滑：

```python
def smooth_root_xz(root_xz_np, margin=0.06):
    """
    root_xz_np: (T, 2) numpy array
    margin: 每帧允许偏离原始轨迹的最大距离（米）
    返回平滑后的 (T, 2) numpy array
    """
    margins = np.full(len(root_xz_np), margin)
    return smooth_signal(root_xz_np, margins)  # Kimodo 的 smooth_signal
```

参数：
- `margin=0.06`：每帧最大 6cm 偏差（来自 Kimodo 原始设置）
- `pos_weight=0`：不强制靠近原始位置，只最小化加速度
- 多分辨率迭代：从粗到细，避免局部最优

**注意**：如果不引入 Kimodo 的依赖，可以用简单的高斯滤波作为近似：

```python
from scipy.ndimage import gaussian_filter1d

def smooth_root_xz_gaussian(root_xz, sigma=2.0):
    """sigma 控制平滑程度，2.0 约等于 Kimodo margin=0.06 的效果"""
    return gaussian_filter1d(root_xz, sigma=sigma, axis=0)
```

---

## 实现步骤

1. **添加平滑函数**到 `utils/traj_batch.py` 或新建 `utils/smooth_traj.py`：
   ```python
   def smooth_root_xz(root_xz: np.ndarray, sigma: float = 2.0) -> np.ndarray:
       from scipy.ndimage import gaussian_filter1d
       return gaussian_filter1d(root_xz.astype(np.float64), sigma=sigma, axis=0).astype(np.float32)
   ```

2. **修改 `datasets/humanml3d.py`**（约第 205 行）：
   ```python
   # 对 traj_features 的 xz 分量做平滑（仅控制信号，不影响 GT loss 计算）
   traj_xyz = output["traj"]                          # (T, 3), raw
   traj_xz_smooth = smooth_root_xz(traj_xyz[:, [0, 2]].numpy())
   traj_xyz_for_cond = traj_xyz.clone()
   traj_xyz_for_cond[:, 0] = torch.from_numpy(traj_xz_smooth[:, 0])
   traj_xyz_for_cond[:, 2] = torch.from_numpy(traj_xz_smooth[:, 1])
   
   traj_features = path_heading_features_from_root_xyz(traj_xyz_for_cond)
   output["traj_features"] = traj_features
   # output["traj"] 保持原始不变（用于 L_control_xz 的 GT）
   ```

3. **添加配置开关**（`configs/ldf.yaml`，`data` 节下）：
   ```yaml
   data:
       smooth_traj_sigma: 2.0   # 0.0 = 关闭（保持原始），>0 = 高斯平滑 σ（单位：帧）
   ```

4. **验证**：
   - 可视化平滑前后的轨迹曲线（尤其是 000021 这种长序列复杂轨迹）
   - 重跑 `eval_control_loss` 对比平滑 vs 原始的 generate control loss
   - 重跑 `viz_traj.py` 对比轨迹跟随视觉效果

---

## 与其他任务的关系

- 方案 A 不需要修改模型架构，与 Scheduled Sampling（Task7）正交，可以同时使用。
- 平滑只影响 **ControlNet 的输入轨迹特征**，不影响 loss 计算的 GT 轨迹。
- 如果后续做 **两阶段去噪**（仿 Kimodo），smooth root 的概念可以直接复用。

---

## 参考

- Kimodo 技术报告：NVIDIA Research, 2026（见 `/home/yuankai/Text2Motion/kimodo/`）
- 平滑实现：`/home/yuankai/Text2Motion/kimodo/kimodo/motion_rep/smooth_root.py`
- Kimodo 动作特征：`smooth_root_pos` 字段，ADMM + multigrid 平滑，margin=0.06m
