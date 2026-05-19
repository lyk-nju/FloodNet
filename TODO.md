# FloodNet TODO：Web Demo Runtime 与 Stream Benchmark

本文件包含两个连续任务：

```text
Task 1:
    先完成 web_demo trajectory runtime 语义闭环，
    包括手动画线、debug 轨迹、repeat、clear、delayed blended update。

Task 2:
    在 Task 1 稳定后，再统一 benchmark runner 和 metrics，
    一条命令跑 step / real / turn / babel 四类 suite。
```

执行原则：

```text
1. 先确保 web_demo 实际送入模型的轨迹语义正确。
2. 再确保 benchmark 和 metric 测的是同一个 target。
3. Task 1 不做 benchmark 大重构。
4. Task 2 不改模型、不改训练。
```

---

# Active Task 001：Web Demo Trajectory Runtime 与 Delayed Blend Update Policy

## Task

统一 web_demo 中手动画线、debug 轨迹、repeat 轨迹、clear 轨迹和 `update_traj` 的 trajectory runtime 语义；实现 delayed blended trajectory update policy，避免 mid-session update 立即污染当前 denoising latent，同时让新轨迹平滑接管旧轨迹。

---

## Status

`open`

---

## Problem

当前 web_demo 中存在多类轨迹输入：

```text
manual drawing
debug HumanML3D trajectory
repeat trajectory
mid-session update trajectory
clear trajectory
```

这些输入的 runtime 语义需要统一，否则会出现：

```text
1. 用户画线后，模型实际收到的轨迹和前端显示的不一致；
2. debug HumanML3D 轨迹和手动画线走不同采样逻辑；
3. repeat 轨迹保持旧世界坐标，导致人物往回走或循环摆动；
4. update_traj 立即写入当前 denoise 区域，造成旧 latent state 和新 trajectory condition 冲突；
5. clear_traj 后仍然残留 trajectory buffer 或 pending update；
6. 前端看到的是新轨迹，但模型实际仍在 delay zone 使用旧轨迹，容易造成误判；
7. 如果 plan 采样使用全局 stream time，新 plan / repeat plan 可能从 plan 中间开始采样；
8. 如果 delayed update 期间消耗 new plan 时间轴，delay 越大，新 plan 越容易跳过前段。
```

已有 diagnose 结果显示，mid-session update 立即生效较差，而 delay 后明显改善。因此 Task 1 先把 web_demo runtime 语义做正确。

---

## Scope

本 task 只做 web_demo runtime 和 trajectory update policy。

包含：

```text
1. 手动画线按弧长重采样；
2. 默认 5s / 0.05s timestamp plan；
3. 手动画线 plan 平移到当前 root；
4. debug HumanML3D 轨迹也走同一套 timestamped plan 逻辑；
5. repeat 轨迹以当前 root 重新 anchor；
6. clear_traj 清空 active / pending / trajectory buffer；
7. update_traj 不立即覆盖当前条件；
8. 实现 delayed blended replace；
9. status/debug metadata 暴露模型实际使用的 blended trajectory；
10. sample_plan_future 使用 plan-local time，而不是 global stream time；
11. delayed update 中 new plan 从 effective_commit_index 才开始计时。
```

不包含：

```text
1. 不训练模型；
2. 不改模型结构；
3. 不做 benchmark / metric 大重构；
4. 不加入 lateral / heading metrics；
5. 不做 BABEL benchmark；
6. 不实现 pred-root closed-loop self-forcing；
7. 不做复杂 SE(2) heading blend；
8. 不把 web_demo 状态逻辑塞进 dataset。
```

---

## Required changes

## 1. 新增统一 trajectory runtime 纯函数

目标文件：

```text
FloodNet/utils/stream_traj.py
```

新增函数：

```python
def ensure_xyz(points: np.ndarray) -> np.ndarray:
    """保证输入为 (N, 3) xyz。"""


def resample_polyline_by_arclength(
    points_xyz: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """按 XZ 弧长均匀重采样。"""


def assign_uniform_timestamps(
    num_points: int,
    waypoint_dt: float,
) -> np.ndarray:
    """生成 0, dt, 2dt, ... 时间戳。"""


def translate_plan_to_current_root(
    plan_points_xyz: np.ndarray,
    current_root_xyz: np.ndarray,
) -> np.ndarray:
    """将 plan 平移，使首点靠近 current_root。"""


def normalize_manual_waypoints(
    raw_points_xyz: np.ndarray,
    *,
    current_root_xyz: np.ndarray,
    waypoint_dt: float,
    manual_duration_seconds: float,
    resample_arclength: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """把 web 手动画线转换成 timestamped world-space plan。"""


def sample_plan_by_time(
    plan_times: np.ndarray,
    plan_points_xyz: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """按时间从 plan 中采样 future target。"""


def sample_plan_future(
    plan: StreamTrajectoryPlan,
    *,
    current_commit: int,
    current_root_xyz: np.ndarray,
    horizon_tokens: int,
    token_dt: float,
    reanchor_to_current_root: bool,
) -> np.ndarray:
    """
    构造当前 step 的 future trajectory condition。

    重要：该函数必须使用 plan-local time，而不是 global stream time。

    语义：
        plan.start_commit_index 对应 plan.times 的 t=0。
        elapsed_tokens = current_commit - plan.start_commit_index
        query_times = elapsed_tokens * token_dt + arange(horizon_tokens) * token_dt
    """


def smoothstep01(u: float) -> float:
    """smoothstep interpolation weight。"""


def blend_future_trajs(
    old_xyz: np.ndarray,
    new_xyz: np.ndarray,
    weight: float,
) -> np.ndarray:
    """对 old/new future trajectory 做 XYZ smooth blend。"""
```

第一版只做 XYZ position blend，不做复杂 SE(2) heading blend。

---

## 2. stream.yaml 增加 runtime 配置

目标文件：

```text
FloodNet/configs/stream.yaml
```

建议字段：

```yaml
traj_mask:
  time_mode: timestamped
  waypoint_dt: 0.05
  manual_duration_seconds: 5.0
  manual_resample_arclength: true
  horizon_tokens: 20
  token_dt: 0.20
  repeat_policy: translate_from_current_root

  update_policy: delayed_blended_replace
  update_delay_tokens: 20
  update_blend_tokens: 4
  update_warmup_tokens: 4   # optional alias
```

读取规则：

```python
delay_tokens = int(cfg.get("update_delay_tokens", horizon_tokens))
blend_tokens = int(cfg.get("update_blend_tokens", cfg.get("update_warmup_tokens", 0)))
```

说明：

```text
1. update_warmup_tokens 只是兼容别名；
2. 代码内部统一使用 blend_tokens；
3. 这里的 warmup 只表示 update transition blend window，不是模型 warmup。
```

---

## 3. 统一 plan-local time 语义

所有 plan 都必须有自己的时间原点：

```text
plan.start_commit_index:
    该 plan 的局部 t=0 对应的 commit index。
```

`sample_plan_future()` 不允许直接使用全局 stream time。

正确采样公式：

```python
elapsed_tokens = max(0, current_commit - plan.start_commit_index)
query_times = (
    elapsed_tokens * token_dt
    + np.arange(horizon_tokens, dtype=np.float32) * token_dt
)
future_xyz = sample_plan_by_time(plan.times, plan.points_xyz, query_times)
```

原因：

```text
manual plan / repeat plan / update new plan 都是从各自激活时刻开始计时，
不是从整个 generation session 的 t=0 开始计时。
```

如果错误使用 global stream time，会导致：

```text
1. repeat plan 从中间开始采样；
2. update new plan 一生效就被采到几秒以后；
3. delay 越长，新 plan 越容易跳过前段。
```

---

## 4. 统一手动画线语义

目标文件：

```text
FloodNet/web_demo/model_manager.py
FloodNet/web_demo/app.py
```

要求：

```text
1. 用户画线后，调用 normalize_manual_waypoints；
2. 默认 manual_duration_seconds = 5.0；
3. 默认 waypoint_dt = 0.05；
4. 默认重采样到 100 / 101 个点；
5. plan 首点平移到 current_root 附近；
6. plan.start_commit_index = 当前 commit index；
7. 前端显示的 plan 与后端 active / pending plan 一致；
8. 后端应保存 timestamped plan，而不是只保存裸空间点。
```

伪代码：

```python
current_commit = self.get_commit_index()
current_root = self.get_current_root_xyz()

times, points = normalize_manual_waypoints(
    raw_points_xyz,
    current_root_xyz=current_root,
    waypoint_dt=self.waypoint_dt,
    manual_duration_seconds=self.manual_duration_seconds,
    resample_arclength=self.manual_resample_arclength,
)

plan = StreamTrajectoryPlan(
    times=times,
    points_xyz=points,
    start_commit_index=current_commit,
    version=self.next_plan_version(),
    source="manual",
)
```

---

## 5. 统一 debug HumanML3D 轨迹语义

目标文件：

```text
FloodNet/web_demo/model_manager.py
FloodNet/web_demo/app.py
```

要求：

```text
1. debug 轨迹只读取 HumanML3D root path 空间点；
2. 不使用 HumanML3D 原始 temporal timestamps；
3. 使用与手动画线相同的 resample_arclength + waypoint_dt；
4. 赋予 0.05s timestamps；
5. plan.start_commit_index = 当前 commit index；
6. 如果 debug 轨迹 repeat，也必须 reanchor 到 current_root。
```

目的：

```text
让 debug 轨迹、手动画线和 diagnose duration waypoint 语义一致。
```

---

## 6. 统一 repeat trajectory 语义

repeat 时必须：

```text
1. 复制原 plan 的空间形状；
2. 平移到当前 root；
3. 重新赋 timestamp；
4. start_commit_index = 当前 commit index；
5. 不能保留旧世界坐标；
6. future condition 应从 current_root 附近开始。
```

禁止：

```text
repeat plan 沿用旧世界坐标
→ 人物往回走
→ 循环摆动
```

伪代码：

```python
def repeat_active_plan():
    current_commit = self.get_commit_index()
    current_root = self.get_current_root_xyz()

    old_points = self.active_traj_plan.points_xyz
    repeated_points = translate_plan_to_current_root(old_points, current_root)
    times = assign_uniform_timestamps(len(repeated_points), self.waypoint_dt)

    self.active_traj_plan = StreamTrajectoryPlan(
        times=times,
        points_xyz=repeated_points,
        start_commit_index=current_commit,
        version=self.next_plan_version(),
        source="repeat",
    )
```

---

## 7. clear_traj 行为

`clear_traj` 必须：

```text
1. self.active_traj_plan = None
2. self.pending_update_event = None
3. reset model trajectory buffer
4. 后续生成走 no-traj / backbone-only path
5. 前端状态显示 no active trajectory
```

伪代码：

```python
def clear_trajectory():
    self.active_traj_plan = None
    self.pending_update_event = None
    self.reset_model_traj_buffer()
    self.trajectory_state = "none"
```

---

## 8. 增加 active plan 和 pending update event 状态

目标文件：

```text
FloodNet/web_demo/model_manager.py
```

建议状态：

```python
self.active_traj_plan = None
self.pending_update_event = None
self.traj_update_delay_tokens = ...
self.traj_update_blend_tokens = ...
self.plan_version = 0
```

建议 dataclass：

```python
@dataclass
class StreamTrajectoryPlan:
    times: np.ndarray
    points_xyz: np.ndarray
    start_commit_index: int
    version: int
    source: str


@dataclass
class TrajectoryUpdateEvent:
    old_plan: StreamTrajectoryPlan | None
    new_plan: StreamTrajectoryPlan
    edit_commit_index: int
    effective_commit_index: int
    delay_tokens: int
    blend_tokens: int
    version: int
```

第一版如果不想引入 dataclass，也可以用 dict，但语义必须一致。

---

## 9. update_trajectory 不直接覆盖 active plan

当前 `update_traj` 不应立即覆盖当前 active plan，而是创建 pending update event。

关键语义：

```text
edit_commit_index:
    用户触发 update 的 commit index。

effective_commit_index:
    新 plan 真正开始消耗时间轴的 commit index。
    effective_commit_index = edit_commit_index + delay_tokens。

new_plan.start_commit_index:
    必须等于 effective_commit_index。
```

伪代码：

```python
def update_trajectory(raw_waypoints):
    edit_commit = self.get_commit_index()
    current_root = self.get_current_root_xyz()

    delay_tokens = self.traj_update_delay_tokens
    effective_commit = edit_commit + delay_tokens

    times, points = normalize_manual_waypoints(
        raw_waypoints,
        current_root_xyz=current_root,
        waypoint_dt=self.waypoint_dt,
        manual_duration_seconds=self.manual_duration_seconds,
        resample_arclength=self.manual_resample_arclength,
    )

    new_plan = StreamTrajectoryPlan(
        times=times,
        points_xyz=points,
        start_commit_index=effective_commit,
        version=self.next_plan_version(),
        source="manual_update",
    )

    self.pending_update_event = TrajectoryUpdateEvent(
        old_plan=self.active_traj_plan,
        new_plan=new_plan,
        edit_commit_index=edit_commit,
        effective_commit_index=effective_commit,
        delay_tokens=delay_tokens,
        blend_tokens=self.traj_update_blend_tokens,
        version=new_plan.version,
    )
```

说明：

```text
delay zone 中：
    old_plan 正常推进；
    new_plan 不消耗时间轴。

blend / replace 阶段：
    new_plan 从局部 t=0 开始采样。
```

---

## 10. 每步生成前计算 delayed blend 权重

伪代码：

```python
def get_update_blend_weight(event, current_commit):
    offset = current_commit - event.edit_commit_index

    if offset < event.delay_tokens:
        return 0.0

    if event.blend_tokens <= 0:
        return 1.0

    u = (offset - event.delay_tokens) / event.blend_tokens
    return smoothstep01(u)
```

语义：

```text
offset < delay_tokens:
    使用 old trajectory

delay_tokens <= offset < delay_tokens + blend_tokens:
    使用 old/new blended trajectory

offset >= delay_tokens + blend_tokens:
    使用 new trajectory，并结束 pending event
```

---

## 11. `_build_stream_traj_input` 支持 pending update

目标文件：

```text
FloodNet/web_demo/model_manager.py
```

伪代码：

```python
def _build_stream_traj_input():
    current_commit = self.get_commit_index()
    current_root = self.get_current_root_xyz()

    event = self.pending_update_event

    if event is None:
        if self.active_traj_plan is None:
            return None

        future_xyz = sample_plan_future(
            self.active_traj_plan,
            current_commit=current_commit,
            current_root_xyz=current_root,
            horizon_tokens=self.traj_horizon_tokens,
            token_dt=self.token_dt,
            reanchor_to_current_root=True,
        )
        self._last_model_used_traj_preview = future_xyz
        self.trajectory_state = "active"
        return future_xyz

    old_future = None
    if event.old_plan is not None:
        old_future = sample_plan_future(
            event.old_plan,
            current_commit=current_commit,
            current_root_xyz=current_root,
            horizon_tokens=self.traj_horizon_tokens,
            token_dt=self.token_dt,
            reanchor_to_current_root=True,
        )

    new_future = sample_plan_future(
        event.new_plan,
        current_commit=current_commit,
        current_root_xyz=current_root,
        horizon_tokens=self.traj_horizon_tokens,
        token_dt=self.token_dt,
        reanchor_to_current_root=True,
    )

    w = get_update_blend_weight(event, current_commit)

    if old_future is None:
        future_xyz = new_future
    elif w <= 0:
        future_xyz = old_future
    elif w >= 1:
        future_xyz = new_future
    else:
        future_xyz = blend_future_trajs(old_future, new_future, w)

    self._last_model_used_traj_preview = future_xyz
    self._last_update_blend_weight = w

    if w <= 0:
        self.trajectory_state = "delay"
    elif w < 1:
        self.trajectory_state = "blend"
    else:
        self.trajectory_state = "replaced"
        self.active_traj_plan = reanchor_plan_to_current_root(event.new_plan, current_root)
        self.pending_update_event = None

    return future_xyz
```

注意：

```text
sample_plan_future(event.new_plan, current_commit=...) 内部会用
new_plan.start_commit_index = event.effective_commit_index
计算 plan-local query time。
```

---

## 12. 前端 / status 暴露实际使用轨迹状态

status/debug metadata 至少暴露：

```json
{
  "active_plan_version": 1,
  "pending_plan_version": 2,
  "trajectory_update_policy": "delayed_blended_replace",
  "update_blend_weight": 0.5,
  "update_delay_tokens": 20,
  "update_blend_tokens": 4,
  "trajectory_state": "delay|blend|replaced|active|none",
  "edit_commit_index": 100,
  "effective_commit_index": 120
}
```

建议额外返回：

```text
model_used_traj_preview
```

即模型当前实际使用的 blended future trajectory，而不只是用户刚画的新 trajectory。

原因：

```text
delay zone 内前端看到新轨迹，但模型实际仍在用旧轨迹。
如果不展示状态，会造成误判。
```

---

## Do not do

本 task 不做：

```text
1. 不训练模型。
2. 不改模型结构。
3. 不做 benchmark / metric 大重构。
4. 不加入 lateral / heading metrics。
5. 不实现 BABEL benchmark。
6. 不实现 pred-root closed-loop self-forcing。
7. 不做复杂 SE(2) heading blend。
8. 不把 web_demo 状态逻辑塞进 dataset。
```

---

## Validation

### 编译

```bash
python3 -m py_compile \
  /home/yuankai/Text2Motion/FloodNet/web_demo/model_manager.py \
  /home/yuankai/Text2Motion/FloodNet/web_demo/app.py \
  /home/yuankai/Text2Motion/FloodNet/utils/stream_traj.py
```

### 最小逻辑测试

至少覆盖：

```text
1. 手动画线 2 点 -> 100 / 101 个弧长均匀点。
2. 手动画线首点远离 current_root -> plan 首点被平移到 current_root 附近。
3. debug HumanML3D root path -> 使用 0.05s timestamps。
4. repeat 轨迹 -> 新 plan 首点接近 current_root。
5. repeat plan 的 start_commit_index = 当前 commit。
6. clear_traj -> active_plan / pending_update_event / traj buffer 均清空。
7. update_traj -> 不立即覆盖 active_plan，而是创建 pending_update_event。
8. update_traj 中 new_plan.start_commit_index = edit_commit + delay_tokens。
9. offset < delay_tokens 时，模型实际使用 old_future。
10. delay_tokens <= offset < delay+blend 时，模型实际使用 blended_future。
11. offset >= delay+blend 时，active_plan 变成 new_plan，pending_update_event 清空。
12. sample_plan_future 使用 plan-local time，而不是 global stream time。
13. status 中能看到 update_blend_weight、edit_commit_index、effective_commit_index 和 trajectory_state。
```

### Web smoke

手动检查：

```text
1. start generation
2. 人物沿初始轨迹前进
3. 用户手动画一条新轨迹
4. 新轨迹首点显示在当前 root 附近
5. update 后 delay zone 内模型不应立即被新轨迹拉扯
6. delay zone 内 new plan 不应被时间轴消耗
7. blend zone 内模型逐渐转向新轨迹
8. repeat 轨迹不会回到旧世界坐标
9. clear trajectory 后模型不再收到轨迹条件
10. 前端显示 active / pending / blend weight 状态
11. model_used_traj_preview 与实际生成方向一致
```

---

## Done criteria

```text
1. web_demo 手动画线、debug 轨迹、repeat 轨迹都使用同一套 stream_traj 纯函数。
2. 默认 5s / 0.05s 行为稳定。
3. 手动画线和 repeat 轨迹都会 reanchor 到当前 root。
4. clear_traj 会清空 active plan、pending event 和 model trajectory buffer。
5. update_traj 不再立即覆盖当前 active plan。
6. pending update event 在 commit + delay_tokens 后开始接管。
7. new_plan.start_commit_index = edit_commit_index + delay_tokens。
8. sample_plan_future 使用 plan-local time。
9. blend_tokens > 0 时，新轨迹平滑接管旧轨迹。
10. blend_tokens = 0 时，退化为 hard delayed replace。
11. delay_tokens = 20 时，web 行为和 diagnose delay20 语义一致。
12. 前端 / status 能解释当前处于 none / active / delay / blend / replaced 哪个阶段。
13. status 能显示模型实际使用的 blended trajectory preview。
14. 不再出现 repeat 轨迹保持旧世界坐标导致人物往回走的问题。
15. 不改模型、不改训练、不做 benchmark 重构。
```

---

## Review update

待实现。

---

# Active Task 002：Unified Stream Benchmark Runner 与 Metrics 重构

## Task

在 Task 001 web_demo trajectory runtime 语义稳定后，实现统一 `stream_benchmark.py`：一次命令运行 `step / real / turn / babel` 四类 benchmark suite，统一 plan target 构造、metric 口径、summary 输出，并新增 lateral / heading 诊断指标。

---

## Status

`open`

---

## Problem

完成 Task 001 后，web_demo 中模型实际收到的 trajectory runtime 语义将统一为：

```text
manual drawing
debug trajectory
repeat trajectory
clear trajectory
delayed blended update
```

接下来需要统一 benchmark 和 metric，使后续每次修改 web_demo、推理策略、训练或 checkpoint 时，都能通过一条命令回答：

```text
1. 基础 step 能力是否退化？
2. pred-root closed-loop gap 是否改善？
3. web-demo-like real plan 是否改善？
4. mid-session update 是否改善？
5. BABEL 长程多文本 / 多轨迹是否可用？
6. root 是否贴轨迹但身体没转向？
7. 是否存在明显侧向滑动？
```

当前 `diagnose_stream_control.py` 已经能跑很多诊断，但入口、case、target、metric、summary 字段仍不够统一。

---

## Scope

本 task 只做 benchmark 和 metric 重构。

包含：

```text
1. 新增统一 benchmark runner；
2. 定义 step / real / turn / babel suite；
3. 统一 ADE / FDE / path_arc / path_chamfer；
4. 明确 path_arc 和 path_chamfer 的固定定义；
5. 新增 lateral_velocity_ratio；
6. 新增 heading_path_error_deg；
7. real / turn / babel 主指标对 sampled / updated plan target 计算；
8. original GT 只作为辅助指标；
9. 输出 summary.json / summary.csv；
10. 可选 render video。
```

不包含：

```text
1. 不改模型结构；
2. 不改训练逻辑；
3. 不实现 delayed blend，Task 001 已完成；
4. 不实现 pending update event，Task 001 已完成；
5. 不实现 pred-root closed-loop self-forcing；
6. 不引入新的 loss；
7. 不修改 checkpoint 结构；
8. 不为了整理 benchmark 大规模重写所有 eval 代码。
```

---

## Required changes

## 1. 新增统一 benchmark runner

目标文件：

```text
FloodNet/eval/stream_benchmark.py
```

职责：

```text
1. 解析命令行参数；
2. 加载 config / model / VAE / dataset；
3. 根据 --suites 或 --preset 选择 benchmark suite；
4. 调用 stream_benchmarks.py 中的 case；
5. 调用 stream_metrics.py 计算指标；
6. 汇总 summary.json / summary.csv；
7. 可选 render video；
8. 第一版可以复用 diagnose_stream_control.py 里的现有函数，避免大重写。
```

推荐 CLI：

```bash
python eval/stream_benchmark.py \
  --config configs/stream.yaml \
  --ckpt outputs/step_460000.ckpt \
  --vae_ckpt outputs/vae_1d_z4_step=300000.ckpt \
  --raw_data_dir /data1/yuankai/text2Motion/FloodDiffusion/raw_data \
  --preset smoke \
  --render_video
```

---

## 2. 定义 benchmark cases 和 suites

目标文件：

```text
FloodNet/eval/stream_benchmarks.py
```

建议结构：

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class StreamBenchmarkCase:
    name: str
    suite: str
    sample_id: str | list[str]
    dataset: str
    mode: str
    config_overrides: dict[str, Any] = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)
```

统一四类 suite：

```text
step
real
turn
babel
```

---

## 3. CLI 设计

### 3.1 默认 smoke

```bash
python eval/stream_benchmark.py \
  --config configs/stream.yaml \
  --ckpt outputs/step_460000.ckpt \
  --vae_ckpt outputs/vae_1d_z4_step=300000.ckpt \
  --raw_data_dir /data1/yuankai/text2Motion/FloodDiffusion/raw_data \
  --preset smoke
```

Smoke preset 至少包含：

```text
step_metric_001168
real_metric_001168
turn_metric_001168_rot30
babel_metric_9797
```

### 3.2 跑所有 suite

```bash
python eval/stream_benchmark.py \
  --config configs/stream.yaml \
  --ckpt outputs/step_460000.ckpt \
  --vae_ckpt outputs/vae_1d_z4_step=300000.ckpt \
  --raw_data_dir /data1/yuankai/text2Motion/FloodDiffusion/raw_data \
  --suites all \
  --render_video
```

### 3.3 只跑部分 suite

```bash
python eval/stream_benchmark.py \
  --config configs/stream.yaml \
  --ckpt outputs/step_460000.ckpt \
  --vae_ckpt outputs/vae_1d_z4_step=300000.ckpt \
  --raw_data_dir /data1/yuankai/text2Motion/FloodDiffusion/raw_data \
  --suites step,real,turn
```

### 3.4 推荐参数

```text
--config
--ckpt
--vae_ckpt
--raw_data_dir
--output_dir
--suites                  # all / step,real,turn,babel
--preset                  # smoke / full
--sample_id
--sample_ids
--dataset                 # humanml3d / babel
--render_video
--duration_waypoint_stride_seconds
--waypoint_dt
--history_length
--traj_horizon_tokens
--turn_angle
--split_token
--update_delay_tokens
--update_blend_tokens
```

规则：

```text
--suites 与 --preset 同时出现时，优先 --suites。
--preset smoke 是默认建议。
```

---

## Benchmark suites

## 4. Step suite

### 用途

检查基础 streaming step 能力和 GT-root / pred-root gap。

### Modes

```text
step_gtroot
step_predroot
step_no_traj
```

### Target

```text
original_gt_root
```

### Metrics

```text
ADE
FDE
path_arc
path_chamfer
lateral_velocity_ratio
heading_path_error_deg
```

### 禁止输出

```text
closed_loop_gap_ADE
closed_loop_gap_FDE
traj_gain_ADE
traj_gain_FDE
```

这些派生字段容易误导，统一 benchmark 中不输出。

---

## 5. Real suite

### 用途

模拟 web-demo-like duration waypoint / sampled plan target。

### Modes

```text
real_gtroot
real_predroot
real_no_traj
```

### 主 target

```text
sampled_plan_target
```

### 辅助 target

```text
original_gt_root
```

### HumanML3D 单样本 plan 构造公式

对 HumanML3D 单样本：

```text
target_frames = len(original_gt_root)
target_duration = (target_frames - 1) / motion_fps
num_points = round(target_duration / waypoint_dt) + 1
plan_points = resample_polyline_by_arclength(original_gt_root, num_points)
plan_times = arange(num_points) * waypoint_dt
```

当：

```text
motion_fps = 20
waypoint_dt = 0.05
```

则：

```text
num_points == target_frames
```

注意：

```text
real_metric 不应默认使用 web_demo 的 manual_duration_seconds = 5.0。
real_metric 的 plan 长度应由 dataset sample 的 target_frames 决定。
```

### 主指标

对 `sampled_plan_target` 计算：

```text
ADE
FDE
path_arc
path_chamfer
lateral_velocity_ratio
heading_path_error_deg
```

### 辅助指标

```text
ADE_vs_original_gt
FDE_vs_original_gt
```

注意：

```text
real_metric 的 ADE / FDE / path_arc / path_chamfer 默认全部对 sampled_plan_target 计算。
original GT 只作为辅助对照。
```

---

## 6. Turn suite

### 用途

测试 mid-session update / rotated suffix / delayed update policy。

### Modes

第一版建议包含：

```text
turn_immediate_rot30
turn_delay20_rot30
turn_delay20_blend4_rot30
```

可选 oracle：

```text
turn_gtroot_delay20_blend4_rot30
```

### Target

```text
updated_plan_target
```

### Metrics

```text
ADE
FDE
path_arc
path_chamfer
lateral_velocity_ratio
heading_path_error_deg
```

### 额外 pre/post 指标

```text
pre_ADE
post_ADE
pre_lateral_velocity_ratio
post_lateral_velocity_ratio
pre_heading_path_error_deg
post_heading_path_error_deg
```

注意：

```text
turn_metric 随着 delay 改变实际更新点和总 frame 长度。
必须先计算完整 target frame 数，再统一截断 pred / target。
不要在原 clip 结束处提前截断。
```

---

## 7. BABEL suite

### 用途

长程多文本 / 多轨迹切换验证。

### Modes

```text
babel_real
babel_timestamped
babel_no_traj
```

### Target

```text
per-segment sampled_plan_target
```

### Metrics

```text
overall ADE
overall FDE
overall path_arc
overall path_chamfer
overall lateral_velocity_ratio
overall heading_path_error_deg
per-segment metrics
```

注意：

```text
BABEL 的 real plan 应在每个子样本内部独立构造 plan，
再拼接 plan target。
不要在拼接后的整体路径上统一采样。
```

---

## Metric design

## 8. 新增 metric 文件

目标文件：

```text
FloodNet/eval/stream_metrics.py
```

---

## 9. 基础 metrics

```python
def compute_ade(
    pred_root_xyz: np.ndarray,
    target_root_xyz: np.ndarray,
) -> float:
    ...


def compute_fde(
    pred_root_xyz: np.ndarray,
    target_root_xyz: np.ndarray,
) -> float:
    ...


def compute_path_arc(
    pred_root_xyz: np.ndarray,
    target_arc_xyz: np.ndarray,
) -> float:
    ...


def compute_path_chamfer(
    pred_root_xyz: np.ndarray,
    target_arc_xyz: np.ndarray,
) -> float:
    ...
```

---

## 10. path_arc 固定定义

`path_arc` 定义为：

```text
将 pred_root path 和 target path 都按弧长重采样到相同点数，
然后计算平均 L2 distance。
```

公式：

```text
path_arc(P, T) = mean_i || arc_resample(P, N)[i] - arc_resample(T, N)[i] ||_2
```

其中：

```text
N 默认取 target_frames 或当前比较窗口长度。
```

用途：

```text
衡量路径形状相似度，减少 temporal alignment 对路径形状评估的影响。
```

---

## 11. path_chamfer 固定定义

`path_chamfer` 默认使用 symmetric Chamfer distance：

```text
Chamfer(P, T) =
0.5 * (
    mean_{p in P} min_{t in T} ||p - t||_2
    +
    mean_{t in T} min_{p in P} ||t - p||_2
)
```

注意：

```text
必须固定为 symmetric Chamfer，除非配置中显式指定 one-sided。
如果后续实现 one-sided，需要在 summary 中写明 chamfer_type。
```

---

## 12. Plan target 构造

复用 Task 001 的 `utils/stream_traj.py`：

```python
def compute_plan_targets(
    plan_times: np.ndarray,
    plan_points_xyz: np.ndarray,
    target_frames: int,
    motion_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 target_time 和 target_arc。"""
```

伪代码：

```python
def compute_plan_targets(plan_times, plan_points, target_frames, motion_fps):
    frame_times = np.arange(target_frames, dtype=np.float32) / motion_fps

    target_time = sample_plan_by_time(
        plan_times,
        plan_points,
        frame_times,
    )

    target_arc = build_plan_arc_target(
        plan_points,
        target_frames,
    )

    return target_time, target_arc
```

---

## 13. Plan metrics 汇总

```python
def build_plan_metrics(
    pred_root_xyz: np.ndarray,
    original_gt_root_xyz: np.ndarray | None,
    plan_times: np.ndarray,
    plan_points_xyz: np.ndarray,
    target_frames: int,
    motion_fps: float,
    motion_263: np.ndarray | None = None,
) -> dict:
    ...
```

伪代码：

```python
def build_plan_metrics(
    pred_root,
    original_gt_root,
    plan_times,
    plan_points,
    target_frames,
    motion_fps,
    motion_263=None,
):
    target_time, target_arc = compute_plan_targets(
        plan_times,
        plan_points,
        target_frames=target_frames,
        motion_fps=motion_fps,
    )

    pred_root = pred_root[:target_frames]
    target_time = target_time[:len(pred_root)]
    target_arc = target_arc[:len(pred_root)]

    metrics = {
        "ADE": compute_ade(pred_root, target_time),
        "FDE": compute_fde(pred_root, target_time),
        "path_arc": compute_path_arc(pred_root, target_arc),
        "path_chamfer": compute_path_chamfer(pred_root, target_arc),
        "target_source": "sampled_plan_target",
        "chamfer_type": "symmetric",
    }

    if original_gt_root is not None:
        gt = original_gt_root[:len(pred_root)]
        metrics["ADE_vs_original_gt"] = compute_ade(pred_root, gt)
        metrics["FDE_vs_original_gt"] = compute_fde(pred_root, gt)

    if motion_263 is not None:
        metrics.update(
            compute_motion_heading_metrics(
                motion_263=motion_263,
                target_root_xyz=target_time,
            )
        )

    return metrics
```

---

## 14. lateral_velocity_ratio

### 用途

衡量人物是否在 body local frame 中侧向滑动。

### 函数

```python
def compute_lateral_velocity_ratio(motion_263: np.ndarray) -> float:
    ...
```

### 伪代码

```python
def compute_lateral_velocity_ratio(motion_263):
    root = extract_root(motion_263)
    body_yaw = estimate_body_yaw(motion_263)

    vel_xz = np.diff(root[:, [0, 2]], axis=0)
    local_vel = rotate_by_minus_yaw(vel_xz, body_yaw[:-1])

    lateral = local_vel[:, 0]
    speed = np.linalg.norm(vel_xz, axis=-1)

    return np.mean(np.abs(lateral)) / (np.mean(speed) + eps)
```

注意：

```text
body local frame 的前向轴 / 侧向轴要和现有 skeleton 坐标定义一致。
如果当前无法可靠估计 body_yaw，第一版可以使用 root rotation 或 pelvis/hip 方向近似。
```

---

## 15. heading_path_error_deg

### 用途

衡量身体朝向是否和目标路径切线方向一致。

### 函数

```python
def compute_heading_path_error_deg(
    motion_263: np.ndarray,
    target_root_xyz: np.ndarray,
) -> float:
    ...
```

### 伪代码

```python
def compute_heading_path_error_deg(motion_263, target_root_xyz):
    body_yaw = estimate_body_yaw(motion_263)

    target_vel = np.diff(target_root_xyz[:, [0, 2]], axis=0)
    path_yaw = np.arctan2(target_vel[:, 1], target_vel[:, 0])

    err = wrap_angle(body_yaw[:-1] - path_yaw)
    valid = np.linalg.norm(target_vel, axis=-1) > speed_eps

    return np.mean(np.abs(err[valid])) * 180.0 / np.pi
```

注意：

```text
该指标只用于诊断，不代表所有动作都应该朝路径方向走。
对 dance / sidestep / turn-in-place，应结合文本和视频人工判断。
```

---

## 16. 输出字段

每个 mode 统一输出：

```json
{
  "lateral_velocity_ratio": 0.0,
  "heading_path_error_deg": 0.0
}
```

turn suite 额外输出：

```json
{
  "pre_lateral_velocity_ratio": 0.0,
  "post_lateral_velocity_ratio": 0.0,
  "pre_heading_path_error_deg": 0.0,
  "post_heading_path_error_deg": 0.0
}
```

---

## Output format

## 17. 推荐输出目录

```text
outputs/stream_benchmark/
└── <run_id>/
    ├── summary.json
    ├── summary.csv
    ├── suites/
    │   ├── step_metric.json
    │   ├── real_metric.json
    │   ├── turn_metric.json
    │   └── babel_metric.json
    └── videos/
```

## 18. summary.json

示例：

```json
{
  "run_id": "2026xxxx_xxxxxx",
  "config": "configs/stream.yaml",
  "ckpt": "outputs/step_460000.ckpt",
  "vae_ckpt": "outputs/vae_1d_z4_step=300000.ckpt",
  "raw_data_dir": "/data1/yuankai/text2Motion/FloodDiffusion/raw_data",
  "waypoint_dt": 0.05,
  "traj_horizon_tokens": 20,
  "history_length": 30,
  "suites": {
    "step": {
      "step_gtroot": {
        "sample_id": "001168",
        "ADE": 0.0,
        "FDE": 0.0,
        "path_arc": 0.0,
        "path_chamfer": 0.0,
        "chamfer_type": "symmetric",
        "lateral_velocity_ratio": 0.0,
        "heading_path_error_deg": 0.0,
        "target_source": "original_gt_root"
      }
    },
    "real": {
      "real_predroot": {
        "sample_id": "001168",
        "ADE": 0.0,
        "FDE": 0.0,
        "path_arc": 0.0,
        "path_chamfer": 0.0,
        "chamfer_type": "symmetric",
        "lateral_velocity_ratio": 0.0,
        "heading_path_error_deg": 0.0,
        "target_source": "sampled_plan_target",
        "ADE_vs_original_gt": 0.0,
        "FDE_vs_original_gt": 0.0
      }
    }
  }
}
```

## 19. summary.csv

每行一个 mode：

```text
suite,mode,sample_id,ADE,FDE,path_arc,path_chamfer,chamfer_type,lateral_velocity_ratio,heading_path_error_deg,target_source,ADE_vs_original_gt,FDE_vs_original_gt
```

---

## Validation

### 20. 编译

```bash
python3 -m py_compile \
  /home/yuankai/Text2Motion/FloodNet/eval/stream_benchmark.py \
  /home/yuankai/Text2Motion/FloodNet/eval/stream_benchmarks.py \
  /home/yuankai/Text2Motion/FloodNet/eval/stream_metrics.py \
  /home/yuankai/Text2Motion/FloodNet/eval/diagnose_stream_control.py
```

### 21. 最小单元测试或脚本检查

至少覆盖：

```text
1. real suite 的 ADE / FDE / path_arc / path_chamfer 对 sampled plan target 计算。
2. real_metric 的 num_points 由 target_frames 和 waypoint_dt 决定，不使用 web 默认 5s。
3. original GT 指标只出现在 ADE_vs_original_gt / FDE_vs_original_gt。
4. step suite 不输出 closed_loop_gap / traj_gain 派生字段。
5. turn suite delay 后先扩展 target frame，再统一截断 pred / target。
6. path_arc 对 pred/target 都做 arc-length resample 后计算。
7. path_chamfer 使用 symmetric Chamfer，并在 summary 中记录 chamfer_type。
8. lateral_velocity_ratio 能正常输出 finite number。
9. heading_path_error_deg 能正常输出 finite number。
10. summary.json 和 summary.csv 都能生成。
11. render_video 时视频 overlay 目标 trajectory。
```

### 22. Smoke benchmark

```bash
cd /home/yuankai/Text2Motion/FloodNet

python eval/stream_benchmark.py \
  --config configs/stream.yaml \
  --ckpt /home/yuankai/Text2Motion/FloodNet/outputs/step_460000.ckpt \
  --vae_ckpt /home/yuankai/Text2Motion/FloodNet/outputs/vae_1d_z4_step=300000.ckpt \
  --raw_data_dir /data1/yuankai/text2Motion/FloodDiffusion/raw_data \
  --preset smoke \
  --render_video
```

应输出：

```text
summary.json
summary.csv
suites/step_metric.json
suites/real_metric.json
suites/turn_metric.json
suites/babel_metric.json
videos/
```

### 23. 只跑部分 suite

```bash
cd /home/yuankai/Text2Motion/FloodNet

python eval/stream_benchmark.py \
  --config configs/stream.yaml \
  --ckpt /home/yuankai/Text2Motion/FloodNet/outputs/step_460000.ckpt \
  --vae_ckpt /home/yuankai/Text2Motion/FloodNet/outputs/vae_1d_z4_step=300000.ckpt \
  --raw_data_dir /data1/yuankai/text2Motion/FloodDiffusion/raw_data \
  --suites step,real,turn
```

---

## Done criteria

```text
1. 一条命令可以跑 step / real / turn / babel 四类 suite。
2. 可以通过 --suites 选择子集。
3. 默认 smoke preset 至少跑：
   - step_metric_001168
   - real_metric_001168
   - turn_metric_001168_rot30
   - babel_metric_9797
4. 所有 mode 输出统一字段：
   - ADE
   - FDE
   - path_arc
   - path_chamfer
   - chamfer_type
   - lateral_velocity_ratio
   - heading_path_error_deg
   - target_source
5. real / turn / babel 的主指标对 sampled / updated plan target 计算。
6. real_metric 的 plan 长度由 dataset sample 的 target_frames 决定，不误用 web 默认 5s。
7. original GT 只作为 ADE_vs_original_gt / FDE_vs_original_gt。
8. path_arc 和 path_chamfer 定义固定且实现一致。
9. 输出 summary.json 和 summary.csv。
10. render_video 时，每个 case 的视频 overlay 目标轨迹。
11. 不输出 closed_loop_gap_ADE / traj_gain_ADE 等派生字段。
12. lateral_velocity_ratio 能解释“侧着走”现象。
13. heading_path_error_deg 能解释“轨迹跟上但身体没转向”现象。
14. 不改模型、不改训练、不实现 delayed blend。
```

---

## Review update

待实现。
