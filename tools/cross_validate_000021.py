"""
交叉验证：手动计算 000021 的 control loss，
对比 eval_control_loss.py 的结果。

方法A：用已存储的 feature (173,263) 提取 XZ → 对比 traj_xz
方法B：重新 decode token (44,4) → motion   → 对比 traj_xz
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch_ema import ExponentialMovingAverage

from utils.initialize import Config, instantiate
from utils.motion_process import extract_root_trajectory_263_torch

BASE = "outputs/20260402_114343_ldf/HumanML3D"
NAME = "000021"
CKPT_300K = "outputs/20260402_114343_ldf/step_step=300000.ckpt"

# ── 加载数据 ──────────────────────────────────────────────────────────────────
traj_xz = torch.from_numpy(np.load(f"{BASE}/traj_xz/{NAME}.npy")).float()   # (173,2) xz
token     = torch.from_numpy(np.load(f"{BASE}/token/{NAME}.npy")).float()        # (44,4)
traj_mask = torch.from_numpy(np.load(f"{BASE}/traj_mask/{NAME}.npy")).float()    # (173,)
feature   = torch.from_numpy(np.load(f"{BASE}/feature/{NAME}.npy")).float()      # (173,263)

print("=" * 55)
print("数据形状确认")
print("=" * 55)
print(f"  token     : {token.shape}  → causal decode = {4*(token.shape[0]-1)+1} frames")
print(f"  traj_xz : {traj_xz.shape}")
print(f"  traj_mask : sum={int(traj_mask.sum())}/{len(traj_mask)}")
print(f"  feature   : {feature.shape}")
print()

# ── 加载 VAE ──────────────────────────────────────────────────────────────────
cfg = Config("configs/ldf.yaml").config
vae = instantiate(target=cfg.test_vae.target, cfg=None, hfstyle=False,
                  **cfg.test_vae.params)
vae_ckpt = torch.load(cfg.test_vae_ckpt, map_location="cpu", weights_only=False)
vae.load_state_dict(vae_ckpt["state_dict"], strict=True)
if "ema_state" in vae_ckpt:
    vae_ema = ExponentialMovingAverage(vae.parameters(), decay=cfg.test_vae.ema_decay)
    vae_ema.load_state_dict(vae_ckpt["ema_state"])
    vae_ema.copy_to(vae.parameters())
vae.eval()
print("[VAE loaded]")
print()

# ── 方法A：直接用存储的 feature ───────────────────────────────────────────────
with torch.no_grad():
    traj_A = extract_root_trajectory_263_torch(feature.unsqueeze(0))[0]  # (173,3)
xz_A  = traj_A[:, [0, 2]]  # (173,2)
T_A   = min(len(xz_A), len(traj_xz))
err_A = ((xz_A[:T_A] - traj_xz[:T_A]) ** 2).sum(dim=-1)
mask  = traj_mask[:T_A]
loss_A = (mask * err_A).sum() / mask.sum()

print("方法A — stored feature → extract XZ → MSE vs traj_xz")
print(f"  frames={T_A}  valid_points={int(mask.sum())}  control_loss_xz = {loss_A.item():.8f}")
print()

# ── 方法B：重新 decode token ──────────────────────────────────────────────────
with torch.no_grad():
    decoded_B = vae.decode(token.unsqueeze(0))[0].float()  # (T_frame, 263)
traj_B = extract_root_trajectory_263_torch(decoded_B.unsqueeze(0))[0]
xz_B   = traj_B[:, [0, 2]]
T_B    = min(len(xz_B), len(traj_xz))
err_B  = ((xz_B[:T_B] - traj_xz[:T_B]) ** 2).sum(dim=-1)
mask_B = traj_mask[:T_B]
loss_B = (mask_B * err_B).sum() / mask_B.sum()

print("方法B — decode token → extract XZ → MSE vs traj_xz")
print(f"  decoded_shape={decoded_B.shape}  frames={T_B}  valid_points={int(mask_B.sum())}")
print(f"  control_loss_xz = {loss_B.item():.8f}")
print()

# ── A vs B 一致性 ──────────────────────────────────────────────────────────────
diff = (xz_A[:T_B] - xz_B[:T_B]).abs().max().item()
print(f"方法A vs 方法B — XZ 最大绝对差: {diff:.6f}")
if diff < 1e-3:
    print("→ feature 与 decode(token) 基本一致 (stored feature 即生成结果)")
else:
    print("→ feature 与 decode(token) 存在差异，需确认 feature 来源")
print()

# ── eval_control_loss.py (forward mode, step=245k ckpt) 参考值 ───────────────
print("=" * 55)
print("对照参考（eval_control_loss.py 报告）")
print("=" * 55)
print("  ckpt  : step=245000  (20260415_114956_ldf)")
print("  mode  : forward (pred_x0_latent_list 窗口内)")
print("  000021 control_loss_xz = 0.06670745  valid_points=17")
print()
print("注意：eval_control_loss.py forward mode 只比较")
print("  active window（最后 chunk_size=5 个 token 对应的帧），")
print("  而方法A/B 是全序列比较，因此数值不直接可比。")
print("  关键对比点：方法A 与方法B 的结果是否一致。")
