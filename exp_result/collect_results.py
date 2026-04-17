#!/usr/bin/env python3
"""
从 exp_result/results/*.log 中提取 control loss 数字，输出汇总表。
用法：
    cd /home/yuankai/Text2Motion/FloodNet
    conda run -n flooddiffusion python exp_result/collect_results.py
"""
import re
import pathlib

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# 精确匹配 eval 脚本输出的 summary 行：
#   "control_loss_xz: 0.12345678"   （行首，无缩进）
SUMMARY_RE = re.compile(r"^control_loss_xz:\s*([\d.]+)", re.MULTILINE)

# 次选：含 "overall ... control_loss" 的行
MEAN_RE = re.compile(
    r"(?:Mean|mean|overall)[^\d]*control[^\d]*loss[^\d]*([\d.]+)", re.IGNORECASE
)

def extract_mean(log_path: pathlib.Path) -> str:
    text = log_path.read_text(errors="replace")
    # 优先：summary 行（取第一个，即 overall）
    m = SUMMARY_RE.search(text)
    if m:
        return m.group(1)
    # 次选
    m = MEAN_RE.search(text)
    if m:
        return m.group(1)
    return "N/A"

def main():
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return

    logs = sorted(RESULTS_DIR.glob("*.log"))
    if not logs:
        print("No .log files found in results/")
        return

    rows = []
    for log in logs:
        name = log.stem          # e.g. A_245k_new_generate
        mean = extract_mean(log)
        rows.append((name, mean))

    # Print formatted table
    print(f"\n{'Log name':<40}  {'Mean xz MSE'}")
    print("-" * 55)
    for name, mean in rows:
        print(f"{name:<40}  {mean}")

    # Print grouped by experiment
    print("\n===== Exp A: Checkpoint 对比 =====")
    print(f"{'Checkpoint':<12} {'forward':>12} {'generate':>12}")
    ckpts = ["240k", "245k_old", "245k_new", "300k"]
    result = {}
    for name, mean in rows:
        result[name] = mean
    for ck in ckpts:
        fwd = result.get(f"A_{ck}_forward", "—")
        gen = result.get(f"A_{ck}_generate", "—")
        print(f"{ck:<12} {fwd:>12} {gen:>12}")

    print("\n===== Exp B: Separated CFG (245k_new, generate) =====")
    print(f"{'cfg_scale_text':>14} {'cfg_scale_traj':>14} {'Mean xz MSE':>12}")
    cfg_combos = [("5.0","0.0"),("5.0","3.0"),("3.0","5.0"),("1.0","7.0"),("3.0","3.0"),("5.0","5.0")]
    for wt, wr in cfg_combos:
        tag = f"B_wt{wt}_wr{wr}_generate"
        mean = result.get(tag, "—")
        print(f"{wt:>14} {wr:>14} {mean:>12}")

    print("\n===== Exp C: Smooth Root (245k_new, generate) =====")
    print(f"{'smooth_traj_sigma':>18} {'Mean xz MSE':>12}")
    for sigma in ["0.0", "2.0"]:
        tag = f"C_sigma{sigma}_generate"
        mean = result.get(tag, "—")
        print(f"{sigma:>18} {mean:>12}")

if __name__ == "__main__":
    main()
