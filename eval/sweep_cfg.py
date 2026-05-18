"""
sweep_cfg.py — CFG grid-search evaluation for FloodNet.

Sweeps cfg_scale_text × cfg_scale_traj combinations by calling
eval_generation_metrics.py for each combination as a subprocess.
Collects metrics.json from each run and prints a comparison table.

Usage examples
--------------
# Quick smoke test (first 3 batches, no T2M):
python eval/sweep_cfg.py \\
    --config configs/ldf.yaml \\
    --ckpt /path/to/step_396000.ckpt \\
    --cfg_text_values 3.0 5.0 7.0 \\
    --cfg_traj_values 0.0 1.5 3.0 \\
    --max_batches 3 \\
    --num_runs 1

# Full evaluation with T2M metrics:
python eval/sweep_cfg.py \\
    --config configs/ldf.yaml \\
    --ckpt /path/to/step_396000.ckpt \\
    --cfg_text_values 3.0 5.0 7.0 \\
    --cfg_traj_values 0.0 1.5 3.0 5.0 \\
    --num_runs 3 \\
    --t2m_metric

# Just print the sweep plan without running anything:
python eval/sweep_cfg.py \\
    --config configs/ldf.yaml \\
    --ckpt /path/to/step_396000.ckpt \\
    --cfg_text_values 3.0 5.0 \\
    --cfg_traj_values 0.0 3.0 \\
    --dry_run
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EVAL_SCRIPT = os.path.join(PROJECT_ROOT, "eval", "eval_generation_metrics.py")

# Metrics to show in the comparison table.
# (display_name, json_key)
_TRAJ_METRICS = [
    ("ADE↓",       "traj/ADE_mean"),
    ("FDE↓",       "traj/FDE_mean"),
    ("MSE↓",       "traj/MSE_mean"),
    ("Jitter↓",    "traj/jitter_mean"),
    ("CtrlΔADE",   "traj/ctrl_delta_ADE_mean"),
]
_CTRL_METRICS = [
    ("L2dist↓",    "control/Control_L2_dist_mean"),
    ("Skate↓",     "control/Skating_Ratio_mean"),
    ("F20cm↓",     "control/traj_fail_20cm_mean"),
    ("F50cm↓",     "control/traj_fail_50cm_mean"),
]
_T2M_METRICS = [
    ("FID↓",       "FID"),
    ("R@1↑",       "R_precision_top_1"),
    ("R@2↑",       "R_precision_top_2"),
    ("R@3↑",       "R_precision_top_3"),
    ("MM_dist↓",   "Matching_score"),
    ("Diversity",  "Diversity"),
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep cfg_scale_text × cfg_scale_traj and compare evaluation metrics."
    )
    p.add_argument("--config", type=str, default="configs/ldf.yaml")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint path. Falls back to cfg.test_ckpt.")
    p.add_argument("--cfg_text_values", nargs="+", type=float, default=[5.0],
                   help="cfg_scale_text values to sweep (default: [5.0])")
    p.add_argument("--cfg_traj_values", nargs="+", type=float, default=[3.0],
                   help="cfg_scale_traj values to sweep (default: [3.0])")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Root directory for sweep outputs (default: eval/sweep_cfg_<timestamp>/).")
    p.add_argument("--num_runs", type=int, default=1,
                   help="Number of generation runs per sample per combination.")
    p.add_argument("--seg_size", type=int, default=20,
                   help="Frame window size for segment/prefix MSE (matches eval_seg_size, default 20).")
    p.add_argument("--forward_control_loss", action="store_true",
                   help="Run model() forward pass to compute active-window XZ control loss "
                        "(matches eval_forward_control_loss).")
    p.add_argument("--max_batches", type=int, default=0,
                   help="If >0, limit each eval to first N batches (for quick smoke tests).")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override batch size per eval run.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no_ema", action="store_true", help="Skip EMA weight application.")
    p.add_argument("--t2m_metric", action="store_true",
                   help="Also run T2M FID/R-Precision (slow).")
    p.add_argument("--traj_ablation", action="store_true",
                   help="Pass --traj_ablation to each eval run.")
    p.add_argument("--viz_traj", action="store_true",
                   help="Pass --viz_traj to each eval run (saves XZ PNG per sample).")
    p.add_argument("--meta_paths", nargs="+", default=None,
                   help="Test meta paths (.txt files) passed as --meta_paths to each eval run. "
                        "Required when cfg.data.test_meta_paths is absent (e.g. ldf.yaml uses "
                        "test_probe_meta_paths instead).")
    p.add_argument("--extra_set", nargs="*", metavar="KEY=VALUE", default=[],
                   help="Extra --set overrides to pass to every eval run.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing them.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip combinations whose metrics.json already exists.")
    p.add_argument("--gpu_ids", nargs="+", type=int, default=None,
                   help="GPU IDs to use in parallel (e.g. --gpu_ids 0 1 2 3 4 5 6 7). "
                        "Combinations are distributed round-robin; at most len(gpu_ids) "
                        "runs execute simultaneously. Omit to run sequentially on whatever "
                        "CUDA_VISIBLE_DEVICES is currently set to.")
    p.add_argument("--python", type=str, default=sys.executable,
                   help="Python interpreter to use (default: same as current).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_dir(out_root: Path, cfg_text: float, cfg_traj: float) -> Path:
    return out_root / f"text{cfg_text:.2f}_traj{cfg_traj:.2f}"


def _build_cmd(
    python: str,
    cfg_text: float,
    cfg_traj: float,
    args,
    run_out_dir: Path,
) -> List[str]:
    # --set takes nargs="*": merge cfg overrides and extra_set into one group
    set_overrides = [
        f"model.params.cfg_scale_text={cfg_text}",
        f"model.params.cfg_scale_traj={cfg_traj}",
    ] + list(args.extra_set or [])

    cmd = [
        python, EVAL_SCRIPT,
        "--config", args.config,
        "--seed", str(args.seed),
        "--num_runs", str(args.num_runs),
        "--out_dir", str(run_out_dir),
        "--probe_tag", f"cfg_t{cfg_text:.2f}_r{cfg_traj:.2f}",
        "--set", *set_overrides,
    ]
    if args.ckpt:
        cmd += ["--ckpt", args.ckpt]
    if args.batch_size is not None:
        cmd += ["--batch_size", str(args.batch_size)]
    if args.max_batches > 0:
        cmd += ["--max_batches", str(args.max_batches)]
    cmd += ["--seg_size", str(args.seg_size)]
    if args.forward_control_loss:
        cmd.append("--forward_control_loss")
    if args.no_ema:
        cmd.append("--no_ema")
    if args.t2m_metric:
        cmd.append("--t2m_metric")
    if args.traj_ablation:
        cmd.append("--traj_ablation")
    if args.viz_traj:
        cmd.append("--viz_traj")
    if args.meta_paths:
        cmd += ["--meta_paths"] + list(args.meta_paths)
    return cmd


def _find_metrics_json(run_out_dir: Path) -> Optional[Path]:
    """Find metrics.json anywhere under run_out_dir (one level deep)."""
    # eval_generation_metrics.py saves to: run_out_dir/eval_<exp>_<probe>_<meta>_seed<N>/metrics.json
    hits = list(run_out_dir.rglob("metrics.json"))
    if not hits:
        return None
    # Pick the one with the most recent mtime if there are multiple.
    return max(hits, key=lambda p: p.stat().st_mtime)


def _load_summary(metrics_path: Path) -> Dict:
    """Load and return the 'summary' dict from a metrics.json file."""
    try:
        data = json.loads(metrics_path.read_text())
        return data.get("summary", {})
    except Exception as e:
        print(f"[sweep] Warning: could not parse {metrics_path}: {e}")
        return {}


def _fmt(v, width=9) -> str:
    if v is None or (isinstance(v, float) and v != v):  # nan
        return "-".center(width)
    if isinstance(v, float):
        return f"{v:.4f}".rjust(width)
    return str(v).rjust(width)


def _print_table(
    combos: List[Tuple[float, float]],
    summaries: Dict[Tuple[float, float], Dict],
    metric_groups: List[List[Tuple[str, str]]],
    group_names: List[str],
    title: str = "",
):
    if title:
        print(f"\n{'=' * 80}")
        print(f" {title}")
        print(f"{'=' * 80}")

    col_w = 9
    label_w = 22

    for group_name, metrics in zip(group_names, metric_groups):
        if not metrics:
            continue
        print(f"\n  ── {group_name} ──")

        # Header: cfg_text/cfg_traj | metric1 metric2 ...
        col_title = "cfg_text/cfg_traj"
        header = f"{col_title:<{label_w}}"
        for name, _ in metrics:
            header += f" {name.center(col_w)}"
        print("  " + header)
        print("  " + "-" * len(header))

        # Group by cfg_text (rows) × cfg_traj (columns)
        cfg_text_vals = sorted(set(ct for ct, _ in combos))
        cfg_traj_vals = sorted(set(cr for _, cr in combos))

        for ct in cfg_text_vals:
            for cr in cfg_traj_vals:
                key = (ct, cr)
                row_label = f"text={ct:.2f} traj={cr:.2f}"
                summary = summaries.get(key, {})
                row = f"  {row_label:<{label_w}}"
                for _, jkey in metrics:
                    v = summary.get(jkey)
                    row += " " + _fmt(v, col_w)
                print(row)

    print()


def _save_sweep_summary(
    out_root: Path,
    combos: List[Tuple[float, float]],
    summaries: Dict[Tuple[float, float], Dict],
    args,
):
    """Save sweep summary as JSON."""
    result = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "ckpt": args.ckpt,
        "seed": args.seed,
        "num_runs": args.num_runs,
        "max_batches": args.max_batches,
        "cfg_text_values": args.cfg_text_values,
        "cfg_traj_values": args.cfg_traj_values,
        "runs": {},
    }
    for ct, cr in combos:
        key = f"text={ct:.2f}_traj={cr:.2f}"
        result["runs"][key] = summaries.get((ct, cr), {})

    path = out_root / "sweep_summary.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"\n[sweep] Summary saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Single-combo worker (called sequentially or from ThreadPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_combo(
    combo_idx: int,
    total: int,
    gpu_id: Optional[int],
    cfg_text: float,
    cfg_traj: float,
    args,
    out_root: Path,
) -> tuple:
    """Run one combination. Returns ((cfg_text, cfg_traj), summary_dict_or_None)."""
    run_out_dir = _run_dir(out_root, cfg_text, cfg_traj)
    run_out_dir.mkdir(parents=True, exist_ok=True)
    gpu_tag = f"GPU{gpu_id}" if gpu_id is not None else "GPU?"
    prefix = f"[sweep] ({combo_idx}/{total}) {gpu_tag} text={cfg_text:.2f} traj={cfg_traj:.2f}"

    if args.skip_existing:
        existing = _find_metrics_json(run_out_dir)
        if existing is not None:
            print(f"{prefix} — SKIP (found existing metrics.json)", flush=True)
            return (cfg_text, cfg_traj), _load_summary(existing)

    cmd = _build_cmd(args.python, cfg_text, cfg_traj, args, run_out_dir)
    log_path = run_out_dir / "run.log"

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"{prefix} — START  log → {log_path}", flush=True)
    try:
        with open(log_path, "w") as log_f:
            ret = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
            )
        if ret.returncode != 0:
            print(f"{prefix} — FAILED (exit {ret.returncode})", flush=True)
            return (cfg_text, cfg_traj), None

        metrics_path = _find_metrics_json(run_out_dir)
        if metrics_path is None:
            print(f"{prefix} — FAILED (metrics.json not found)", flush=True)
            return (cfg_text, cfg_traj), None

        summary = _load_summary(metrics_path)
        ade = summary.get("traj/ADE_mean", float("nan"))
        fid = summary.get("FID", float("nan"))
        fid_str = f"  FID={fid:.3f}" if fid == fid else ""
        print(f"{prefix} — DONE  ADE={ade:.4f}{fid_str}", flush=True)
        return (cfg_text, cfg_traj), summary

    except Exception as e:
        print(f"{prefix} — EXCEPTION: {e}", flush=True)
        return (cfg_text, cfg_traj), None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_root = Path(args.out_dir)
    else:
        out_root = Path(PROJECT_ROOT) / "eval" / f"sweep_cfg_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    combos: List[Tuple[float, float]] = [
        (ct, cr)
        for ct in sorted(args.cfg_text_values)
        for cr in sorted(args.cfg_traj_values)
    ]

    gpu_ids = args.gpu_ids
    max_workers = len(gpu_ids) if gpu_ids else 1
    # Round-robin GPU assignment: combo i → gpu_ids[i % n_gpus]
    gpu_cycle = [gpu_ids[i % len(gpu_ids)] if gpu_ids else None for i in range(len(combos))]

    print(f"\n[sweep] Output root : {out_root}")
    print(f"[sweep] Config      : {args.config}")
    print(f"[sweep] Checkpoint  : {args.ckpt or '(from config)'}")
    print(f"[sweep] Combinations: {len(combos)}")
    print(f"[sweep] cfg_text    : {sorted(args.cfg_text_values)}")
    print(f"[sweep] cfg_traj    : {sorted(args.cfg_traj_values)}")
    print(f"[sweep] num_runs    : {args.num_runs}")
    if gpu_ids:
        print(f"[sweep] GPUs        : {gpu_ids}  (parallel workers={max_workers})")
    else:
        print(f"[sweep] GPUs        : (sequential, using ambient CUDA_VISIBLE_DEVICES)")
    if args.max_batches > 0:
        print(f"[sweep] max_batches : {args.max_batches}  (quick-test mode)")

    # ── Dry run: just print commands ─────────────────────────────────────────
    if args.dry_run:
        print("[sweep] *** DRY RUN — commands will be printed but not executed ***")
        for i, (cfg_text, cfg_traj) in enumerate(combos):
            run_out_dir = _run_dir(out_root, cfg_text, cfg_traj)
            cmd = _build_cmd(args.python, cfg_text, cfg_traj, args, run_out_dir)
            gpu_tag = f"GPU{gpu_cycle[i]}" if gpu_cycle[i] is not None else "GPU?"
            env_str = f"CUDA_VISIBLE_DEVICES={gpu_cycle[i]} " if gpu_cycle[i] is not None else ""
            print(f"\n[sweep] ({i+1}/{len(combos)}) {gpu_tag} "
                  f"text={cfg_text:.2f} traj={cfg_traj:.2f}")
            print(f"[sweep] cmd: {env_str}{' '.join(cmd)}")
        print("\n[sweep] Dry run complete. No commands were executed.")
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    summaries: Dict[Tuple[float, float], Dict] = {}
    failed: List[Tuple[float, float]] = []

    if max_workers > 1:
        print(f"\n[sweep] Launching {len(combos)} combinations across "
              f"{max_workers} parallel workers …\n")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_single_combo,
                    i + 1, len(combos), gpu_cycle[i], cfg_text, cfg_traj, args, out_root,
                ): (cfg_text, cfg_traj)
                for i, (cfg_text, cfg_traj) in enumerate(combos)
            }
            try:
                for future in as_completed(futures):
                    key, summary = future.result()
                    if summary is not None:
                        summaries[key] = summary
                    else:
                        failed.append(key)
            except KeyboardInterrupt:
                print("\n[sweep] Interrupted — waiting for in-flight jobs to finish …")
    else:
        for i, (cfg_text, cfg_traj) in enumerate(combos):
            try:
                key, summary = _run_single_combo(
                    i + 1, len(combos), gpu_cycle[i], cfg_text, cfg_traj, args, out_root
                )
                if summary is not None:
                    summaries[key] = summary
                else:
                    failed.append(key)
            except KeyboardInterrupt:
                print("\n[sweep] Interrupted by user.")
                break

    # Collect summaries for any combos that were skipped / succeeded
    for ct, cr in combos:
        if (ct, cr) not in summaries:
            run_out_dir = _run_dir(out_root, ct, cr)
            m = _find_metrics_json(run_out_dir)
            if m is not None:
                summaries[(ct, cr)] = _load_summary(m)

    # Print comparison table
    done_combos = [c for c in combos if c in summaries]
    if done_combos:
        metric_groups = [_TRAJ_METRICS, _CTRL_METRICS]
        group_names = ["Trajectory metrics", "Control metrics"]
        if any("FID" in s for s in summaries.values()):
            metric_groups.append(_T2M_METRICS)
            group_names.append("T2M metrics (FID / R-Precision)")

        _print_table(
            done_combos,
            summaries,
            metric_groups=metric_groups,
            group_names=group_names,
            title=f"CFG sweep results  ({len(done_combos)}/{len(combos)} combinations)",
        )
        _save_sweep_summary(out_root, done_combos, summaries, args)
    else:
        print("[sweep] No completed runs to summarise.")

    if failed:
        print(f"\n[sweep] Failed combinations ({len(failed)}):")
        for ct, cr in failed:
            print(f"  cfg_scale_text={ct:.2f}  cfg_scale_traj={cr:.2f}")


if __name__ == "__main__":
    main()
