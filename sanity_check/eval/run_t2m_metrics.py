"""
Parallel T2M evaluation across multiple GPUs.

Each GPU handles a disjoint shard of the val set, saves per-shard embeddings,
then a merge step loads all shards and computes the final FID / R-Precision / Diversity.

Usage:
    cd /home/yuankai/Text2Motion/FloodNet
    conda run -n flooddiffusion python eval/run_t2m_metrics.py \\
        --config configs/ldf.yaml \\
        --ckpt outputs/20260506_012427_ldf/last.ckpt \\
        --gpu_ids 0 1 2 3 4 \\
        --val_bs 16
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   required=True,       help="Path to ldf.yaml")
    p.add_argument("--ckpt",     required=True,       help="Checkpoint path")
    p.add_argument("--gpu_ids",  nargs="+", type=int, required=True,
                   help="GPU ids to use, e.g. --gpu_ids 0 1 2 3 4")
    p.add_argument("--val_bs",   type=int, default=16, help="Val batch size per GPU")
    p.add_argument("--shards_dir", type=str, default=None,
                   help="Directory to store shard .npz files (default: auto under eval/)")
    p.add_argument("--seed",     type=int, default=1234)
    p.add_argument("--no_ema",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    num_shards = len(args.gpu_ids)

    shards_dir = Path(args.shards_dir) if args.shards_dir else (
        Path(__file__).parent / "t2m_parallel_shards"
    )
    shards_dir.mkdir(parents=True, exist_ok=True)
    print(f"[launcher] Shard dir : {shards_dir}")
    print(f"[launcher] GPUs      : {args.gpu_ids}")
    print(f"[launcher] Num shards: {num_shards}")

    script = str(Path(__file__).parent / "eval_generation_metrics.py")
    conda_python = sys.executable

    procs = []
    for i, gpu_id in enumerate(args.gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            conda_python, script,
            "--config",            args.config,
            "--ckpt",              args.ckpt,
            "--t2m_metric",
            "--skip_test_pass",
            "--val_num_shards",    str(num_shards),
            "--val_shard_idx",     str(i),
            "--t2m_shards_save_dir", str(shards_dir),
            "--set", f"data.val_bs={args.val_bs}",
            "--set", "test_setting.render=false",
            "--seed", str(args.seed),
        ]
        if args.no_ema:
            cmd.append("--no_ema")

        log_path = shards_dir / f"shard_{i}_gpu{gpu_id}.log"
        print(f"[launcher] Shard {i}/{num_shards} on GPU {gpu_id} → {log_path}")
        log_f = open(log_path, "w")
        proc  = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((i, gpu_id, proc, log_path, log_f))

    print(f"[launcher] All {num_shards} shard workers launched. Waiting...")

    failed = []
    for i, gpu_id, proc, log_path, log_f in procs:
        ret = proc.wait()
        log_f.close()
        if ret != 0:
            print(f"[launcher] FAILED: shard {i} GPU {gpu_id} (exit {ret}) → see {log_path}")
            failed.append(i)
        else:
            print(f"[launcher] Done  : shard {i} GPU {gpu_id}")

    if failed:
        print(f"[launcher] {len(failed)} shard(s) failed: {failed}. Aborting merge.")
        sys.exit(1)

    print("\n[launcher] All shards complete. Running merge + FID computation...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids[0])
    merge_cmd = [
        conda_python, script,
        "--config",         args.config,
        "--t2m_merge_dir",  str(shards_dir),
        "--seed",           str(args.seed),
    ]
    ret = subprocess.call(merge_cmd, env=env)
    if ret != 0:
        print("[launcher] Merge step failed!")
        sys.exit(1)

    results_path = shards_dir / "t2m_results.json"
    print(f"\n[launcher] Done! Results → {results_path}")
    if results_path.exists():
        results = json.loads(results_path.read_text())
        print("\n=== Final T2M Metrics ===")
        for k, v in results.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
