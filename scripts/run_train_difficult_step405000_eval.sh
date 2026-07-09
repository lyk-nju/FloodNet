#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="/data/home/shengqiuProf_user_yuankai/Floodcontrol"
PYTHON_BIN="/data/home/shengqiuProf_user_yuankai/miniconda3/envs/flooddiffusion/bin/python"
CONFIG="$BASE_PATH/configs/ldf_test.yaml"
CKPT="$BASE_PATH/outputs/20260703_200127_ldf/step_430000.ckpt"
VAE_CKPT="/data/home/shengqiuProf_user_yuankai/FloodDiffusion/outputs/vae_1d_z4_step=300000.ckpt"
META_PATH="/data/home/shengqiuProf_user_yuankai/FloodDiffusion/raw_data/HumanML3D/train_difficult.txt"

CKPT_TAG="$(basename "$CKPT" .ckpt)"
OUT_DIR="$BASE_PATH/eval/output_eval/ldf/train_difficult_${CKPT_TAG}_all"

DEVICES="0,1,2,3,4,5,6,7"
NUM_DEVICES=8
NUM_WORKERS=4
NUM_RUNS=1
ADE_THRESHOLDS=("0.3" "0.5")
RUN_NAME="${CKPT_TAG}_train_difficult_stream_step_h30_hor20_cfg_t1p25_r3p00_runs${NUM_RUNS}"

# Set to 0 if you want to keep generated npy/png artifacts for every sample.
CLEAN_LARGE_ARTIFACTS=1

# Set to 0 to force re-running eval even when summary.json already exists.
SKIP_EVAL_IF_SUMMARY_EXISTS=1

cd "$BASE_PATH"
export PYTHONPATH="$BASE_PATH:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/floodnet_mpl}"
mkdir -p "$MPLCONFIGDIR" "$OUT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python env not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -s "$CONFIG" ]]; then
    echo "Config file is missing or empty: $CONFIG" >&2
    exit 1
fi
if [[ ! -f "$CKPT" ]]; then
    echo "Checkpoint not found: $CKPT" >&2
    exit 1
fi
if [[ ! -f "$VAE_CKPT" ]]; then
    echo "VAE checkpoint not found: $VAE_CKPT" >&2
    exit 1
fi
if [[ ! -f "$META_PATH" ]]; then
    echo "Meta file not found: $META_PATH" >&2
    exit 1
fi

echo "Starting train_difficult stream_generate_step eval"
echo "  ckpt:       $CKPT"
echo "  meta:       $META_PATH"
echo "  devices:    $DEVICES"
echo "  num_runs:   $NUM_RUNS"
echo "  out_dir:    $OUT_DIR/$RUN_NAME"

SUMMARY="$OUT_DIR/$RUN_NAME/summary.json"

if [[ "$SKIP_EVAL_IF_SUMMARY_EXISTS" == "1" && -f "$SUMMARY" ]]; then
    echo "Found existing summary, skipping eval and regenerating reports:"
    echo "  $SUMMARY"
    ELAPSED_SEC=0
else
    START_TS=$(date +%s)

    CUDA_VISIBLE_DEVICES="$DEVICES" "$PYTHON_BIN" -m eval.ldf.stream_metrics \
        --config "$CONFIG" \
        --ckpt "$CKPT" \
        --vae_ckpt "$VAE_CKPT" \
        --stream_mode stream_generate_step \
        --batch_size 1 \
        --num_workers "$NUM_WORKERS" \
        --num_runs "$NUM_RUNS" \
        --max_samples 0 \
        --max_batches 0 \
        --seed 1234 \
        --meta_paths "$META_PATH" \
        --probe_tag train_difficult \
        --no_compute_offline_baseline \
        --no_compute_no_traj_baseline \
        --no_save_plots \
        --out_dir "$OUT_DIR" \
        --run_name "$RUN_NAME" \
        --devices "$NUM_DEVICES" \
        --set \
            eval.history_length=30 \
            eval.traj_horizon_tokens=20 \
            eval.token_dt=0.20 \
            eval.frames_per_token=4 \
            eval.save_feature_npy=false \
            eval.save_latent_npy=false \
            eval.save_plots=false \
            eval.render_video=false \
            model.params.cfg_scale_text=1.25 \
            model.params.cfg_scale_traj=3.0

    END_TS=$(date +%s)
    ELAPSED_SEC=$((END_TS - START_TS))
fi

if [[ ! -f "$SUMMARY" ]]; then
    echo "summary.json not found after eval: $SUMMARY" >&2
    exit 1
fi

for ADE_THRESHOLD in "${ADE_THRESHOLDS[@]}"; do
REPORT_DIR="$OUT_DIR/$RUN_NAME/ade_gt_${ADE_THRESHOLD/./p}"

"$PYTHON_BIN" - "$SUMMARY" "$REPORT_DIR" "$ADE_THRESHOLD" "$ELAPSED_SEC" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
report_dir = Path(sys.argv[2])
threshold = float(sys.argv[3])
elapsed_sec = int(sys.argv[4])

payload = json.loads(summary_path.read_text())
samples = payload.get("samples", [])

def first_finite(record, *keys):
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None

rows = []
for sample in samples:
    ade = first_finite(sample, "ade", "traj/ADE_mean", "stream_gt/root_ADE")
    if ade is None:
        continue
    if ade <= threshold:
        continue
    fde = first_finite(sample, "fde", "traj/FDE_mean", "stream_gt/root_FDE")
    rows.append(
        {
            "sample_index": sample.get("_sample_index"),
            "name": sample.get("name"),
            "dataset": sample.get("dataset"),
            "ade_mean": ade,
            "ade_std": first_finite(sample, "ade_std", "traj/ADE_std"),
            "fde_mean": fde,
            "fde_std": first_finite(sample, "fde_std", "traj/FDE_std"),
            "stream_yaw_error": sample.get("stream_yaw_error"),
            "stream_root_jump_mean": sample.get("stream_root_jump_mean"),
            "num_runs": sample.get("num_runs"),
        }
    )

rows.sort(key=lambda item: item["ade_mean"], reverse=True)
report_dir.mkdir(parents=True, exist_ok=True)

threshold_tag = str(threshold).replace(".", "p")
csv_path = report_dir / f"ade_gt_{threshold_tag}.csv"
json_path = report_dir / f"ade_gt_{threshold_tag}.json"
txt_path = report_dir / "report.txt"

fieldnames = [
    "sample_index",
    "name",
    "dataset",
    "ade_mean",
    "ade_std",
    "fde_mean",
    "fde_std",
    "stream_yaw_error",
    "stream_root_jump_mean",
    "num_runs",
]
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

json_path.write_text(json.dumps(rows, indent=2))

summary = payload.get("summary", {})
lines = [
    f"summary: {summary_path}",
    f"num_samples: {payload.get('num_samples')}",
    f"num_runs: {payload.get('num_runs')}",
    f"elapsed_sec: {elapsed_sec}",
    f"overall_ADE_mean: {summary.get('traj/ADE_mean')}",
    f"overall_FDE_mean: {summary.get('traj/FDE_mean')}",
    f"ADE_threshold: {threshold}",
    f"num_ADE_gt_threshold: {len(rows)}",
    "",
    "Top ADE samples:",
]
for row in rows[:50]:
    lines.append(
        f"{row['name']}\tidx={row['sample_index']}\t"
        f"ADE={row['ade_mean']:.6f}\tFDE={row['fde_mean']}"
    )
txt_path.write_text("\n".join(lines) + "\n")

print(f"Wrote {csv_path}")
print(f"Wrote {json_path}")
print(f"Wrote {txt_path}")
print(f"ADE > {threshold}: {len(rows)} / {payload.get('num_samples')}")
PY
done

if [[ "$CLEAN_LARGE_ARTIFACTS" == "1" ]]; then
    echo "Cleaning large per-sample feature/plot artifacts; keeping summary and metrics."
    rm -rf "$OUT_DIR/$RUN_NAME/samples"
    rm -rf "$OUT_DIR/HumanML3D/feature/train_difficult/$CKPT_TAG"
    rm -rf "$OUT_DIR/HumanML3D/token/train_difficult/$CKPT_TAG"
    rm -rf "$OUT_DIR/HumanML3D/traj_xz/train_difficult/$CKPT_TAG"
    rm -rf "$OUT_DIR/HumanML3D/traj_mask/train_difficult/$CKPT_TAG"
    rm -rf "$OUT_DIR/HumanML3D/frames/train_difficult/$CKPT_TAG"
    rm -rf "$OUT_DIR/HumanML3D/condition_compare/train_difficult/$CKPT_TAG"
    rm -rf "$OUT_DIR/HumanML3D/video/train_difficult/$CKPT_TAG"
    rm -rf "$OUT_DIR/HumanML3D/composite/train_difficult/$CKPT_TAG"
fi

echo "Done."
echo "Summary: $SUMMARY"
echo "Reports:"
for ADE_THRESHOLD in "${ADE_THRESHOLDS[@]}"; do
    echo "  $OUT_DIR/$RUN_NAME/ade_gt_${ADE_THRESHOLD/./p}"
done
