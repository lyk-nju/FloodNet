#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="/data/home/shengqiuProf_user_yuankai/Floodcontrol"
PYTHON_BIN="/data/home/shengqiuProf_user_yuankai/miniconda3/envs/flooddiffusion/bin/python"
CONFIG="$BASE_PATH/configs/ldf_test.yaml"
CKPT="$BASE_PATH/outputs/20260703_200127_ldf/step_430000.ckpt"
VAE_CKPT="/data/home/shengqiuProf_user_yuankai/FloodDiffusion/outputs/vae_1d_z4_step=300000.ckpt"
META_PATH="/data/home/shengqiuProf_user_yuankai/FloodDiffusion/raw_data/HumanML3D/train_difficult.txt"

CKPT_TAG="$(basename "$CKPT" .ckpt)"
RUN_NAME="${CKPT_TAG}_train_difficult_stream_step_h30_hor20_cfg_t1p25_r3p00_runs1"
OUT_DIR="$BASE_PATH/eval/output_eval/ldf/train_difficult_${CKPT_TAG}_all"

SUMMARY="$OUT_DIR/$RUN_NAME/summary.json"
METRICS_DIR="$OUT_DIR/HumanML3D/metrics/train_difficult/$CKPT_TAG"
REPORT_BASE="$OUT_DIR/$RUN_NAME"
ADE_THRESHOLDS=("0.3" "0.5")

RUN_GENERATION=1
FORCE_RERUN=0
DEVICES="0,1,2,3,4,5,6,7"
NUM_DEVICES=8
NUM_WORKERS=4
NUM_RUNS=1

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python env not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [[ "$RUN_GENERATION" == "1" ]]; then
    if [[ -f "$SUMMARY" && "$FORCE_RERUN" != "1" ]]; then
        echo "summary.json already exists; skip generation because FORCE_RERUN=0:"
        echo "  $SUMMARY"
    else
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

        cd "$BASE_PATH"
        export PYTHONPATH="$BASE_PATH:${PYTHONPATH:-}"
        export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/floodnet_mpl}"
        mkdir -p "$MPLCONFIGDIR" "$OUT_DIR"

        echo "Starting train_difficult stream_generate_step eval"
        echo "  ckpt:       $CKPT"
        echo "  meta:       $META_PATH"
        echo "  devices:    $DEVICES"
        echo "  num_runs:   $NUM_RUNS"
        echo "  out_dir:    $OUT_DIR/$RUN_NAME"

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
    fi
fi

if [[ ! -f "$SUMMARY" && ! -d "$METRICS_DIR" ]]; then
    echo "Neither summary nor metrics dir exists:" >&2
    echo "  summary: $SUMMARY" >&2
    echo "  metrics: $METRICS_DIR" >&2
    exit 1
fi

"$PYTHON_BIN" - "$SUMMARY" "$METRICS_DIR" "$REPORT_BASE" "${ADE_THRESHOLDS[@]}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
metrics_dir = Path(sys.argv[2])
report_base = Path(sys.argv[3])
thresholds = [float(value) for value in sys.argv[4:]]


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


def load_payload_and_samples():
    if summary_path.exists():
        payload = json.loads(summary_path.read_text())
        samples = payload.get("samples") or []
        if samples:
            return payload, samples, f"summary:{summary_path}"

    samples = []
    if metrics_dir.exists():
        for path in sorted(metrics_dir.glob("*.json")):
            if path.name == "summary.json":
                continue
            sample = json.loads(path.read_text())
            samples.append(sample)
    payload = {
        "summary": {},
        "num_samples": len(samples),
        "num_runs": samples[0].get("num_runs") if samples else None,
    }
    return payload, samples, f"metrics_dir:{metrics_dir}"


payload, samples, source = load_payload_and_samples()
summary = payload.get("summary") or {}

all_ades = [
    first_finite(sample, "ade", "traj/ADE_mean", "stream_gt/root_ADE")
    for sample in samples
]
all_ades = [value for value in all_ades if value is not None]
all_fdes = [
    first_finite(sample, "fde", "traj/FDE_mean", "stream_gt/root_FDE")
    for sample in samples
]
all_fdes = [value for value in all_fdes if value is not None]

overall_ade = summary.get("traj/ADE_mean")
if overall_ade is None and all_ades:
    overall_ade = sum(all_ades) / len(all_ades)
overall_fde = summary.get("traj/FDE_mean")
if overall_fde is None and all_fdes:
    overall_fde = sum(all_fdes) / len(all_fdes)

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

for threshold in thresholds:
    rows = []
    for sample in samples:
        ade = first_finite(sample, "ade", "traj/ADE_mean", "stream_gt/root_ADE")
        if ade is None or ade <= threshold:
            continue
        rows.append(
            {
                "sample_index": sample.get("_sample_index"),
                "name": sample.get("name"),
                "dataset": sample.get("dataset"),
                "ade_mean": ade,
                "ade_std": first_finite(sample, "ade_std", "traj/ADE_std"),
                "fde_mean": first_finite(sample, "fde", "traj/FDE_mean", "stream_gt/root_FDE"),
                "fde_std": first_finite(sample, "fde_std", "traj/FDE_std"),
                "stream_yaw_error": sample.get("stream_yaw_error"),
                "stream_root_jump_mean": sample.get("stream_root_jump_mean"),
                "num_runs": sample.get("num_runs"),
            }
        )

    rows.sort(key=lambda item: item["ade_mean"], reverse=True)
    threshold_tag = str(threshold).replace(".", "p")
    report_dir = report_base / f"ade_gt_{threshold_tag}"
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / f"ade_gt_{threshold_tag}.csv"
    json_path = report_dir / f"ade_gt_{threshold_tag}.json"
    txt_path = report_dir / "report.txt"
    train_txt_path = report_base / f"train_hard_ade_gt{threshold_tag}.txt"
    train_txt_copy_path = report_dir / f"train_hard_ade_gt{threshold_tag}.txt"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2))
    train_lines = [str(row["name"]) for row in rows if row.get("name")]
    train_txt_path.write_text("\n".join(train_lines) + ("\n" if train_lines else ""))
    train_txt_copy_path.write_text(
        "\n".join(train_lines) + ("\n" if train_lines else "")
    )

    lines = [
        f"source: {source}",
        f"summary: {summary_path}",
        f"metrics_dir: {metrics_dir}",
        f"num_samples: {len(samples)}",
        f"num_runs: {payload.get('num_runs')}",
        f"overall_ADE_mean: {overall_ade}",
        f"overall_FDE_mean: {overall_fde}",
        f"ADE_threshold: {threshold}",
        f"num_ADE_gt_threshold: {len(rows)}",
        f"train_txt: {train_txt_path}",
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
    print(f"Wrote {train_txt_path}")
    print(f"Wrote {train_txt_copy_path}")
    print(f"Wrote {txt_path}")
    print(f"ADE > {threshold}: {len(rows)} / {len(samples)}")
PY

echo "Done."
echo "Reports:"
for ADE_THRESHOLD in "${ADE_THRESHOLDS[@]}"; do
    echo "  $REPORT_BASE/ade_gt_${ADE_THRESHOLD/./p}"
done
echo "Train txt:"
for ADE_THRESHOLD in "${ADE_THRESHOLDS[@]}"; do
    echo "  $REPORT_BASE/train_hard_ade_gt${ADE_THRESHOLD/./p}.txt"
done
