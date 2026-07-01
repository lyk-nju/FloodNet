#!/usr/bin/env bash
set -euo pipefail

# Evaluate every step_*.ckpt in CKPT_DIR with eval.ldf.stream_metrics.
#
# Typical usage:
#   CKPT_DIR=/path/to/outputs/20260617_020237_ldf \
#   VAE_CKPT=/path/to/vae_1d_z4_step=300000.ckpt \
#   META_PATHS="/path/to/HumanML3D/test_difficult.txt" \
#   DEVICES=0,1,2,3,4,5,6,7 \
#   NUM_RUNS=1 \
#   scripts/run_all_stream_eval_ckpts.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BASE_PATH="${BASE_PATH:-$(cd "$SCRIPT_DIR/.." && pwd -P)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-$BASE_PATH/configs/ldf_test.yaml}"
CKPT_DIR="${CKPT_DIR:-$BASE_PATH/outputs}"
VAE_CKPT="${VAE_CKPT:-$BASE_PATH/outputs/vae_1d_z4_step=300000.ckpt}"
OUT_DIR="${OUT_DIR:-$BASE_PATH/eval/result}"

DEVICES="${DEVICES:-0}"
STREAM_METRICS_DEVICES="${STREAM_METRICS_DEVICES:-}"
STREAM_MODE="${STREAM_MODE:-stream_generate_step}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_RUNS="${NUM_RUNS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_BATCHES="${MAX_BATCHES:-0}"
NUM_WORKERS="${NUM_WORKERS:-}"
NUM_DENOISE_STEPS="${NUM_DENOISE_STEPS:-}"
META_PATHS="${META_PATHS:-}"
PROBE_TAG="${PROBE_TAG:-}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"
SKIP_DONE="${SKIP_DONE:-false}"
COMPUTE_OFFLINE_BASELINE="${COMPUTE_OFFLINE_BASELINE:-false}"
COMPUTE_NO_TRAJ_BASELINE="${COMPUTE_NO_TRAJ_BASELINE:-false}"

cd "$BASE_PATH"
# Keep Floodmain first and isolated. stream_metrics has a compatibility import
# path that tries "FloodNet.*" before local modules; inheriting a parent
# PYTHONPATH that contains a sibling FloodNet checkout can silently run the
# wrong implementation.
export PYTHONPATH="$BASE_PATH${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/floodnet_mpl}"
mkdir -p "$MPLCONFIGDIR" "$OUT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python env not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -s "$CONFIG" ]]; then
    echo "Config file is missing or empty: $CONFIG" >&2
    exit 1
fi

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Checkpoint directory not found: $CKPT_DIR" >&2
    exit 1
fi

if [[ ! -f "$VAE_CKPT" ]]; then
    echo "VAE checkpoint not found: $VAE_CKPT" >&2
    exit 1
fi

if [[ -z "$STREAM_METRICS_DEVICES" ]]; then
    if [[ "$DEVICES" == *,* ]]; then
        IFS=',' read -r -a _device_parts <<< "$DEVICES"
        STREAM_METRICS_DEVICES="${#_device_parts[@]}"
    else
        STREAM_METRICS_DEVICES="1"
    fi
fi

declare -a CKPTS=()
if [[ -n "${STEPS:-}" ]]; then
    read -r -a _steps <<< "$STEPS"
    for step in "${_steps[@]}"; do
        CKPTS+=("$CKPT_DIR/step_${step}.ckpt")
    done
else
    while IFS= read -r ckpt; do
        CKPTS+=("$ckpt")
    done < <(find "$CKPT_DIR" -maxdepth 1 -type f -name 'step_*.ckpt' | sort -V)
fi

if [[ "${#CKPTS[@]}" -eq 0 ]]; then
    echo "No checkpoints found in $CKPT_DIR matching step_*.ckpt" >&2
    exit 1
fi

declare -a COMMON_ARGS=(
    --config "$CONFIG"
    --vae_ckpt "$VAE_CKPT"
    --stream_mode "$STREAM_MODE"
    --batch_size "$BATCH_SIZE"
    --num_runs "$NUM_RUNS"
    --max_samples "$MAX_SAMPLES"
    --max_batches "$MAX_BATCHES"
    --out_dir "$OUT_DIR"
    --devices "$STREAM_METRICS_DEVICES"
)

if [[ -n "$NUM_WORKERS" ]]; then
    COMMON_ARGS+=(--num_workers "$NUM_WORKERS")
fi

if [[ -n "$NUM_DENOISE_STEPS" ]]; then
    COMMON_ARGS+=(--num_denoise_steps "$NUM_DENOISE_STEPS")
fi

if [[ -n "$META_PATHS" ]]; then
    read -r -a _meta_paths <<< "$META_PATHS"
    COMMON_ARGS+=(--meta_paths "${_meta_paths[@]}")
fi

if [[ -n "$PROBE_TAG" ]]; then
    COMMON_ARGS+=(--probe_tag "$PROBE_TAG")
fi

if [[ "$COMPUTE_OFFLINE_BASELINE" == "true" ]]; then
    COMMON_ARGS+=(--compute_offline_baseline)
else
    COMMON_ARGS+=(--no_compute_offline_baseline)
fi

if [[ "$COMPUTE_NO_TRAJ_BASELINE" == "true" ]]; then
    COMMON_ARGS+=(--compute_no_traj_baseline)
else
    COMMON_ARGS+=(--no_compute_no_traj_baseline)
fi

for CKPT in "${CKPTS[@]}"; do
    if [[ ! -f "$CKPT" ]]; then
        echo "Checkpoint not found: $CKPT" >&2
        exit 1
    fi

    ckpt_stem="$(basename "$CKPT" .ckpt)"
    run_name="${RUN_NAME_PREFIX}${ckpt_stem}"
    if [[ "$SKIP_DONE" == "true" && -f "$OUT_DIR/$run_name/summary.json" ]]; then
        echo "Skipping completed stream evaluation: $run_name"
        continue
    fi

    echo "Starting stream evaluation: ckpt=$CKPT run_name=$run_name devices=$DEVICES workers=$STREAM_METRICS_DEVICES"
    CUDA_VISIBLE_DEVICES="$DEVICES" "$PYTHON_BIN" -m eval.ldf.stream_metrics \
        "${COMMON_ARGS[@]}" \
        --ckpt "$CKPT" \
        --run_name "$run_name" \
        "$@"
done

echo "Finished stream evaluation for ${#CKPTS[@]} checkpoint(s)."
