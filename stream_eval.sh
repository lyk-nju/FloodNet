#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="/home/yuankai/Text2Motion/FloodNet"
PYTHON_BIN="/home/yuankai/.conda/envs/flooddiffusion/bin/python"
CONFIG="$BASE_PATH/configs/ldf_test.yaml"
CKPT_DIR="$BASE_PATH/outputs"
VAE_CKPT="$BASE_PATH/outputs/vae_1d_z4_step=300000.ckpt"
OUT_DIR="$BASE_PATH/eval/result"
DEVICES="0,1,2,3,4"
NUM_DEVICES=5
STEPS=("485000")

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

if [[ ! -f "$VAE_CKPT" ]]; then
    echo "VAE checkpoint not found: $VAE_CKPT" >&2
    exit 1
fi

for STEP in "${STEPS[@]}"
do
    CKPT="$CKPT_DIR/step_$STEP.ckpt"
    if [[ ! -f "$CKPT" ]]; then
        echo "Checkpoint not found: $CKPT" >&2
        exit 1
    fi

    echo "Starting stream evaluation for step: $STEP"

    CUDA_VISIBLE_DEVICES="$DEVICES" "$PYTHON_BIN" -m eval.ldf.stream_metrics \
        --config "$CONFIG" \
        --ckpt "$CKPT" \
        --vae_ckpt "$VAE_CKPT" \
        --stream_mode stream_generate_step \
        --batch_size 1 \
        --max_samples 8 \
        --num_runs 5 \
        --compute_offline_baseline \
        --compute_no_traj_baseline \
        --out_dir "$OUT_DIR" \
        --run_name "step_$STEP" \
        --devices "$NUM_DEVICES"
done
