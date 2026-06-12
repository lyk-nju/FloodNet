#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="/home/yuankai/Text2Motion/FloodNet"
PYTHON_BIN="/home/yuankai/.conda/envs/flooddiffusion/bin/python"
CONFIG="$BASE_PATH/configs/ldf.yaml"
CKPT_DIR="$BASE_PATH/outputs"
ARTIFACT_ROOT="$BASE_PATH/eval/eval_output"
DEVICES="0,1,2,3,4"
NUM_DEVICES=5
STEPS=("485000")

cd "$BASE_PATH"
export PYTHONPATH="$BASE_PATH:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/floodnet_mpl}"
mkdir -p "$MPLCONFIGDIR" "$ARTIFACT_ROOT"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python env not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -s "$CONFIG" ]]; then
    echo "Config file is missing or empty: $CONFIG" >&2
    exit 1
fi

for STEP in "${STEPS[@]}"
do
    CKPT="$CKPT_DIR/step_$STEP.ckpt"
    if [[ ! -f "$CKPT" ]]; then
        echo "Checkpoint not found: $CKPT" >&2
        exit 1
    fi

    echo "Starting evaluation for step: $STEP"

    CUDA_VISIBLE_DEVICES="$DEVICES" "$PYTHON_BIN" eval/run_eval.py \
        --config "$CONFIG" \
        --ckpt "$CKPT" \
        --artifact_root "$ARTIFACT_ROOT" \
        --default_root_dir "$ARTIFACT_ROOT/async_eval" \
        --devices "$NUM_DEVICES"
done
