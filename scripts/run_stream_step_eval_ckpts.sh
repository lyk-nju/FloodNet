#!/usr/bin/env bash
set -euo pipefail

BASE_PATH="/data/home/shengqiuProf_user_yuankai/FloodNet"
PYTHON_BIN="/data/home/shengqiuProf_user_yuankai/miniconda3/envs/flooddiffusion/bin/python"
CONFIG="$BASE_PATH/configs/ldf_test.yaml"
CKPT_DIR="/data/home/shengqiuProf_user_yuankai/FloodNet/outputs/20260630_132040_ldf"
VAE_CKPT="/data/home/shengqiuProf_user_yuankai/FloodDiffusion/outputs/vae_1d_z4_step=300000.ckpt"
OUT_DIR="$BASE_PATH/eval/result"
DEVICES="0,1,2,3,4,5,6,7"
NUM_DEVICES=8
STEPS=(
    "531000" "532000" "533000" "534000" "535000"
    "536000" "537000" "538000" "539000" "540000"
    "541000" "542000" "543000" "544000" "545000"
    "546000" "547000" "548000" "549000" "550000"
    "551000" "552000" "553000" "554000" "555000"
    "556000" "557000" "558000" "559000" "560000"
    "561000" "562000" "563000" "564000" "565000"
)

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

for STEP in "${STEPS[@]}"; do
    CKPT="$CKPT_DIR/step_$STEP.ckpt"
    if [[ ! -f "$CKPT" ]]; then
        echo "Checkpoint not found: $CKPT" >&2
        exit 1
    fi

    echo "Starting stream_generate_step evaluation for step: $STEP"

    CUDA_VISIBLE_DEVICES="$DEVICES" "$PYTHON_BIN" -m eval.ldf.stream_metrics \
        --config "$CONFIG" \
        --ckpt "$CKPT" \
        --vae_ckpt "$VAE_CKPT" \
        --stream_mode stream_generate_step \
        --batch_size 1 \
        --num_runs 5 \
        --max_samples 0 \
        --max_batches 0 \
        --seed 1234 \
        --meta_paths "/data/home/shengqiuProf_user_yuankai/FloodDiffusion/raw_data/HumanML3D/test_min.txt" \
        --no_compute_offline_baseline \
        --no_compute_no_traj_baseline \
        --out_dir "$OUT_DIR" \
        --run_name "step_$STEP" \
        --devices "$NUM_DEVICES" \
        --save_feature_npy \
        --set \
            eval.history_length=30 \
            eval.traj_horizon_tokens=20 \
            eval.token_dt=0.20 \
            eval.frames_per_token=4 \
            model.params.cfg_scale_text=1.25 \
            model.params.cfg_scale_traj=3.0
done
