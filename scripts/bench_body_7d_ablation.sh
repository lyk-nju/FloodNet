#!/usr/bin/env bash
# T_B_12: body 7D fine-tune ablation sweep + predroot stream benchmark.
#
# RUN ON THE DATA MACHINE (needs data + VAE + GPU). For each of the 6 ablations
# (all_on, no_corruption, no_horizon_sim, no_anchor_canonical, no_heading_loss,
# no_7d) it (1) fine-tunes a variant from the base ckpt with that ablation's
# config overrides, then (2) runs stream_benchmark predroot on the result. The
# override sets come from utils/training/ablation.py (single source of truth).
#
# Usage:  scripts/bench_body_7d_ablation.sh [RESUME_CKPT] [MAX_STEPS] [VAE_CKPT] [RAW_DATA_DIR]
set -euo pipefail

PY="${PY:-/home/lai/anaconda3/envs/floodiffusion/bin/python}"
RESUME_CKPT="${1:-outputs/step_460000.ckpt}"
MAX_STEPS="${2:-510000}"            # ABSOLUTE global_step (460000 resume + 50000); see finetune_body_7d.sh
VAE_CKPT="${3:?pass the VAE ckpt path}"
RAW_DATA_DIR="${4:?pass the raw_data dir}"
Z_STATS_DIR="${Z_STATS_DIR:-deps/body_stats}"   # finite z stats (compute_z_stats --skip_nonfinite)
SUITES="${SUITES:-step_predroot,real_predroot}"
ROOT_OUT="${ROOT_OUT:-outputs/body_finetune_ablation}"

cd "$(dirname "$0")/.."

ABLATIONS="all_on no_corruption no_horizon_sim no_anchor_canonical no_heading_loss no_7d"

for name in $ABLATIONS; do
  echo "==================== ablation: ${name} ===================="
  overrides="$("$PY" -m utils.training.ablation "$name")"
  exp="body_ft_${name}"
  out="${ROOT_OUT}/${name}"

  # (1) fine-tune this ablation variant
  # shellcheck disable=SC2086
  "$PY" train_ldf.py --config configs/ldf.yaml --override \
    "resume_ckpt=${RESUME_CKPT}" "trainer.max_steps=${MAX_STEPS}" \
    "history_corruption.z_stats_dir=${Z_STATS_DIR}" \
    "exp_name=${exp}" $overrides

  # (2) benchmark predroot on the produced ckpt
  ckpt="$(ls -t ${ROOT_OUT}/*_${exp}/step_*.ckpt 2>/dev/null | head -1 || true)"
  if [[ -z "${ckpt}" ]]; then
    echo "WARN: no ckpt found for ${exp}; skipping benchmark" >&2
    continue
  fi
  "$PY" eval/stream_benchmark.py \
    --config configs/ldf.yaml \
    --ckpt "${ckpt}" \
    --vae_ckpt "${VAE_CKPT}" \
    --raw_data_dir "${RAW_DATA_DIR}" \
    --suites "${SUITES}" \
    --output_dir "${out}/bench"
done

echo "ablation sweep done → ${ROOT_OUT}"
