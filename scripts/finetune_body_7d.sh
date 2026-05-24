#!/usr/bin/env bash
# T_B_11: body 7D fine-tune from step_460000.ckpt (5 subitems on).
#
# RUN ON THE DATA MACHINE (needs HumanML3D data + VAE + GPU). The dev box has no
# data, so this only launches training; it does not run here.
#
# CLI note: train_ldf.py takes `--config` + `--override key=value ...` (there is
# no --resume/--max_steps/--output_dir flag). Output lands in
# <cfg.save_dir>/<run_time>_<exp_name>/. 7D needs BOTH traj flags = 7 (a mismatch
# raises via validate_traj_dim_consistency, T_B_10).
#
# Usage:  scripts/finetune_body_7d.sh [RESUME_CKPT] [MAX_STEPS] [EXP_NAME]
set -euo pipefail

PY="${PY:-/home/lai/anaconda3/envs/floodiffusion/bin/python}"
RESUME_CKPT="${1:-outputs/step_460000.ckpt}"
MAX_STEPS="${2:-50000}"
EXP_NAME="${3:-body_finetune_v1}"

cd "$(dirname "$0")/.."

exec "$PY" train_ldf.py --config configs/ldf.yaml --override \
  "resume_ckpt=${RESUME_CKPT}" \
  "trainer.max_steps=${MAX_STEPS}" \
  "exp_name=${EXP_NAME}" \
  "model.params.traj_encoder_in_dim=7" \
  "data.traj_feat_dim=7" \
  "body_aux_loss.enabled=true" \
  "anchor_canonicalize.enabled=true" \
  "history_corruption.enabled=true" \
  "horizon_sim.enabled=true"
