#!/usr/bin/env bash
# T_B_11: body 7D fine-tune from step_460000.ckpt (5 subitems on).
#
# RUN ON THE DATA MACHINE (needs HumanML3D data + VAE + a working CUDA build —
# flash-attention asserts on CPU). The dev box has no data; this only launches.
#
# CLI note: train_ldf.py takes `--config` + `--override key=value ...` (there is
# no --resume/--max_steps/--output_dir flag). Output lands in
# <cfg.save_dir>/<run_time>_<exp_name>/. 7D needs BOTH traj flags = 7 (a mismatch
# raises via validate_traj_dim_consistency, T_B_10).
#
# ⚠ trainer.max_steps is ABSOLUTE (Lightning compares global_step). Resuming from
#   step_460000 means: for +50k steps pass MAX_STEPS=510000; for a 1-step smoke
#   pass MAX_STEPS=460001 (max_steps=1 is already in the past → trains nothing).
#
# ⚠ Z_STATS_DIR must point at finite z_mean.npy/z_std.npy from compute_z_stats.py
#   (run it with --skip_nonfinite first; 2 real latents are NaN). Without this the
#   model keeps z_std=1 and history-corruption noise is uncalibrated (B-P0-1).
#
# Usage:  scripts/finetune_body_7d.sh [RESUME_CKPT] [MAX_STEPS_ABS] [EXP_NAME] [Z_STATS_DIR]
#   smoke:  scripts/finetune_body_7d.sh outputs/step_460000.ckpt 460001 body_ft_smoke deps/body_stats
set -euo pipefail

PY="${PY:-/home/lai/anaconda3/envs/floodiffusion/bin/python}"
RESUME_CKPT="${1:-outputs/step_460000.ckpt}"
MAX_STEPS="${2:-510000}"            # ABSOLUTE: 460000 (resume) + 50000 fine-tune
EXP_NAME="${3:-body_finetune_v1}"
Z_STATS_DIR="${4:-deps/body_stats}"

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
  "history_corruption.z_stats_dir=${Z_STATS_DIR}"
