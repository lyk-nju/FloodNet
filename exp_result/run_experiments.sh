#!/usr/bin/env bash
# =============================================================================
# FloodNet ControlNet 综合实验脚本
# 覆盖：checkpoint 对比 + Separated CFG 调参
# 结果写入 exp_result/results/
# 用法：cd /home/yuankai/Text2Motion/FloodNet && bash exp_result/run_experiments.sh
# =============================================================================

set -e
cd "$(dirname "$0")/.."   # 切到 FloodNet 根目录

CONDA_ENV="flooddiffusion"
PYTHON="conda run -n ${CONDA_ENV} python"
EVAL="tools/eval_control_loss.py"
CFG="configs/ldf.yaml"
SEED=1234
OUT="exp_result/results"
mkdir -p "$OUT"

# GPU: 单卡推理，选空闲的 GPU（默认 GPU 0）
export CUDA_VISIBLE_DEVICES=0

CKPT_240K="outputs/20251107_021814_ldf_stream/step_step=240000.ckpt"
CKPT_245K_OLD="outputs/20260415_114956_ldf/step_step=245000.ckpt"
CKPT_245K_NEW="outputs/20260416_000936_ldf/step_step=245000.ckpt"
CKPT_300K="outputs/20260402_114343_ldf/step_step=300000.ckpt"

echo "========================================================"
echo " FloodNet 实验开始  $(date)"
echo "========================================================"

# ===========================================================================
# 实验 A：四个 Checkpoint 的 forward / generate control loss 对比
# ===========================================================================
echo ""
echo "--- Exp A: Checkpoint 对比 ---"

for MODE in forward generate; do
  for TAG_CKPT in \
    "240k:${CKPT_240K}" \
    "245k_old:${CKPT_245K_OLD}" \
    "245k_new:${CKPT_245K_NEW}" \
    "300k:${CKPT_300K}"; do

    TAG="${TAG_CKPT%%:*}"
    CKPT="${TAG_CKPT#*:}"
    LOG="${OUT}/A_${TAG}_${MODE}.log"

    echo "[A] ${TAG} | mode=${MODE} -> ${LOG}"
    $PYTHON $EVAL \
      --config $CFG \
      --ckpt   "$CKPT" \
      --eval_mode $MODE \
      --seed $SEED \
      --topk 3 \
      2>&1 | tee "$LOG"
  done
done

# ===========================================================================
# 实验 B：Separated CFG 调参（在 245k_new 上）
# ===========================================================================
echo ""
echo "--- Exp B: Separated CFG 调参 (245k_new, generate mode) ---"

# (cfg_scale_text, cfg_scale_traj) 组合
CFG_COMBOS=(
  "5.0:0.0"    # baseline: 纯文本 CFG（无轨迹引导）
  "5.0:3.0"    # 轻度轨迹增强
  "3.0:5.0"    # 轨迹优先
  "1.0:7.0"    # 极端轨迹引导（几乎不引导文本）
  "3.0:3.0"    # 均衡
  "5.0:5.0"    # 两者均强
)

for COMBO in "${CFG_COMBOS[@]}"; do
  W_TEXT="${COMBO%%:*}"
  W_TRAJ="${COMBO#*:}"
  TAG="wt${W_TEXT}_wr${W_TRAJ}"
  LOG="${OUT}/B_${TAG}_generate.log"

  echo "[B] cfg_scale_text=${W_TEXT} cfg_scale_traj=${W_TRAJ} -> ${LOG}"
  $PYTHON $EVAL \
    --config $CFG \
    --ckpt   "$CKPT_245K_NEW" \
    --eval_mode generate \
    --seed $SEED \
    --topk 3 \
    --set model.params.cfg_scale_text=${W_TEXT} \
          model.params.cfg_scale_traj=${W_TRAJ} \
    2>&1 | tee "$LOG"
done

# ===========================================================================
# 实验 C：Smooth Root 效果验证（245k_new，sigma=0 vs 2.0）
# ===========================================================================
echo ""
echo "--- Exp C: Smooth Root 效果 (245k_new, generate mode) ---"

for SIGMA in 0.0 2.0; do
  TAG="sigma${SIGMA}"
  LOG="${OUT}/C_${TAG}_generate.log"

  echo "[C] smooth_traj_sigma=${SIGMA} -> ${LOG}"
  $PYTHON $EVAL \
    --config $CFG \
    --ckpt   "$CKPT_245K_NEW" \
    --eval_mode generate \
    --seed $SEED \
    --set data.smooth_traj_sigma=${SIGMA} \
    2>&1 | tee "$LOG"
done

echo ""
echo "========================================================"
echo " 全部实验完成  $(date)"
echo " 结果目录: exp_result/results/"
echo "========================================================"
