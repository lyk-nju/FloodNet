conda run -n flooddiffusion python tools/eval_generation_metrics.py \
  --config configs/ldf_copy5.yaml \
  --ckpt outputs/20260421_174423_ldf/step_step=244000.ckpt \
  --set test_vae_ckpt=outputs/vae_1d_z4_step=300000.ckpt exp_name=ldf_174423 \
  --seed 1234 \
  --viz_traj