import torch

ckpt = torch.load("/data/home/shengqiuProf_user_yuankai/FloodNet/outputs/20260508_001434_ldf/step_392000.ckpt", map_location='cpu', weights_only=False)

print('has ema_state:', 'ema_state' in ckpt)
print('global_step:', ckpt.get('global_step'))

if 'ema_state' in ckpt:
    print('shadow_params count:', len(ckpt['ema_state']['shadow_params']))