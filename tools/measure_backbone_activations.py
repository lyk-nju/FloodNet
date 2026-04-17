import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from utils.initialize import Config, get_function, instantiate

cfg = Config('configs/ldf.yaml').config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASELINE_CKPT = '/home/yuankai/Text2Motion/FloodDiffusion/outputs/20251107_021814_ldf_stream/step_step=240000.ckpt'

model_params = dict(cfg.model.params)
model_params['use_controlnet_traj'] = False
model_params['freeze_backbone_for_controlnet'] = False
model_params['controlnet_init_from_backbone'] = False

model = instantiate(target=cfg.model.target, cfg=None, hfstyle=False, **model_params)
ckpt = torch.load(BASELINE_CKPT, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=False)
if 'ema_state' in ckpt:
    ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
    try:
        ema.load_state_dict(ckpt['ema_state'])
        ema.copy_to(model.parameters())
        print('[EMA applied]')
    except Exception as e:
        print(f'[EMA skip: {e}]')
model.to(device).eval()
print(f'Blocks: {len(model.model.blocks)}')

layer_outputs = {}
hooks = []
for i, block in enumerate(model.model.blocks):
    def make_hook(idx):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if t.dim() == 5:
                t = t[:, :, :, 0, 0].permute(0, 2, 1)  # (B,C,T,1,1) -> (B,T,C)
            norm = t.detach().cpu().float().norm(dim=-1).mean().item()
            layer_outputs.setdefault(idx, []).append(norm)
        return hook
    hooks.append(block.register_forward_hook(make_hook(i)))

collate_fn = get_function(cfg.data.collate_fn) if cfg.data.get('collate_fn') else None
dataset = instantiate(cfg.data.get('test_target', cfg.data.target), cfg=cfg, split='test')
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

n_done = 0
with torch.no_grad():
    for batch in loader:
        if n_done >= 5: break
        mb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        mb['feature'] = mb.pop('token') if 'token' in mb else mb['feature']
        mb['feature_length'] = mb.pop('token_length') if 'token_length' in mb else mb['feature_length']
        if 'token_text_end' in batch:
            mb['feature_text_end'] = batch['token_text_end']
        model(mb)
        n_done += 1

for h in hooks: h.remove()

print('\n===== Baseline backbone 各层 hidden state L2 norm =====')
print(f'  {"layer":>6}  {"mean_L2_norm":>14}')
print('  ' + '-'*24)
for i in sorted(layer_outputs.keys()):
    print(f'  {i:>6}  {np.mean(layer_outputs[i]):>14.4f}')
