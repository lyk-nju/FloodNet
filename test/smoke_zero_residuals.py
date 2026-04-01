"""
Smoke test: ControlNet zero residuals correctness.

This script is intentionally pytest-free (per project convention).

What it checks:
1) After init_from_backbone(), WanControlNet residual heads are exactly zero.
2) If CUDA is available, forward residuals are exactly zero AND injecting them into WanModel
   produces identical outputs (zero-init equivalence).

Run:
  conda activate flooddiffusion
  python FloodNet/test/smoke_zero_residuals.py
"""

import sys


def _fail(msg: str) -> None:
    raise SystemExit(f"[FAIL] {msg}")


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def main() -> int:
    try:
        import torch
    except Exception as e:
        _fail(f"torch import failed: {e}")

    from FloodNet.models.tools.wan_model import WanModel
    from FloodNet.models.tools.wan_controlnet import WanControlNet

    torch.manual_seed(0)

    # Keep shapes small; must still satisfy Wan constraints (dim/heads).
    B = 2
    seq_len = 8
    C_in = 16
    C_out = 16
    dim = 64
    ffn_dim = 128
    freq_dim = 32
    text_dim = 48
    text_len = 16
    num_heads = 4
    num_layers = 3

    # Backbone: unconditional (no traj tokens). ControlNet-only conditioning is verified via residual injection.
    backbone = WanModel(
        model_type="t2v",
        patch_size=(1, 1, 1),
        text_len=text_len,
        in_dim=C_in,
        dim=dim,
        ffn_dim=ffn_dim,
        freq_dim=freq_dim,
        text_dim=text_dim,
        out_dim=C_out,
        num_heads=num_heads,
        num_layers=num_layers,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        causal=False,
        traj_enc_dim=0,
    )

    # ControlNet: in this smoke test we keep traj_enc_dim=0 as well; we only verify zero-init residual property.
    control = WanControlNet(
        model_type="t2v",
        patch_size=(1, 1, 1),
        text_len=text_len,
        in_dim=C_in,
        dim=dim,
        ffn_dim=ffn_dim,
        freq_dim=freq_dim,
        text_dim=text_dim,
        out_dim=C_out,
        num_heads=num_heads,
        num_layers=num_layers,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        causal=False,
        traj_enc_dim=0,
    )
    control.init_from_backbone(backbone)

    # Check residual heads are exactly zero.
    for i, m in enumerate(control.zero_out):
        if not torch.all(m.weight == 0):
            _fail(f"zero_out[{i}].weight is not all-zero")
        if not torch.all(m.bias == 0):
            _fail(f"zero_out[{i}].bias is not all-zero")
    _ok("zero_out heads are exactly zero after init_from_backbone()")

    if not torch.cuda.is_available():
        print(
            "[SKIP] CUDA not available; forward equivalence test requires GPU because "
            "FloodNet/models/tools/attention.py asserts CUDA for flash_attention."
        )
        return 0

    device = torch.device("cuda")
    backbone = backbone.to(device)
    control = control.to(device)

    x = [torch.randn(C_in, seq_len, 1, 1, device=device) for _ in range(B)]
    t = torch.rand(B, seq_len, device=device)
    context = [torch.randn(text_len, text_dim, device=device) for _ in range(B)]

    with torch.no_grad():
        residuals = control(x, t, context, seq_len)
        mx_res = max(r.abs().max().item() for r in residuals)
        if mx_res != 0.0:
            _fail(f"control forward residuals are not exactly zero (max={mx_res})")
        _ok("control forward residuals are exactly zero (GPU)")

        out0 = backbone(x, t, context, seq_len)
        out1 = backbone(x, t, context, seq_len, controlnet_residuals=residuals)

    mx = 0.0
    for a, b in zip(out0, out1):
        mx = max(mx, (a - b).abs().max().item())
    if mx != 0.0:
        _fail(f"injecting zero residuals changed backbone output (max diff={mx})")
    _ok("injecting zero residuals leaves backbone output unchanged (GPU)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

