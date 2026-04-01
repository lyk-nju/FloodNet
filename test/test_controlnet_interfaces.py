import inspect
import os
import tempfile
from pathlib import Path

import pytest
import torch

from FloodNet.models.diffusion_forcing_wan import DiffForcingWanModel
from FloodNet.models.tools.wan_controlnet import WanControlNet
from FloodNet.models.tools.wan_model import WanModel


def test_wanmodel_forward_has_controlnet_residuals_param():
    sig = inspect.signature(WanModel.forward)
    assert "controlnet_residuals" in sig.parameters


def test_controlnet_zero_heads_are_zero_after_init_from_backbone():
    # Small dims; no forward needed (CPU ok).
    backbone = WanModel(
        model_type="t2v",
        patch_size=(1, 1, 1),
        text_len=8,
        in_dim=16,
        dim=32,
        ffn_dim=64,
        freq_dim=16,
        text_dim=48,
        out_dim=16,
        num_heads=4,
        num_layers=2,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        causal=False,
        traj_enc_dim=0,
    )
    control = WanControlNet(
        model_type="t2v",
        patch_size=(1, 1, 1),
        text_len=8,
        in_dim=16,
        dim=32,
        ffn_dim=64,
        freq_dim=16,
        text_dim=48,
        out_dim=16,
        num_heads=4,
        num_layers=2,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        causal=False,
        traj_enc_dim=0,
    )
    control.init_from_backbone(backbone)
    for m in control.zero_out:
        assert torch.all(m.weight == 0)
        assert torch.all(m.bias == 0)


def test_diff_forcing_model_builds_controlnet_when_enabled():
    # Build-only (no forward): use pretokenized T5 table if present, else a tiny temp table.
    # Prefer env or sibling FloodDiffusion raw_data (same layout as configs/ldf.yaml dirs.raw_data).
    repo_root = Path(__file__).resolve().parents[2]
    default_pt = (
        repo_root / "FloodDiffusion" / "raw_data" / "HumanML3D" / "t5_text_embeddings.pt"
    )
    env_pt = os.environ.get("FLOODNET_PRETOKENIZED_T5", "").strip()
    if env_pt and Path(env_pt).is_file():
        path, cleanup = Path(env_pt), False
    elif default_pt.is_file():
        path, cleanup = default_pt, False
    else:
        text_dim = 4096
        fd, tmp = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.save(
            {"embeddings": {"": torch.zeros(4, text_dim)}, "text_dim": text_dim},
            tmp,
        )
        path, cleanup = Path(tmp), True
    try:
        m = DiffForcingWanModel(
            input_dim=16,
            hidden_dim=32,
            ffn_dim=64,
            freq_dim=16,
            num_heads=4,
            num_layers=2,
            text_len=8,
            use_text_cond=False,
            use_traj_cond=False,
            use_controlnet_traj=True,
            controlnet_init_from_backbone=False,
            use_precomputed_text_emb=True,
            precomputed_text_emb_path=str(path),
        )
        assert m.controlnet is not None
    finally:
        if cleanup:
            os.unlink(path)


@pytest.mark.gpu
def test_gpu_zero_init_equivalence_if_available():
    if os.environ.get("RUN_GPU_TESTS", "") != "1":
        pytest.skip("Set RUN_GPU_TESTS=1 to enable GPU integration tests.")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")

    device = torch.device("cuda")
    torch.manual_seed(0)

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
    ).to(device)

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
    ).to(device)
    control.init_from_backbone(backbone)

    x = [torch.randn(C_in, seq_len, 1, 1, device=device) for _ in range(B)]
    t = torch.rand(B, seq_len, device=device)
    context = [torch.randn(text_len, text_dim, device=device) for _ in range(B)]

    with torch.no_grad():
        residuals = control(x, t, context, seq_len)
        assert max(r.abs().max().item() for r in residuals) == 0.0

        out0 = backbone(x, t, context, seq_len)
        out1 = backbone(x, t, context, seq_len, controlnet_residuals=residuals)

    mx = 0.0
    for a, b in zip(out0, out1):
        mx = max(mx, (a - b).abs().max().item())
    assert mx == 0.0

