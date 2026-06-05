import types

import torch

from models.diffusion_forcing_wan import DiffForcingWanModel


class _RecordingBackbone:
    def __init__(self, outputs_by_call):
        self.outputs_by_call = outputs_by_call
        self.calls = []

    def __call__(
        self,
        noisy_input,
        t_scaled,
        context,
        seq_len,
        y=None,
        traj_emb=None,
        traj_seq_lens=None,
        controlnet_residuals=None,
    ):
        self.calls.append(
            {
                "noisy_input": noisy_input,
                "t_scaled": t_scaled,
                "context": context,
                "seq_len": seq_len,
                "traj_emb": traj_emb,
                "traj_seq_lens": traj_seq_lens,
                "controlnet_residuals": controlnet_residuals,
            }
        )
        scalars = self.outputs_by_call[len(self.calls) - 1]
        return [
            torch.full_like(sample, float(scalar))
            for sample, scalar in zip(noisy_input, scalars)
        ]


class _RecordingControlNet:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        noisy_input,
        t_scaled,
        text_context,
        seq_len,
        traj_emb,
        traj_seq_lens,
        traj_token_mask=None,
    ):
        residuals = [object()]
        self.calls.append(
            {
                "noisy_input": noisy_input,
                "t_scaled": t_scaled,
                "text_context": text_context,
                "seq_len": seq_len,
                "traj_emb": traj_emb,
                "traj_seq_lens": traj_seq_lens,
                "traj_token_mask": traj_token_mask,
                "residuals": residuals,
            }
        )
        return residuals


class _TensorRecordingControlNet:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        noisy_input,
        t_scaled,
        text_context,
        seq_len,
        traj_emb,
        traj_seq_lens,
        traj_token_mask=None,
    ):
        value = float(len(self.calls) + 1)
        residuals = [
            torch.full(
                (len(noisy_input), seq_len, 1),
                value,
                dtype=noisy_input[0].dtype,
                device=noisy_input[0].device,
            )
        ]
        self.calls.append(
            {
                "noisy_input": noisy_input,
                "t_scaled": t_scaled,
                "text_context": text_context,
                "seq_len": seq_len,
                "traj_emb": traj_emb,
                "traj_seq_lens": traj_seq_lens,
                "traj_token_mask": traj_token_mask,
                "residuals": residuals,
            }
        )
        return residuals


def _make_harness(cfg_scale_text, backbone, controlnet):
    model = DiffForcingWanModel.__new__(DiffForcingWanModel)
    model.cfg_scale_text = cfg_scale_text
    model.cfg_scale_traj = 999.0
    model.model = backbone
    model._controlnet_forward = controlnet
    model._concat_text_for_cfg = types.MethodType(
        DiffForcingWanModel._concat_text_for_cfg, model
    )
    return model


def test_no_traj_double_batch_text_cfg_runs_controlnet_null():
    seq_len = 4
    noisy = [torch.zeros(2, seq_len, 1, 1)]
    t_scaled = torch.tensor([0.25])
    backbone = _RecordingBackbone(outputs_by_call=[[10.0, 3.0]])
    controlnet = _RecordingControlNet()
    model = _make_harness(cfg_scale_text=2.0, backbone=backbone, controlnet=controlnet)

    out = DiffForcingWanModel._denoise_with_cfg(
        model,
        noisy,
        t_scaled,
        text_cond_ctx=["cond"],
        text_null_ctx=["null"],
        traj_emb=None,
        traj_seq_lens=None,
        seq_len=seq_len,
        batch_size=1,
        traj_token_mask=None,
    )

    assert torch.allclose(out[0], torch.full_like(noisy[0], 17.0))
    assert len(controlnet.calls) == 1
    assert controlnet.calls[0]["text_context"] == ["cond", "null"]
    assert len(controlnet.calls[0]["noisy_input"]) == 2
    assert torch.equal(controlnet.calls[0]["t_scaled"], torch.tensor([0.25, 0.25]))
    assert controlnet.calls[0]["traj_emb"] is None
    assert controlnet.calls[0]["traj_seq_lens"] is None
    assert controlnet.calls[0]["traj_token_mask"] is None
    assert len(backbone.calls) == 1
    assert backbone.calls[0]["controlnet_residuals"] is controlnet.calls[0]["residuals"]


def test_separated_traj_cfg_uncond_branch_runs_controlnet_null():
    seq_len = 4
    noisy = [torch.zeros(2, seq_len, 1, 1)]
    t_scaled = torch.tensor([0.25])
    traj_emb = torch.ones(1, seq_len, 2)
    traj_seq_lens = torch.tensor([seq_len])
    traj_token_mask = torch.ones(1, seq_len)
    backbone = _RecordingBackbone(outputs_by_call=[[10.0, 3.0, 1.0]])
    controlnet = _TensorRecordingControlNet()
    model = _make_harness(cfg_scale_text=2.0, backbone=backbone, controlnet=controlnet)
    model.cfg_scale_traj = 4.0

    DiffForcingWanModel._denoise_with_cfg(
        model,
        noisy,
        t_scaled,
        text_cond_ctx=["cond"],
        text_null_ctx=["null"],
        traj_emb=traj_emb,
        traj_seq_lens=traj_seq_lens,
        seq_len=seq_len,
        batch_size=1,
        traj_token_mask=traj_token_mask,
    )

    assert len(controlnet.calls) == 2
    assert controlnet.calls[0]["traj_emb"] is not None
    assert controlnet.calls[1]["traj_emb"] is None
    assert controlnet.calls[1]["traj_seq_lens"] is None
    assert controlnet.calls[1]["traj_token_mask"] is None
    triple_residual = backbone.calls[0]["controlnet_residuals"][0]
    assert torch.allclose(triple_residual[:2], torch.ones(2, seq_len, 1))
    assert torch.allclose(triple_residual[2:], torch.full((1, seq_len, 1), 2.0))
    assert backbone.calls[0]["traj_emb"] is None
    assert backbone.calls[0]["traj_seq_lens"] is None


def test_no_traj_fallback_text_cfg_runs_separate_null_controlnet_branch():
    seq_len = 4
    noisy = [torch.zeros(2, seq_len, 1, 1)]
    t_scaled = torch.tensor([0.25])
    cond_ctx = ["frame-a", "frame-b"]
    null_ctx = ["null"]
    backbone = _RecordingBackbone(outputs_by_call=[[10.0], [3.0]])
    controlnet = _RecordingControlNet()
    model = _make_harness(cfg_scale_text=2.0, backbone=backbone, controlnet=controlnet)

    out = DiffForcingWanModel._denoise_with_cfg(
        model,
        noisy,
        t_scaled,
        text_cond_ctx=cond_ctx,
        text_null_ctx=null_ctx,
        traj_emb=None,
        traj_seq_lens=None,
        seq_len=seq_len,
        batch_size=1,
        traj_token_mask=None,
    )

    assert torch.allclose(out[0], torch.full_like(noisy[0], 17.0))
    assert len(controlnet.calls) == 2
    assert controlnet.calls[0]["text_context"] is cond_ctx
    assert controlnet.calls[1]["text_context"] is null_ctx
    assert controlnet.calls[0]["traj_emb"] is None
    assert controlnet.calls[1]["traj_emb"] is None
    assert controlnet.calls[0]["traj_seq_lens"] is None
    assert controlnet.calls[1]["traj_seq_lens"] is None
    assert controlnet.calls[0]["traj_token_mask"] is None
    assert controlnet.calls[1]["traj_token_mask"] is None
    assert backbone.calls[0]["controlnet_residuals"] is controlnet.calls[0]["residuals"]
    assert backbone.calls[1]["controlnet_residuals"] is controlnet.calls[1]["residuals"]
    assert backbone.calls[0]["controlnet_residuals"] is not backbone.calls[1]["controlnet_residuals"]


def test_no_traj_without_text_cfg_runs_controlnet_null_once():
    seq_len = 4
    noisy = [torch.zeros(2, seq_len, 1, 1)]
    t_scaled = torch.tensor([0.25])
    backbone = _RecordingBackbone(outputs_by_call=[[7.0]])
    controlnet = _RecordingControlNet()
    model = _make_harness(cfg_scale_text=1.0, backbone=backbone, controlnet=controlnet)

    out = DiffForcingWanModel._denoise_with_cfg(
        model,
        noisy,
        t_scaled,
        text_cond_ctx=["cond"],
        text_null_ctx=["null"],
        traj_emb=None,
        traj_seq_lens=None,
        seq_len=seq_len,
        batch_size=1,
        traj_token_mask=None,
    )

    assert torch.allclose(out[0], torch.full_like(noisy[0], 7.0))
    assert len(controlnet.calls) == 1
    assert len(backbone.calls) == 1
    assert controlnet.calls[0]["text_context"] == ["cond"]
    assert controlnet.calls[0]["traj_emb"] is None
    assert controlnet.calls[0]["traj_seq_lens"] is None
    assert controlnet.calls[0]["traj_token_mask"] is None
    assert backbone.calls[0]["controlnet_residuals"] is controlnet.calls[0]["residuals"]
