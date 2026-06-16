import torch

from models.diffusion_forcing_wan import DiffForcingWanModel
from utils.training.ldf.conditioning import PreparedCondition
from utils.training.ldf.model_step import run_training_window


class _ConstantBackbone:
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
        values = self.outputs_by_call[len(self.calls) - 1]
        return [
            torch.full_like(sample, float(value))
            for sample, value in zip(noisy_input, values)
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


def _bare_model():
    model = DiffForcingWanModel.__new__(DiffForcingWanModel)
    torch.nn.Module.__init__(model)
    return model


def test_run_training_window_contract_for_vel_prediction_and_x0_estimates():
    model = _bare_model()
    model.chunk_size = 2
    model.prediction_type = "vel"
    model.time_embedding_scale = 3.0

    clean = torch.arange(2 * 5 * 2, dtype=torch.float32).view(2, 5, 2) / 10.0
    noise = torch.linspace(-0.4, 0.55, steps=2 * 5 * 2).view(2, 5, 2)
    noise_level = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.40, 0.50],
            [0.15, 0.25, 0.35, 0.45, 0.55],
        ],
        dtype=torch.float32,
    )
    noisy = clean + noise

    model._get_noise_levels = lambda device, seq_len, time_steps: noise_level.to(device)
    model.add_noise = lambda clean_feature, level: (noisy, noise)

    controlnet = _RecordingControlNet()
    backbone = _ConstantBackbone(outputs_by_call=[[0.75, -0.25]])
    model._controlnet_forward = controlnet
    model.model = backbone

    text_context = [object(), object()]
    traj_emb = torch.ones(2, 5, 3)
    traj_seq_lens = torch.tensor([3, 5])
    traj_token_mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],
        dtype=torch.float32,
    )
    batch = {
        "feature_length": torch.tensor([4, 5]),
        "traj_features": torch.zeros(2, 5, 7),
    }
    condition = PreparedCondition(
        text_context=text_context,
        text_dropped_flags=[False, False],
        traj_emb=traj_emb,
        traj_seq_lens=traj_seq_lens,
        traj_dropped=False,
        traj_token_mask=traj_token_mask,
    )

    out = run_training_window(
        model,
        batch,
        clean_feature=clean,
        time_steps=torch.tensor([1.0, 2.0]),
        condition=condition,
    )

    clean_pre = model.preprocess(clean)
    noise_pre = model.preprocess(noise)
    noisy_pre = model.preprocess(noisy)
    pred_0 = torch.full_like(clean_pre[0, :, :3], 0.75)
    pred_1 = torch.full_like(clean_pre[1, :, :5], -0.25)
    vel_0 = clean_pre[0, :, :3] - noise_pre[0, :, :3]
    vel_1 = clean_pre[1, :, :5] - noise_pre[1, :, :5]
    loss_0 = ((pred_0[:, -2:] - vel_0[:, -2:]) ** 2).mean()
    loss_1 = ((pred_1[:, -2:] - vel_1[:, -2:]) ** 2).mean()

    torch.testing.assert_close(out["loss"], (loss_0 + loss_1) / 2)
    assert out["end_indices"] == [3, 5]
    assert out["pred_x0_latent_list"] is not None
    assert len(out["pred_x0_latent_list"]) == 2
    assert len(out["x0_latent_list"]) == 2

    expected_loss_x0_0 = (pred_0 + noise_pre[0, :, :3])[:, :, 0, 0].permute(1, 0)
    expected_loss_x0_1 = (pred_1 + noise_pre[1, :, :5])[:, :, 0, 0].permute(1, 0)
    expected_sf_x0_0 = (
        noisy_pre[0, :, :3]
        + noise_level[0, :3].view(1, -1, 1, 1) * pred_0
    )[:, :, 0, 0].permute(1, 0)
    expected_sf_x0_1 = (
        noisy_pre[1, :, :5]
        + noise_level[1, :5].view(1, -1, 1, 1) * pred_1
    )[:, :, 0, 0].permute(1, 0)

    torch.testing.assert_close(out["pred_x0_latent_list"][0], expected_loss_x0_0)
    torch.testing.assert_close(out["pred_x0_latent_list"][1], expected_loss_x0_1)
    torch.testing.assert_close(out["x0_latent_list"][0], expected_sf_x0_0)
    torch.testing.assert_close(out["x0_latent_list"][1], expected_sf_x0_1)

    assert len(controlnet.calls) == 1
    control_call = controlnet.calls[0]
    assert control_call["text_context"] is text_context
    assert control_call["traj_emb"] is traj_emb
    assert control_call["traj_seq_lens"] is traj_seq_lens
    assert control_call["traj_token_mask"] is traj_token_mask
    torch.testing.assert_close(control_call["t_scaled"], noise_level * 3.0)
    assert [sample.shape[1] for sample in control_call["noisy_input"]] == [3, 5]

    assert len(backbone.calls) == 1
    backbone_call = backbone.calls[0]
    assert backbone_call["controlnet_residuals"] is control_call["residuals"]
    assert backbone_call["traj_emb"] is None
    assert backbone_call["traj_seq_lens"] is None


def test_denoise_with_cfg_contract_for_separated_text_and_traj_guidance():
    model = _bare_model()
    model.cfg_scale_text = 2.0
    model.cfg_scale_traj = 4.0

    controlnet = _RecordingControlNet()
    backbone = _ConstantBackbone(outputs_by_call=[[10.0, 3.0, 1.0]])
    model._controlnet_forward = controlnet
    model.model = backbone

    seq_len = 4
    noisy_input = [torch.zeros(2, seq_len, 1, 1)]
    t_scaled = torch.tensor([0.25])
    text_cond_ctx = ["cond"]
    text_null_ctx = ["null"]
    traj_emb = torch.arange(seq_len * 3, dtype=torch.float32).view(1, seq_len, 3)
    traj_seq_lens = torch.tensor([seq_len])
    traj_token_mask = torch.ones(1, seq_len)

    out = model._denoise_with_cfg(
        noisy_input,
        t_scaled,
        text_cond_ctx=text_cond_ctx,
        text_null_ctx=text_null_ctx,
        traj_emb=traj_emb,
        traj_seq_lens=traj_seq_lens,
        seq_len=seq_len,
        batch_size=1,
        traj_token_mask=traj_token_mask,
    )

    expected = torch.full_like(noisy_input[0], 23.0)
    torch.testing.assert_close(out[0], expected)

    assert len(controlnet.calls) == 2
    cond_call, uncond_call = controlnet.calls
    assert cond_call["text_context"] == ["cond", "null"]
    assert cond_call["traj_emb"].shape == (2, seq_len, 3)
    assert cond_call["traj_seq_lens"].tolist() == [seq_len, seq_len]
    assert cond_call["traj_token_mask"].shape == (2, seq_len)
    assert uncond_call["text_context"] is text_null_ctx
    assert uncond_call["traj_emb"] is None
    assert uncond_call["traj_seq_lens"] is None
    assert uncond_call["traj_token_mask"] is None

    assert len(backbone.calls) == 1
    backbone_call = backbone.calls[0]
    assert len(backbone_call["noisy_input"]) == 3
    torch.testing.assert_close(
        backbone_call["t_scaled"],
        torch.tensor([0.25, 0.25, 0.25]),
    )
    assert backbone_call["context"] == ["cond", "null", "null"]
    residuals = backbone_call["controlnet_residuals"][0]
    assert residuals.shape == (3, seq_len, 1)
    torch.testing.assert_close(residuals[:2], torch.ones(2, seq_len, 1))
    torch.testing.assert_close(residuals[2:], torch.full((1, seq_len, 1), 2.0))
    assert backbone_call["traj_emb"] is None
    assert backbone_call["traj_seq_lens"] is None
