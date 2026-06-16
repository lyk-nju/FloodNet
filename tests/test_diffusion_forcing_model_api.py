from __future__ import annotations

import inspect
import torch

import models.diffusion_forcing_wan as diffusion_wan_mod
from models.diffusion_forcing_wan import DiffForcingWanModel
from utils.ldf_condition import LDFCondition
from utils.training.ldf.model_factory import (
    install_precomputed_text_embeddings,
    prepare_model_params,
)


def test_model_constructor_has_no_training_strategy_params():
    params = inspect.signature(DiffForcingWanModel.__init__).parameters
    assert "self_forcing_enabled" not in params
    assert "self_forcing_stride_tokens" not in params
    assert "self_forcing_detach_between_steps" not in params
    assert "self_forcing_k_schedule" not in params
    assert "self_forcing_start_step" not in params


def test_model_constructor_has_no_precomputed_text_file_params():
    params = inspect.signature(DiffForcingWanModel.__init__).parameters
    assert "use_precomputed_text_emb" not in params
    assert "precomputed_text_emb_path" not in params


def test_precomputed_text_embeddings_are_installed_outside_model(tmp_path):
    path = tmp_path / "text.pt"
    torch.save(
        {
            "text_dim": 4096,
            "embeddings": {
                "": torch.zeros(1, 4096),
                "walk forward ": torch.ones(1, 4096),
            },
        },
        path,
    )
    params, text_emb_path = prepare_model_params(
        {
            "hidden_dim": 16,
            "use_precomputed_text_emb": True,
            "precomputed_text_emb_path": str(path),
        }
    )
    assert params == {"hidden_dim": 16, "build_text_encoder": False}
    assert text_emb_path == str(path)

    class StubModel:
        text_encoder = object()
        _precomputed_text_emb = None

    model = StubModel()
    install_precomputed_text_embeddings(model, text_emb_path, expected_text_dim=4096)

    assert model.text_encoder is None
    assert "" in model._precomputed_text_emb
    assert "walk forward" in model._precomputed_text_emb


def test_model_does_not_keep_condition_wrapper_methods():
    model_attrs = vars(DiffForcingWanModel)
    assert "_concat_text_for_cfg" not in model_attrs
    assert "_build_traj_emb" not in model_attrs
    assert "_get_traj_seq_lens" not in model_attrs
    assert ("_build_stream_direct_" + "traj_condition") not in model_attrs
    assert ("_extend_stream_text_context_" + "for_attention") not in model_attrs
    assert "load_state_dict" not in model_attrs


def test_model_does_not_import_inference_traj_buffer():
    source = inspect.getsource(diffusion_wan_mod)
    assert "TrajStreamBuffer" not in source
    assert "utils.inference" not in source
    assert "utils.inference.buffer" not in source
    assert "encode_traj_batch" not in source
    assert "get_traj_seq_lens" not in source


def test_stream_state_buffer_is_injected_from_inference_layer():
    model = DiffForcingWanModel.__new__(DiffForcingWanModel)
    torch.nn.Module.__init__(model)
    model.seq_len = None
    model.batch_size = None
    model.noise_steps = 4
    model.chunk_size = 2
    model.input_dim = 3
    model.preprocess = lambda x: x
    buffer = object()

    model.init_generated(5, batch_size=2, num_denoise_steps=4, traj_buffer=buffer)

    assert model._traj_buf is buffer


def test_forward_is_network_forward_not_training_step():
    model = DiffForcingWanModel.__new__(DiffForcingWanModel)
    torch.nn.Module.__init__(model)
    model.recorded = {}

    def controlnet_forward(
        noisy_input,
        t_scaled,
        text_context,
        seq_len,
        traj_emb,
        traj_seq_lens,
        traj_token_mask=None,
    ):
        model.recorded["controlnet"] = {
            "noisy_input": noisy_input,
            "t_scaled": t_scaled,
            "text_context": text_context,
            "seq_len": seq_len,
            "traj_emb": traj_emb,
            "traj_seq_lens": traj_seq_lens,
            "traj_token_mask": traj_token_mask,
        }
        return ["residual"]

    class Backbone:
        def __call__(
            self,
            noisy_input,
            t_scaled,
            text_context,
            seq_len,
            y=None,
            traj_emb=None,
            traj_seq_lens=None,
            controlnet_residuals=None,
        ):
            model.recorded["backbone"] = {
                "noisy_input": noisy_input,
                "t_scaled": t_scaled,
                "text_context": text_context,
                "seq_len": seq_len,
                "traj_emb": traj_emb,
                "traj_seq_lens": traj_seq_lens,
                "controlnet_residuals": controlnet_residuals,
            }
            return ["pred"]

    model._controlnet_forward = controlnet_forward
    model.model = Backbone()

    noisy_input = [torch.zeros(2, 3, 1, 1)]
    t_scaled = torch.zeros(1, 3)
    text_context = ["text"]
    traj_emb = torch.zeros(1, 3, 4)
    traj_seq_lens = torch.tensor([3])
    traj_token_mask = torch.ones(1, 3)

    out = model(
        noisy_input,
        t_scaled,
        text_context,
        3,
        traj_emb=traj_emb,
        traj_seq_lens=traj_seq_lens,
        traj_token_mask=traj_token_mask,
    )

    assert out == ["pred"]
    assert model.recorded["controlnet"]["traj_emb"] is traj_emb
    assert model.recorded["controlnet"]["traj_seq_lens"] is traj_seq_lens
    assert model.recorded["controlnet"]["traj_token_mask"] is traj_token_mask
    assert model.recorded["backbone"]["controlnet_residuals"] == ["residual"]
    assert model.recorded["backbone"]["traj_emb"] is None
    assert model.recorded["backbone"]["traj_seq_lens"] is None


def test_generate_consumes_prepared_condition_without_raw_condition_adapters():
    model = DiffForcingWanModel.__new__(DiffForcingWanModel)
    torch.nn.Module.__init__(model)
    model._dummy_param = torch.nn.Parameter(torch.zeros(()))
    model.input_dim = 2
    model.chunk_size = 1
    model.noise_steps = 1
    model.time_embedding_scale = 1.0
    model.prediction_type = "vel"
    model.recorded = {}

    def denoise(
        noisy_input,
        t_scaled,
        text_cond_ctx,
        text_null_ctx,
        traj_emb,
        traj_seq_lens,
        seq_len,
        batch_size,
        traj_token_mask=None,
    ):
        model.recorded["condition"] = {
            "text_context": text_cond_ctx,
            "text_null_context": text_null_ctx,
            "traj_emb": traj_emb,
            "traj_seq_lens": traj_seq_lens,
            "traj_token_mask": traj_token_mask,
            "seq_len": seq_len,
        }
        return [torch.zeros_like(noisy_input[0])]

    def forbidden_encode_text(*args, **kwargs):
        raise AssertionError("generate should consume prepared text_context")

    model._denoise_with_cfg = denoise
    model.encode_text_with_cache = forbidden_encode_text
    model.preprocess = lambda x: x.permute(0, 2, 1)[:, :, :, None, None]
    model.postprocess = lambda x: x.squeeze(-1).squeeze(-1).permute(0, 2, 1)

    traj_emb = torch.ones(1, 3, 4)
    traj_seq_lens = torch.tensor([3])
    traj_token_mask = torch.ones(1, 3)
    condition = LDFCondition(
        text_context=["cond"],
        text_null_context=["null"],
        traj_emb=traj_emb,
        traj_seq_lens=traj_seq_lens,
        traj_token_mask=traj_token_mask,
        seq_len=3,
        attn_len=3,
    )
    batch = {"feature_length": torch.tensor([2])}

    out = model.generate(batch, condition=condition, num_denoise_steps=1)

    assert out["generated"][0].shape == (2, 2)
    assert model.recorded["condition"]["text_context"] == ["cond"]
    assert model.recorded["condition"]["text_null_context"] == ["null"]
    assert model.recorded["condition"]["traj_emb"] is traj_emb
    assert model.recorded["condition"]["traj_seq_lens"] is traj_seq_lens
    assert model.recorded["condition"]["traj_token_mask"] is traj_token_mask
    assert model.recorded["condition"]["seq_len"] == 3
