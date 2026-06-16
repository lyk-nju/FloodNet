"""7D-path tests that need real data / VAE — RUN ON THE DATA MACHINE.

These are skipped automatically when the HumanML3D data dir is absent (e.g. on a
dev box). On a machine with data they verify the parts of the flag-gated 4D→7D
migration that cannot be unit-tested without data:

  - the dataset actually emits `traj_cond_7d` [T,7] in 7D mode (T_B_09 __getitem__).

Run with:  ./scripts/run_pytest.sh tests/test_7d_datamachine.py -v

Manual runtime checks NOT auto-tested here (verify by launching a short 7D
fine-tune; see docs/TODO.md T_B_11):
  - step_460000.ckpt loads into the 7D model via the T_B_08 hook (no shape error);
  - one SF step: total loss + body_aux/* (root_xz/root_y/heading/fwd_delta/
    yaw_delta) are finite and decreasing;
  - anchor_canonicalize/valid_frac ~ 1.0 on normal-length clips;
  - history_corruption/applied and stream_training/runtime_horizon_tokens log sane values;
  - traj_encoder_in_dim=7 with body_aux_loss.enabled=false raises at startup.
"""

from __future__ import annotations

import os
import random

import pytest
import torch

from torch.utils.data import DataLoader
from utils.token_frame import num_tokens_for_frame_len

_CFG_PATH = "configs/ldf.yaml"


def _resolve_raw_data_dir() -> str | None:
    """B-P1-1: locate the HumanML3D data root. Priority:
      1. $FLOODNET_RAW_DATA_DIR (override on any machine);
      2. the config's resolved dirs.raw_data.
    Returns None if neither yields an existing HumanML3D/train.txt.
    """
    candidates = []
    env = os.environ.get("FLOODNET_RAW_DATA_DIR")
    if env:
        candidates.append(env)
    try:
        from utils.initialize import Config

        candidates.append(str(Config(_CFG_PATH).config.dirs.raw_data))
    except Exception:
        pass
    for root in candidates:
        if os.path.exists(os.path.join(root, "HumanML3D", "train.txt")):
            return root
    return None


_RAW_DIR = _resolve_raw_data_dir()

pytestmark = pytest.mark.skipif(
    _RAW_DIR is None,
    reason="HumanML3D data absent (set FLOODNET_RAW_DATA_DIR; run on data machine)",
)


def _make_cfg(traj_feat_dim: int):
    """Build a resolved ldf cfg pinned to <raw_dir>/HumanML3D/{train,val}.txt
    (NOT the config's train_difficult.txt, which may not exist on this host)."""
    from omegaconf import OmegaConf

    from utils.initialize import Config

    cfg = OmegaConf.create(OmegaConf.to_container(Config(_CFG_PATH).config, resolve=True))
    hml = os.path.join(_RAW_DIR, "HumanML3D")
    OmegaConf.update(cfg, "dirs.raw_data", _RAW_DIR)
    OmegaConf.update(cfg, "data.train_meta_paths", [os.path.join(hml, "train.txt")])
    OmegaConf.update(cfg, "data.val_meta_paths", [os.path.join(hml, "val.txt")])
    OmegaConf.update(cfg, "data.traj_feat_dim", traj_feat_dim)
    return cfg


class _RecordingOnlineVAE:
    def __init__(self, latent_dim: int = 4):
        self.latent_dim = int(latent_dim)
        self.encode_inputs = []

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.encode_inputs.append(x.detach().clone())
        n_tokens = num_tokens_for_frame_len(int(x.shape[1]))
        return x.new_zeros(x.shape[0], n_tokens, self.latent_dim)


class _RecordingLDFModel:
    def __init__(self):
        from models.tools.traj_encoder import LocalTrajEncoder, TrajEncoder

        self.training = True
        self.use_text_cond = False
        self.text_dropout = 0.0
        self.traj_dropout = 0.0
        self.param_dtype = torch.float32
        self.chunk_size = 5
        self.time_embedding_scale = 1.0
        self.prediction_type = "vel"
        self.local_traj_encoder = LocalTrajEncoder(in_dim=7)
        self.traj_encoder = TrajEncoder(out_dim=128)
        self.controlnet_calls = []
        self.backbone_calls = []
        self.model = self._backbone_forward

    def __call__(
        self,
        noisy_input,
        t_scaled,
        text_context,
        seq_len,
        *,
        traj_emb=None,
        traj_seq_lens=None,
        traj_token_mask=None,
        controlnet_residuals=None,
    ):
        if controlnet_residuals is None:
            controlnet_residuals = self._controlnet_forward(
                noisy_input,
                t_scaled,
                text_context,
                seq_len,
                traj_emb,
                traj_seq_lens,
                traj_token_mask=traj_token_mask,
            )
        return self.model(
            noisy_input,
            t_scaled,
            text_context,
            seq_len,
            y=None,
            traj_emb=None,
            traj_seq_lens=None,
            controlnet_residuals=controlnet_residuals,
        )

    def encode_text_with_cache(self, text_list, device):
        return [torch.zeros(1, 4, device=device) for _ in text_list]

    def _get_noise_levels(self, device, seq_len, time_steps):
        return torch.zeros(time_steps.shape[0], seq_len, device=device)

    def add_noise(self, clean_feature, noise_level):
        return clean_feature, torch.zeros_like(clean_feature)

    def preprocess(self, x):
        return x.permute(0, 2, 1)[:, :, :, None, None]

    def _controlnet_forward(
        self,
        noisy_input,
        t_scaled,
        text_context,
        seq_len,
        traj_emb,
        traj_seq_lens,
        traj_token_mask=None,
    ):
        self.controlnet_calls.append(
            {
                "noisy_input_lengths": [int(item.shape[1]) for item in noisy_input],
                "seq_len": int(seq_len),
                "traj_emb_shape": tuple(traj_emb.shape),
                "traj_seq_lens": traj_seq_lens.detach().cpu().clone(),
                "traj_token_mask_shape": tuple(traj_token_mask.shape),
                "traj_token_mask": traj_token_mask.detach().cpu().clone(),
            }
        )
        return [traj_emb.new_zeros(len(noisy_input), seq_len, 1)]

    def _backbone_forward(
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
        self.backbone_calls.append({"seq_len": int(seq_len)})
        return [torch.zeros_like(item) for item in noisy_input]


def test_7d_dataset_emits_traj_cond_7d():
    from datasets.humanml3d import HumanML3DDataset

    ds = HumanML3DDataset(_make_cfg(7), split="train")
    assert len(ds) > 0
    item = ds[0]
    assert "traj_cond_7d" in item, "7D dataset must emit traj_cond_7d"
    arr = item["traj_cond_7d"]
    assert arr.shape[-1] == 7
    # first-frame fwd_delta / yaw_delta are zero (root_to_traj_feats_7d v1 rule)
    assert abs(float(arr[0, 5])) < 1e-4
    assert abs(float(arr[0, 6])) < 1e-4


def test_4d_default_dataset_has_no_traj_cond_7d():
    from datasets.humanml3d import HumanML3DDataset

    ds = HumanML3DDataset(_make_cfg(4), split="train")
    item = ds[0]
    assert "traj_cond_7d" not in item            # 4D path unchanged
    assert "traj_features" in item               # legacy 4D feature still emitted


def test_stream_training_bs8_contract_exposes_horizon_traj_to_controlnet():
    from datasets.humanml3d import HumanML3DDataset, collate_fn
    from omegaconf import OmegaConf
    from utils.training.ldf.conditioning import prepare_condition
    from utils.training.ldf.model_step import run_training_window
    from utils.training.ldf.sample_creator import SampleCreator

    random.seed(1234)
    torch.manual_seed(1234)

    cfg = _make_cfg(7)
    hml = os.path.join(_RAW_DIR, "HumanML3D")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.update(cfg, "debug", True)
    OmegaConf.update(cfg, "data.train_meta_paths", [os.path.join(hml, "train.txt")])
    if not os.path.isdir(os.path.join(hml, str(cfg.data.get("token_path", "")))):
        OmegaConf.update(cfg, "data.token_path", None)

    dataset = HumanML3DDataset(cfg, split="train")
    if len(dataset) < 8:
        pytest.skip(f"need at least 8 HumanML3D train samples, got {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    raw_batch = next(iter(loader))

    latent_dim = (
        int(raw_batch["token"].shape[-1])
        if "token" in raw_batch and torch.is_tensor(raw_batch["token"])
        else 4
    )
    vae = _RecordingOnlineVAE(latent_dim=latent_dim)
    creator = SampleCreator(
        stream_enabled=True,
        context_tokens=int(cfg.stream_training.context_tokens),
        window_sampling=OmegaConf.to_container(
            cfg.stream_training.window_sampling,
            resolve=True,
        ),
        chunk_size=5,
        rollout_span=4,
    )
    model_batch = creator.create(raw_batch, vae=vae)

    latent_lengths = model_batch["feature_length"].to(dtype=torch.long)
    horizons = model_batch["_window_sampling_horizon_tokens"].to(dtype=torch.long)
    traj_tokens = model_batch["traj_num_tokens"].to(dtype=torch.long)
    latent_pad_len = int(model_batch["feature"].shape[1])
    traj_pad_len = int(traj_tokens.max().item())

    assert int(model_batch["feature"].shape[0]) == 8
    assert torch.equal(traj_tokens, latent_lengths + horizons)
    assert latent_pad_len == int(latent_lengths.max().item())
    assert torch.equal(model_batch["token_length"].to(dtype=torch.long), latent_lengths)
    for b, encoded_window in enumerate(vae.encode_inputs):
        assert int(encoded_window.shape[0]) == 1
        assert num_tokens_for_frame_len(int(encoded_window.shape[1])) == int(
            latent_lengths[b].item()
        )

    model = _RecordingLDFModel()
    condition = prepare_condition(
        model,
        model_batch,
        latent_pad_len,
        model_batch["feature"].device,
    )
    assert condition.traj_emb.shape[:2] == (8, traj_pad_len)
    assert tuple(condition.traj_token_mask.shape) == (8, traj_pad_len)
    assert torch.equal(condition.traj_seq_lens.cpu(), traj_tokens.cpu())

    run_training_window(
        model,
        model_batch,
        clean_feature=model_batch["feature"],
        time_steps=torch.full((8,), 999.0, device=model_batch["feature"].device),
        condition=condition,
    )

    assert len(model.controlnet_calls) == 1
    call = model.controlnet_calls[0]
    assert call["seq_len"] == latent_pad_len
    assert call["traj_emb_shape"][:2] == (8, traj_pad_len)
    assert torch.equal(call["traj_seq_lens"], traj_tokens.cpu())
    assert call["traj_token_mask_shape"] == (8, traj_pad_len)
    assert call["noisy_input_lengths"] == [int(v.item()) for v in latent_lengths]
    for b in range(8):
        valid = int(traj_tokens[b].item())
        assert bool(call["traj_token_mask"][b, :valid].all())
        assert not bool(call["traj_token_mask"][b, valid:].any())
