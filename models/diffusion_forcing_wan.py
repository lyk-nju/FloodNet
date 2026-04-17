import warnings

import numpy as np
import torch
import torch.nn as nn

from .tools.t5 import T5EncoderModel
from utils.traj_batch import root_to_traj_feats
from .tools.traj_encoder import LocalTrajEncoder, TrajEncoder
from .tools.wan_model import WanModel
from .tools.wan_controlnet import WanControlNet


def _expand_precomputed_caption_keys(emb: dict) -> dict:
    """Alias strip() keys so table matches HumanML3D captions after .strip()."""
    out = dict(emb)
    for k, v in emb.items():
        s = k.strip()
        if s not in out:
            out[s] = v
    return out


class DiffForcingWanModel(nn.Module):
    def __init__(
        self,
        checkpoint_path="deps/t5_umt5-xxl-enc-bf16/models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_path="deps/t5_umt5-xxl-enc-bf16/google/umt5-xxl",
        input_dim=256,
        hidden_dim=1024,
        ffn_dim=2048,
        freq_dim=256,
        num_heads=8,
        num_layers=8,
        time_embedding_scale=1.0,
        chunk_size=5,
        noise_steps=10,
        use_text_cond=True,
        text_len=512,
        dropout=0.1,
        cfg_scale_text=5.0,
        cfg_scale_traj=0.0,
        prediction_type="vel",  # "vel", "x0", "noise"
        causal=False,
        # Deprecated for ControlNet training: backbone stays unconditional on traj.
        # Keep for backwards-compat only (FlexTraj mode when controlnet disabled).
        use_traj_cond=False,
        traj_out_dim=64,
        traj_in_dim=4,
        traj_encoder_in_dim=None,
        traj_dropout=0.1,
        use_traj_emb_cache=False,
        use_traj_kv_cache=None,
        control_loss_weight=1.0,  # used by train_ldf, not by model
        freeze_backbone_for_traj=False,
        use_controlnet_traj=False,
        controlnet_init_from_backbone=True,
        freeze_backbone_for_controlnet=False,
        use_precomputed_text_emb=False,
        precomputed_text_emb_path=None,
        scheduled_sampling_prob=0.0,
        **kwargs,
    ):
        kwargs.pop("traj_lora_rank", None)
        kwargs.pop("lora_rank_traj", None)
        # Backward compat: old checkpoints stored the param as cfg_scale.
        _legacy_cfg_scale = kwargs.pop("cfg_scale", None)
        if _legacy_cfg_scale is not None:
            cfg_scale_text = _legacy_cfg_scale
        if kwargs:
            raise TypeError(
                "DiffForcingWanModel: unexpected keyword arguments "
                f"{sorted(kwargs.keys())}"
            )
        super().__init__()
        if traj_encoder_in_dim is not None:
            traj_in_dim = traj_encoder_in_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.time_embedding_scale = time_embedding_scale
        self.chunk_size = chunk_size
        self.noise_steps = noise_steps
        self.use_text_cond = use_text_cond
        self.dropout = dropout
        self.cfg_scale_text = cfg_scale_text
        self.cfg_scale_traj = float(cfg_scale_traj)
        self.prediction_type = prediction_type
        self.causal = causal
        self.use_traj_cond = use_traj_cond
        self.traj_out_dim = traj_out_dim
        self.traj_in_dim = traj_in_dim
        self.traj_dropout = traj_dropout
        self.freeze_backbone_for_traj = freeze_backbone_for_traj
        self.use_controlnet_traj = bool(use_controlnet_traj)
        self.controlnet_init_from_backbone = bool(controlnet_init_from_backbone)
        self.freeze_backbone_for_controlnet = bool(freeze_backbone_for_controlnet)
        if self.use_controlnet_traj and use_traj_cond:
            warnings.warn(
                "use_traj_cond=True is ignored when use_controlnet_traj=True. "
                "Backbone stays unconditional; traj is consumed by ControlNet only.",
                stacklevel=2,
            )
        if self.use_controlnet_traj and freeze_backbone_for_traj:
            warnings.warn(
                "freeze_backbone_for_traj=True is a legacy FlexTraj setting and is ignored "
                "when use_controlnet_traj=True. Use freeze_backbone_for_controlnet instead.",
                stacklevel=2,
            )
        if use_traj_kv_cache is not None:
            warnings.warn(
                "`use_traj_kv_cache` is deprecated; use `use_traj_emb_cache`.",
                stacklevel=2,
            )
            use_traj_emb_cache = bool(use_traj_kv_cache)
        self.use_traj_emb_cache = use_traj_emb_cache
        self.scheduled_sampling_prob = float(scheduled_sampling_prob)

        self.text_dim = 4096
        self.text_len = text_len
        self.use_precomputed_text_emb = bool(use_precomputed_text_emb)
        self._precomputed_text_emb = None
        self.text_encoder = None

        if self.use_precomputed_text_emb:
            if not precomputed_text_emb_path:
                raise ValueError(
                    "use_precomputed_text_emb=True requires precomputed_text_emb_path "
                    "(run pretokenize_t5_text.py to build the .pt)."
                )
            blob = torch.load(precomputed_text_emb_path, map_location="cpu", weights_only=False)
            self._precomputed_text_emb = _expand_precomputed_caption_keys(blob["embeddings"])
            if "" not in self._precomputed_text_emb:
                raise KeyError(
                    'precomputed embeddings must include empty string key "" for CFG / dropout.'
                )
            td = int(blob.get("text_dim", self.text_dim))
            if td != self.text_dim:
                raise ValueError(f"precomputed text_dim {td} != model text_dim {self.text_dim}")
        else:
            self.text_encoder = T5EncoderModel(
                text_len=self.text_len,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                checkpoint_path=checkpoint_path,
                tokenizer_path=tokenizer_path,
                shard_fn=None,
            )

        # Text encoding cache (only used when running live T5)
        self.text_cache = {}
        # Default (recommended) behavior: backbone matches original FloodDiffusion (no traj tokens),
        # ControlNet is the ONLY consumer of trajectory condition and injects residuals into backbone.
        traj_enc_dim_backbone = 0
        traj_enc_dim_controlnet = self.traj_out_dim if self.use_controlnet_traj else 0
        self.model = WanModel(
            model_type="t2v",
            patch_size=(1, 1, 1),
            text_len=self.text_len,
            in_dim=self.input_dim,
            dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            freq_dim=self.freq_dim,
            text_dim=self.text_dim,
            out_dim=self.input_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
            causal=self.causal,
            traj_enc_dim=traj_enc_dim_backbone,
        )

        self.controlnet = None
        if self.use_controlnet_traj:
            self.controlnet = WanControlNet(
                model_type="t2v",
                patch_size=(1, 1, 1),
                text_len=self.text_len,
                in_dim=self.input_dim,
                dim=self.hidden_dim,
                ffn_dim=self.ffn_dim,
                freq_dim=self.freq_dim,
                text_dim=self.text_dim,
                out_dim=self.input_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=True,
                eps=1e-6,
                causal=self.causal,
                traj_enc_dim=traj_enc_dim_controlnet,
            )
            if self.controlnet_init_from_backbone:
                self.controlnet.init_from_backbone(self.model)

        # Trajectory encoder is only needed for ControlNet conditioning (recommended path).
        # FlexTraj mode (backbone_use_traj) is kept as legacy and should not be mixed with ControlNet.
        if self.use_controlnet_traj or (self.use_traj_cond and not self.use_controlnet_traj):
            # Local within-token encoder: compress masked 4-frame traj feats -> token-level 4D.
            self.local_traj_encoder = LocalTrajEncoder(hidden_dim=32)
            self.traj_encoder = TrajEncoder(
                in_dim=self.traj_in_dim, hidden_dim=64, out_dim=self.traj_out_dim
            )
        else:
            self.local_traj_encoder = None
            self.traj_encoder = None
        self.param_dtype = torch.float32
        self._traj_stream_version = 0
        self._traj_emb_cache = {}

        if freeze_backbone_for_traj:
            trainable_substrings = (
                "traj_in_proj",
                "traj_type_embed",
            )
            for name, p in self.model.named_parameters():
                if any(s in name for s in trainable_substrings):
                    continue
                p.requires_grad = False

            if self.traj_encoder is not None:
                for p in self.traj_encoder.parameters():
                    p.requires_grad = True

        if self.freeze_backbone_for_controlnet:
            for p in self.model.parameters():
                p.requires_grad = False
            if self.controlnet is not None:
                for p in self.controlnet.parameters():
                    p.requires_grad = True
            if self.traj_encoder is not None:
                for p in self.traj_encoder.parameters():
                    p.requires_grad = True

    def _controlnet_forward(
        self,
        noisy_input,
        t_scaled,
        text_context,
        seq_len,
        traj_emb,
        traj_seq_lens,
    ):
        if self.controlnet is None:
            return None
        return self.controlnet(
            noisy_input,
            t_scaled,
            text_context,
            seq_len,
            y=None,
            traj_emb=traj_emb,
            traj_seq_lens=traj_seq_lens,
        )

    def _build_cfg_2b_text(
        self, text_context, text_null_per_sample, batch_size, model_seq_len
    ):
        """Build text context list for MotionLCM-style 2B CFG (cond || uncond).

        WanModel/WanControlNet accept either one global caption per sample (len == B)
        or frame-aligned captions (len == B * model_seq_len). Returns None if the
        layout does not match either (caller falls back to two backbone forwards).
        """
        b = batch_size
        n = len(text_context)
        if n == b:
            return list(text_context) + list(text_null_per_sample)
        if n == b * model_seq_len:
            null_flat = []
            for i in range(b):
                ni = text_null_per_sample[i]
                for _ in range(model_seq_len):
                    null_flat.append(ni)
            return list(text_context) + null_flat
        return None

    def _uncond_backbone_forward(
        self, noisy_input, t_scaled, text_null_context, seq_len,
        traj_emb_backbone, traj_seq_lens_backbone,
    ):
        """Single backbone forward with null text and no ControlNet residuals.

        Used by separated CFG to obtain the fully unconditional prediction
        (text OFF, traj OFF).  Returns a list of per-sample tensors.
        """
        return self.model(
            noisy_input,
            t_scaled,
            text_null_context,
            seq_len,
            y=None,
            traj_emb=traj_emb_backbone,
            traj_seq_lens=traj_seq_lens_backbone,
            controlnet_residuals=None,
        )

    def encode_text_with_cache(self, text_list, device):
        """Encode text using cache
        Args:
            text_list: List[str], list of texts
            device: torch.device
        Returns:
            List[Tensor]: List of encoded text features
        """
        if self._precomputed_text_emb is not None:
            out = []
            d = self._precomputed_text_emb
            for text in text_list:
                row = d.get(text)
                if row is None:
                    row = d.get(text.strip())
                if row is None:
                    preview = text.replace("\n", "\\n")
                    if len(preview) > 160:
                        preview = preview[:157] + "..."
                    raise KeyError(
                        "Caption not in precomputed T5 table. "
                        f"len={len(text)} preview={preview!r}. "
                        "Re-run pretokenize_t5_text.py with the same config "
                        "(include val/test meta paths), or set use_precomputed_text_emb=false."
                    )
                out.append(row.to(device))
            return out

        text_features = []
        indices_to_encode = []
        texts_to_encode = []

        # Check cache
        for i, text in enumerate(text_list):
            if text in self.text_cache:
                # Get from cache and move to correct device
                cached_feature = self.text_cache[text].to(device)
                text_features.append(cached_feature)
            else:
                # Need to encode
                text_features.append(None)
                indices_to_encode.append(i)
                texts_to_encode.append(text)

        # Batch encode uncached texts
        if texts_to_encode:
            self.text_encoder.model.to(device)
            encoded = self.text_encoder(texts_to_encode, device)

            # Store in cache and update results
            for idx, text, feature in zip(indices_to_encode, texts_to_encode, encoded):
                # Cache to CPU to save GPU memory
                self.text_cache[text] = feature.cpu()
                text_features[idx] = feature

        return text_features

    def preprocess(self, x):
        # (bs, T, C) -> (bs, C, T, 1, 1)
        x = x.permute(0, 2, 1)[:, :, :, None, None]
        return x

    def postprocess(self, x):
        # (bs, C, T, 1, 1) ->  (bs, T, C)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(x.size(0), x.size(2), -1)
        return x

    def _build_traj_emb(self, x, seq_len, device, training_dropout=False):
        # Standard ControlNet conditioning: build token-level traj features from frame-level traj_features
        # using a local within-token encoder over 4 frames, then apply TrajEncoder.
        if (not (self.use_controlnet_traj or self.use_traj_cond)) or self.traj_encoder is None:
            return None
        if training_dropout and np.random.rand() <= self.traj_dropout:
            return None

        # Prefer frame-level traj_features if present; else derive from traj xyz.
        if "traj_features" in x and x["traj_features"] is not None:
            feats_frame = x["traj_features"].to(device)  # (B, T_frame, 4)
        elif "traj" in x and x["traj"] is not None:
            feats_frame = root_to_traj_feats(x["traj"].to(device))
        else:
            return None

        # Frame-level mask.
        mask_frame = None
        if "traj_mask" in x and x["traj_mask"] is not None:
            mask_frame = x["traj_mask"].to(device=device, dtype=torch.float32)
        elif "token_mask" in x and x["token_mask"] is not None:
            # Build frame mask with causal VAE convention:
            #   token 0 → frame 0; token k (k≥1) → frames [4k-3, 4k]
            tm = x["token_mask"].to(device=device, dtype=torch.float32)
            B_tm, N_tm = tm.shape
            tf = feats_frame.shape[1]
            mask_frame = tm.new_zeros(B_tm, tf)
            mask_frame[:, 0] = tm[:, 0]
            for k in range(1, N_tm):
                sf = 4 * k - 3
                ef = min(4 * k + 1, tf)
                if sf < tf:
                    mask_frame[:, sf:ef] = tm[:, k : k + 1].expand(-1, ef - sf)
        if mask_frame is not None:
            tf = feats_frame.shape[1]
            if mask_frame.shape[1] < tf:
                pad = mask_frame.new_zeros(mask_frame.shape[0], tf - mask_frame.shape[1])
                mask_frame = torch.cat([mask_frame, pad], dim=1)
            mask_frame = mask_frame[:, :tf]
            feats_frame = feats_frame * mask_frame.unsqueeze(-1).to(dtype=feats_frame.dtype)

        # If traj_features is already token-level (B, seq_len, 4), skip local 4-frame encoder.
        # Otherwise, treat it as frame-level (B, T_frame, 4) and compress 4 frames per token.
        if feats_frame.shape[1] == seq_len:
            feats_tok = feats_frame
        else:
            # Causal VAE convention: N tokens → 4*(N-1)+1 frames
            #   token 0 → frame 0 only
            #   token k (k≥1) → frames [4k-3, 4k]
            total_causal = 4 * (seq_len - 1) + 1 if seq_len > 1 else 1
            tf = feats_frame.shape[1]
            if tf < total_causal:
                pad = feats_frame.new_zeros(
                    feats_frame.shape[0], total_causal - tf, feats_frame.shape[2]
                )
                feats_frame = torch.cat([feats_frame, pad], dim=1)
            feats_frame = feats_frame[:, :total_causal, :]
            # Token 0: repeat frame 0 four times (causal seed frame)
            tok0 = feats_frame[:, 0:1, :].unsqueeze(2).expand(-1, -1, 4, -1)  # (B,1,4,C)
            if seq_len > 1:
                # Tokens 1..seq_len-1: frames [1, 4*(seq_len-1)] in consecutive groups of 4
                rest = feats_frame[:, 1:, :].reshape(
                    feats_frame.shape[0], seq_len - 1, 4, feats_frame.shape[2]
                )  # (B, seq_len-1, 4, C)
                feats_4 = torch.cat([tok0, rest], dim=1)  # (B, seq_len, 4, C)
            else:
                feats_4 = tok0  # (B, 1, 4, C)
            if self.local_traj_encoder is None:
                return None
            feats_tok = self.local_traj_encoder(feats_4)  # (B, seq_len, 4)

        # Token-level mask (gate at token granularity).
        if "token_mask" in x and x["token_mask"] is not None:
            tm = x["token_mask"].to(device=device, dtype=torch.float32)
            if tm.shape[1] < seq_len:
                pad = tm.new_zeros(tm.shape[0], seq_len - tm.shape[1])
                tm = torch.cat([tm, pad], dim=1)
            tm = tm[:, :seq_len]
            feats_tok = feats_tok * tm.unsqueeze(-1).to(dtype=feats_tok.dtype)

        return self.traj_encoder(feats_tok)

    def _get_traj_seq_lens(self, x, seq_len, device):
        if "traj_features_length" in x and x["traj_features_length"] is not None:
            return (
                x["traj_features_length"]
                .to(device=device, dtype=torch.long)
                .clamp(min=0, max=seq_len)
            )
        if "traj_length" in x and x["traj_length"] is not None:
            # Causal: N tokens → 4*(N-1)+1 frames, so frames→tokens = (T-1)//4 + 1
            tl = x["traj_length"].to(device=device, dtype=torch.long)
            return ((tl - 1) // 4 + 1).clamp(min=0, max=seq_len)
        return None

    def _get_noise_levels(self, device, seq_len, time_steps):
        """Get noise levels (Paper: vectorized schedule).
        β^k_t = 1 - α^k_t, with α^k_t = clamp(t - k/n_s, 0, 1). So β^k_t = clamp(1 + k/n_s - t, 0, 1).
        chunk_size = n_s (streaming step-size). Left of active window → β≈0 (clean), right → β≈1 (noise).
        """
        # noise_level[k] = β^k_t for position k
        noise_level = torch.clamp(
            1
            + torch.arange(seq_len, device=device) / self.chunk_size
            - time_steps.unsqueeze(1),
            min=0.0,
            max=1.0,
        )
        return noise_level

    def add_noise(self, x, noise_level):
        """Add noise (Paper Eq.: x_t = α_t ⊙ z + β_t ⊙ ε).
        Args:
            x: (B, T, D) clean latent z
            noise_level: (B, T) β_t
        """
        noise = torch.randn_like(x)
        noise_level = noise_level.unsqueeze(-1)
        noisy_x = x * (1 - noise_level) + noise_level * noise  # α*z + β*ε
        return noisy_x, noise

    def forward(self, x):
        feature = x["feature"]  # (B, T, C)
        feature_length = x["feature_length"]  # (B,)
        batch_size, seq_len, _ = feature.shape
        device = feature.device

        # Randomly use a time step
        time_steps = []
        for i in range(batch_size):
            valid_len = feature_length[i].item()
            # Random float from 0 to valid_len/chunk_size, not an integer
            max_time = valid_len / self.chunk_size
            # max_time = valid_len / self.chunk_size + 1
            time_steps.append(torch.FloatTensor(1).uniform_(0, max_time).item())
        time_steps = torch.tensor(time_steps, device=device)  # (B,)
        noise_level = self._get_noise_levels(device, seq_len, time_steps)  # (B, T)

        # Add noise to entire sequence
        noisy_feature, noise = self.add_noise(feature, noise_level)  # (B, T, D)

        feature = self.preprocess(feature)  # (B, C, T, 1, 1)
        noisy_feature = self.preprocess(noisy_feature)  # (B, C, T, 1, 1)
        noise = self.preprocess(noise)  # (B, C, T, 1, 1)

        feature_ref = []
        noise_ref = []
        noisy_feature_input = []
        for i in range(batch_size):
            t = time_steps[i].item()
            end_index = int(self.chunk_size * t) + 1
            valid_len = feature_length[i].item()
            end_index = min(valid_len, end_index)
            feature_ref.append(feature[i, :, :end_index, ...])
            noise_ref.append(noise[i, :, :end_index, ...])
            noisy_feature_input.append(noisy_feature[i, :, :end_index, ...])

        # Encode text condition (using cache)
        if self.use_text_cond and "text" in x:
            text_list = x["text"]  # List[str] or List[List[str]]
            if isinstance(text_list[0], list):
                text_end_list = x["feature_text_end"]
                all_text_context = []
                for single_text_list, single_text_end_list in zip(
                    text_list, text_end_list
                ):
                    # Paper: classifier-free guidance — with prob dropout replace text by "" at train time
                    if np.random.rand() > self.dropout:
                        single_text_end_list = [0] + [
                            min(t, seq_len) for t in single_text_end_list
                        ]
                    else:
                        single_text_list = [""]
                        single_text_end_list = [0, seq_len]
                    single_text_length_list = [
                        t - b
                        for t, b in zip(
                            single_text_end_list[1:], single_text_end_list[:-1]
                        )
                    ]
                    single_text_context = self.encode_text_with_cache(
                        single_text_list, device
                    )
                    single_text_context = [
                        u.to(self.param_dtype) for u in single_text_context
                    ]
                    # Paper: frame-wise text conditioning — one embedding per frame so each motion frame attends only to "the text prompt active at that time"
                    for u, duration in zip(
                        single_text_context, single_text_length_list
                    ):
                        all_text_context.extend([u for _ in range(duration)])
                    all_text_context.extend(
                        [
                            single_text_context[-1]
                            for _ in range(seq_len - single_text_end_list[-1])
                        ]
                    )
            else:
                all_text_context = [
                    (u if np.random.rand() > self.dropout else "") for u in text_list
                ]
                all_text_context = self.encode_text_with_cache(all_text_context, device)
                all_text_context = [u.to(self.param_dtype) for u in all_text_context]
        else:
            all_text_context = [""] * batch_size
            all_text_context = self.encode_text_with_cache(all_text_context, device)
            all_text_context = [u.to(self.param_dtype) for u in all_text_context]

        traj_emb = None
        traj_seq_lens = None
        if self.use_controlnet_traj or self.use_traj_cond:
            # If only traj/control branch is trainable, dropping traj condition can make loss
            # independent of all trainable params and break backward.
            traj_training_dropout = not (
                self.freeze_backbone_for_traj or self.freeze_backbone_for_controlnet
            )
            traj_emb = self._build_traj_emb(
                x, seq_len, device, training_dropout=traj_training_dropout
            )
            traj_seq_lens = self._get_traj_seq_lens(x, seq_len, device)

        # Backbone stays unconditional on traj in standard ControlNet mode.
        traj_emb_backbone = traj_emb if (self.use_traj_cond and not self.use_controlnet_traj) else None
        traj_seq_lens_backbone = traj_seq_lens if (self.use_traj_cond and not self.use_controlnet_traj) else None

        # ── Scheduled Sampling ──────────────────────────────────────────────
        # With probability ss_prob, replace context tokens (active-window prefix,
        # noise_level ≈ 0) with the model's own x0 prediction (no_grad pass).
        # This exposes the model to its own prediction errors during training,
        # closing the teacher-forcing → generate exposure-bias gap.
        if self.scheduled_sampling_prob > 0.0 and np.random.rand() < self.scheduled_sampling_prob:
            with torch.no_grad():
                cn_res_ss = self._controlnet_forward(
                    noisy_feature_input,
                    noise_level * self.time_embedding_scale,
                    all_text_context,
                    seq_len,
                    traj_emb,
                    traj_seq_lens,
                )
                pred_ss = self.model(
                    noisy_feature_input,
                    noise_level * self.time_embedding_scale,
                    all_text_context,
                    seq_len,
                    y=None,
                    traj_emb=traj_emb_backbone,
                    traj_seq_lens=traj_seq_lens_backbone,
                    controlnet_residuals=cn_res_ss,
                )
            # Replace context tokens (prefix before active window) with x0_hat.
            for b in range(batch_size):
                t_len = noisy_feature_input[b].shape[1]  # end_index for this sample
                ctx_len = t_len - self.chunk_size        # tokens before active window
                if ctx_len <= 0:
                    continue
                if self.prediction_type == "vel":
                    # vel = x0 - noise  →  x0_hat = pred_vel + noise
                    x0_hat = (pred_ss[b] + noise_ref[b]).detach()
                elif self.prediction_type == "x0":
                    x0_hat = pred_ss[b].detach()
                else:
                    continue  # "noise" type: skip SS for this sample
                noisy_feature_input[b] = torch.cat(
                    [x0_hat[:, :ctx_len], noisy_feature_input[b][:, ctx_len:]], dim=1
                )
        # ────────────────────────────────────────────────────────────────────

        # Through WanModel
        controlnet_residuals = self._controlnet_forward(
            noisy_feature_input,
            noise_level * self.time_embedding_scale,
            all_text_context,
            seq_len,
            traj_emb,
            traj_seq_lens,
        )
        predicted_result = self.model(
            noisy_feature_input,
            noise_level * self.time_embedding_scale,
            all_text_context,
            seq_len,
            y=None,
            traj_emb=traj_emb_backbone,
            traj_seq_lens=traj_seq_lens_backbone,
            controlnet_residuals=controlnet_residuals,
        )  # (B, C, T, 1, 1)

        loss = 0.0
        # Paper: only compute loss on active window [m(t), n(t)) (last chunk_size positions)
        for b in range(batch_size):
            if self.prediction_type == "vel":
                vel = feature_ref[b] - noise_ref[b]  # (C, input_length, 1, 1)
                squared_error = (
                    predicted_result[b][:, -self.chunk_size :, ...]
                    - vel[:, -self.chunk_size :, ...]
                ) ** 2
            elif self.prediction_type == "x0":
                squared_error = (
                    predicted_result[b][:, -self.chunk_size :, ...]
                    - feature_ref[b][:, -self.chunk_size :, ...]
                ) ** 2
            elif self.prediction_type == "noise":
                squared_error = (
                    predicted_result[b][:, -self.chunk_size :, ...]
                    - noise_ref[b][:, -self.chunk_size :, ...]
                ) ** 2
            sample_loss = squared_error.sum().mean()
            loss += sample_loss
        loss = loss / batch_size

        loss_dict = {"total": loss, "mse": loss}

        # MotionLCM-style control loss: prepare pred_x0_latent for decoding (train_ldf will add L_control)
        if ("traj" in x) and (self.prediction_type in ("vel", "x0")) and (
            self.use_traj_cond or self.use_controlnet_traj
        ):
            pred_x0_latent_list = []
            for b in range(batch_size):
                if self.prediction_type == "vel":
                    pred_x0 = predicted_result[b] + noise_ref[b]
                else:
                    pred_x0 = predicted_result[b]
                # (C, T, 1, 1) -> (T, C) for VAE decode
                p = pred_x0[:, :, 0, 0].permute(1, 0)
                pred_x0_latent_list.append(p)
            loss_dict["control_aux"] = {"pred_x0_latent_list": pred_x0_latent_list}

        return loss_dict

    def generate(self, x, num_denoise_steps=None):
        """
        Generation - Diffusion Forcing inference
        Uses triangular noise schedule, progressively generating from left to right

        Generation process:
        1. Start from t=0, gradually increase t
        2. Each t corresponds to a noise schedule: clean on left, noisy on right, gradient in middle
        3. After each denoising step, t increases slightly and continues
        """
        feature_length = x["feature_length"]
        batch_size = len(feature_length)
        seq_len = max(feature_length).item()

        if num_denoise_steps is None:
            num_denoise_steps = self.noise_steps
        assert num_denoise_steps % self.chunk_size == 0

        device = next(self.parameters()).device

        # Initialize entire sequence as pure noise
        generated = torch.randn(
            batch_size, seq_len + self.chunk_size, self.input_dim, device=device
        )
        generated = self.preprocess(generated)  # (B, C, T, 1, 1)

        # Calculate total number of time steps needed
        max_t = 1 + (seq_len - 1) / self.chunk_size

        # Step size for each advancement
        dt = 1 / num_denoise_steps
        total_steps = int(max_t / dt)

        # Encode text condition (using cache)
        if self.use_text_cond and "text" in x:
            text_list = x["text"]  # List[str] or List[List[str]]
            if isinstance(text_list[0], list):
                generated_length = []
                text_end_list = x["feature_text_end"]
                full_text = []
                all_text_context = []
                for single_text_list, single_text_end_list in zip(
                    text_list, text_end_list
                ):
                    single_text_end_list = [0] + [
                        min(t, seq_len) for t in single_text_end_list
                    ]
                    generated_length.append(single_text_end_list[-1])
                    single_text_length_list = [
                        t - b
                        for t, b in zip(
                            single_text_end_list[1:], single_text_end_list[:-1]
                        )
                    ]
                    full_text.append(
                        " ////////// ".join(
                            [
                                f"{u} //dur:{t}"
                                for u, t in zip(
                                    single_text_list, single_text_length_list
                                )
                            ]
                        )
                    )
                    single_text_context = self.encode_text_with_cache(
                        single_text_list, device
                    )
                    single_text_context = [
                        u.to(self.param_dtype) for u in single_text_context
                    ]
                    for u, duration in zip(
                        single_text_context, single_text_length_list
                    ):
                        all_text_context.extend([u for _ in range(duration)])
                    all_text_context.extend(
                        [
                            single_text_context[-1]
                            for _ in range(
                                seq_len + self.chunk_size - single_text_end_list[-1]
                            )
                        ]
                    )
            else:
                generated_length = feature_length
                full_text = text_list
                all_text_context = self.encode_text_with_cache(text_list, device)
                all_text_context = [u.to(self.param_dtype) for u in all_text_context]
        else:
            generated_length = feature_length
            full_text = [""] * batch_size
            all_text_context = [""] * batch_size
            all_text_context = self.encode_text_with_cache(all_text_context, device)
            all_text_context = [u.to(self.param_dtype) for u in all_text_context]

        # Get empty text condition encoding (for CFG)
        text_null_list = [""] * batch_size
        text_null_context = self.encode_text_with_cache(text_null_list, device)
        text_null_context = [u.to(self.param_dtype) for u in text_null_context]

        gen_seq_len = seq_len + self.chunk_size
        traj_emb = None
        traj_seq_lens = None
        if self.use_controlnet_traj or self.use_traj_cond:
            traj_emb = self._build_traj_emb(
                x, gen_seq_len, device, training_dropout=False
            )
            traj_seq_lens = self._get_traj_seq_lens(x, gen_seq_len, device)
        # ControlNet mode: backbone stays unconditional on traj; only ControlNet branch consumes it.
        traj_emb_backbone = traj_emb if (self.use_traj_cond and not self.use_controlnet_traj) else None
        traj_seq_lens_backbone = traj_seq_lens if (self.use_traj_cond and not self.use_controlnet_traj) else None

        # Progressively advance from t=0 to t=max_t
        for step in range(total_steps):
            # Current time step
            t = step * dt
            start_index = max(0, int(self.chunk_size * (t - 1)) + 1)
            end_index = int(self.chunk_size * t) + 1
            time_steps = torch.full((batch_size,), t, device=device)

            # Calculate current noise schedule
            noise_level = self._get_noise_levels(
                device, seq_len + self.chunk_size, time_steps
            )  # (B, T)

            # Predict noise through WanModel
            noisy_input = []
            for i in range(batch_size):
                noisy_input.append(generated[i, :, :end_index, ...])

            gen_sl = seq_len + self.chunk_size
            t_scaled = noise_level * self.time_embedding_scale
            ctx_2b = (
                self._build_cfg_2b_text(
                    all_text_context, text_null_context, batch_size, gen_sl
                )
                if self.cfg_scale_text != 1.0
                else None
            )
            if ctx_2b is not None:
                noisy_2b = list(noisy_input) + list(noisy_input)
                t_2b = torch.cat([t_scaled, t_scaled], dim=0)
                traj_cn_2b = (
                    torch.cat([traj_emb, traj_emb], dim=0)
                    if traj_emb is not None
                    else None
                )
                traj_sl_2b = (
                    torch.cat([traj_seq_lens, traj_seq_lens], dim=0)
                    if traj_seq_lens is not None
                    else None
                )
                traj_bb_2b = (
                    torch.cat([traj_emb_backbone, traj_emb_backbone], dim=0)
                    if traj_emb_backbone is not None
                    else None
                )
                traj_bb_sl_2b = (
                    torch.cat(
                        [traj_seq_lens_backbone, traj_seq_lens_backbone], dim=0
                    )
                    if traj_seq_lens_backbone is not None
                    else None
                )
                controlnet_residuals = self._controlnet_forward(
                    noisy_2b,
                    t_2b,
                    ctx_2b,
                    gen_sl,
                    traj_cn_2b,
                    traj_sl_2b,
                )
                pred_2b = self.model(
                    noisy_2b,
                    t_2b,
                    ctx_2b,
                    gen_sl,
                    y=None,
                    traj_emb=traj_bb_2b,
                    traj_seq_lens=traj_bb_sl_2b,
                    controlnet_residuals=controlnet_residuals,
                )
                if self.cfg_scale_traj > 0.0:
                    # Separated CFG: reuse 2-batch (out_full, out_null_text);
                    # one extra backbone forward for out_uncond (text OFF, traj OFF).
                    # Formula:
                    #   out = out_uncond
                    #       + w_text * (out_full - out_null_text)
                    #       + w_traj * (out_null_text - out_uncond)
                    pred_uncond = self._uncond_backbone_forward(
                        noisy_input, t_scaled, text_null_context, gen_sl,
                        traj_emb_backbone, traj_seq_lens_backbone,
                    )
                    predicted_result = [
                        pred_uncond[i]
                        + self.cfg_scale_text * (pred_2b[i] - pred_2b[i + batch_size])
                        + self.cfg_scale_traj * (pred_2b[i + batch_size] - pred_uncond[i])
                        for i in range(batch_size)
                    ]
                else:
                    predicted_result = [
                        self.cfg_scale_text * pred_2b[i]
                        - (self.cfg_scale_text - 1) * pred_2b[i + batch_size]
                        for i in range(batch_size)
                    ]
            else:
                controlnet_residuals = self._controlnet_forward(
                    noisy_input,
                    t_scaled,
                    all_text_context,
                    gen_sl,
                    traj_emb,
                    traj_seq_lens,
                )
                predicted_result = self.model(
                    noisy_input,
                    t_scaled,
                    all_text_context,
                    gen_sl,
                    y=None,
                    traj_emb=traj_emb_backbone,
                    traj_seq_lens=traj_seq_lens_backbone,
                    controlnet_residuals=controlnet_residuals,
                )
                if self.cfg_scale_text != 1.0:
                    # Re-compute ControlNet residuals with null text so the uncond
                    # branch is truly unconditioned on text (Bug fix: previously
                    # reused cond residuals, making CFG uncond branch not truly null).
                    controlnet_residuals_null = self._controlnet_forward(
                        noisy_input,
                        t_scaled,
                        text_null_context,
                        gen_sl,
                        traj_emb,
                        traj_seq_lens,
                    )
                    predicted_result_null = self.model(
                        noisy_input,
                        t_scaled,
                        text_null_context,
                        gen_sl,
                        y=None,
                        traj_emb=traj_emb_backbone,
                        traj_seq_lens=traj_seq_lens_backbone,
                        controlnet_residuals=controlnet_residuals_null,
                    )
                    predicted_result = [
                        self.cfg_scale_text * pv - (self.cfg_scale_text - 1) * pvn
                        for pv, pvn in zip(predicted_result, predicted_result_null)
                    ]

            for i in range(batch_size):
                predicted_result_i = predicted_result[i]  # (C, input_length, 1, 1)
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, start_index:end_index, ...]
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "x0":
                    predicted_vel = (
                        predicted_result_i[:, start_index:end_index, ...]
                        - generated[i, :, start_index:end_index, ...]
                    ) / (
                        noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "noise":
                    predicted_vel = (
                        generated[i, :, start_index:end_index, ...]
                        - predicted_result_i[:, start_index:end_index, ...]
                    ) / (
                        1
                        + dt
                        - noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt

        generated = self.postprocess(generated)  # (B, T, C)
        y_hat_out = []
        for i in range(batch_size):
            # cut off the padding
            single_generated = generated[i, : generated_length[i], :]
            y_hat_out.append(single_generated)
        out = {}
        out["generated"] = y_hat_out
        out["text"] = full_text

        return out

    @torch.no_grad()
    def stream_generate(self, x, num_denoise_steps=None):
        """
        Streaming generation - Diffusion Forcing inference
        Uses triangular noise schedule, progressively generating from left to right

        Generation process:
        1. Start from t=0, gradually increase t
        2. Each t corresponds to a noise schedule: clean on left, noisy on right, gradient in middle
        3. After each denoising step, t increases slightly and continues
        """
        feature_length = x["feature_length"]
        batch_size = len(feature_length)
        seq_len = max(feature_length).item()

        if num_denoise_steps is None:
            num_denoise_steps = self.noise_steps
        assert num_denoise_steps % self.chunk_size == 0

        device = next(self.parameters()).device

        # Initialize entire sequence as pure noise
        generated = torch.randn(
            batch_size, seq_len + self.chunk_size, self.input_dim, device=device
        )
        generated = self.preprocess(generated)  # (B, C, T, 1, 1)

        # Calculate total number of time steps needed
        max_t = 1 + (seq_len - 1) / self.chunk_size

        # Step size for each advancement
        dt = 1 / num_denoise_steps
        total_steps = int(max_t / dt)

        # Encode text condition (using cache)
        if self.use_text_cond and "text" in x:
            text_list = x["text"]  # List[str] or List[List[str]]
            if isinstance(text_list[0], list):
                generated_length = []
                text_end_list = x["feature_text_end"]
                full_text = []
                all_text_context = []
                for single_text_list, single_text_end_list in zip(
                    text_list, text_end_list
                ):
                    single_text_end_list = [0] + [
                        min(t, seq_len) for t in single_text_end_list
                    ]
                    generated_length.append(single_text_end_list[-1])
                    single_text_length_list = [
                        t - b
                        for t, b in zip(
                            single_text_end_list[1:], single_text_end_list[:-1]
                        )
                    ]
                    full_text.append(
                        " ////////// ".join(
                            [
                                f"{u} //dur:{t}"
                                for u, t in zip(
                                    single_text_list, single_text_length_list
                                )
                            ]
                        )
                    )
                    single_text_context = self.encode_text_with_cache(
                        single_text_list, device
                    )
                    single_text_context = [
                        u.to(self.param_dtype) for u in single_text_context
                    ]
                    for u, duration in zip(
                        single_text_context, single_text_length_list
                    ):
                        all_text_context.extend([u for _ in range(duration)])
                    all_text_context.extend(
                        [
                            single_text_context[-1]
                            for _ in range(
                                seq_len + self.chunk_size - single_text_end_list[-1]
                            )
                        ]
                    )
            else:
                generated_length = feature_length
                full_text = text_list
                all_text_context = self.encode_text_with_cache(text_list, device)
                all_text_context = [u.to(self.param_dtype) for u in all_text_context]
        else:
            generated_length = feature_length
            full_text = [""] * batch_size
            all_text_context = [""] * batch_size
            all_text_context = self.encode_text_with_cache(all_text_context, device)
            all_text_context = [u.to(self.param_dtype) for u in all_text_context]

        # Get empty text condition encoding (for CFG)
        text_null_list = [""] * batch_size
        text_null_context = self.encode_text_with_cache(text_null_list, device)
        text_null_context = [u.to(self.param_dtype) for u in text_null_context]

        gen_seq_len = seq_len + self.chunk_size
        traj_emb = None
        traj_seq_lens = None
        if self.use_controlnet_traj or self.use_traj_cond:
            traj_emb = self._build_traj_emb(
                x, gen_seq_len, device, training_dropout=False
            )
            traj_seq_lens = self._get_traj_seq_lens(x, gen_seq_len, device)
        traj_emb_backbone = traj_emb if (self.use_traj_cond and not self.use_controlnet_traj) else None
        traj_seq_lens_backbone = traj_seq_lens if (self.use_traj_cond and not self.use_controlnet_traj) else None

        commit_index = 0
        # Progressively advance from t=0 to t=max_t
        for step in range(total_steps):
            # Current time step
            t = step * dt
            start_index = max(0, int(self.chunk_size * (t - 1)) + 1)
            end_index = int(self.chunk_size * t) + 1
            time_steps = torch.full((batch_size,), t, device=device)

            # Calculate current noise schedule
            noise_level = self._get_noise_levels(
                device, seq_len + self.chunk_size, time_steps
            )  # (B, T)

            # Predict noise through WanModel
            noisy_input = []
            for i in range(batch_size):
                noisy_input.append(generated[i, :, :end_index, ...])

            gen_sl = seq_len + self.chunk_size
            t_scaled = noise_level * self.time_embedding_scale
            ctx_2b = (
                self._build_cfg_2b_text(
                    all_text_context, text_null_context, batch_size, gen_sl
                )
                if self.cfg_scale_text != 1.0
                else None
            )
            if ctx_2b is not None:
                noisy_2b = list(noisy_input) + list(noisy_input)
                t_2b = torch.cat([t_scaled, t_scaled], dim=0)
                traj_cn_2b = (
                    torch.cat([traj_emb, traj_emb], dim=0)
                    if traj_emb is not None
                    else None
                )
                traj_sl_2b = (
                    torch.cat([traj_seq_lens, traj_seq_lens], dim=0)
                    if traj_seq_lens is not None
                    else None
                )
                traj_bb_2b = (
                    torch.cat([traj_emb_backbone, traj_emb_backbone], dim=0)
                    if traj_emb_backbone is not None
                    else None
                )
                traj_bb_sl_2b = (
                    torch.cat(
                        [traj_seq_lens_backbone, traj_seq_lens_backbone], dim=0
                    )
                    if traj_seq_lens_backbone is not None
                    else None
                )
                controlnet_residuals = self._controlnet_forward(
                    noisy_2b,
                    t_2b,
                    ctx_2b,
                    gen_sl,
                    traj_cn_2b,
                    traj_sl_2b,
                )
                pred_2b = self.model(
                    noisy_2b,
                    t_2b,
                    ctx_2b,
                    gen_sl,
                    y=None,
                    traj_emb=traj_bb_2b,
                    traj_seq_lens=traj_bb_sl_2b,
                    controlnet_residuals=controlnet_residuals,
                )
                if self.cfg_scale_traj > 0.0:
                    # Separated CFG: reuse 2-batch (out_full, out_null_text);
                    # one extra backbone forward for out_uncond (text OFF, traj OFF).
                    # Formula:
                    #   out = out_uncond
                    #       + w_text * (out_full - out_null_text)
                    #       + w_traj * (out_null_text - out_uncond)
                    pred_uncond = self._uncond_backbone_forward(
                        noisy_input, t_scaled, text_null_context, gen_sl,
                        traj_emb_backbone, traj_seq_lens_backbone,
                    )
                    predicted_result = [
                        pred_uncond[i]
                        + self.cfg_scale_text * (pred_2b[i] - pred_2b[i + batch_size])
                        + self.cfg_scale_traj * (pred_2b[i + batch_size] - pred_uncond[i])
                        for i in range(batch_size)
                    ]
                else:
                    predicted_result = [
                        self.cfg_scale_text * pred_2b[i]
                        - (self.cfg_scale_text - 1) * pred_2b[i + batch_size]
                        for i in range(batch_size)
                    ]
            else:
                controlnet_residuals = self._controlnet_forward(
                    noisy_input,
                    t_scaled,
                    all_text_context,
                    gen_sl,
                    traj_emb,
                    traj_seq_lens,
                )
                predicted_result = self.model(
                    noisy_input,
                    t_scaled,
                    all_text_context,
                    gen_sl,
                    y=None,
                    traj_emb=traj_emb_backbone,
                    traj_seq_lens=traj_seq_lens_backbone,
                    controlnet_residuals=controlnet_residuals,
                )
                if self.cfg_scale_text != 1.0:
                    # Re-compute ControlNet residuals with null text so the uncond
                    # branch is truly unconditioned on text (Bug fix: previously
                    # reused cond residuals, making CFG uncond branch not truly null).
                    controlnet_residuals_null = self._controlnet_forward(
                        noisy_input,
                        t_scaled,
                        text_null_context,
                        gen_sl,
                        traj_emb,
                        traj_seq_lens,
                    )
                    predicted_result_null = self.model(
                        noisy_input,
                        t_scaled,
                        text_null_context,
                        gen_sl,
                        y=None,
                        traj_emb=traj_emb_backbone,
                        traj_seq_lens=traj_seq_lens_backbone,
                        controlnet_residuals=controlnet_residuals_null,
                    )
                    predicted_result = [
                        self.cfg_scale_text * pv - (self.cfg_scale_text - 1) * pvn
                        for pv, pvn in zip(predicted_result, predicted_result_null)
                    ]

            for i in range(batch_size):
                predicted_result_i = predicted_result[i]  # (C, input_length, 1, 1)
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, start_index:end_index, ...]
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "x0":
                    predicted_vel = (
                        predicted_result_i[:, start_index:end_index, ...]
                        - generated[i, :, start_index:end_index, ...]
                    ) / (
                        noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "noise":
                    predicted_vel = (
                        generated[i, :, start_index:end_index, ...]
                        - predicted_result_i[:, start_index:end_index, ...]
                    ) / (
                        1
                        + dt
                        - noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt

            if commit_index < start_index:
                output = generated[:, :, commit_index:start_index, ...]
                output = self.postprocess(output)  # (B, T, C)
                y_hat_out = []
                for i in range(batch_size):
                    if commit_index < generated_length[i]:
                        y_hat_out.append(
                            output[i, : generated_length[i] - commit_index, ...]
                        )
                    else:
                        y_hat_out.append(None)

                out = {}
                out["generated"] = y_hat_out
                yield out
                commit_index = start_index

        output = generated[:, :, commit_index:, ...]
        output = self.postprocess(output)  # (B, T_remain, C)
        y_hat_out = []
        for i in range(batch_size):
            if commit_index < generated_length[i]:
                y_hat_out.append(output[i, : generated_length[i] - commit_index, ...])
            else:
                y_hat_out.append(None)
        out = {}
        out["generated"] = y_hat_out
        yield out

    def init_generated(self, seq_len, batch_size=1, num_denoise_steps=None):
        self.seq_len = seq_len
        self.batch_size = batch_size
        if num_denoise_steps is None:
            self.num_denoise_steps = self.noise_steps
        else:
            self.num_denoise_steps = num_denoise_steps
        assert self.num_denoise_steps % self.chunk_size == 0
        self.dt = 1 / self.num_denoise_steps
        self.current_step = 0
        self.text_condition_list = [[] for _ in range(self.batch_size)]
        self.generated = torch.randn(
            self.batch_size, self.seq_len * 2 + self.chunk_size, self.input_dim
        )
        self.generated = self.preprocess(self.generated)  # (B, C, T, 1, 1)
        self.commit_index = 0
        # Trajectory buffer for streaming: (B, seq_len*2+chunk_size, 3); clear so reset() stops conditioning on old traj
        self.traj_buffer = None
        self.traj_features_buffer = None
        self.token_mask_buffer = None
        self._traj_stream_version = 0
        self._traj_emb_cache = {}

    def _stream_update_traj_buffers(self, x, device):
        if not (self.use_controlnet_traj or self.use_traj_cond) or self.traj_encoder is None:
            return
        buf_len = self.seq_len * 2 + self.chunk_size
        if self.commit_index >= buf_len:
            return
        wrote = False
        no_traj_input = ("traj" not in x or x["traj"] is None) and (
            "traj_features" not in x or x["traj_features"] is None
        )
        if no_traj_input:
            if (
                self.traj_buffer is not None
                or self.traj_features_buffer is not None
                or self.token_mask_buffer is not None
            ):
                self.traj_buffer = None
                self.traj_features_buffer = None
                self.token_mask_buffer = None
                self._traj_stream_version += 1
                self._traj_emb_cache = {}
            return

        if "traj_features" in x and x["traj_features"] is not None:
            tf = x["traj_features"]
            if isinstance(tf, np.ndarray):
                tf = torch.from_numpy(tf).float().to(device)
            if tf.dim() == 2:
                tf = tf.unsqueeze(0)
            if tf.dim() == 3 and tf.size(0) == self.batch_size:
                if self.traj_features_buffer is None:
                    self.traj_features_buffer = torch.zeros(
                        self.batch_size, buf_len, tf.size(-1), device=device, dtype=tf.dtype
                    )
                else:
                    self.traj_features_buffer = self.traj_features_buffer.to(device)
                k = max(0, min(tf.size(1), buf_len - self.commit_index))
                if k > 0:
                    self.traj_features_buffer[:, self.commit_index : self.commit_index + k, :] = tf[:, :k, :]
                    wrote = True

                if "token_mask" in x and x["token_mask"] is not None:
                    tm = x["token_mask"]
                    if isinstance(tm, np.ndarray):
                        tm = torch.from_numpy(tm).float().to(device)
                    if tm.dim() == 1:
                        tm = tm.unsqueeze(0)
                    if tm.dim() == 2 and tm.size(0) == self.batch_size:
                        if self.token_mask_buffer is None:
                            self.token_mask_buffer = torch.zeros(
                                self.batch_size, buf_len, device=device, dtype=tm.dtype
                            )
                        else:
                            self.token_mask_buffer = self.token_mask_buffer.to(device)
                        if k > 0:
                            self.token_mask_buffer[:, self.commit_index : self.commit_index + k] = tm[:, :k]

        if "traj" in x and x["traj"] is not None:
            traj_in = x["traj"]
            if isinstance(traj_in, np.ndarray):
                traj_in = torch.from_numpy(traj_in).float().to(device)
            if traj_in.dim() == 2:
                traj_in = traj_in.unsqueeze(0)
            if traj_in.dim() == 1:
                traj_in = traj_in.unsqueeze(0).unsqueeze(0)
            if traj_in.dim() == 3 and traj_in.size(0) == self.batch_size:
                if self.traj_buffer is None:
                    self.traj_buffer = torch.zeros(
                        self.batch_size, buf_len, 3, device=device, dtype=torch.float32
                    )
                else:
                    self.traj_buffer = self.traj_buffer.to(device)
                k = max(0, min(traj_in.size(1), buf_len - self.commit_index))
                if k > 0:
                    self.traj_buffer[:, self.commit_index : self.commit_index + k, :] = traj_in[:, :k, :].to(device)
                    wrote = True

        if wrote:
            self._traj_stream_version += 1
            self._traj_emb_cache = {}

    def _stream_build_traj_emb(self, x, end_index, device):
        if not (self.use_controlnet_traj or self.use_traj_cond) or self.traj_encoder is None:
            return None
        ctx_len = min(end_index, self.seq_len)
        start_t = max(0, end_index - self.seq_len)

        if self.traj_features_buffer is not None:
            key = ("feat", start_t, end_index, self._traj_stream_version)
            if self.use_traj_emb_cache and key in self._traj_emb_cache:
                return self._traj_emb_cache[key]
            feats = self.traj_features_buffer[:, start_t:end_index, :]
            mask = None
            if self.token_mask_buffer is not None:
                mask = self.token_mask_buffer[:, start_t:end_index]
            if feats.size(1) < ctx_len:
                pad_len = ctx_len - feats.size(1)
                feats = torch.cat([torch.zeros(self.batch_size, pad_len, feats.size(-1), device=device, dtype=feats.dtype), feats], dim=1)
                if mask is not None:
                    mask = torch.cat([torch.zeros(self.batch_size, pad_len, device=device, dtype=mask.dtype), mask], dim=1)
            if mask is not None:
                feats = feats * mask.unsqueeze(-1).to(dtype=feats.dtype)
            emb = self.traj_encoder(feats)
            if self.use_traj_emb_cache:
                self._traj_emb_cache[key] = emb
            return emb

        if self.traj_buffer is not None:
            key = ("xyz", start_t, end_index, self._traj_stream_version)
            if self.use_traj_emb_cache and key in self._traj_emb_cache:
                return self._traj_emb_cache[key]
            traj_slice = self.traj_buffer[:, start_t:end_index, :]
            if traj_slice.size(1) < ctx_len:
                pad_len = ctx_len - traj_slice.size(1)
                traj_slice = torch.cat(
                    [
                        torch.zeros(self.batch_size, pad_len, 3, device=traj_slice.device, dtype=traj_slice.dtype),
                        traj_slice,
                    ],
                    dim=1,
                )
            feats_frame = root_to_traj_feats(traj_slice)  # (B, T_frames, 4)
            # Apply the same LocalTrajEncoder grouping as the training path (_build_traj_emb).
            # Causal VAE: N tokens → 4*(N-1)+1 frames; group frames into (B, ctx_len, 4, 4).
            total_causal = 4 * (ctx_len - 1) + 1 if ctx_len > 1 else 1
            tf = feats_frame.shape[1]
            if tf < total_causal:
                pad = feats_frame.new_zeros(feats_frame.shape[0], total_causal - tf, feats_frame.shape[2])
                feats_frame = torch.cat([feats_frame, pad], dim=1)
            feats_frame = feats_frame[:, :total_causal, :]
            tok0 = feats_frame[:, 0:1, :].unsqueeze(2).expand(-1, -1, 4, -1)  # (B,1,4,4)
            if ctx_len > 1:
                rest = feats_frame[:, 1:, :].reshape(feats_frame.shape[0], ctx_len - 1, 4, feats_frame.shape[2])
                feats_4 = torch.cat([tok0, rest], dim=1)  # (B, ctx_len, 4, 4)
            else:
                feats_4 = tok0
            if self.local_traj_encoder is None:
                return None
            feats_tok = self.local_traj_encoder(feats_4)  # (B, ctx_len, 4)
            emb = self.traj_encoder(feats_tok)
            if self.use_traj_emb_cache:
                self._traj_emb_cache[key] = emb
            return emb

        return self._build_traj_emb(x, ctx_len, device, training_dropout=False)

    @torch.no_grad()
    def stream_generate_step(self, x, first_chunk=True):
        """
        Streaming generation step - Diffusion Forcing inference
        Uses triangular noise schedule, progressively generating from left to right

        Generation process:
        1. Start from t=0, gradually increase t
        2. Each t corresponds to a noise schedule: clean on left, noisy on right, gradient in middle
        3. After each denoising step, t increases slightly and continues
        """

        device = next(self.parameters()).device
        if first_chunk:
            self.generated = self.generated.to(device)
        self._stream_update_traj_buffers(x, device)

        # Encode text condition (using cache)
        if self.use_text_cond and "text" in x:
            text_list = x["text"]  # List[str]
            new_text_context = self.encode_text_with_cache(text_list, device)
            new_text_context = [u.to(self.param_dtype) for u in new_text_context]
        else:
            new_text_context = [""] * self.batch_size
            new_text_context = self.encode_text_with_cache(new_text_context, device)
            new_text_context = [u.to(self.param_dtype) for u in new_text_context]

        # Get empty text condition encoding (for CFG)
        text_null_list = [""] * self.batch_size
        text_null_context = self.encode_text_with_cache(text_null_list, device)
        text_null_context = [u.to(self.param_dtype) for u in text_null_context]

        for i in range(self.batch_size):
            if first_chunk:
                self.text_condition_list[i].extend(
                    [new_text_context[i]] * self.chunk_size
                )
            else:
                self.text_condition_list[i].extend([new_text_context[i]])

        end_step = (
            (self.commit_index + self.chunk_size)
            * self.num_denoise_steps
            / self.chunk_size
        )
        while self.current_step < end_step:
            current_time = self.current_step * self.dt
            start_index = max(0, int(self.chunk_size * (current_time - 1)) + 1)
            end_index = int(self.chunk_size * current_time) + 1
            time_steps = torch.full((self.batch_size,), current_time, device=device)

            noise_level = self._get_noise_levels(device, end_index, time_steps)[
                :, -self.seq_len :
            ]  # (B, T)

            # Predict noise through WanModel
            noisy_input = []
            for i in range(self.batch_size):
                noisy_input.append(
                    self.generated[i, :, :end_index, ...][:, -self.seq_len :]
                )  # (C, T, 1, 1)

            text_condition = []
            for i in range(self.batch_size):
                text_condition.extend(
                    self.text_condition_list[i][:end_index][-self.seq_len :]
                )  # (T, D, 4096)

            traj_emb = self._stream_build_traj_emb(x, end_index, device)
            traj_seq_lens = (
                torch.full(
                    (self.batch_size,),
                    min(end_index, self.seq_len),
                    device=device,
                    dtype=torch.long,
                )
                if traj_emb is not None
                else None
            )
            # ControlNet mode: backbone stays unconditional on traj; only ControlNet branch consumes it.
            traj_emb_backbone = traj_emb if (self.use_traj_cond and not self.use_controlnet_traj) else None
            traj_seq_lens_backbone = traj_seq_lens if (self.use_traj_cond and not self.use_controlnet_traj) else None

            model_sl = min(end_index, self.seq_len)
            t_scaled = noise_level * self.time_embedding_scale
            ctx_2b = (
                self._build_cfg_2b_text(
                    text_condition, text_null_context, self.batch_size, model_sl
                )
                if self.cfg_scale_text != 1.0
                else None
            )
            if ctx_2b is not None:
                noisy_2b = list(noisy_input) + list(noisy_input)
                t_2b = torch.cat([t_scaled, t_scaled], dim=0)
                traj_cn_2b = (
                    torch.cat([traj_emb, traj_emb], dim=0)
                    if traj_emb is not None
                    else None
                )
                traj_sl_2b = (
                    torch.cat([traj_seq_lens, traj_seq_lens], dim=0)
                    if traj_seq_lens is not None
                    else None
                )
                traj_bb_2b = (
                    torch.cat([traj_emb_backbone, traj_emb_backbone], dim=0)
                    if traj_emb_backbone is not None
                    else None
                )
                traj_bb_sl_2b = (
                    torch.cat(
                        [traj_seq_lens_backbone, traj_seq_lens_backbone], dim=0
                    )
                    if traj_seq_lens_backbone is not None
                    else None
                )
                controlnet_residuals = self._controlnet_forward(
                    noisy_2b,
                    t_2b,
                    ctx_2b,
                    model_sl,
                    traj_cn_2b,
                    traj_sl_2b,
                )
                pred_2b = self.model(
                    noisy_2b,
                    t_2b,
                    ctx_2b,
                    model_sl,
                    y=None,
                    traj_emb=traj_bb_2b,
                    traj_seq_lens=traj_bb_sl_2b,
                    controlnet_residuals=controlnet_residuals,
                )
                if self.cfg_scale_traj > 0.0:
                    # Separated CFG: reuse 2-batch (out_full, out_null_text);
                    # one extra backbone forward for out_uncond (text OFF, traj OFF).
                    # Formula:
                    #   out = out_uncond
                    #       + w_text * (out_full - out_null_text)
                    #       + w_traj * (out_null_text - out_uncond)
                    pred_uncond = self._uncond_backbone_forward(
                        noisy_input, t_scaled, text_null_context, model_sl,
                        traj_emb_backbone, traj_seq_lens_backbone,
                    )
                    predicted_result = [
                        pred_uncond[i]
                        + self.cfg_scale_text * (pred_2b[i] - pred_2b[i + self.batch_size])
                        + self.cfg_scale_traj * (pred_2b[i + self.batch_size] - pred_uncond[i])
                        for i in range(self.batch_size)
                    ]
                else:
                    predicted_result = [
                        self.cfg_scale_text * pred_2b[i]
                        - (self.cfg_scale_text - 1) * pred_2b[i + self.batch_size]
                        for i in range(self.batch_size)
                    ]
            else:
                controlnet_residuals = self._controlnet_forward(
                    noisy_input,
                    t_scaled,
                    text_condition,
                    model_sl,
                    traj_emb,
                    traj_seq_lens,
                )
                predicted_result = self.model(
                    noisy_input,
                    t_scaled,
                    text_condition,
                    model_sl,
                    y=None,
                    traj_emb=traj_emb_backbone,
                    traj_seq_lens=traj_seq_lens_backbone,
                    controlnet_residuals=controlnet_residuals,
                )
                if self.cfg_scale_text != 1.0:
                    controlnet_residuals_null = self._controlnet_forward(
                        noisy_input,
                        t_scaled,
                        text_null_context,
                        model_sl,
                        traj_emb,
                        traj_seq_lens,
                    )
                    predicted_result_null = self.model(
                        noisy_input,
                        t_scaled,
                        text_null_context,
                        model_sl,
                        y=None,
                        traj_emb=traj_emb_backbone,
                        traj_seq_lens=traj_seq_lens_backbone,
                        controlnet_residuals=controlnet_residuals_null,
                    )
                    predicted_result = [
                        self.cfg_scale_text * pv - (self.cfg_scale_text - 1) * pvn
                        for pv, pvn in zip(predicted_result, predicted_result_null)
                    ]

            for i in range(self.batch_size):
                predicted_result_i = predicted_result[i]  # (C, input_length, 1, 1)
                if end_index > self.seq_len:
                    predicted_result_i = torch.cat(
                        [
                            torch.zeros(
                                predicted_result_i.shape[0],
                                end_index - self.seq_len,
                                predicted_result_i.shape[2],
                                predicted_result_i.shape[3],
                                device=device,
                            ),
                            predicted_result_i,
                        ],
                        dim=1,
                    )
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, start_index:end_index, ...]
                    self.generated[i, :, start_index:end_index, ...] += (
                        predicted_vel * self.dt
                    )
                elif self.prediction_type == "x0":
                    predicted_vel = (
                        predicted_result_i[:, start_index:end_index, ...]
                        - self.generated[i, :, start_index:end_index, ...]
                    ) / (
                        noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    self.generated[i, :, start_index:end_index, ...] += (
                        predicted_vel * self.dt
                    )
                elif self.prediction_type == "noise":
                    predicted_vel = (
                        self.generated[i, :, start_index:end_index, ...]
                        - predicted_result_i[:, start_index:end_index, ...]
                    ) / (
                        1
                        + self.dt
                        - noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    self.generated[i, :, start_index:end_index, ...] += (
                        predicted_vel * self.dt
                    )
            self.current_step += 1
        output = self.generated[:, :, self.commit_index : self.commit_index + 1, ...]
        output = self.postprocess(output)  # (B, 1, C)
        out = {}
        out["generated"] = output
        self.commit_index += 1

        if self.commit_index == self.seq_len * 2:
            self.generated = torch.cat(
                [
                    self.generated[:, :, self.seq_len :, ...],
                    torch.randn(
                        self.batch_size,
                        self.input_dim,
                        self.seq_len,
                        1,
                        1,
                        device=device,
                    ),
                ],
                dim=2,
            )
            if self.traj_buffer is not None:
                self.traj_buffer = torch.cat(
                    [
                        self.traj_buffer[:, self.seq_len :, :],
                        torch.zeros(
                            self.batch_size,
                            self.seq_len,
                            3,
                            device=device,
                            dtype=self.traj_buffer.dtype,
                        ),
                    ],
                    dim=1,
                )
            if self.traj_features_buffer is not None:
                self.traj_features_buffer = torch.cat(
                    [
                        self.traj_features_buffer[:, self.seq_len :, :],
                        torch.zeros(
                            self.batch_size,
                            self.seq_len,
                            self.traj_features_buffer.size(-1),
                            device=device,
                            dtype=self.traj_features_buffer.dtype,
                        ),
                    ],
                    dim=1,
                )
            if self.token_mask_buffer is not None:
                self.token_mask_buffer = torch.cat(
                    [
                        self.token_mask_buffer[:, self.seq_len :],
                        torch.zeros(
                            self.batch_size,
                            self.seq_len,
                            device=device,
                            dtype=self.token_mask_buffer.dtype,
                        ),
                    ],
                    dim=1,
                )
            self._traj_stream_version += 1
            self._traj_emb_cache = {}
            self.current_step -= self.seq_len * self.num_denoise_steps / self.chunk_size
            self.commit_index -= self.seq_len
            for i in range(self.batch_size):
                self.text_condition_list[i] = self.text_condition_list[i][
                    self.seq_len :
                ]
        return out
