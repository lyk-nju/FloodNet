import warnings

import numpy as np
import torch
import torch.nn as nn

from .tools.t5 import T5EncoderModel
from .tools.traj_encoder import LocalTrajEncoder, TrajEncoder
from .tools.wan_model import WanModel
from .tools.wan_controlnet import WanControlNet

try:
    from FloodNet.utils.traj_batch import encode_traj_batch
    from FloodNet.utils.traj_stream_buffer import TrajStreamBuffer
except ImportError:  # pragma: no cover - script entrypoints use top-level imports
    from utils.traj_batch import encode_traj_batch
    from utils.traj_stream_buffer import TrajStreamBuffer


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
        text_dropout=0.1,
        cfg_scale_text=5.0,
        cfg_scale_traj=0.0,
        prediction_type="vel",  # "vel", "x0", "noise"
        causal=False,
        traj_out_dim=64,
        traj_in_dim=4,
        traj_encoder_in_dim=None,
        traj_dropout=0.1,
        use_traj_emb_cache=False,
        use_traj_kv_cache=None,
        control_loss_weight=1.0,
        freeze_backbone=True,
        use_precomputed_text_emb=False,
        precomputed_text_emb_path=None,
        scheduled_sampling_prob=0.0,
        self_forcing_enabled=False,
        self_forcing_stride_tokens=1,
        self_forcing_detach_between_steps=True,
        self_forcing_k_schedule=((0.0, 2), (0.4, 3), (0.7, 5)),
        self_forcing_start_step=None,
    ):
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
        self.text_dropout = text_dropout
        self.cfg_scale_text = cfg_scale_text
        self.cfg_scale_traj = float(cfg_scale_traj)
        self.prediction_type = prediction_type
        self.causal = causal
        self.traj_out_dim = traj_out_dim
        self.traj_in_dim = traj_in_dim
        self.traj_dropout = traj_dropout
        self.freeze_backbone = bool(freeze_backbone)
        if use_traj_kv_cache is not None:
            warnings.warn(
                "`use_traj_kv_cache` is deprecated; use `use_traj_emb_cache`.",
                stacklevel=2,
            )
            use_traj_emb_cache = bool(use_traj_kv_cache)
        self.use_traj_emb_cache = use_traj_emb_cache
        self.scheduled_sampling_prob = float(scheduled_sampling_prob)
        self.self_forcing_enabled = bool(self_forcing_enabled)
        self.self_forcing_stride_tokens = int(self_forcing_stride_tokens)
        self.self_forcing_detach_between_steps = bool(
            self_forcing_detach_between_steps
        )
        self.self_forcing_k_schedule = [
            (float(p), int(k)) for p, k in self_forcing_k_schedule
        ]
        if self_forcing_start_step is not None:
            warnings.warn(
                "`self_forcing_start_step` is deprecated and ignored. "
                "When self_forcing_enabled=True, self-forcing now starts from the first step.",
                stacklevel=2,
            )

        if self.self_forcing_stride_tokens != 1:
            raise ValueError(
                "v1 self-forcing only supports self_forcing_stride_tokens == 1"
            )
        if self.self_forcing_enabled and not self.self_forcing_detach_between_steps:
            raise NotImplementedError(
                "v1 self-forcing only supports detach_between_steps=True"
            )
        if self.self_forcing_enabled and self.prediction_type not in ("vel", "x0"):
            raise ValueError(
                "self-forcing only supports prediction_type in {'vel', 'x0'}"
            )
        if not self.self_forcing_k_schedule:
            raise ValueError("self_forcing_k_schedule must not be empty")
        self.self_forcing_k_schedule.sort(key=lambda x: x[0])

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
        # Backbone is unconditional on traj; ControlNet is the sole trajectory consumer.
        traj_enc_dim_backbone = 0
        traj_enc_dim_controlnet = self.traj_out_dim
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
        self.controlnet.init_from_backbone(self.model)

        # Local within-token encoder: compress masked 4-frame traj feats -> token-level 4D.
        self.local_traj_encoder = LocalTrajEncoder(hidden_dim=32)
        self.traj_encoder = TrajEncoder(
            in_dim=self.traj_in_dim, hidden_dim=64, out_dim=self.traj_out_dim
        )
        self.param_dtype = torch.float32

        if self.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.controlnet.parameters():
                p.requires_grad = True
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
        return self.controlnet(
            noisy_input,
            t_scaled,
            text_context,
            seq_len,
            y=None,
            traj_emb=traj_emb,
            traj_seq_lens=traj_seq_lens,
        )

    def _concat_text_for_cfg(
        self, text_context, text_null_per_sample, batch_size, model_seq_len
    ):
        """Build text context list for MotionLCM-style double-batch CFG (cond || uncond).

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
            traj_emb=None,
            traj_seq_lens=None,
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

    def _build_traj_emb(self, x, seq_len, device):
        if self.traj_encoder is None:
            return None
        return encode_traj_batch(
            x, seq_len, device, self.local_traj_encoder, self.traj_encoder
        )

    def _get_traj_seq_lens(self, x, seq_len, device):
        # Prefer feature_length (= token_length) — exact, no rounding error.
        if "feature_length" in x and x["feature_length"] is not None:
            return (
                x["feature_length"]
                .to(device=device, dtype=torch.long)
                .clamp(min=0, max=seq_len)
            )
        if "traj_features_length" in x and x["traj_features_length"] is not None:
            return (
                x["traj_features_length"]
                .to(device=device, dtype=torch.long)
                .clamp(min=0, max=seq_len)
            )
        if "traj_length" in x and x["traj_length"] is not None:
            # Fallback: match dataset formula token_end = (feature_length + 2) // 4
            tl = x["traj_length"].to(device=device, dtype=torch.long)
            return ((tl + 2) // 4 + 1).clamp(min=0, max=seq_len)
        return None

    def _prepare_text_context(self, x, seq_len, device):
        if self.use_text_cond and "text" in x:
            text_list = x["text"]  # List[str] or List[List[str]]
            if isinstance(text_list[0], list):
                text_end_list = x["feature_text_end"]
                all_text_context = []
                for single_text_list, single_text_end_list in zip(
                    text_list, text_end_list
                ):
                    if (not self.training) or (np.random.rand() > self.text_dropout):
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
                if self.training:
                    all_text_context = [
                        (u if np.random.rand() > self.text_dropout else "") for u in text_list
                    ]
                else:
                    all_text_context = list(text_list)
                all_text_context = self.encode_text_with_cache(all_text_context, device)
                all_text_context = [u.to(self.param_dtype) for u in all_text_context]
        else:
            all_text_context = [""] * x["feature"].shape[0]
            all_text_context = self.encode_text_with_cache(all_text_context, device)
            all_text_context = [u.to(self.param_dtype) for u in all_text_context]
        return all_text_context

    def _decide_traj_dropout(self, device):
        traj_dropped = False
        if self.training:
            drop = torch.empty(1, device=device).uniform_()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast(drop, src=0)
            traj_dropped = drop.item() < self.traj_dropout
        return traj_dropped

    def _prepare_traj_condition(
        self, x, seq_len, device, traj_dropped_override=None
    ):
        traj_emb = None
        traj_seq_lens = None
        if traj_dropped_override is None:
            traj_dropped = self._decide_traj_dropout(device)
        else:
            traj_dropped = bool(traj_dropped_override)

        if not traj_dropped:
            traj_emb = self._build_traj_emb(x, seq_len, device)
            traj_seq_lens = self._get_traj_seq_lens(x, seq_len, device)

        return traj_emb, traj_seq_lens, traj_dropped

    def _slice_diffusion_window(self, clean_feature, feature_length, time_steps):
        batch_size, seq_len, _ = clean_feature.shape
        device = clean_feature.device

        noise_level = self._get_noise_levels(device, seq_len, time_steps)
        noisy_feature, noise = self.add_noise(clean_feature, noise_level)

        feature = self.preprocess(clean_feature)
        noisy_feature = self.preprocess(noisy_feature)
        noise = self.preprocess(noise)

        feature_ref = []
        noise_ref = []
        noisy_feature_input = []
        end_indices = []
        for i in range(batch_size):
            end_index = int(self.chunk_size * time_steps[i].item()) + 1
            valid_len = int(feature_length[i].item())
            end_index = min(valid_len, end_index)
            feature_ref.append(feature[i, :, :end_index, ...])
            noise_ref.append(noise[i, :, :end_index, ...])
            noisy_feature_input.append(noisy_feature[i, :, :end_index, ...])
            end_indices.append(end_index)

        return noise_level, feature_ref, noise_ref, noisy_feature_input, end_indices

    def _forward_single_window(
        self,
        x,
        clean_feature,
        time_steps,
        all_text_context,
        traj_emb,
        traj_seq_lens,
        traj_dropped,
        enable_scheduled_sampling=True,
    ):
        feature_length = x["feature_length"]
        batch_size, seq_len, _ = clean_feature.shape
        device = clean_feature.device

        (
            noise_level,
            feature_ref,
            noise_ref,
            noisy_feature_input,
            end_indices,
        ) = self._slice_diffusion_window(clean_feature, feature_length, time_steps)

        if (
            enable_scheduled_sampling
            and self.training
            and self.scheduled_sampling_prob > 0.0
            and np.random.rand() < self.scheduled_sampling_prob
        ):
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
                    traj_emb=None,
                    traj_seq_lens=None,
                    controlnet_residuals=cn_res_ss,
                )
            for b in range(batch_size):
                t_len = noisy_feature_input[b].shape[1]
                ctx_len = t_len - self.chunk_size
                if ctx_len <= 0:
                    continue
                if self.prediction_type == "vel":
                    # Same low-β stability issue as the self-forcing rollout:
                    # use z = noisy_x + β·vel instead of vel + ε.  See the
                    # comment in _forward_single_window for details.
                    beta_b = noise_level[b, :t_len].view(1, -1, 1, 1)
                    x0_hat = (
                        noisy_feature_input[b] + beta_b * pred_ss[b]
                    ).detach()
                elif self.prediction_type == "x0":
                    x0_hat = pred_ss[b].detach()
                else:
                    continue
                noisy_feature_input[b] = torch.cat(
                    [x0_hat[:, :ctx_len], noisy_feature_input[b][:, ctx_len:]], dim=1
                )

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
            traj_emb=None,
            traj_seq_lens=None,
            controlnet_residuals=controlnet_residuals,
        )

        loss = 0.0
        # Two lists with different x0-recovery formulas, each tuned to its
        # consumer (both formulas are mathematically valid; they differ only in
        # how prediction error δ propagates):
        #
        #   loss_x0_list  (Formula 1, "z = pred_vel + ε"):
        #     error  = δ           (β-independent, fully exposes the model's
        #                           velocity-prediction error)
        #     ∂x0/∂pred_vel = 1    (full-strength gradient at every position)
        #     → used by control loss so the gradient signal is not damped at
        #       low β, where Formula 2 would let noisy_x carry the loss with
        #       almost no gradient flowing back into the model.
        #
        #   sf_x0_list  (Formula 2, "z = noisy_x + β·pred_vel"):
        #     error  = β·δ         (low-variance estimate of z, especially
        #                           important at low β)
        #     ∂x0/∂pred_vel = β    (β-attenuated gradient — irrelevant here,
        #                           because SF rollout consumes the *value*
        #                           after .detach())
        #     → used by self-forcing rollout to substitute the chunk's
        #       leftmost token where β ≈ 1/cs is small; Formula 1 there
        #       collapses to z + ε (pure noise injection) and corrupts the
        #       next rollout step's context.
        loss_x0_list = []
        sf_x0_list = []
        for b in range(batch_size):
            t_b = noisy_feature_input[b].shape[1]
            if self.prediction_type == "vel":
                vel = feature_ref[b] - noise_ref[b]
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
                loss_x0_list.append(None)
                sf_x0_list.append(None)
            else:
                raise ValueError(
                    f"Unsupported prediction_type={self.prediction_type!r}"
                )
            sample_loss = squared_error.mean()
            loss += sample_loss
            if self.prediction_type == "vel":
                # Formula 1 — full-gradient estimate, for control loss.
                pred_x0_loss = predicted_result[b] + noise_ref[b]
                loss_x0_list.append(pred_x0_loss[:, :, 0, 0].permute(1, 0))
                # Formula 2 — low-variance estimate, for SF rollout.
                beta_b = noise_level[b, :t_b].view(1, -1, 1, 1)
                pred_x0_sf = noisy_feature_input[b] + beta_b * predicted_result[b]
                sf_x0_list.append(pred_x0_sf[:, :, 0, 0].permute(1, 0))
            elif self.prediction_type == "x0":
                # x0-prediction: model directly outputs z, so the two formulas
                # coincide. Share the same tensor for both consumers.
                pred_x0 = predicted_result[b]
                latent = pred_x0[:, :, 0, 0].permute(1, 0)
                loss_x0_list.append(latent)
                sf_x0_list.append(latent)
        loss = loss / batch_size

        pred_x0_latent_list = None
        if ("traj" in x) and (self.prediction_type in ("vel", "x0")) and not traj_dropped:
            pred_x0_latent_list = loss_x0_list

        return {
            "loss": loss,
            # Consumed by control loss → Formula 1 (full-gradient).
            "pred_x0_latent_list": pred_x0_latent_list,
            # Consumed by self-forcing rollout → Formula 2 (low-variance).
            "x0_latent_list": sf_x0_list,
            "end_indices": end_indices,
        }

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

        time_steps_override = x.get("_time_steps_override", None)
        if time_steps_override is not None:
            if not torch.is_tensor(time_steps_override):
                time_steps = torch.as_tensor(
                    time_steps_override, device=device, dtype=torch.float32
                )
            else:
                time_steps = time_steps_override.to(device=device, dtype=torch.float32)
            if time_steps.ndim == 0:
                time_steps = time_steps.repeat(batch_size)
            if time_steps.shape[0] != batch_size:
                raise ValueError(
                    f"_time_steps_override batch mismatch: got {tuple(time_steps.shape)} "
                    f"for batch_size={batch_size}"
                )
        else:
            # Randomly use a time step
            time_steps = []
            for i in range(batch_size):
                valid_len = feature_length[i].item()
                # Random float from 0 to valid_len/chunk_size, not an integer
                max_time = valid_len / self.chunk_size
                time_steps.append(torch.FloatTensor(1).uniform_(0, max_time).item())
            time_steps = torch.tensor(time_steps, device=device)  # (B,)
        all_text_context = self._prepare_text_context(x, seq_len, device)
        traj_emb, traj_seq_lens, traj_dropped = self._prepare_traj_condition(
            x, seq_len, device
        )

        single_result = self._forward_single_window(
            x,
            feature,
            time_steps,
            all_text_context,
            traj_emb,
            traj_seq_lens,
            traj_dropped,
            enable_scheduled_sampling=True,
        )

        loss_dict = {"total": single_result["loss"], "mse": single_result["loss"]}
        if single_result["pred_x0_latent_list"] is not None:
            loss_dict["control_aux"] = {
                "pred_x0_latent_list": single_result["pred_x0_latent_list"]
            }
        return loss_dict

    def _denoise_with_cfg(
        self,
        noisy_input: list,
        t_scaled: torch.Tensor,
        text_cond_ctx: list,
        text_null_ctx: list,
        traj_emb,
        traj_seq_lens,
        seq_len: int,
        batch_size: int,
    ) -> list:
        """Unified CFG denoising step shared by generate / stream_generate / stream_generate_step.

        Handles three modes transparently:
          - 2-batch text CFG (cfg_scale_text != 1) + optional separated traj CFG
          - Single-batch with post-hoc null-text forward (cfg_scale_text != 1, no double-batch context)
          - Unconditioned (cfg_scale_text == 1)
        Returns a list of per-sample predicted tensors (C, T, 1, 1).
        """
        ctx_double = (
            self._concat_text_for_cfg(text_cond_ctx, text_null_ctx, batch_size, seq_len)
            if self.cfg_scale_text != 1.0
            else None
        )

        if ctx_double is not None:
            noisy_double = list(noisy_input) + list(noisy_input)
            t_double = torch.cat([t_scaled, t_scaled], dim=0)
            traj_double = (
                torch.cat([traj_emb, traj_emb], dim=0) if traj_emb is not None else None
            )
            traj_sl_double = (
                torch.cat([traj_seq_lens, traj_seq_lens], dim=0)
                if traj_seq_lens is not None
                else None
            )
            residuals = self._controlnet_forward(
                noisy_double, t_double, ctx_double, seq_len, traj_double, traj_sl_double
            )
            pred_double = self.model(
                noisy_double, t_double, ctx_double, seq_len,
                y=None, traj_emb=None, traj_seq_lens=None, controlnet_residuals=residuals,
            )
            if self.cfg_scale_traj > 0.0:
                # Separated CFG:
                #   out = out_uncond
                #       + w_text * (out_full - out_null_text)
                #       + w_traj * (out_null_text - out_uncond)
                pred_uncond = self._uncond_backbone_forward(
                    noisy_input, t_scaled, text_null_ctx, seq_len
                )
                return [
                    pred_uncond[i]
                    + self.cfg_scale_text * (pred_double[i] - pred_double[i + batch_size])
                    + self.cfg_scale_traj * (pred_double[i + batch_size] - pred_uncond[i])
                    for i in range(batch_size)
                ]
            else:
                return [
                    self.cfg_scale_text * pred_double[i]
                    - (self.cfg_scale_text - 1) * pred_double[i + batch_size]
                    for i in range(batch_size)
                ]
        else:
            residuals = self._controlnet_forward(
                noisy_input, t_scaled, text_cond_ctx, seq_len, traj_emb, traj_seq_lens
            )
            pred = self.model(
                noisy_input, t_scaled, text_cond_ctx, seq_len,
                y=None, traj_emb=None, traj_seq_lens=None, controlnet_residuals=residuals,
            )
            if self.cfg_scale_text != 1.0:
                # Re-compute ControlNet residuals with null text so the uncond branch is
                # truly unconditioned (Bug fix: reusing cond residuals made CFG uncond
                # branch not truly null).
                residuals_null = self._controlnet_forward(
                    noisy_input, t_scaled, text_null_ctx, seq_len, traj_emb, traj_seq_lens
                )
                pred_null = self.model(
                    noisy_input, t_scaled, text_null_ctx, seq_len,
                    y=None, traj_emb=None, traj_seq_lens=None,
                    controlnet_residuals=residuals_null,
                )
                return [
                    self.cfg_scale_text * pv - (self.cfg_scale_text - 1) * pvn
                    for pv, pvn in zip(pred, pred_null)
                ]
            return pred

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
        traj_emb = self._build_traj_emb(x, gen_seq_len, device)
        traj_seq_lens = self._get_traj_seq_lens(x, gen_seq_len, device)

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
            predicted_result = self._denoise_with_cfg(
                noisy_input, t_scaled,
                all_text_context, text_null_context,
                traj_emb, traj_seq_lens, gen_sl, batch_size,
            )

            for i in range(batch_size):
                predicted_result_i = predicted_result[i]  # (C, input_length, 1, 1)
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, start_index:end_index, ...]
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "x0":
                    nl = (
                        noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                        .clamp(min=1e-6)
                    )
                    predicted_vel = (
                        predicted_result_i[:, start_index:end_index, ...]
                        - generated[i, :, start_index:end_index, ...]
                    ) / nl
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "noise":
                    denom = (
                        1
                        + dt
                        - noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    ).clamp(min=1e-6)
                    predicted_vel = (
                        generated[i, :, start_index:end_index, ...]
                        - predicted_result_i[:, start_index:end_index, ...]
                    ) / denom
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
        traj_emb = self._build_traj_emb(x, gen_seq_len, device)
        traj_seq_lens = self._get_traj_seq_lens(x, gen_seq_len, device)

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
            predicted_result = self._denoise_with_cfg(
                noisy_input, t_scaled,
                all_text_context, text_null_context,
                traj_emb, traj_seq_lens, gen_sl, batch_size,
            )

            for i in range(batch_size):
                predicted_result_i = predicted_result[i]  # (C, input_length, 1, 1)
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, start_index:end_index, ...]
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "x0":
                    nl = (
                        noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                        .clamp(min=1e-6)
                    )
                    predicted_vel = (
                        predicted_result_i[:, start_index:end_index, ...]
                        - generated[i, :, start_index:end_index, ...]
                    ) / nl
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "noise":
                    denom = (
                        1
                        + dt
                        - noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    ).clamp(min=1e-6)
                    predicted_vel = (
                        generated[i, :, start_index:end_index, ...]
                        - predicted_result_i[:, start_index:end_index, ...]
                    ) / denom
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
        self._traj_buf = TrajStreamBuffer(
            batch_size=batch_size,
            buf_len=self.seq_len * 2 + self.chunk_size,
            local_traj_encoder=self.local_traj_encoder,
            traj_encoder=self.traj_encoder,
            use_emb_cache=self.use_traj_emb_cache,
        )

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
        self._traj_buf.update(x, self.commit_index, device)

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

            traj_emb = self._traj_buf.build_traj_emb(end_index, self.seq_len, device)
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
            model_sl = min(end_index, self.seq_len)
            t_scaled = noise_level * self.time_embedding_scale
            predicted_result = self._denoise_with_cfg(
                noisy_input, t_scaled,
                text_condition, text_null_context,
                traj_emb, traj_seq_lens, model_sl, self.batch_size,
            )

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
                    nl = (
                        noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                        .clamp(min=1e-6)
                    )
                    predicted_vel = (
                        predicted_result_i[:, start_index:end_index, ...]
                        - self.generated[i, :, start_index:end_index, ...]
                    ) / nl
                    self.generated[i, :, start_index:end_index, ...] += (
                        predicted_vel * self.dt
                    )
                elif self.prediction_type == "noise":
                    denom = (
                        1
                        + self.dt
                        - noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    ).clamp(min=1e-6)
                    predicted_vel = (
                        self.generated[i, :, start_index:end_index, ...]
                        - predicted_result_i[:, start_index:end_index, ...]
                    ) / denom
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
            self._traj_buf.roll(self.seq_len, device)
            self.current_step -= self.seq_len * self.num_denoise_steps / self.chunk_size
            self.commit_index -= self.seq_len
            for i in range(self.batch_size):
                self.text_condition_list[i] = self.text_condition_list[i][
                    self.seq_len :
                ]
        return out
