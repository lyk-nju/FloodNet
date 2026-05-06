import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModel

from utils.traj_batch import build_traj_emb, root_to_traj_feats
from .tools.traj_encoder import TrajEncoder
from .tools.wan_model import WanModel


def _expand_precomputed_caption_keys(emb: dict) -> dict:
    """Alias strip() keys so table matches HumanML3D captions after .strip()."""
    out = dict(emb)
    for k, v in emb.items():
        s = k.strip()
        if s not in out:
            out[s] = v
    return out


class HFT5Encoder:
    """Wrapper for HuggingFace T5 encoder, compatible with original T5EncoderModel interface"""
    def __init__(self, text_len, dtype=torch.float32, device=torch.device("cpu"), model_name="google/umt5-base"):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        
        print(f"Loading {model_name} from HuggingFace...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, 
            dtype=dtype
        ).encoder  # Only use the encoder part
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)
    
    def __call__(self, texts, device):
        """Encode texts, returns list of tensors (one per text, with padding removed)"""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.text_len,
            return_tensors="pt"
        )
        ids = inputs.input_ids.to(device)
        mask = inputs.attention_mask.to(device)
        
        # Encode (model should already be on device via external .model.to(device) call)
        context = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        
        # Get sequence lengths (excluding padding)
        seq_lens = mask.sum(dim=1).long()
        
        # Return list of tensors with padding removed (same as original T5EncoderModel)
        return [u[:v] for u, v in zip(context, seq_lens)]


class DiffForcingWanModel(nn.Module):
    def __init__(
        self,
        model_name="google/umt5-base",  # HuggingFace model name
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
        drop_out=0.1,
        cfg_scale_text=5.0,
        prediction_type="vel",  # "vel", "x0", "noise"
        causal=False,
        use_traj_cond=False,
        traj_out_dim=64,
        traj_in_dim=4,
        traj_encoder_in_dim=None,
        traj_drop_out=0.1,
        use_traj_emb_cache=False,
        use_traj_kv_cache=None,
        control_loss_weight=1.0,  # used by train_ldf, not by model
        freeze_backbone_for_traj=False,
        use_precomputed_text_emb=False,
        precomputed_text_emb_path=None,
        **kwargs,
    ):
        kwargs.pop("traj_lora_rank", None)
        kwargs.pop("lora_rank_traj", None)
        # Backward compat: old checkpoints stored the param as cfg_scale.
        _legacy_cfg_scale = kwargs.pop("cfg_scale", None)
        if _legacy_cfg_scale is not None:
            cfg_scale_text = _legacy_cfg_scale
        kwargs.pop("freeze_backbone_for_controlnet", None)
        kwargs.pop("use_controlnet_traj", None)
        kwargs.pop("controlnet_init_from_backbone", None)
        if "dropout" in kwargs:
            drop_out = kwargs.pop("dropout")
        if "traj_dropout" in kwargs:
            traj_drop_out = kwargs.pop("traj_dropout")
        if kwargs:
            raise TypeError(
                "DiffForcingWanModel: unexpected keyword arguments "
                f"{sorted(kwargs.keys())}"
            )
        super().__init__()
        if traj_encoder_in_dim is not None:
            warnings.warn(
                "`traj_encoder_in_dim` is preferred; it overrides `traj_in_dim`.",
                stacklevel=2,
            )
            traj_in_dim = traj_encoder_in_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_traj_cond = use_traj_cond
        self.traj_out_dim = traj_out_dim
        self.traj_in_dim = traj_in_dim
        self.traj_drop_out = traj_drop_out
        if use_traj_kv_cache is not None:
            warnings.warn(
                "`use_traj_kv_cache` is deprecated; use `use_traj_emb_cache`.",
                stacklevel=2,
            )
            use_traj_emb_cache = bool(use_traj_kv_cache)
        self.use_traj_emb_cache = use_traj_emb_cache
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.time_embedding_scale = time_embedding_scale
        self.chunk_size = chunk_size
        self.noise_steps = noise_steps
        self.use_text_cond = use_text_cond
        self.drop_out = drop_out
        self.cfg_scale_text = cfg_scale_text
        self.prediction_type = prediction_type
        self.causal = causal

        self.text_len = text_len
        self.model_name = model_name
        self.use_precomputed_text_emb = bool(use_precomputed_text_emb)
        self._precomputed_text_emb = None
        self.text_encoder = None
        self.text_cache = {}

        if self.use_precomputed_text_emb:
            if not precomputed_text_emb_path:
                raise ValueError(
                    "use_precomputed_text_emb=True requires precomputed_text_emb_path "
                    "(run pretokenize_t5_text_tiny.py)."
                )
            blob = torch.load(
                precomputed_text_emb_path, map_location="cpu", weights_only=False
            )
            self._precomputed_text_emb = _expand_precomputed_caption_keys(
                blob["embeddings"]
            )
            if "" not in self._precomputed_text_emb:
                raise KeyError(
                    'precomputed embeddings must include empty string key "" for CFG / dropout.'
                )
            self.text_dim = int(blob.get("text_dim", 768))
            print(
                f"Loaded precomputed text embeddings from {precomputed_text_emb_path}: "
                f"{len(self._precomputed_text_emb)} keys, text_dim={self.text_dim}"
            )
        else:
            hf_cfg = AutoConfig.from_pretrained(model_name)
            self.text_dim = int(
                getattr(hf_cfg, "d_model", None)
                or getattr(hf_cfg, "hidden_size", None)
                or 768
            )
            print(f"Loading {model_name} from HuggingFace...")
            self.text_encoder = HFT5Encoder(
                text_len=text_len,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                model_name=model_name,
            )
        traj_enc_dim = self.traj_out_dim if self.use_traj_cond else 0
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
            traj_enc_dim=traj_enc_dim,
        )
        if self.use_traj_cond:
            self.traj_encoder = TrajEncoder(
                in_dim=self.traj_in_dim, hidden_dim=64, out_dim=self.traj_out_dim
            )
        else:
            self.traj_encoder = None
        self.param_dtype = torch.float32
        self._traj_stream_version = 0
        self._traj_emb_cache = {}

        # Optionally freeze backbone when adding trajectory branch, so we only train traj-related parts
        if freeze_backbone_for_traj:
            # 1) HuggingFace encoder 已在 HFT5Encoder 内部 requires_grad_(False)，且包装类无 .parameters()，这里不再遍历冻结

            # 2) Freeze WanModel backbone; keep traj token projections trainable
            trainable_substrings = (
                "traj_in_proj",
                "traj_type_embed",
            )
            for name, p in self.model.named_parameters():
                if any(s in name for s in trainable_substrings):
                    continue
                p.requires_grad = False

            # 3) Keep TrajEncoder trainable (trajectory branch)
            if self.traj_encoder is not None:
                for p in self.traj_encoder.parameters():
                    p.requires_grad = True

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
                        "Caption not in precomputed T5 (tiny) table. "
                        f"len={len(text)} preview={preview!r}. "
                        "Re-run pretokenize_t5_text_tiny.py with the same config "
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
        return build_traj_emb(
            x,
            seq_len,
            device,
            self.traj_encoder,
            self.use_traj_cond,
            self.traj_drop_out,
            training_dropout,
        )

    def _get_traj_seq_lens(self, x, seq_len, device):
        if "traj_features_length" in x and x["traj_features_length"] is not None:
            return (
                x["traj_features_length"]
                .to(device=device, dtype=torch.long)
                .clamp(min=0, max=seq_len)
            )
        if "traj_length" in x and x["traj_length"] is not None:
            return (
                (x["traj_length"].to(device=device, dtype=torch.long) // 4)
                .clamp(min=0, max=seq_len)
            )
        return None

    def _get_noise_levels(self, device, seq_len, time_steps):
        """Get noise levels"""
        # noise_level[i] = clip(1 + i / chunk_size - time_steps, 0, 1)
        noise_level = torch.clamp(
            1
            + torch.arange(seq_len, device=device) / self.chunk_size
            - time_steps.unsqueeze(1),
            min=0.0,
            max=1.0,
        )
        return noise_level

    def add_noise(self, x, noise_level):
        """Add noise
        Args:
            x: (B, T, D)
            noise_level: (B, T)
        """
        noise = torch.randn_like(x)
        # noise_level: (B, T) -> (B, T, 1)
        noise_level = noise_level.unsqueeze(-1)
        noisy_x = x * (1 - noise_level) + noise_level * noise
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

        # # Debug: Print noise levels
        # print("Time steps and corresponding noise levels:")
        # for i in range(batch_size):
        #     t = time_steps[i].item()
        #     # Get noise level at each position
        #     start_idx = int(self.chunk_size * (t - 1))
        #     end_idx = int(self.chunk_size * t) + 2
        #     # Limit to valid range
        #     start_idx = max(0, start_idx)
        #     end_idx = min(seq_len, end_idx)
        #     print(time_steps[i])
        #     print(noise_level[i, start_idx:end_idx])

        # Add noise to entire sequence
        noisy_feature, noise = self.add_noise(feature, noise_level)  # (B, T, D)

        # Debug: Print noise addition information
        # print("Added noise levels at chunk positions:")
        # for i in range(batch_size):
        #     t = time_steps[i].item()
        #     start_idx = int(self.chunk_size * (t - 1))
        #     end_idx = int(self.chunk_size * t) + 2
        #     # Limit to valid range
        #     start_idx = max(0, start_idx)
        #     end_idx = min(seq_len, end_idx)
        #     test1 = (
        #         feature[i, start_idx:end_idx, :] - noisy_feature[i, start_idx:end_idx, :]
        #     )
        #     test2 = (
        #         noise[i, start_idx:end_idx, :] - noisy_feature[i, start_idx:end_idx, :]
        #     )
        #     # Compute length on last dimension
        #     print(test1.norm(dim=-1))
        #     print(test2.norm(dim=-1))

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
                    if np.random.rand() > self.drop_out:
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
                all_text_context = [
                    (u if np.random.rand() > self.drop_out else "") for u in text_list
                ]
                all_text_context = self.encode_text_with_cache(all_text_context, device)
                all_text_context = [u.to(self.param_dtype) for u in all_text_context]
        else:
            all_text_context = [""] * batch_size
            all_text_context = self.encode_text_with_cache(all_text_context, device)
            all_text_context = [u.to(self.param_dtype) for u in all_text_context]

        traj_emb = self._build_traj_emb(x, seq_len, device, training_dropout=True)
        traj_seq_lens = self._get_traj_seq_lens(x, seq_len, device)

        # Through WanModel
        predicted_result = self.model(
            noisy_feature_input,
            noise_level * self.time_embedding_scale,
            all_text_context,
            seq_len,
            y=None,
            traj_emb=traj_emb,
            traj_seq_lens=traj_seq_lens,
        )  # (B, C, T, 1, 1)

        loss = 0.0
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

        # MotionLCM-style control loss: prepare pred_x0_latent for decoding
        if (
            self.use_traj_cond
            and "traj" in x
            and self.prediction_type in ("vel", "x0")
        ):
            pred_x0_latent_list = []
            for b in range(batch_size):
                if self.prediction_type == "vel":
                    # Use z = noisy_x + β·vel (numerically stable at all β);
                    # `vel + ε` collapses to z + ε at low β because the model
                    # cannot recover ε from a near-clean input.
                    t_b = noisy_feature_input[b].shape[1]
                    beta_b = noise_level[b, :t_b].view(1, -1, 1, 1)
                    pred_x0 = noisy_feature_input[b] + beta_b * predicted_result[b]
                else:
                    pred_x0 = predicted_result[b]
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

        # # debug
        # x["text"] = [["walk forward.", "sit down.", "stand up."] for _ in range(batch_size)]
        # x["feature_text_end"] = [[1, 2, 3] for _ in range(batch_size)]
        # text = x["text"]
        # text_end = x["feature_text_end"]
        # print(text)
        # print(text_end)
        # print(batch_size, seq_len, self.chunk_size)

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
        traj_emb = self._build_traj_emb(x, gen_seq_len, device, training_dropout=False)
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

            predicted_result = self.model(
                noisy_input,
                noise_level * self.time_embedding_scale,
                all_text_context,
                seq_len + self.chunk_size,
                y=None,
                traj_emb=traj_emb,
                traj_seq_lens=traj_seq_lens,
            )  # (B, C, T, 1, 1)

            # Adjust using CFG
            if self.cfg_scale_text != 1.0:
                predicted_result_null = self.model(
                    noisy_input,
                    noise_level * self.time_embedding_scale,
                    text_null_context,
                    seq_len + self.chunk_size,
                    y=None,
                    traj_emb=traj_emb,
                    traj_seq_lens=traj_seq_lens,
                )  # (B, C, T, 1, 1)
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

        # # debug
        # x["text"] = [["walk forward.", "sit down.", "stand up."] for _ in range(batch_size)]
        # x["feature_text_end"] = [[1, 2, 3] for _ in range(batch_size)]
        # text = x["text"]
        # text_end = x["feature_text_end"]
        # print(text)
        # print(text_end)
        # print(batch_size, seq_len, self.chunk_size)

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
        traj_emb = self._build_traj_emb(x, gen_seq_len, device, training_dropout=False)
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

            predicted_result = self.model(
                noisy_input,
                noise_level * self.time_embedding_scale,
                all_text_context,
                seq_len + self.chunk_size,
                y=None,
                traj_emb=traj_emb,
                traj_seq_lens=traj_seq_lens,
            )  # (B, C, T, 1, 1)

            # Adjust using CFG
            if self.cfg_scale_text != 1.0:
                predicted_result_null = self.model(
                    noisy_input,
                    noise_level * self.time_embedding_scale,
                    text_null_context,
                    seq_len + self.chunk_size,
                    y=None,
                    traj_emb=traj_emb,
                    traj_seq_lens=traj_seq_lens,
                )  # (B, C, T, 1, 1)
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
        self.traj_buffer = None
        self.traj_features_buffer = None
        self.token_mask_buffer = None
        self._traj_stream_version = 0
        self._traj_emb_cache = {}

    def _stream_update_traj_buffers(self, x, device):
        if not self.use_traj_cond or self.traj_encoder is None:
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
        if not self.use_traj_cond or self.traj_encoder is None:
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
            emb = self.traj_encoder(root_to_traj_feats(traj_slice))
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

            # print("////////////////////")
            # print("current step: ", self.current_step)
            # print("chunk size: ", self.chunk_size)
            # print("start_index: ", start_index)
            # print("end_index: ", end_index)
            # print("noisy_input shape: ", noisy_input[0].shape)
            # print("noise_level: ", noise_level[0, start_index:end_index])
            # print("text_condition shape: ", len(text_condition))
            # print("commit_index: ", self.commit_index)
            # print("////////////////////")

            ctx_len = min(end_index, self.seq_len)
            traj_emb = self._stream_build_traj_emb(x, end_index, device)
            traj_seq_lens = (
                torch.full(
                    (self.batch_size,),
                    ctx_len,
                    device=device,
                    dtype=torch.long,
                )
                if traj_emb is not None
                else None
            )

            predicted_result = self.model(
                noisy_input,
                noise_level * self.time_embedding_scale,
                text_condition,
                ctx_len,
                y=None,
                traj_emb=traj_emb,
                traj_seq_lens=traj_seq_lens,
            )  # (B, C, T, 1, 1)

            # Adjust using CFG
            if self.cfg_scale_text != 1.0:
                predicted_result_null = self.model(
                    noisy_input,
                    noise_level * self.time_embedding_scale,
                    text_null_context,
                    ctx_len,
                    y=None,
                    traj_emb=traj_emb,
                    traj_seq_lens=traj_seq_lens,
                )  # (B, C, T, 1, 1)
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
