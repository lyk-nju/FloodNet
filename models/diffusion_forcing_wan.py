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


_SCHEDULED_SAMPLING_WARNED = False


def warn_scheduled_sampling_deprecated(value) -> bool:
    """Warn once per process that ``scheduled_sampling_prob`` is deprecated.

    T_B_01 removed scheduled sampling (replaced by history_corruption, see
    design.md §2.1.4). The config key is kept for backward-compat (so existing
    configs load without raising) but ignored.

    Only warns when a **nonzero** value was requested — 0.0 (the historical
    default in every shipped config) means scheduled sampling was already off,
    so there is nothing to migrate and warning on it every run would be pure
    noise. Returns True iff a warning was emitted.
    """
    global _SCHEDULED_SAMPLING_WARNED
    try:
        nonzero = float(value) != 0.0
    except (TypeError, ValueError):
        nonzero = False
    if nonzero and not _SCHEDULED_SAMPLING_WARNED:
        warnings.warn(
            "scheduled_sampling_prob is deprecated and ignored (removed in "
            "T_B_01); replaced by history_corruption.apply_prob (see "
            "design.md §2.1.4).",
            DeprecationWarning,
            stacklevel=2,
        )
        _SCHEDULED_SAMPLING_WARNED = True
        return True
    return False


# Optional fields added by later phases (T_B_02: history-corruption support).
# Old (4D) checkpoints predate them, so a strict load would raise on the missing
# keys. We back-fill them from the freshly-initialized module so older ckpts load
# unchanged — the new fields simply keep their init values (mask_emb at its
# random init; z_mean/z_std at 0/1 until load_z_stats is called).
# Matched by leaf name so both bare (`mask_emb`, WanModel loaded directly) and
# prefixed (`model.mask_emb`, WanModel nested in DiffForcingWanModel) keys work.
_BACKWARD_COMPAT_OPTIONAL_NAMES = ("mask_emb", "z_mean", "z_std")


def _is_optional_compat_key(key: str) -> bool:
    """True if `key`'s final path component is a T_B_02 optional field name."""
    return key.rsplit(".", 1)[-1] in _BACKWARD_COMPAT_OPTIONAL_NAMES


def backfill_compat_state_dict(state_dict: dict, own_state: dict):
    """Return (filled_state_dict, n_backfilled).

    Copies `state_dict`, then for every key in `own_state` that (a) is absent
    from the incoming `state_dict` and (b) is an optional T_B_02 field (matched
    by leaf name), fills it from `own_state`. All other keys are left untouched
    so a subsequent strict load still catches genuine missing / unexpected /
    shape-mismatch errors.
    """
    filled = dict(state_dict)
    n = 0
    for k, v in own_state.items():
        if k not in filled and _is_optional_compat_key(k):
            filled[k] = v.clone() if hasattr(v, "clone") else v
            n += 1
    return filled, n


def _expand_precomputed_caption_keys(emb: dict) -> dict:
    """Alias strip() keys so table matches HumanML3D captions after .strip()."""
    out = dict(emb)
    for k, v in emb.items():
        s = k.strip()
        if s not in out:
            out[s] = v
    return out


class DiffForcingWanModel(nn.Module):
    """Diffusion Forcing with streaming trajectory ControlNet conditioning.

    Triangular noise schedule (left-clean, right-noisy) with per-sample time
    steps. Supports velocity / x0 / noise prediction with CFG over text and
    trajectory conditions.
    """
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
        traj_out_dim=128,
        traj_in_dim=7,
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
        # T_B_01: scheduled_sampling is removed (replaced by history_corruption,
        # design.md §2.1.4). The constructor param is retained so existing
        # configs that still carry `scheduled_sampling_prob` load without raising
        # (lightning_module splats **cfg.model.params), but it is ignored — warn
        # if a caller actually requested a nonzero value.
        warn_scheduled_sampling_deprecated(scheduled_sampling_prob)
        self.scheduled_sampling_prob = 0.0
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

        # Local within-token Conv1d encoder over the 4 frames of a token, then
        # a token-level LayerNorm + MLP. 7D-only — see models/tools/traj_encoder.py.
        self.local_traj_encoder = LocalTrajEncoder(in_dim=self.traj_in_dim)
        self.traj_encoder = TrajEncoder(out_dim=self.traj_out_dim)
        self.param_dtype = torch.float32

        if self.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.controlnet.parameters():
                p.requires_grad = True
            for p in self.traj_encoder.parameters():
                p.requires_grad = True
            for p in self.local_traj_encoder.parameters():
                p.requires_grad = True
            # B-P0-2: mask_emb is the LEARNABLE history-corruption replacement
            # embedding (T_B_02/T_B_03); it lives inside self.model so the freeze
            # loop above froze it. Keep it trainable for the 7D fine-tune.
            if hasattr(self.model, "mask_emb"):
                self.model.mask_emb.requires_grad_(True)

    def load_state_dict(self, state_dict, strict=True):
        """Backward-compatible load: when loading an older ckpt (strict=True)
        that predates the T_B_02 optional fields (model.mask_emb / model.z_mean /
        model.z_std), back-fill them from this module's init values so the load
        doesn't raise on missing keys. Genuine missing / unexpected / mismatched
        keys are still caught by the strict load.
        """
        if strict:
            state_dict, n = backfill_compat_state_dict(state_dict, self.state_dict())
            if n:
                warnings.warn(
                    f"load_state_dict: back-filled {n} optional field(s) "
                    f"(mask_emb/z_mean/z_std) from init — loading a checkpoint that "
                    f"predates T_B_02. New fields keep init values until "
                    f"load_z_stats() is called.",
                    stacklevel=2,
                )
        return super().load_state_dict(state_dict, strict=strict)

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
        return self.controlnet(
            noisy_input,
            t_scaled,
            text_context,
            seq_len,
            y=None,
            traj_emb=traj_emb,
            traj_seq_lens=traj_seq_lens,
            traj_token_mask=traj_token_mask,
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

    def _build_traj_emb(self, x, seq_len, device, horizon_tokens=None,
                        horizon_active_end=0, return_token_mask=False):
        if self.traj_encoder is None:
            return (None, None) if return_token_mask else None
        return encode_traj_batch(
            x, seq_len, device, self.local_traj_encoder, self.traj_encoder,
            horizon_tokens=horizon_tokens,
            horizon_active_end_token=horizon_active_end,
            return_token_mask=return_token_mask,
        )

    def _get_traj_seq_lens(self, x, seq_len, device, horizon_tokens=None,
                           horizon_active_end=0):
        # Base token length from feature_length / traj_length (exact).
        base = None
        if "feature_length" in x and x["feature_length"] is not None:
            base = x["feature_length"].to(device=device, dtype=torch.long).clamp(min=0, max=seq_len)
        elif "traj_features_length" in x and x["traj_features_length"] is not None:
            base = x["traj_features_length"].to(device=device, dtype=torch.long).clamp(min=0, max=seq_len)
        elif "traj_length" in x and x["traj_length"] is not None:
            tl = x["traj_length"].to(device=device, dtype=torch.long)
            # Vectorized mirror of token_frame.num_tokens_for_frame_len
            # (= frame_idx_to_token_idx(tl-1)+1): 0 for tl<=0, 1 for tl==1,
            # (tl-2)//4 + 2 for tl>=2. Replaces the old opaque (tl+2)//4+1.
            tokens = torch.where(
                tl <= 1, tl.clamp(min=0, max=1), (tl - 2) // 4 + 2,
            )
            base = tokens.clamp(min=0, max=seq_len)
        if base is None:
            return None

        # B-P0-1: mask-aware truncation so ControlNet attention ignores
        # out-of-horizon / overflow tail tokens (their traj_type_embed would leak
        # otherwise). Reuses the SAME token mask as encode_traj_batch
        # (build_traj_token_mask — single source). prefix_len_from_tail_invalid
        # truncates ONLY a pure-suffix invalid region; a middle hole returns
        # seq_len so min() keeps base (sparse holes must NOT shorten attention —
        # they are handled by per-token embedding zeroing). The end-to-end
        # ControlNet-residual value-invariance check (real VAE-latent shapes) is
        # run on the runtime box; the truncation logic is unit-tested here.
        from utils.token_frame import prefix_len_from_tail_invalid
        from utils.traj_batch import build_traj_token_mask
        token_mask = build_traj_token_mask(
            x, seq_len, device, horizon_tokens=horizon_tokens,
            horizon_active_end_token=horizon_active_end,
        )
        if token_mask is None:
            return base
        prefix = prefix_len_from_tail_invalid(token_mask).to(device=device)
        return torch.minimum(base, prefix)

    def _decide_text_dropout(self, batch_size: int, device) -> list:
        """Sample per-sample text-dropout flags, synced across DDP ranks."""
        if not self.training:
            return [False] * batch_size
        drop = torch.empty(batch_size, device=device).uniform_()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(drop, src=0)
        return (drop < self.text_dropout).tolist()

    def _prepare_text_context(self, x, seq_len, device, text_dropped_flags=None):
        if self.use_text_cond and "text" in x:
            text_list = x["text"]  # List[str] or List[List[str]]
            if isinstance(text_list[0], list):
                text_end_list = x["feature_text_end"]
                all_text_context = []
                for i, (single_text_list, single_text_end_list) in enumerate(
                    zip(text_list, text_end_list)
                ):
                    sample_dropped = (
                        text_dropped_flags[i]
                        if text_dropped_flags is not None
                        else False
                    )
                    if (not self.training) or (not sample_dropped):
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
                if self.training and text_dropped_flags is not None:
                    all_text_context = [
                        ("" if text_dropped_flags[i] else u)
                        for i, u in enumerate(text_list)
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
        self, x, seq_len, device, traj_dropped_override=None, horizon_tokens=None,
        horizon_active_end=0,
    ):
        # T_B_04 / B-P0-2: horizon_tokens (token-level) + horizon_active_end (the
        # per-sample active-window token position, [B] tensor or int) are computed
        # by the training outer loop (SelfForcingTrainer) and passed in — the model
        # never reads global_step. horizon_active_end defaults to 0 (clip start)
        # for non-SF callers; SF passes the supervised window position so horizon
        # truncates relative to the current window, not the clip start.
        traj_emb = None
        traj_seq_lens = None
        traj_token_mask = None
        if traj_dropped_override is None:
            traj_dropped = self._decide_traj_dropout(device)
        else:
            traj_dropped = bool(traj_dropped_override)

        if not traj_dropped:
            traj_emb, traj_token_mask = self._build_traj_emb(
                x, seq_len, device, horizon_tokens=horizon_tokens,
                horizon_active_end=horizon_active_end,
                return_token_mask=True,
            )
            traj_seq_lens = self._get_traj_seq_lens(
                x, seq_len, device, horizon_tokens=horizon_tokens,
                horizon_active_end=horizon_active_end,
            )

        return traj_emb, traj_seq_lens, traj_dropped, traj_token_mask

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
        traj_token_mask=None,
    ):
        feature_length = x["feature_length"]
        batch_size, seq_len, _ = clean_feature.shape

        (
            noise_level,
            feature_ref,
            noise_ref,
            noisy_feature_input,
            end_indices,
        ) = self._slice_diffusion_window(clean_feature, feature_length, time_steps)

        # (T_B_01: scheduled-sampling forward branch removed — it was dead since
        # scheduled_sampling_prob defaulted to 0.0 everywhere, so do_ss was always
        # False and noisy_feature_input passed through unchanged. The
        # `_scheduled_sampling_override` input key is no longer consumed.)

        # Always call ControlNet so gradients flow when backbone is frozen.
        # When traj_dropped=True, traj_emb is already None; ControlNet learns
        # to produce near-zero residuals for null-traj input, which closely
        # approximates the zero residuals used by pred_uncond at inference.
        controlnet_residuals = self._controlnet_forward(
            noisy_feature_input,
            noise_level * self.time_embedding_scale,
            all_text_context,
            seq_len,
            traj_emb,
            traj_seq_lens,
            traj_token_mask=traj_token_mask,
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
        text_dropped_flags = self._decide_text_dropout(batch_size, device)
        all_text_context = self._prepare_text_context(x, seq_len, device, text_dropped_flags)
        traj_emb, traj_seq_lens, traj_dropped, traj_token_mask = self._prepare_traj_condition(
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
            traj_token_mask=traj_token_mask,
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
        traj_token_mask=None,
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

        if traj_emb is None:
            if ctx_double is not None:
                noisy_double = list(noisy_input) + list(noisy_input)
                t_double = torch.cat([t_scaled, t_scaled], dim=0)
                residuals_double = self._controlnet_forward(
                    noisy_double,
                    t_double,
                    ctx_double,
                    seq_len,
                    traj_emb=None,
                    traj_seq_lens=None,
                    traj_token_mask=None,
                )
                pred_double = self.model(
                    noisy_double, t_double, ctx_double, seq_len,
                    y=None, traj_emb=None, traj_seq_lens=None,
                    controlnet_residuals=residuals_double,
                )
                return [
                    self.cfg_scale_text * pred_double[i]
                    - (self.cfg_scale_text - 1) * pred_double[i + batch_size]
                    for i in range(batch_size)
                ]
            residuals = self._controlnet_forward(
                noisy_input,
                t_scaled,
                text_cond_ctx,
                seq_len,
                traj_emb=None,
                traj_seq_lens=None,
                traj_token_mask=None,
            )
            pred = self.model(
                noisy_input, t_scaled, text_cond_ctx, seq_len,
                y=None, traj_emb=None, traj_seq_lens=None,
                controlnet_residuals=residuals,
            )
            if self.cfg_scale_text != 1.0:
                residuals_null = self._controlnet_forward(
                    noisy_input,
                    t_scaled,
                    text_null_ctx,
                    seq_len,
                    traj_emb=None,
                    traj_seq_lens=None,
                    traj_token_mask=None,
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
            traj_mask_double = (
                torch.cat([traj_token_mask, traj_token_mask], dim=0)
                if traj_token_mask is not None
                else None
            )
            residuals = self._controlnet_forward(
                noisy_double, t_double, ctx_double, seq_len, traj_double, traj_sl_double,
                traj_token_mask=traj_mask_double,
            )
            if self.cfg_scale_traj > 0.0:
                # Separated CFG — batch all 3 passes into a single 3B backbone forward.
                # ControlNet runs on 2B with traj and on B with null traj so the
                # uncond slot matches the project-wide no-traj semantics.
                #   out = out_uncond
                #       + w_text * (out_full - out_null_text+traj)   ← pure text effect, traj fixed
                #       + w_traj * (out_null_text+traj - out_uncond) ← pure traj effect, text=null
                noisy_triple = list(noisy_double) + list(noisy_input)
                t_triple = torch.cat([t_double, t_scaled], dim=0)
                if len(ctx_double) == 2 * batch_size:
                    # simple mode: one context per sample
                    ctx_triple = list(ctx_double) + list(text_null_ctx)
                else:
                    # frame-aligned mode: seq_len contexts per sample in ctx_double
                    # expand each null ctx over seq_len token positions so ctx_triple = 3*B*seq_len
                    null_flat_uncond = [ni for ni in text_null_ctx for _ in range(seq_len)]
                    ctx_triple = list(ctx_double) + null_flat_uncond
                residuals_uncond = self._controlnet_forward(
                    noisy_input,
                    t_scaled,
                    text_null_ctx,
                    seq_len,
                    traj_emb=None,
                    traj_seq_lens=None,
                    traj_token_mask=None,
                )
                residuals_triple = [
                    torch.cat([r, r_uncond], dim=0)
                    for r, r_uncond in zip(residuals, residuals_uncond)
                ]
                pred_triple = self.model(
                    noisy_triple, t_triple, ctx_triple, seq_len,
                    y=None, traj_emb=None, traj_seq_lens=None,
                    controlnet_residuals=residuals_triple,
                )
                return [
                    pred_triple[i + 2 * batch_size]
                    + self.cfg_scale_text * (pred_triple[i] - pred_triple[i + batch_size])
                    + self.cfg_scale_traj * (pred_triple[i + batch_size] - pred_triple[i + 2 * batch_size])
                    for i in range(batch_size)
                ]
            else:
                pred_double = self.model(
                    noisy_double, t_double, ctx_double, seq_len,
                    y=None, traj_emb=None, traj_seq_lens=None, controlnet_residuals=residuals,
                )
                return [
                    self.cfg_scale_text * pred_double[i]
                    - (self.cfg_scale_text - 1) * pred_double[i + batch_size]
                    for i in range(batch_size)
                ]
        else:
            residuals = self._controlnet_forward(
                noisy_input, t_scaled, text_cond_ctx, seq_len, traj_emb, traj_seq_lens,
                traj_token_mask=traj_token_mask,
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
                    noisy_input, t_scaled, text_null_ctx, seq_len, traj_emb, traj_seq_lens,
                    traj_token_mask=traj_token_mask,
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
        traj_emb, traj_token_mask = self._build_traj_emb(
            x, gen_seq_len, device, return_token_mask=True,
        )
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
                traj_token_mask=traj_token_mask,
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
        traj_token_mask = None
        traj_emb, traj_token_mask = self._build_traj_emb(
            x, gen_seq_len, device, return_token_mask=True,
        )
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
                traj_token_mask=traj_token_mask,
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

    def _build_stream_direct_traj_condition(
        self,
        x: dict,
        model_sl: int,
        window_start_token: int,
        device,
        traj_sl: int | None = None,
    ):
        """Encode an explicit frame-level 7D stream trajectory payload.

        The payload must already be window-relative and body-window-local. This
        helper only validates shape/start-token consistency and routes it
        through the same 7D encoder path used by training.
        """
        subpayloads = x.get("traj_substep_payloads")
        if subpayloads:
            selected = None
            for subpayload in subpayloads:
                if int(subpayload.get("traj_start_token", -1)) == int(window_start_token):
                    selected = subpayload
                    break
            if selected is None:
                starts = [
                    int(subpayload.get("traj_start_token", -1))
                    for subpayload in subpayloads
                ]
                raise ValueError(
                    "stream_generate_step 7D payload has no substep payload "
                    f"for window_start_token={window_start_token}; available "
                    f"starts={starts}."
                )
            x = selected

        from utils.token_frame import (
            frame_idx_to_token_idx,
            prefix_len_from_tail_invalid,
            token_range_to_frame_slice,
            token_start_frame,
        )

        traj_frame = x["traj_cond_7d_frame"]
        if isinstance(traj_frame, np.ndarray):
            traj_frame = torch.from_numpy(traj_frame).float()
        traj_frame = traj_frame.to(device=device)
        if traj_frame.dim() == 2:
            traj_frame = traj_frame.unsqueeze(0)
        if traj_frame.dim() != 3 or traj_frame.shape[-1] != 7:
            raise ValueError(
                "traj_cond_7d_frame must be [B,T_frame,7] or [T_frame,7], "
                f"got {tuple(traj_frame.shape)}"
            )
        if traj_frame.shape[0] != self.batch_size:
            raise ValueError(
                f"traj_cond_7d_frame batch size {traj_frame.shape[0]} does not "
                f"match stream batch_size {self.batch_size}"
            )

        payload_local_start = int(x.get("traj_start_token", window_start_token))
        payload_abs_start = int(x.get("traj_abs_start_token", payload_local_start))
        if payload_local_start > window_start_token:
            raise ValueError(
                "stream_generate_step 7D payload starts after current latent "
                "window start; got traj_start_token="
                f"{payload_local_start}, window_start_token={window_start_token}. "
                "Build direct 7D payloads from the earliest denoise substep "
                "window start or earlier."
            )
        payload_num_tokens = x.get("traj_num_tokens", None)
        if payload_num_tokens is not None:
            payload_num_tokens = int(payload_num_tokens)
            if payload_num_tokens < model_sl:
                raise ValueError(
                    "stream traj_num_tokens must be >= model_sl; got "
                    f"traj_num_tokens={payload_num_tokens}, model_sl={model_sl}."
                )
        if traj_sl is None:
            if payload_num_tokens is not None:
                traj_sl = payload_num_tokens
            elif traj_frame.shape[1] <= 0:
                traj_sl = model_sl
            else:
                origin_frame = token_start_frame(payload_abs_start)
                payload_last_frame = origin_frame + int(traj_frame.shape[1]) - 1
                payload_end_token = frame_idx_to_token_idx(payload_last_frame) + 1
                window_abs_start = payload_abs_start + (
                    window_start_token - payload_local_start
                )
                traj_sl = max(model_sl, payload_end_token - window_abs_start)
        traj_sl = int(traj_sl)
        if traj_sl < model_sl:
            raise ValueError(
                f"stream traj_sl must be >= model_sl; got traj_sl={traj_sl}, "
                f"model_sl={model_sl}."
            )

        window_abs_start = payload_abs_start + (
            window_start_token - payload_local_start
        )
        if payload_local_start < window_start_token:
            crop_tokens = window_start_token - payload_local_start
            if payload_num_tokens is not None:
                traj_sl = max(model_sl, payload_num_tokens - crop_tokens)
            origin_frame = token_start_frame(payload_abs_start)
            needed = token_range_to_frame_slice(window_abs_start, traj_sl)
            rel_start = needed.start - origin_frame
            rel_stop = needed.stop - origin_frame
            if rel_start >= traj_frame.shape[1]:
                traj_frame = traj_frame[:, :0, :]
            else:
                traj_frame = traj_frame[:, max(0, rel_start):min(rel_stop, traj_frame.shape[1]), :]
        else:
            window_abs_start = payload_abs_start

        traj_payload = {
            "traj_features": traj_frame,
            "traj_start_token": window_abs_start,
        }
        traj_mask = x.get("traj_cond_frame_mask", x.get("traj_cond_mask"))
        if traj_mask is not None:
            if isinstance(traj_mask, np.ndarray):
                traj_mask = torch.from_numpy(traj_mask).float()
            traj_mask = traj_mask.to(device=device)
            if traj_mask.dim() == 1:
                traj_mask = traj_mask.unsqueeze(0)
            if traj_mask.shape[0] != self.batch_size:
                raise ValueError(
                    f"traj_cond_frame_mask batch size {traj_mask.shape[0]} does not "
                    f"match stream batch_size {self.batch_size}"
                )
            if payload_local_start < window_start_token:
                if rel_start >= traj_mask.shape[1]:
                    traj_mask = traj_mask[:, :0]
                else:
                    traj_mask = traj_mask[:, max(0, rel_start):min(rel_stop, traj_mask.shape[1])]
            traj_payload["traj_cond_mask"] = traj_mask

        traj_emb, traj_token_mask = encode_traj_batch(
            traj_payload,
            traj_sl,
            device,
            self.local_traj_encoder,
            self.traj_encoder,
            return_token_mask=True,
        )
        if traj_emb is None:
            return None, None, None
        if traj_token_mask is not None:
            traj_seq_lens = prefix_len_from_tail_invalid(traj_token_mask).to(
                device=device
            )
        else:
            traj_seq_lens = torch.full(
                (self.batch_size,),
                traj_sl,
                device=device,
                dtype=torch.long,
            )
        return traj_emb, traj_seq_lens, traj_token_mask

    @staticmethod
    def _extend_stream_text_context_for_attention(
        text_condition: list,
        batch_size: int,
        model_sl: int,
        attn_sl: int,
    ) -> list:
        """Pad frame-aligned stream text context to the attention length.

        The latent rolling window provides ``model_sl`` text slots. Direct 7D
        trajectory conditioning can extend attention to future trajectory tokens,
        so frame-aligned text context must be padded to ``attn_sl``. Future
        slots reuse the last visible text context for each sample.
        """
        if attn_sl <= model_sl:
            return text_condition
        if len(text_condition) != batch_size * model_sl:
            if len(text_condition) == batch_size:
                return text_condition
            return text_condition
        out = []
        for i in range(batch_size):
            segment = list(text_condition[i * model_sl : (i + 1) * model_sl])
            if not segment:
                continue
            out.extend(segment)
            out.extend([segment[-1]] * (attn_sl - model_sl))
        return out

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

            noise_level_full = self._get_noise_levels(device, end_index, time_steps)
            noise_level = noise_level_full[:, -self.seq_len :]  # (B, seq_len)
            noise_level_for_update = noise_level_full  # (B, end_index), matches pred shape

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

            model_sl = min(end_index, self.seq_len)
            window_start_token = max(0, end_index - model_sl)
            attn_sl = model_sl
            if x.get("traj_cond_7d_frame") is not None:
                traj_emb, traj_seq_lens, traj_token_mask = (
                    self._build_stream_direct_traj_condition(
                        x,
                        model_sl,
                        window_start_token,
                        device,
                    )
                )
                if traj_emb is not None:
                    attn_sl = max(model_sl, int(traj_emb.shape[1]))
            else:
                traj_emb = self._traj_buf.build_traj_emb(end_index, self.seq_len, device)
                if traj_emb is not None:
                    valid_lens = self._traj_buf.get_traj_valid_lens(
                        end_index, self.seq_len, device
                    )
                    traj_seq_lens = (
                        valid_lens
                        if valid_lens is not None
                        else torch.full(
                            (self.batch_size,),
                            model_sl,
                            device=device,
                            dtype=torch.long,
                        )
                    )
                    traj_token_mask = self._traj_buf.get_traj_token_mask(
                        end_index, self.seq_len, device
                    )
                else:
                    traj_seq_lens = None
                    traj_token_mask = None
            text_condition_attn = self._extend_stream_text_context_for_attention(
                text_condition,
                self.batch_size,
                model_sl,
                attn_sl,
            )
            if attn_sl == model_sl:
                noise_level_for_attn = noise_level
            else:
                noise_level_attn_full = self._get_noise_levels(
                    device,
                    window_start_token + attn_sl,
                    time_steps,
                )
                noise_level_for_attn = noise_level_attn_full[:, -attn_sl:]
            t_scaled = noise_level_for_attn * self.time_embedding_scale
            predicted_result = self._denoise_with_cfg(
                noisy_input, t_scaled,
                text_condition_attn, text_null_context,
                traj_emb, traj_seq_lens, attn_sl, self.batch_size,
                traj_token_mask=traj_token_mask,
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
                        noise_level_for_update[i, start_index:end_index]
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
                        - noise_level_for_update[i, start_index:end_index]
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
