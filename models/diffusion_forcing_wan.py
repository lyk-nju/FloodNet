import warnings

import torch
import torch.nn as nn

from .tools.t5 import T5EncoderModel
from .tools.traj_encoder import TrajectoryEncoder
from .tools.wan_model import WanModel
from .tools.wan_controlnet import WanControlNet
from utils.ldf_condition import LDFCondition


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
        traj_out_dim=128,
        traj_in_dim=7,
        traj_encoder_in_dim=None,
        traj_dropout=0.1,
        use_traj_emb_cache=False,
        use_traj_kv_cache=None,
        control_loss_weight=1.0,
        freeze_backbone=True,
        build_text_encoder=True,
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
        self.freeze_backbone = bool(freeze_backbone)

        self.traj_out_dim = traj_out_dim
        self.traj_in_dim = traj_in_dim
        self.traj_dropout = traj_dropout

        if use_traj_kv_cache is not None:
            warnings.warn(
                "`use_traj_kv_cache` is deprecated; use `use_traj_emb_cache`.",
                stacklevel=2,
            )
            use_traj_emb_cache = bool(use_traj_kv_cache)
        self.use_traj_emb_cache = use_traj_emb_cache

        self.text_dim = 4096
        self.text_len = text_len
        self._precomputed_text_emb = None
        self.text_encoder = None

        if build_text_encoder:
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

        self.traj_encoder = TrajectoryEncoder(
            in_dim=self.traj_in_dim,
            out_dim=self.traj_out_dim,
        )
        self.param_dtype = torch.float32

        if self.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.controlnet.parameters():
                p.requires_grad = True
            for p in self.traj_encoder.parameters():
                p.requires_grad = True
            # mask_emb lives in the frozen backbone but is trained by history
            # corruption, so keep it trainable.
            if hasattr(self.model, "mask_emb"):
                self.model.mask_emb.requires_grad_(True)

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

    def _resolve_condition(
        self,
        batch,
        condition: LDFCondition | None,
        *,
        batch_size: int,
    ) -> LDFCondition:
        if condition is None:
            condition = batch.get("ldf_condition") if isinstance(batch, dict) else None
        if condition is None:
            raise ValueError(
                "DiffForcingWanModel requires prepared LDFCondition. "
                "Build it outside the model before calling generate or stream_generate_step."
            )
        if condition.text_null_context is None:
            raise ValueError("LDFCondition.text_null_context is required for CFG.")
        condition.validate(batch_size=batch_size)
        return condition

    def _resolve_stream_condition(
        self,
        batch,
        condition,
        *,
        end_index: int,
        model_sl: int,
        window_start_token: int,
        time_steps,
        device,
    ) -> LDFCondition:
        if condition is None and isinstance(batch, dict):
            condition = batch.get("ldf_condition_provider", batch.get("ldf_condition"))
        if callable(condition):
            condition = condition(
                end_index=end_index,
                model_sl=model_sl,
                window_start_token=window_start_token,
                time_steps=time_steps,
                device=device,
            )
        return self._resolve_condition({}, condition, batch_size=self.batch_size)

    def encode_text_with_cache(self, text_list, device):
        if self._precomputed_text_emb is not None:
            out = []
            for text in text_list:
                row = self._precomputed_text_emb.get(text)
                if row is None:
                    row = self._precomputed_text_emb.get(text.strip())
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

        for i, text in enumerate(text_list):
            if text in self.text_cache:
                text_features.append(self.text_cache[text].to(device))
            else:
                text_features.append(None)
                indices_to_encode.append(i)
                texts_to_encode.append(text)

        if texts_to_encode:
            self.text_encoder.model.to(device)
            encoded = self.text_encoder(texts_to_encode, device)
            for idx, text, feature in zip(indices_to_encode, texts_to_encode, encoded):
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

    def _get_noise_levels(self, device, seq_len, time_steps):
        """Get vectorized triangular noise levels."""
        noise_level = torch.clamp(
            1
            + torch.arange(seq_len, device=device) / self.chunk_size
            - time_steps.unsqueeze(1),
            min=0.0,
            max=1.0,
        )
        return noise_level

    def add_noise(self, x, noise_level):
        """Add noise with x_t = alpha_t * z + beta_t * eps.

        Args:
            x: (B, T, D) clean latent z
            noise_level: (B, T) beta_t
        """
        noise = torch.randn_like(x)
        noise_level = noise_level.unsqueeze(-1)
        noisy_x = x * (1 - noise_level) + noise_level * noise
        return noisy_x, noise

    def forward(
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
        ctx_double = None
        if self.cfg_scale_text != 1.0:
            # Text CFG supports per-sample context (B) and token-expanded
            # stream context (B * seq_len).
            n_text_ctx = len(text_cond_ctx)
            if n_text_ctx == batch_size:
                ctx_double = list(text_cond_ctx) + list(text_null_ctx)
            elif n_text_ctx == batch_size * seq_len:
                null_flat = []
                for i in range(batch_size):
                    for _ in range(seq_len):
                        null_flat.append(text_null_ctx[i])
                ctx_double = list(text_cond_ctx) + null_flat

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
                # Separated CFG: batch all 3 passes into one 3B backbone forward.
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

    def generate(self, x, *, condition: LDFCondition | None = None, num_denoise_steps=None):
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
        condition = self._resolve_condition(x, condition, batch_size=batch_size)

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

        gen_seq_len = seq_len + self.chunk_size
        generated_length = feature_length
        full_text = x.get("output_text", x.get("text", [""] * batch_size))
        if isinstance(full_text, list) and full_text and isinstance(full_text[0], list):
            full_text = [" ////////// ".join(map(str, item)) for item in full_text]

        # Progressively advance from t=0 to t=max_t
        latent_attn_len = condition.seq_len
        if latent_attn_len is None:
            latent_attn_len = gen_seq_len
        latent_attn_len = int(latent_attn_len)
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

            if latent_attn_len == gen_seq_len:
                noise_level_for_attn = noise_level
            else:
                noise_level_for_attn = self._get_noise_levels(
                    device, latent_attn_len, time_steps
                )
            t_scaled = noise_level_for_attn * self.time_embedding_scale
            predicted_result = self._denoise_with_cfg(
                noisy_input, t_scaled,
                condition.text_context, condition.text_null_context,
                condition.traj_emb, condition.traj_seq_lens, latent_attn_len, batch_size,
                traj_token_mask=condition.traj_token_mask,
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
    def stream_generate(self, x, *, condition: LDFCondition | None = None, num_denoise_steps=None):
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
        condition = self._resolve_condition(x, condition, batch_size=batch_size)

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

        gen_seq_len = seq_len + self.chunk_size
        generated_length = feature_length
        full_text = x.get("output_text", x.get("text", [""] * batch_size))
        if isinstance(full_text, list) and full_text and isinstance(full_text[0], list):
            full_text = [" ////////// ".join(map(str, item)) for item in full_text]

        commit_index = 0
        # Progressively advance from t=0 to t=max_t
        latent_attn_len = condition.seq_len
        if latent_attn_len is None:
            latent_attn_len = gen_seq_len
        latent_attn_len = int(latent_attn_len)
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

            if latent_attn_len == gen_seq_len:
                noise_level_for_attn = noise_level
            else:
                noise_level_for_attn = self._get_noise_levels(
                    device, latent_attn_len, time_steps
                )
            t_scaled = noise_level_for_attn * self.time_embedding_scale
            predicted_result = self._denoise_with_cfg(
                noisy_input, t_scaled,
                condition.text_context, condition.text_null_context,
                condition.traj_emb, condition.traj_seq_lens, latent_attn_len, batch_size,
                traj_token_mask=condition.traj_token_mask,
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

    def init_generated(
        self,
        seq_len,
        batch_size=1,
        num_denoise_steps=None,
        traj_buffer=None,
    ):
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
        self._traj_buf = traj_buffer

    @torch.no_grad()
    def stream_generate_step(self, x=None, first_chunk=True, condition=None):
        """
        Streaming generation step - Diffusion Forcing inference
        Uses triangular noise schedule, progressively generating from left to right

        Generation process:
        1. Start from t=0, gradually increase t
        2. Each t corresponds to a noise schedule: clean on left, noisy on right, gradient in middle
        3. After each denoising step, t increases slightly and continues
        """

        if x is None:
            x = {}
        device = next(self.parameters()).device
        if first_chunk:
            self.generated = self.generated.to(device)

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

            model_sl = min(end_index, self.seq_len)
            window_start_token = max(0, end_index - model_sl)
            step_condition = self._resolve_stream_condition(
                x,
                condition,
                end_index=end_index,
                model_sl=model_sl,
                window_start_token=window_start_token,
                time_steps=time_steps,
                device=device,
            )
            condition_seq_len = step_condition.seq_len
            if condition_seq_len is not None and int(condition_seq_len) != int(model_sl):
                raise ValueError(
                    "stream condition seq_len must match the latent window length: "
                    f"got {condition_seq_len}, expected {model_sl}"
                )
            t_scaled = noise_level * self.time_embedding_scale
            predicted_result = self._denoise_with_cfg(
                noisy_input, t_scaled,
                step_condition.text_context, step_condition.text_null_context,
                step_condition.traj_emb, step_condition.traj_seq_lens, model_sl,
                self.batch_size,
                traj_token_mask=step_condition.traj_token_mask,
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
            if self._traj_buf is not None:
                self._traj_buf.roll(self.seq_len, device)
            self.current_step -= self.seq_len * self.num_denoise_steps / self.chunk_size
            self.commit_index -= self.seq_len
            for i in range(self.batch_size):
                self.text_condition_list[i] = self.text_condition_list[i][
                    self.seq_len :
                ]
        return out
