# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from __future__ import annotations

import os
import warnings

import torch

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import torch.nn.functional as F

__all__ = [
    "flash_attention",
    "attention",
    "flextraj_self_attn_bias",
    "flextraj_query_valid",
    "flextraj_sdpa_self_attention",
    "flextraj_flash_split_self_attention",
    "flextraj_self_attention",
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
            device=q.device, non_blocking=True
        )
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
            device=k.device, non_blocking=True
        )
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            "Flash attention 3 is not available, use flash attention 2 instead."
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
            .cumsum(0, dtype=torch.int32)
            .to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        ).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                "Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance."
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p
        )

        out = out.transpose(1, 2).contiguous()
        return out


def _flextraj_vl_vt(
    seq_lens: torch.Tensor,
    n_latent: int,
    device: torch.device,
    *,
    n_traj: int | None = None,
    latent_lens: torch.Tensor | None = None,
    traj_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_traj = n_latent if n_traj is None else int(n_traj)
    if latent_lens is not None or traj_lens is not None:
        assert latent_lens is not None and traj_lens is not None
        vl = latent_lens.to(device=device, dtype=torch.long).clamp(min=0, max=n_latent)
        vt = traj_lens.to(device=device, dtype=torch.long).clamp(min=0, max=n_traj)
        return vl, vt
    seq_lens = seq_lens.to(device=device, dtype=torch.long)
    half = seq_lens // 2
    vl = torch.minimum(half, torch.full_like(half, n_latent))
    vt = torch.minimum(seq_lens - half, torch.full_like(half, n_traj))
    return vl, vt


def flextraj_query_valid(
    seq_lens: torch.Tensor,
    n_latent: int,
    total_len: int,
    latent_lens: torch.Tensor | None = None,
    traj_lens: torch.Tensor | None = None,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """(B, L, 1) — 有效 query 行为 1，pad query 为 0。"""
    b = int(seq_lens.shape[0])
    L = total_len
    n_traj = max(0, L - n_latent)
    vl, vt = _flextraj_vl_vt(
        seq_lens,
        n_latent,
        device,
        n_traj=n_traj,
        latent_lens=latent_lens,
        traj_lens=traj_lens,
    )
    idx_i = torch.arange(L, device=device)
    qv = (
        (idx_i.view(1, L) < vl.view(b, 1))
        | (
            (idx_i.view(1, L) >= n_latent)
            & (idx_i.view(1, L) < (n_latent + vt.view(b, 1)))
        )
    ).to(dtype=dtype)
    return qv.unsqueeze(-1)


def flextraj_self_attn_bias(
    seq_lens: torch.Tensor,
    n_latent: int,
    total_len: int,
    latent_lens: torch.Tensor | None = None,
    traj_lens: torch.Tensor | None = None,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Task4 块稀疏：latent query 可看 latent+traj 有效 key；traj 段任意 query（含 traj pad 槽位）
    不能看 latent key；padding key 列对全体 query 关闭。

    Args:
        seq_lens: (B,) 每条样本 **有效 token 总数**（与 WanModel 一致：valid_latent + valid_traj，常为 2×帧数）。
        n_latent: latent 段 pad 长度；traj 段长度由 ``total_len - n_latent`` 推导。
        total_len: self-attn 序列长度 L。
        causal: 为 True 时在掩码上再叠加标准因果（j > i 禁止）。

    Returns:
        attn_bias: (B, 1, L, L)，加到 attention logits（0 允许，大负数禁止）。
        query_valid: (B, L, 1)，有效 query 行 1，pad query 行 0（乘到输出上避免全 -inf 行）。
    """
    b = int(seq_lens.shape[0])
    L = total_len
    if L < n_latent:
        raise ValueError(
            f"total_len must be >= n_latent, got total_len={L}, n_latent={n_latent}"
        )
    n_traj = L - n_latent

    vl, vt = _flextraj_vl_vt(
        seq_lens,
        n_latent,
        device,
        n_traj=n_traj,
        latent_lens=latent_lens,
        traj_lens=traj_lens,
    )

    neg_large = torch.finfo(torch.float32).min / 4
    idx_i = torch.arange(L, device=device)
    idx_j = torch.arange(L, device=device)

    # [B, L, L]
    j_grid = idx_j.view(1, 1, L)
    key_invalid_latent_pad = (j_grid >= vl.view(b, 1, 1)) & (j_grid < n_latent)
    key_invalid_traj_pad = j_grid >= (n_latent + vt.view(b, 1, 1))
    traj_segment_query = idx_i.view(1, L, 1) >= n_latent
    latent_key = idx_j.view(1, 1, L) < n_latent
    traj_cannot_see_latent = traj_segment_query & latent_key

    block = key_invalid_latent_pad | key_invalid_traj_pad | traj_cannot_see_latent

    if causal:
        causal_block = idx_j.view(1, 1, L) > idx_i.view(1, L, 1)
        block = block | causal_block

    bias = torch.zeros(b, L, L, device=device, dtype=dtype)
    bias = bias.masked_fill(block, neg_large)
    attn_bias = bias.unsqueeze(1)

    query_valid = flextraj_query_valid(
        seq_lens,
        n_latent,
        L,
        latent_lens=latent_lens,
        traj_lens=traj_lens,
        device=device,
        dtype=dtype,
    )
    return attn_bias, query_valid


def flextraj_sdpa_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor,
    n_latent: int,
    latent_lens: torch.Tensor | None = None,
    traj_lens: torch.Tensor | None = None,
    causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    FlexTraj self-attn：``q,k,v`` 形状 (B, L, H, D)。内部用 ``scaled_dot_product_attention`` + Task4 掩码。
    与 flash-attn varlen 不同，此处为稠密 L×L（L=2×n_latent），适合中等序列长度。
    """
    b, lq, n_heads, head_dim = q.shape
    assert k.shape[:3] == (b, lq, n_heads) and v.shape[:3] == (b, lq, n_heads)

    compute_dtype = torch.float32
    attn_bias, query_valid = flextraj_self_attn_bias(
        seq_lens,
        n_latent,
        lq,
        latent_lens=latent_lens,
        traj_lens=traj_lens,
        device=q.device,
        dtype=compute_dtype,
        causal=causal,
    )

    qh = q.transpose(1, 2).contiguous()
    kh = k.transpose(1, 2).contiguous()
    vh = v.transpose(1, 2).contiguous()
    qh_f = qh.to(compute_dtype)
    kh_f = kh.to(compute_dtype)
    vh_f = vh.to(compute_dtype)

    out = F.scaled_dot_product_attention(
        qh_f,
        kh_f,
        vh_f,
        attn_mask=attn_bias,
        dropout_p=dropout_p,
        is_causal=False,
        scale=None,
    )
    out = out.transpose(1, 2).to(q.dtype)
    qv = query_valid.to(device=q.device, dtype=q.dtype).view(b, lq, 1, 1)
    out = out * qv
    return out


def _flextraj_flash_available() -> bool:
    return FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE


def flextraj_flash_split_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor,
    n_latent: int,
    latent_lens: torch.Tensor | None = None,
    traj_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Task4 与双路 Flash-Attn：在 **causal=False** 且与 Task4 掩码等价的前提下，拆成两次 varlen flash：

    1. latent query → keys ``[K_lat_valid ‖ K_traj_valid]``（长度 ``vl+vt``）；
    2. traj query → keys 仅 traj 段（长度 ``vt``）。

    按 batch **逐条** 调用 ``flash_attention``，因现有 varlen 封装在 ``q_lens`` 逐样本不等时
    无法与 ``unflatten(B, max_len)`` 对齐。

    需 CUDA 且已安装 flash-attn 2 或 3。
    """
    b, L, n_heads, head_dim = q.shape
    n = n_latent
    if L < n:
        raise ValueError(f"total_len must be >= n_latent, got total_len={L}, n_latent={n}")
    n_traj = L - n
    assert k.shape[:3] == (b, L, n_heads) and v.shape[:3] == (b, L, n_heads)
    assert q.is_cuda

    vl, vt = _flextraj_vl_vt(
        seq_lens,
        n,
        q.device,
        n_traj=n_traj,
        latent_lens=latent_lens,
        traj_lens=traj_lens,
    )
    max_vl = int(vl.max().item())
    max_vt = int(vt.max().item())

    out = torch.zeros_like(q)
    h, d = n_heads, head_dim
    fa_dtype = torch.bfloat16

    if max_vl > 0:
        for bi in range(b):
            vlb, vtb = int(vl[bi].item()), int(vt[bi].item())
            if vlb == 0:
                continue
            Q1 = q[bi : bi + 1, :vlb]
            kl_len = vlb + vtb
            if kl_len == 0:
                continue
            K1 = q.new_zeros(1, kl_len, h, d)
            V1 = v.new_zeros(1, kl_len, h, d)
            K1[0, :vlb] = k[bi, :vlb]
            V1[0, :vlb] = v[bi, :vlb]
            if vtb > 0:
                K1[0, vlb : vlb + vtb] = k[bi, n : n + vtb]
                V1[0, vlb : vlb + vtb] = v[bi, n : n + vtb]

            out_l = flash_attention(
                Q1,
                K1,
                V1,
                q_lens=torch.tensor([vlb], dtype=torch.int32, device=q.device),
                k_lens=torch.tensor([kl_len], dtype=torch.int32, device=q.device),
                dropout_p=dropout_p,
                causal=False,
                window_size=(-1, -1),
                dtype=fa_dtype,
            )
            out[bi, :vlb] = out_l[0, :vlb]

    if max_vt > 0:
        for bi in range(b):
            vtb = int(vt[bi].item())
            if vtb == 0:
                continue
            Qt = q[bi : bi + 1, n : n + vtb]
            Kt = k[bi : bi + 1, n : n + vtb]
            Vt = v[bi : bi + 1, n : n + vtb]
            out_t = flash_attention(
                Qt,
                Kt,
                Vt,
                q_lens=torch.tensor([vtb], dtype=torch.int32, device=q.device),
                k_lens=torch.tensor([vtb], dtype=torch.int32, device=q.device),
                dropout_p=dropout_p,
                causal=False,
                window_size=(-1, -1),
                dtype=fa_dtype,
            )
            out[bi, n : n + vtb] = out_t[0, :vtb]

    qv = flextraj_query_valid(
        seq_lens,
        n,
        L,
        latent_lens=latent_lens,
        traj_lens=traj_lens,
        device=q.device,
        dtype=q.dtype,
    )
    out = out * qv.view(b, L, 1, 1)
    return out


def flextraj_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor,
    n_latent: int,
    latent_lens: torch.Tensor | None = None,
    traj_lens: torch.Tensor | None = None,
    causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    FlexTraj self-attn 统一入口。默认 ``FLEXTRAJ_ATTN_BACKEND=auto``：

    - ``causal=False``、CUDA、已安装 flash-attn → **双路 flash**（``flextraj_flash_split_self_attention``）；
    - 否则 → **稠密 SDPA + 显式 bias**（``flextraj_sdpa_self_attention``）。

    环境变量 ``FLEXTRAJ_ATTN_BACKEND``：``auto`` | ``flash`` | ``sdpa``。
    ``flash`` 在条件不满足时抛错，便于排查。
    """
    backend = os.environ.get("FLEXTRAJ_ATTN_BACKEND", "auto").strip().lower()
    sdpa_fn = flextraj_sdpa_self_attention
    flash_fn = flextraj_flash_split_self_attention

    if backend == "sdpa":
        return sdpa_fn(
            q,
            k,
            v,
            seq_lens,
            n_latent,
            latent_lens=latent_lens,
            traj_lens=traj_lens,
            causal=causal,
            dropout_p=dropout_p,
        )

    can_flash_split = (
        (not causal)
        and q.is_cuda
        and _flextraj_flash_available()
    )

    if backend == "flash":
        if not can_flash_split:
            raise RuntimeError(
                "FLEXTRAJ_ATTN_BACKEND=flash 需要 causal=False、CUDA 张量且已安装 flash-attn。"
            )
        return flash_fn(
            q,
            k,
            v,
            seq_lens,
            n_latent,
            latent_lens=latent_lens,
            traj_lens=traj_lens,
            dropout_p=dropout_p,
        )

    if backend not in ("auto", ""):
        warnings.warn(
            f"Unknown FLEXTRAJ_ATTN_BACKEND={backend!r}, using auto.",
            stacklevel=2,
        )

    if can_flash_split:
        return flash_fn(
            q,
            k,
            v,
            seq_lens,
            n_latent,
            latent_lens=latent_lens,
            traj_lens=traj_lens,
            dropout_p=dropout_p,
        )
    return sdpa_fn(
        q,
        k,
        v,
        seq_lens,
        n_latent,
        latent_lens=latent_lens,
        traj_lens=traj_lens,
        causal=causal,
        dropout_p=dropout_p,
    )
