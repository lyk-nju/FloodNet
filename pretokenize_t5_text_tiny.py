"""
Offline: encode every unique caption (HumanML3D / Babel-style texts/*.txt) with the same
HuggingFace encoder as ``models.diffusion_forcing_wan_tiny`` (default ``google/umt5-base``).

Output format matches ``pretokenize_t5_text.py`` except ``text_dim`` is 768 (base) and
``model_name`` is recorded instead of checkpoint_path/tokenizer_path.

Training: in ``configs/ldf_tiny.yaml`` set::

    model.params.use_precomputed_text_emb: true
    model.params.precomputed_text_emb_path: /path/to/umt5_tiny_text_embeddings.pt

Multi-GPU (each rank encodes a shard, rank 0 merges)::

    torchrun --standalone --nproc_per_node=4 pretokenize_t5_text_tiny.py --config configs/ldf_tiny.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoConfig

from models.diffusion_forcing_wan_tiny import HFT5Encoder
from pretokenize_t5_text import (
    batched,
    collect_unique_captions,
    _dist_world,
    _meta_paths_from_data_block,
    _parse_overrides,
    _resolve_config_path,
)
from utils.initialize import load_config

DEFAULT_TINY_CONFIG = os.path.join("configs", "ldf_tiny.yaml")


def _log0(msg: str, rank: int) -> None:
    if rank == 0:
        rank_zero_info(msg)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Offline HF T5 (tiny / umt5-base) text embeddings for DiffForcingWanModel (tiny)."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_TINY_CONFIG,
        help=f"YAML with model.params.model_name, model.params.text_len, data (default {DEFAULT_TINY_CONFIG})",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="+",
        default=None,
        help="Same as train_ldf, e.g. model.params.text_len=256",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pt (default: <dirname(first train meta)>/umt5_tiny_text_embeddings.pt)",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Single process: cuda | cuda:0 | cpu (default cuda). torchrun: uses LOCAL_RANK.",
    )
    args = parser.parse_args()

    world_size, rank, local_rank = _dist_world()
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU pretokenize needs CUDA.")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = _resolve_config_path(args.config)
    override_args = _parse_overrides(args.override)
    _log0(f"pretokenize_t5_text_tiny: load config {config_path}", rank)
    if override_args:
        _log0(f"pretokenize_t5_text_tiny: --override {override_args}", rank)

    cfg = load_config(config_path, override_args)
    oc = cfg.config

    model_name = OmegaConf.select(oc, "model.params.model_name", default="google/umt5-base")
    text_len = int(OmegaConf.select(oc, "model.params.text_len", default=512))
    hf_cfg = AutoConfig.from_pretrained(model_name)
    text_dim = int(
        getattr(hf_cfg, "d_model", None)
        or getattr(hf_cfg, "hidden_size", None)
        or 768
    )
    _log0(
        f"HF encoder: model_name={model_name!r}, text_len={text_len}, text_dim={text_dim}",
        rank,
    )

    if distributed:
        if rank == 0:
            _bucket: List[Optional[List[str]]] = [sorted(collect_unique_captions(oc))]
        else:
            _bucket = [None]
        dist.broadcast_object_list(_bucket, src=0)
        ordered = _bucket[0]
        assert ordered is not None
        _log0(f"Unique captions (incl. empty): {len(ordered)}", rank)
    else:
        captions = collect_unique_captions(oc)
        ordered = sorted(captions)
        rank_zero_info(f"Unique captions (incl. empty): {len(ordered)}")

    out_path = args.output
    if out_path is None:
        first = None
        if getattr(oc.data, "datasets", None) is not None and len(oc.data.datasets) > 0:
            blocks = list(oc.data.datasets)
        else:
            blocks = [oc.data]
        for block in blocks:
            for p in _meta_paths_from_data_block(block):
                if os.path.isfile(p):
                    first = p
                    break
            if first:
                break
        if first is None:
            raise RuntimeError("Could not resolve default --output: no existing meta file found.")
        out_path = os.path.join(os.path.dirname(first), "umt5_tiny_text_embeddings.pt")

    if rank == 0:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if distributed:
        dist.barrier()

    n = len(ordered)
    start = rank * n // world_size
    end = (rank + 1) * n // world_size
    local_ordered = ordered[start:end]

    encoder = HFT5Encoder(
        text_len=text_len,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        model_name=model_name,
    )
    encoder.model.to(device)
    encoder.model.eval()

    emb_dict: Dict[str, torch.Tensor] = {}
    batches = list(batched(local_ordered, args.batch_size))
    pbar = tqdm(
        batches,
        desc=f"HF T5 tiny encode rank {rank}/{world_size}",
        disable=(distributed and rank != 0),
    )
    for batch in pbar:
        encoded = encoder(batch, device)
        for s, t in zip(batch, encoded):
            emb_dict[s] = t.detach().cpu()

    encoder.model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if distributed:
        if rank == 0:
            gather_list: List[Optional[Dict[str, torch.Tensor]]] = [None] * world_size
        else:
            gather_list = None
        dist.gather_object(emb_dict, gather_list, dst=0)
        if rank == 0:
            assert gather_list is not None
            merged: Dict[str, torch.Tensor] = {}
            for part in gather_list:
                assert part is not None
                merged.update(part)
            payload = {
                "embeddings": merged,
                "text_dim": text_dim,
                "text_len": text_len,
                "model_name": str(model_name),
            }
            torch.save(payload, out_path)
            rank_zero_info(f"Saved {len(merged)} entries to {out_path}")
        dist.barrier()
        dist.destroy_process_group()
    else:
        payload = {
            "embeddings": emb_dict,
            "text_dim": text_dim,
            "text_len": text_len,
            "model_name": str(model_name),
        }
        torch.save(payload, out_path)
        rank_zero_info(f"Saved {len(emb_dict)} entries to {out_path}")


if __name__ == "__main__":
    main()
