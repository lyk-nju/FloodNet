import argparse
import os
import sys
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_ldf import CustomLightningModule  # noqa: E402
from utils.initialize import get_function, load_config  # noqa: E402
from utils.training import build_probe_loaders, load_resume_step_offset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FloodNet inline generation eval from a checkpoint outside the training loop."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--artifact_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--default_root_dir", type=str, default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    artifact_root = args.artifact_root or args.output_dir
    if artifact_root is None:
        raise ValueError("Either --artifact_root or --output_dir must be provided.")
    artifact_root = os.path.abspath(artifact_root)
    default_root_dir = os.path.abspath(args.default_root_dir or artifact_root)

    os.makedirs(artifact_root, exist_ok=True)
    os.makedirs(default_root_dir, exist_ok=True)

    cfg = load_config(
        args.config,
        {
            "train": "false",
            "save_dir": artifact_root,
            "validation.test_mode": "inline",
        },
    )
    OmegaConf.update(cfg.config, "save_dir", artifact_root)

    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    collate_fn = (
        get_function(cfg.data.collate_fn) if cfg.data.get("collate_fn", None) else None
    )
    test_probe_loaders, test_loader_tags, total_probe_samples = build_probe_loaders(
        cfg, collate_fn
    )
    rank_zero_info(f"[async-inline-eval] total probe samples: {total_probe_samples}")

    model = CustomLightningModule(cfg=cfg.config)
    model.test_loader_tags = test_loader_tags
    model._resume_step_offset = int(load_resume_step_offset(args.ckpt))

    accelerator = args.accelerator or (
        "gpu" if torch.cuda.is_available() and args.devices > 0 else "cpu"
    )
    precision = cfg.trainer.precision if accelerator == "gpu" else "32-true"

    trainer = Trainer(
        accelerator=accelerator,
        devices=args.devices if accelerator == "gpu" else 1,
        strategy="auto",
        logger=None,
        enable_checkpointing=False,
        inference_mode=True,
        default_root_dir=default_root_dir,
        precision=precision,
    )

    trainer.test(
        model,
        dataloaders=test_probe_loaders,
        ckpt_path=args.ckpt,
        weights_only=False,
    )


if __name__ == "__main__":
    main()
