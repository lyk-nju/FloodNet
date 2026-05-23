"""Standalone Lightning training script for RootRefiner (T_A_08).

Fully decoupled from train_ldf.py. Trains the Refiner on RefinerDataset samples
with a 6-term loss:
    num_token (CE) + xyz (SmoothL1) + heading (cosine) + fwd_delta (SmoothL1)
    + yaw_delta (SmoothL1) + smoothness (2nd-order-diff L2)

loss dict keys are aligned field-by-field with configs/root_refiner.yaml's
loss_weights (round 6 P1-4 / P1-6: keys are fwd_delta / yaw_delta / smoothness;
NO legacy "speed").

References:
- docs/TODO.md §T_A_08 lines 1354-1420.
- docs/design.md §0.2.1 (per-frame delta naming convention).

The text encoder is pluggable; a deterministic frozen stub is used by default
so the training pipeline runs standalone. At integration time, pass the
ldf-shared frozen text encoder via `text_encoder=`.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.root_refiner import RootRefiner   # noqa: E402


# ---------------------------------------------------------------------------
# Masked loss helpers
# ---------------------------------------------------------------------------


def smooth_l1_masked(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
    """SmoothL1 over valid frames. pred/gt: [B, T, C]; mask: [B, T] bool.
    Returns scalar; 0 when no valid frames."""
    mask_f = mask.unsqueeze(-1).to(pred.dtype)              # [B, T, 1]
    denom = mask_f.sum() * pred.shape[-1]
    if denom <= 0:
        return pred.new_zeros(())
    diff = F.smooth_l1_loss(pred, gt, reduction="none") * mask_f
    return diff.sum() / denom


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of `values` [B, T] over valid positions. 0 when no valid frames."""
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum()
    if denom <= 0:
        return values.new_zeros(())
    return (values * mask_f).sum() / denom


def second_order_diff_l2(values: Tensor, mask: Tensor) -> Tensor:
    """L2 on 2nd-order frame differences of `values` [B, T, C], masked.

    diff[t] = values[t] - 2*values[t-1] + values[t-2]; only counted where all
    three frames (t, t-1, t-2) are valid. Returns 0 when fewer than 3 frames
    or no valid triples.
    """
    if values.shape[1] < 3:
        return values.new_zeros(())
    diff = values[:, 2:] - 2 * values[:, 1:-1] + values[:, :-2]       # [B, T-2, C]
    valid = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2])              # [B, T-2]
    valid_f = valid.unsqueeze(-1).to(values.dtype)
    denom = valid_f.sum() * values.shape[-1]
    if denom <= 0:
        return values.new_zeros(())
    return ((diff ** 2) * valid_f).sum() / denom


# ---------------------------------------------------------------------------
# Frozen text encoder stub (replace with ldf-shared encoder at integration)
# ---------------------------------------------------------------------------


class FrozenStubTextEncoder(nn.Module):
    """Deterministic per-text embedding via hashing + frozen embedding table.

    Standalone placeholder so the training pipeline runs without the real
    ldf text encoder. At integration time, pass a module with an `.encode(
    texts: list[str]) -> [B, text_emb_dim]` method instead.
    """

    def __init__(self, emb_dim: int, vocab: int = 4096):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab = vocab
        self.table = nn.Embedding(vocab, emb_dim)
        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _stable_id(text: str, vocab: int) -> int:
        """Process-stable hash of `text` → [0, vocab).

        ⚠ Must NOT use builtin hash(): Python str hashing is salted per process
        (PYTHONHASHSEED), so the same caption would map to different embedding
        rows across runs / between training and benchmarking — silently
        defeating any text conditioning when the encoder is reloaded. hashlib
        is deterministic across processes.
        """
        digest = hashlib.md5(text.encode("utf-8")).digest()[:8]
        return int.from_bytes(digest, "big") % vocab

    @torch.no_grad()
    def encode(self, texts: list[str], device=None) -> Tensor:
        ids = torch.tensor(
            [self._stable_id(t, self.vocab) for t in texts], dtype=torch.long,
        )
        if device is not None:
            ids = ids.to(device)
        return self.table(ids)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def refiner_collate(batch: list[dict]) -> dict:
    """Stack RefinerDataset dict samples into a batch (text stays a list)."""
    out = {
        "text": [s["text"] for s in batch],
        "mode": [s["mode"] for s in batch],
    }
    for key in ("xz_path", "path_mask", "path_stats", "current_motion",
                 "history_mask", "target_waypoints", "target_mask", "num_tokens"):
        out[key] = torch.stack([s[key] for s in batch])
    return out


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class RefinerLightningModule(pl.LightningModule):
    def __init__(self, cfg: dict, text_encoder: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        model_cfg = dict(cfg["model"])
        self.refiner = RootRefiner(**model_cfg)
        self.min_tokens = model_cfg["min_tokens"]
        self.max_tokens = model_cfg["max_tokens"]
        text_emb_dim = model_cfg.get("text_emb_dim", 512)
        self.text_encoder = text_encoder or FrozenStubTextEncoder(text_emb_dim)
        self.loss_weights = dict(cfg.get("loss_weights", {}))
        self.heading_form = cfg.get("loss", {}).get("heading_form", "cosine")
        # Don't pickle the (possibly large) text encoder into hparams.
        self.save_hyperparameters(ignore=["text_encoder"])

    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        text_emb = self.text_encoder.encode(batch["text"], device=self.device)
        return self.refiner(
            text_emb=text_emb,
            xz_path=batch["xz_path"],
            path_mask=batch["path_mask"],
            path_stats=batch["path_stats"],
            current_motion=batch["current_motion"],
            history_mask=batch["history_mask"],
        )

    def _compute_loss(self, out: dict, batch: dict) -> dict:
        target_wp = batch["target_waypoints"]
        target_mask = batch["target_mask"]

        # num_token CE: target class = num_tokens - min_tokens (clamped to logits range).
        n_classes = out["num_token_logits"].shape[-1]
        target_class = (batch["num_tokens"] - self.min_tokens).clamp(0, n_classes - 1)
        L_num = F.cross_entropy(out["num_token_logits"], target_class)

        # xyz SmoothL1 (valid only).
        L_xyz = smooth_l1_masked(out["waypoints"][..., 0:3], target_wp[..., 0:3], target_mask)

        # heading cosine (pred already unit-norm; gt assumed unit-norm).
        pred_h = F.normalize(out["waypoints"][..., 3:5], dim=-1, eps=1e-6)
        gt_h = target_wp[..., 3:5]
        if self.heading_form == "cosine":
            head_term = 1.0 - (pred_h * gt_h).sum(-1)              # [B, T]
            L_head = masked_mean(head_term, target_mask)
        else:
            L_head = smooth_l1_masked(pred_h, gt_h, target_mask)

        # per-frame fwd_delta / yaw_delta SmoothL1 (channels 5, 6).
        L_fwd_delta = smooth_l1_masked(
            out["waypoints"][..., 5:6], target_wp[..., 5:6], target_mask,
        )
        L_yaw_delta = smooth_l1_masked(
            out["waypoints"][..., 6:7], target_wp[..., 6:7], target_mask,
        )

        # smoothness L2 on 2nd-order diff of [fwd_delta, yaw_delta].
        L_smooth = second_order_diff_l2(out["waypoints"][..., 5:7], target_mask)

        w = self.loss_weights
        loss = (
            w.get("num_token", 1.0) * L_num
            + w.get("xyz", 5.0) * L_xyz
            + w.get("heading", 1.0) * L_head
            + w.get("fwd_delta", 0.5) * L_fwd_delta
            + w.get("yaw_delta", 0.5) * L_yaw_delta
            + w.get("smoothness", 0.0) * L_smooth
        )
        return {
            "loss": loss,
            "num_token": L_num,
            "xyz": L_xyz,
            "heading": L_head,
            "fwd_delta": L_fwd_delta,
            "yaw_delta": L_yaw_delta,
            "smoothness": L_smooth,
        }

    def training_step(self, batch: dict, batch_idx: int):
        out = self(batch)
        losses = self._compute_loss(out, batch)
        for k, v in losses.items():
            self.log(f"train/{k}", v, prog_bar=(k == "loss"), on_step=True, on_epoch=False)
        return losses["loss"]

    def validation_step(self, batch: dict, batch_idx: int):
        out = self(batch)
        losses = self._compute_loss(out, batch)
        for k, v in losses.items():
            self.log(f"val/{k}", v, prog_bar=(k == "loss"), on_step=False, on_epoch=True)
        return losses["loss"]

    def configure_optimizers(self):
        tr = self.cfg.get("training", {})
        lr = float(tr.get("lr", 1e-4))
        weight_decay = float(tr.get("weight_decay", 0.01))
        return torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=lr, weight_decay=weight_decay,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_cfg(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_one_dataset(cfg: dict, split_file: str, *, seed: int | None = None):
    """Build a single RefinerDataset for a given split using the real loader."""
    from datasets.refiner_dataset import RefinerDataset
    from scripts.compute_5d_stats import load_clips_from_dir

    data_cfg = cfg.get("data", {})
    raw_dir = data_cfg["raw_data_dir"]
    stats_dir = data_cfg.get("stats_dir")
    clips = load_clips_from_dir(
        raw_dir,
        dataset=data_cfg.get("dataset", "humanml3d"),
        split_file=split_file,
        feature_path=data_cfg.get("feature_path"),
        text_path=data_cfg.get("text_path"),
    )
    model_cfg = cfg["model"]
    return RefinerDataset(
        clips,
        n_hist=model_cfg["n_hist"],
        n_path=model_cfg["n_path"],
        max_tokens=model_cfg["max_tokens"],
        min_tokens=model_cfg["min_tokens"],
        frames_per_token=model_cfg["frames_per_token"],
        full_plan_ratio=cfg.get("training", {}).get("sampling_mode_full_ratio", 0.5),
        normalize=stats_dir is not None,
        stats_dir=stats_dir,
        seed=seed,
    )


def build_datasets(cfg: dict):
    """Build (train_ds, val_ds) RefinerDatasets from cfg.data via the real
    HumanML3D/BABEL loader. val_ds is None if no val_split_file is configured.
    """
    data_cfg = cfg.get("data", {})
    train_split = data_cfg.get("train_split_file", "train.txt")
    val_split = data_cfg.get("val_split_file")
    train_ds = _build_one_dataset(cfg, train_split)
    val_ds = _build_one_dataset(cfg, val_split, seed=0) if val_split else None
    return train_ds, val_ds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/root_refiner.yaml")
    parser.add_argument("--max_steps", type=int, default=None,
                         help="Override training.total_steps (smoke runs).")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default=None,
                         help="Resume-from-checkpoint path.")
    parser.add_argument("--output_dir", type=str, default="outputs/root_refiner")
    args = parser.parse_args(argv)

    cfg = _load_cfg(args.config)
    train_cfg = cfg.get("training", {})
    max_steps = args.max_steps if args.max_steps is not None else train_cfg.get("total_steps", 100000)

    train_ds, val_ds = build_datasets(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=refiner_collate,
        drop_last=True,
    )
    # Build a val loader when a val split is configured, so the module's
    # validation_step actually runs (review finding: previously only the train
    # loader was passed and validation silently never ran).
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=train_cfg.get("val_batch_size", train_cfg.get("batch_size", 64)),
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 4),
            collate_fn=refiner_collate,
            drop_last=False,
        )

    module = RefinerLightningModule(cfg)

    trainer = pl.Trainer(
        max_steps=max_steps,
        devices=args.devices,
        accelerator="auto",
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        default_root_dir=args.output_dir,
        log_every_n_steps=10,
    )
    trainer.fit(module, train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
