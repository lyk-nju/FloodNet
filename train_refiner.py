"""Standalone Lightning training script for RootRefiner (T_A_08).

Fully decoupled from train_ldf.py. Trains the Refiner on HumanML3DRefinerDataset
samples with a duration / waypoint / path-control loss:
    num_token (CE) + xyz (SmoothL1) + heading (cosine) + fwd_delta (SmoothL1)
    + yaw_delta (SmoothL1) + path_control + smoothness (2nd-order-diff L2)

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
import copy
import logging
import os
import time
from pathlib import Path
import sys

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datasets.humanml3d_refiner import refiner_collate  # noqa: E402
from models.root_refiner import RootRefiner   # noqa: E402
from utils.refiner.config_validate import validate_refiner_config  # noqa: E402
from utils.refiner.losses import (  # noqa: E402
    dense_path_control_loss,
    goal_point_control_loss,
    masked_mean,
    second_order_diff_l2,
    smooth_l1_masked,
    sparse_path_control_loss,
)
from utils.text_encoder_resolver import resolve_text_encoder  # noqa: E402
# Re-exported for backward-compatible `from train_refiner import FrozenStubTextEncoder`.
from utils.text_encoder_resolver import FrozenStubTextEncoder  # noqa: E402,F401

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frozen text encoder stub (replace with ldf-shared encoder at integration)
# ---------------------------------------------------------------------------


# FrozenStubTextEncoder now lives in utils/text_encoder_resolver.py (shared with
# the benchmark) and is re-exported above for backward-compatible imports.


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------


class RefinerLightningModule(pl.LightningModule):
    def __init__(self, cfg: dict, text_encoder: nn.Module | None = None):
        super().__init__()
        validate_refiner_config(cfg)
        self.cfg = cfg
        model_cfg = dict(cfg["model"])
        self.refiner = RootRefiner(**model_cfg)
        self.min_tokens = model_cfg["min_tokens"]
        self.max_tokens = model_cfg["max_tokens"]
        text_emb_dim = model_cfg.get("text_emb_dim", 512)
        self.text_encoder = self._resolve_text_encoder(cfg, text_encoder, text_emb_dim)
        self.loss_weights = dict(cfg.get("loss_weights", {}))
        self.heading_form = cfg.get("loss", {}).get("heading_form", "cosine")
        # Don't pickle the (possibly large) text encoder into hparams.
        self.save_hyperparameters(ignore=["text_encoder"])

    @staticmethod
    def _resolve_text_encoder(cfg: dict, text_encoder, text_emb_dim: int):
        """Delegate to the shared resolver (utils/text_encoder_resolver) so train
        and benchmark build the identical encoder. Supports an explicit encoder,
        precomputed_t5_pool (real training), debug_stub (smoke/tests), else raise.
        """
        return resolve_text_encoder(cfg, text_encoder, text_emb_dim)

    # ------------------------------------------------------------------

    def forward(self, batch: dict, *, duration_mode: str = "groundtruth_duration") -> dict:
        if duration_mode not in {"groundtruth_duration", "pred_duration"}:
            raise ValueError(
                "duration_mode must be 'groundtruth_duration' or 'pred_duration', "
                f"got {duration_mode!r}."
            )
        text_emb = self.text_encoder.encode(batch["text"], device=self.device)
        path = batch.get("path", batch.get("xz_path"))
        path_valid_mask = batch.get("path_valid_mask", batch.get("path_mask"))
        path_features = batch.get("path_features", batch.get("path_stats"))
        history_motion = batch.get("history_motion", batch.get("current_motion"))
        num_tokens = batch.get("num_tokens") if duration_mode == "groundtruth_duration" else None
        return self.refiner(
            text_emb=text_emb,
            path=path,
            path_valid_mask=path_valid_mask,
            path_control_mask=batch.get("path_control_mask"),
            path_features=path_features,
            history_motion=history_motion,
            history_mask=batch["history_mask"],
            # Teacher-force the horizon with GT num_tokens during training; the
            # model falls back to its own argmax at eval / when absent.
            num_tokens=num_tokens,
        )

    def _compute_loss(self, out: dict, batch: dict) -> dict:
        target_wp = batch.get("waypoints")
        if target_wp is None:
            target_wp = batch["target_waypoints"][..., :5]
        target_mask = batch.get("waypoints_mask", batch.get("target_mask"))

        # num_token CE: target class = num_tokens - min_tokens (NO silent clamp).
        # An out-of-range target means a config/data mismatch; F.cross_entropy
        # already validates target ∈ [0, C) and fails loudly (device-side assert on
        # CUDA), so we don't add explicit min()/max() asserts here — those force a
        # GPU→CPU host sync every step, crash on an empty batch, and vanish under
        # `python -O`. The dataset shares min/max_tokens with the model, so in the
        # normal path the range always holds.
        n_classes = out["num_token_logits"].shape[-1]
        target_class = batch["num_tokens"] - self.min_tokens
        L_num = F.cross_entropy(out["num_token_logits"], target_class)

        # Ordinal-aware auxiliary term: SmoothL1/Huber between the soft-argmax
        # EXPECTED class (sum_k p_k * k, fully differentiable through softmax) and
        # the target class. CE alone treats off-by-1 the same as off-by-40; this
        # distance-aware term gives the head a gradient proportional to how far
        # the predicted distribution's mean is from the true token count, which is
        # what we want for an ordinal horizon. Weighted by loss_weights.num_token_soft.
        probs = out["num_token_logits"].softmax(dim=-1)                  # [B, K]
        class_idx = torch.arange(
            n_classes, device=probs.device, dtype=probs.dtype,
        )
        expected_class = (probs * class_idx).sum(dim=-1)                 # [B], differentiable
        L_num_soft = F.smooth_l1_loss(expected_class, target_class.to(probs.dtype))

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

        # Same-space delta supervision. The model now emits 5D in NORMALIZED
        # space (the dataset z-scores xyz; cos/sin stay raw). Re-deriving deltas
        # from `target_wp[..., :5]` gives a NORMALIZED-space gt that matches
        # the model's normalized-space pred deltas — comparing those to the
        # dataset's stored `target_wp[..., 5:7]` (PHYSICAL-then-z-scored) would
        # mix two different scales/offsets and silently miscalibrate the speed
        # channels at both training and inference. Frame 0's delta is
        # structurally 0 (no preceding frame), excluded from the mask.
        from utils.motion_process import append_traj_deltas_5d_to_7d
        pred5 = torch.cat([out["waypoints"][..., :3], pred_h], dim=-1)
        gt5 = torch.cat([target_wp[..., :3],
                         F.normalize(target_wp[..., 3:5], dim=-1, eps=1e-6)], dim=-1)
        pred_delta = append_traj_deltas_5d_to_7d(pred5)[..., 5:7]
        gt_delta = append_traj_deltas_5d_to_7d(gt5)[..., 5:7]
        delta_mask = target_mask.clone()
        delta_mask[:, 0] = False
        L_fwd_delta = smooth_l1_masked(
            pred_delta[..., 0:1], gt_delta[..., 0:1], delta_mask,
        )
        L_yaw_delta = smooth_l1_masked(
            pred_delta[..., 1:2], gt_delta[..., 1:2], delta_mask,
        )

        # smoothness L2 on 2nd-order diff of pred deltas (same-space).
        L_smooth = second_order_diff_l2(pred_delta, delta_mask)

        L_path_control = self._compute_path_control_loss(out, batch, target_mask)

        w = self.loss_weights
        loss = (
            w.get("num_token", 1.0) * L_num
            + w.get("num_token_soft", 0.1) * L_num_soft
            + w.get("xyz", 5.0) * L_xyz
            + w.get("heading", 1.0) * L_head
            + w.get("fwd_delta", 0.5) * L_fwd_delta
            + w.get("yaw_delta", 0.5) * L_yaw_delta
            + w.get("path_control", 0.0) * L_path_control
            + w.get("smoothness", 0.0) * L_smooth
        )
        # num_token diagnostics (NOT part of the weighted loss): exact / ±1 / ±2
        # argmax accuracy, discrete (argmax) MAE, and continuous (soft-argmax)
        # expected-token MAE. argmax is non-differentiable so the argmax-based
        # ones carry no gradient; all are logged only.
        with torch.no_grad():
            pred_class = out["num_token_logits"].argmax(dim=-1)
            err = (pred_class - target_class).abs().float()
            soft_err = (expected_class - target_class.to(expected_class.dtype)).abs()
        out_losses = {
            "loss": loss,
            "num_token": L_num,
            "num_token_soft": L_num_soft,
            "xyz": L_xyz,
            "heading": L_head,
            "fwd_delta": L_fwd_delta,
            "yaw_delta": L_yaw_delta,
            "path_control": L_path_control,
            "smoothness": L_smooth,
            "num_token_acc": (err == 0).float().mean(),
            "num_token_acc_pm1": (err <= 1).float().mean(),
            "num_token_acc_pm2": (err <= 2).float().mean(),
            "num_token_mae": err.mean(),
            "num_token_soft_mae": soft_err.mean(),
        }
        return out_losses

    def _compute_path_control_loss(self, out: dict, batch: dict, target_mask: torch.Tensor) -> torch.Tensor:
        if "path" not in batch or "path_control_mask" not in batch:
            return out["waypoints"].new_zeros(())

        path_modes = batch.get("path_mode")
        if path_modes is None:
            path_modes = ["dense_path"] * out["waypoints"].shape[0]
        offset_start_frames = batch.get("offset_start_frames")
        if offset_start_frames is None:
            offset_start_frames = torch.zeros(
                out["waypoints"].shape[0],
                dtype=torch.long,
                device=out["waypoints"].device,
            )
        else:
            offset_start_frames = offset_start_frames.to(out["waypoints"].device)

        losses = []
        for mode in ("dense_path", "sparse_path", "goal_point"):
            idx = [i for i, sample_mode in enumerate(path_modes) if sample_mode == mode]
            if not idx:
                continue
            index = torch.as_tensor(idx, dtype=torch.long, device=out["waypoints"].device)
            pred = out["waypoints"].index_select(0, index)
            path = batch["path"].to(out["waypoints"].device).index_select(0, index)
            control = batch["path_control_mask"].to(out["waypoints"].device).index_select(0, index)
            if mode == "dense_path":
                supervision = batch.get("path_supervision_mask", target_mask)
                losses.append(
                    dense_path_control_loss(
                        pred,
                        path,
                        supervision.to(out["waypoints"].device).index_select(0, index),
                    )
                )
            elif mode == "sparse_path":
                supervision = batch.get("path_supervision_mask", target_mask)
                losses.append(
                    sparse_path_control_loss(
                        pred,
                        path,
                        control,
                        supervision.to(out["waypoints"].device).index_select(0, index),
                        offset_start_frames.index_select(0, index),
                    )
                )
            else:
                losses.append(
                    goal_point_control_loss(
                        pred,
                        target_mask.to(out["waypoints"].device).index_select(0, index),
                        path,
                        control,
                    )
                )
        if not losses:
            return out["waypoints"].new_zeros(())
        return torch.stack(losses).mean()

    # Logged-only diagnostic keys (everything else returned by _compute_loss is a
    # weighted loss term with a matching loss_weights entry).
    METRIC_KEYS = (
        "num_token_acc", "num_token_acc_pm1", "num_token_acc_pm2",
        "num_token_mae", "num_token_soft_mae",
    )

    def training_step(self, batch: dict, batch_idx: int):
        out = self(batch)
        losses = self._compute_loss(out, batch)
        loss = losses["loss"]
        # NaN/Inf guard. A single non-finite loss back-props non-finite grads, and
        # the AdamW step then poisons EVERY parameter — from there on every loss
        # term reads NaN (gradient clipping does not help: clipping NaN grads
        # stays NaN). Returning None makes Lightning skip the optimizer step for
        # this batch instead of corrupting the weights. Costs one scalar host-sync
        # per step, which is an acceptable price for the safety. (Single-device
        # assumption: under DDP a None return on only some ranks desyncs the
        # backward all-reduce — run detect_anomaly to locate the source instead.)
        if not torch.isfinite(loss):
            log.warning(
                "non-finite train loss (%s) at global_step=%d batch_idx=%d; "
                "skipping optimizer step for this batch.",
                loss.detach().item(), self.global_step, batch_idx,
            )
            self.log("train/nonfinite_skip", 1.0, prog_bar=False,
                     on_step=True, on_epoch=False)
            return None
        # Show every per-term loss (num_token/xyz/heading/fwd_delta/yaw_delta +
        # total) on the tqdm bar, not just the total — so directional terms are
        # watchable during training.
        for k, v in losses.items():
            self.log(f"train/{k}", v, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        modes = (self.cfg.get("validation") or {}).get(
            "eval_modes",
            ["groundtruth_duration", "pred_duration"],
        )
        first_loss = None
        for mode in modes:
            out = self(batch, duration_mode=mode)
            losses = self._compute_loss(out, batch)
            if first_loss is None:
                first_loss = losses["loss"]
            for k, v in losses.items():
                self.log(
                    f"val_{mode}/{k}",
                    v,
                    prog_bar=(k == "loss" and mode == "groundtruth_duration"),
                    on_step=False,
                    on_epoch=True,
                )
        return first_loss

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

    path = Path(config_path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    base_config = cfg.pop("base_config", None)
    if not base_config:
        return cfg
    base_path = Path(base_config)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base = _load_cfg(str(base_path))
    return _deep_merge_dicts(base, cfg)


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dicts(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_paths_default() -> dict:
    """Read configs/paths_default.yaml (dirs / wandb_info / refiner_wandb_info).
    Returns {} if absent/unreadable. This is the same file train_ldf merges for
    its ${...} interpolation sources."""
    import yaml

    p = _REPO_ROOT / "configs" / "paths_default.yaml"
    if not p.is_file():
        return {}
    try:
        with p.open() as f:
            return yaml.safe_load(f) or {}
    except Exception:   # noqa: BLE001 — interpolation sources are best-effort
        return {}


def resolve_cfg_interpolations(cfg: dict) -> dict:
    """Resolve OmegaConf-style ${...} interpolations in a plain config dict
    (A-P0-1). yaml.safe_load leaves e.g. precomputed_text_emb_path and
    logger.wandb.* as literals like '${data.raw_data_dir}/...' / '${wandb_info.key}';
    this substitutes them against the cfg's own values PLUS the interpolation
    sources from configs/paths_default.yaml (wandb_info / refiner_wandb_info /
    dirs) — exactly like train_ldf merges paths_default. Call AFTER CLI overrides.

    The injected paths_default blocks are stripped from the result, so the
    returned cfg keeps its original top-level shape (now with resolved values).
    """
    from omegaconf import OmegaConf

    extras = _load_paths_default()
    # Guarantee the wandb interpolation sources exist so a missing/customized
    # paths_default.yaml degrades gracefully — ${wandb_info.*}/${refiner_wandb_info.*}
    # resolve to "" (→ wandb simply skipped) instead of raising InterpolationKeyError
    # and aborting the run (incl. the debug/smoke config, which must run standalone).
    extras.setdefault("wandb_info", {})
    for _k in ("key", "project", "entity"):
        extras["wandb_info"].setdefault(_k, "")
    extras.setdefault("refiner_wandb_info", {})
    extras["refiner_wandb_info"].setdefault("project", "")

    injected = [k for k in extras if k not in cfg]   # only add what cfg lacks
    merged = {**{k: extras[k] for k in injected}, **cfg}
    resolved = OmegaConf.to_container(OmegaConf.create(merged), resolve=True)
    # Strip interpolation-source blocks: the ones we injected PLUS the credential
    # blocks even when the cfg itself defined them — they carry the raw API key and
    # must never survive into ckpt hparams or the wandb run config.
    for k in set(injected) | {"wandb_info", "refiner_wandb_info"}:
        resolved.pop(k, None)
    return resolved


def path_aug_kwargs(cfg: dict) -> dict:
    """Map cfg.path_aug → RefinerDataset path-augmentation kwargs (P1-2).

    Absent keys are omitted so the dataset's own defaults apply.
    """
    aug = cfg.get("path_aug", {}) or {}
    kwargs = {}
    if "trim_prob" in aug:
        kwargs["path_trim_prob"] = float(aug["trim_prob"])
    if "trim_max_frames" in aug:
        kwargs["path_trim_max_frames"] = int(aug["trim_max_frames"])
    if "sparse_prob" in aug:
        kwargs["path_sparse_prob"] = float(aug["sparse_prob"])
    if "sparse_range" in aug:
        kwargs["path_sparse_range"] = tuple(aug["sparse_range"])
    return kwargs


# ---------------------------------------------------------------------------
# Run config: seed / resume / wandb / checkpoint (parity with train_ldf.py)
# ---------------------------------------------------------------------------


def resolve_seed(cfg: dict, cli_seed: int | None = None) -> int:
    """Seed precedence: CLI --seed > top-level cfg.seed (LDF style) >
    cfg.training.seed > 1234 default. Used for pl.seed_everything + dataset RNG.
    """
    if cli_seed is not None:
        return int(cli_seed)
    if cfg.get("seed") is not None:
        return int(cfg["seed"])
    return int((cfg.get("training") or {}).get("seed", 1234))


def resolve_resume_ckpt(cfg: dict, cli_ckpt: str | None = None) -> str | None:
    """Resume precedence: CLI --ckpt_path > cfg.resume_ckpt (LDF style) > None.
    Empty string in either place means "no resume"."""
    if cli_ckpt:
        return cli_ckpt
    rc = cfg.get("resume_ckpt")
    return rc or None


def _read_wandb_info_from_paths_default() -> dict:
    """Best-effort wandb credentials from configs/paths_default.yaml.

    Base = `wandb_info` (the same block train_ldf.py resolves `${wandb_info.*}`
    against → project "FloodNet"). `refiner_wandb_info` is then merged ON TOP so
    the Refiner shares the key/entity but logs to its own project. Returns {} if
    the file is absent/unreadable.
    """
    import yaml

    p = _REPO_ROOT / "configs" / "paths_default.yaml"
    if not p.is_file():
        return {}
    try:
        with p.open() as f:
            d = yaml.safe_load(f) or {}
        base = d.get("wandb_info", {}) or {}
        refiner = d.get("refiner_wandb_info", {}) or {}
        return {**base, **refiner}   # refiner overrides (project); inherits key/entity
    except Exception:   # noqa: BLE001 — credentials are optional, never fatal
        return {}


def _literal_or_none(v):
    """Return v only if it's a usable literal string (non-empty, not an
    unresolved ${...} interpolation); else None so a fallback kicks in."""
    if isinstance(v, str) and v.strip() and not v.startswith("${"):
        return v
    return None


def build_wandb_logger(cfg: dict, run_name: str, save_dir: str, api_key: str | None = None):
    """Build a WandbLogger, mirroring train_ldf.py's gating: OFF when cfg.debug
    is true (smoke), else ON when a cfg.logger.wandb block exists and an API key
    is resolvable. `logger.wandb.enabled: false` is an explicit override. Returns
    None (Trainer keeps its default logger) when disabled or no key is found.

    With the ${wandb_info.*} interpolation style, project/entity arrive already
    resolved in cfg.logger.wandb; `api_key` carries the resolved key separately
    (main() scrubs it out of cfg so it is never saved to ckpt hparams / wandb
    config). Falls back to configs/paths_default.yaml + env WANDB_API_KEY.
    """
    if cfg.get("debug", False):
        return None
    wb = (cfg.get("logger") or {}).get("wandb")
    if wb is None or wb.get("enabled") is False:
        return None
    info = _read_wandb_info_from_paths_default()
    key = (
        api_key
        or _literal_or_none(wb.get("wandb_key"))
        or info.get("key")
        or os.environ.get("WANDB_API_KEY")
    )
    project = _literal_or_none(wb.get("project")) or info.get("project")
    entity = _literal_or_none(wb.get("entity")) or info.get("entity")
    if not key:
        log.warning("wandb requested (debug=false, logger.wandb present) but no "
                    "API key found (cfg / paths_default.yaml / $WANDB_API_KEY) — "
                    "skipping wandb.")
        return None
    os.environ["WANDB_API_KEY"] = key
    from lightning.pytorch.loggers import WandbLogger

    # Don't leak the API key into the logged run config.
    safe_cfg = copy.deepcopy(cfg)
    try:
        safe_cfg["logger"]["wandb"].pop("wandb_key", None)
    except (KeyError, TypeError, AttributeError):
        pass
    return WandbLogger(
        project=project, entity=entity, name=run_name, save_dir=save_dir,
        config=safe_cfg,
    )


def build_checkpoint_callback(cfg: dict, output_dir: str):
    """Build a periodic ModelCheckpoint from cfg.checkpoint (or LDF-style
    cfg.validation.save_every_n_steps). Returns None when no cadence is set, so
    the Trainer falls back to Lightning's default end-of-run save.
    """
    ck = cfg.get("checkpoint") or {}
    val = cfg.get("validation") or {}
    every = ck.get("save_every_n_steps", val.get("save_every_n_steps"))
    if not every:
        return None
    from lightning.pytorch.callbacks import ModelCheckpoint

    # monitor=None means "periodic keep-all"; Lightning forbids a positive
    # finite save_top_k without a monitored metric, so coerce it to -1 (keep
    # every periodic ckpt). Set checkpoint.monitor (+ mode) to keep top-k by a
    # logged metric instead (e.g. "val/loss").
    # Keys may live in either the `checkpoint` block or the LDF-style `validation`
    # block (the shipped configs use the latter), so read both with checkpoint first.
    monitor = ck.get("monitor", val.get("monitor"))
    save_top_k = int(ck.get("save_top_k", val.get("save_top_k", -1)))
    if monitor is None and save_top_k not in (-1, 0):
        log.warning("save_top_k=%d needs a monitor (checkpoint/validation.monitor); "
                    "keeping all periodic ckpts (save_top_k=-1) instead.", save_top_k)
        save_top_k = -1

    return ModelCheckpoint(
        dirpath=ck.get("dirpath", output_dir),
        filename=ck.get("filename", "refiner_step_{step:06d}"),
        every_n_train_steps=int(every),
        save_top_k=save_top_k,
        monitor=monitor,
        mode=ck.get("mode", val.get("mode", "min")),
        save_last=bool(ck.get("save_last", val.get("save_last", True))),
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )


def _num_devices(devices) -> int:
    """Device count used to decide whether DDP is needed. `devices` may be an int
    (>=0), -1 (= all), a list of indices, or "auto"/str (→ visible CUDA count)."""
    if isinstance(devices, (list, tuple)):
        return len(devices)
    if isinstance(devices, int) and devices >= 0:
        return devices
    try:
        return torch.cuda.device_count()
    except Exception:   # noqa: BLE001 — defensive; assume single device
        return 1


def safe_precision(accelerator: str, precision, *, cuda_available: bool):
    """Downgrade a mixed/low precision to 32-true when the run will land on CPU
    (accelerator='cpu', or 'auto' with no CUDA visible) so a GPU-tuned config
    (e.g. bf16-mixed) stays host-portable instead of erroring / crawling on CPU.
    Returns precision unchanged on GPU, or None when none was requested."""
    if precision is None:
        return None
    on_cpu = accelerator == "cpu" or (accelerator == "auto" and not cuda_available)
    fp32 = {"32", "32-true", "64", "64-true", 32, 64}
    if on_cpu and precision not in fp32:
        log.warning("precision=%s requested but the run resolves to CPU; "
                    "using 32-true instead.", precision)
        return "32-true"
    return precision


def _build_one_dataset(cfg: dict, split_file: str, *, seed: int | None = None,
                       randomize_caption: bool = True):
    """Build a single RefinerDataset for a given split using the real loader."""
    from datasets.humanml3d_refiner import HumanML3DRefinerDataset
    from scripts.compute_5d_stats import load_clips_from_dir

    data_cfg = cfg.get("data", {})
    raw_dir = data_cfg["raw_data_dir"]
    stats_dir = data_cfg.get("stats_dir")
    # ⚠ Explicit normalize switch (P1-5): default False so a missing stats_dir
    # doesn't blow up the smoke. If normalize is requested, stats_dir must exist.
    normalize = bool(data_cfg.get("normalize", False))
    if normalize:
        if not stats_dir or not Path(stats_dir).is_dir():
            raise FileNotFoundError(
                f"data.normalize is true but stats_dir={stats_dir!r} does not exist. "
                f"Run scripts/compute_5d_stats.py first, or set data.normalize: false."
            )
    clips = load_clips_from_dir(
        raw_dir,
        dataset=data_cfg.get("dataset", "humanml3d"),
        split_file=split_file,
        feature_path=data_cfg.get("feature_path"),
        text_path=data_cfg.get("text_path"),
    )
    model_cfg = cfg["model"]
    sampling_cfg = cfg.get("sampling") or {}
    path_condition_cfg = (sampling_cfg.get("path_condition") or {})
    offset_cfg = (path_condition_cfg.get("offset_start") or {})
    sparse_cfg = (path_condition_cfg.get("sparse_path") or {})
    return HumanML3DRefinerDataset(
        clips,
        n_hist=model_cfg["n_hist"],
        n_path=model_cfg["n_path"],
        max_tokens=model_cfg["max_tokens"],
        min_tokens=model_cfg["min_tokens"],
        frames_per_token=model_cfg["frames_per_token"],
        full_plan_ratio=cfg.get("training", {}).get("sampling_mode_full_ratio", 0.5),
        horizon_policy=sampling_cfg.get("horizon_policy", "random"),
        path_condition_policy=path_condition_cfg.get("policy", "dense_path"),
        path_condition_ratios=path_condition_cfg.get("ratios"),
        offset_start_enabled=bool(offset_cfg.get("enabled", False)),
        offset_start_prob=float(offset_cfg.get("prob", 0.0)),
        offset_start_max_frames=int(offset_cfg.get("max_frames", 40)),
        offset_start_apply_to=tuple(offset_cfg.get("apply_to", ("dense_path", "sparse_path"))),
        sparse_path_point_range=tuple(sparse_cfg.get("point_range", (3, 8))),
        normalize=normalize,
        stats_dir=stats_dir if normalize else None,
        seed=seed,
        randomize_caption=randomize_caption,
    )


def build_datasets(cfg: dict, seed: int | None = None):
    """Build (train_ds, val_ds) RefinerDatasets from cfg.data via the real
    HumanML3D/BABEL loader. val_ds is None if no val_split_file is configured.

    `seed` is threaded into the train dataset's augmentation RNG for
    reproducibility (matters with num_workers=0; with workers each gets a
    distinct RNG via refiner_worker_init_fn). val uses a fixed seed so the
    val set is identical across runs.

    Train randomizes the caption per sample (text augmentation); val pins the
    first caption (randomize_caption=False) so val/loss stays comparable across
    epochs instead of wobbling with caption choice.
    """
    data_cfg = cfg.get("data", {})
    train_split = data_cfg.get("train_split_file", "train.txt")
    val_split = data_cfg.get("val_split_file")
    train_ds = _build_one_dataset(cfg, train_split, seed=seed)
    val_ds = (_build_one_dataset(cfg, val_split, seed=0, randomize_caption=False)
              if val_split else None)
    return train_ds, val_ds


def apply_fixed_overfit_datasets(train_ds, val_ds, cfg: dict):
    """Replace train/val datasets with pre-sampled deterministic diagnostics."""
    fixed_cfg = cfg.get("fixed_overfit") or {}
    if not fixed_cfg.get("enabled", False):
        return train_ds, val_ds

    from datasets.refiner_fixed import (
        FixedRefinerSampleDataset,
        build_fixed_refiner_samples,
    )

    kwargs = {
        "num_samples": int(fixed_cfg.get("num_samples", 64)),
        "mode_policy": fixed_cfg.get("mode_policy", "mixed"),
        "force_no_path_aug": bool(fixed_cfg.get("force_no_path_aug", True)),
        "force_text_idx": fixed_cfg.get("force_text_idx", 0),
    }
    train_samples = build_fixed_refiner_samples(train_ds, **kwargs)
    fixed_train = FixedRefinerSampleDataset(train_samples)

    if bool(fixed_cfg.get("val_on_train", True)) or val_ds is None:
        fixed_val = FixedRefinerSampleDataset(train_samples)
    else:
        val_samples = build_fixed_refiner_samples(val_ds, **kwargs)
        fixed_val = FixedRefinerSampleDataset(val_samples)

    log.info(
        "fixed_overfit enabled: samples=%d mode_policy=%s force_no_path_aug=%s "
        "val_on_train=%s",
        len(fixed_train),
        kwargs["mode_policy"],
        kwargs["force_no_path_aug"],
        bool(fixed_cfg.get("val_on_train", True)),
    )
    return fixed_train, fixed_val


def apply_default_fixed_validation_dataset(train_ds, val_ds):
    """Replace validation with one deterministic sample per val item.

    Train remains stochastic. Validation is always a fixed clean cache built
    from val.txt so val/loss is comparable across epochs. If no val split is
    configured, leave validation absent.
    """
    if val_ds is None:
        return train_ds, val_ds
    from datasets.refiner_fixed import (
        FixedRefinerSampleDataset,
        build_fixed_refiner_samples,
    )
    if isinstance(val_ds, FixedRefinerSampleDataset):
        return train_ds, val_ds

    val_samples = build_fixed_refiner_samples(
        val_ds,
        num_samples=len(val_ds),
        mode_policy="random",
        force_no_path_aug=True,
        force_text_idx=0,
    )
    fixed_val = FixedRefinerSampleDataset(val_samples)
    log.info(
        "default fixed validation: samples=%d mode_policy=random force_no_path_aug=True",
        len(fixed_val),
    )
    return train_ds, fixed_val


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/root_refiner.yaml")
    parser.add_argument("--max_steps", type=int, default=None,
                         help="Override trainer.max_steps / training.total_steps (smoke runs).")
    parser.add_argument("--devices", type=int, default=None,
                         help="Override trainer.devices.")
    parser.add_argument("--ckpt_path", type=str, default=None,
                         help="Resume-from-checkpoint path (overrides cfg.resume_ckpt).")
    parser.add_argument("--seed", type=int, default=None,
                         help="Override cfg.seed (RNG seed for torch + dataset aug).")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Checkpoint/log dir (overrides cfg.save_dir; "
                              "default outputs/root_refiner).")
    parser.add_argument("--raw_data_dir", type=str, default=None,
                         help="Override data.raw_data_dir (e.g. on a host where the "
                              "config's training-box path doesn't exist).")
    parser.add_argument("--stats_dir", type=str, default=None,
                         help="Override data.stats_dir.")
    parser.add_argument("--normalize", type=str, default=None,
                         choices=["true", "false"],
                         help="Override data.normalize (true/false).")
    args = parser.parse_args(argv)

    cfg = _load_cfg(args.config)
    # CLI overrides for single-host portability (config ships training-box paths).
    cfg.setdefault("data", {})
    if args.raw_data_dir is not None:
        cfg["data"]["raw_data_dir"] = args.raw_data_dir
    if args.stats_dir is not None:
        cfg["data"]["stats_dir"] = args.stats_dir
    if args.normalize is not None:
        cfg["data"]["normalize"] = (args.normalize == "true")
    # A-P0-1: resolve ${data.raw_data_dir}, ${wandb_info.*} etc. AFTER overrides so
    # e.g. text_encoder.precomputed_text_emb_path and logger.wandb become real values.
    cfg = resolve_cfg_interpolations(cfg)
    validate_refiner_config(cfg)
    # Scrub the resolved WandB API key out of cfg (it came from ${wandb_info.key})
    # BEFORE it can be captured by RefinerLightningModule.save_hyperparameters or
    # logged into the wandb run config. Keep it only in a local for the logger.
    wandb_api_key = None
    _wb = (cfg.get("logger") or {}).get("wandb")
    if isinstance(_wb, dict):
        wandb_api_key = _literal_or_none(_wb.get("wandb_key"))
        _wb["wandb_key"] = None
    train_cfg = cfg.get("training") or {}
    trainer_cfg = cfg.get("trainer") or {}
    # max_steps precedence: CLI > trainer.max_steps > training.total_steps.
    max_steps = (
        args.max_steps if args.max_steps is not None
        else trainer_cfg.get("max_steps", train_cfg.get("total_steps", 100000))
    )
    output_dir = args.output_dir or cfg.get("save_dir") or "outputs/root_refiner"

    # Reproducibility: seed torch/numpy/python (+ Lightning workers) and thread
    # the same seed into the dataset augmentation RNG.
    seed = resolve_seed(cfg, args.seed)
    pl.seed_everything(seed, workers=True)
    if bool(cfg.get("deterministic", False)):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    log.info("seed=%d deterministic=%s", seed, bool(cfg.get("deterministic", False)))

    train_ds, val_ds = build_datasets(cfg, seed=seed)
    train_ds, val_ds = apply_fixed_overfit_datasets(train_ds, val_ds, cfg)
    train_ds, val_ds = apply_default_fixed_validation_dataset(train_ds, val_ds)
    from datasets.humanml3d_refiner import refiner_worker_init_fn
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=refiner_collate,
        drop_last=True,
        worker_init_fn=refiner_worker_init_fn,   # P1-3: distinct aug RNG per worker
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

    # Logger (wandb, parity with train_ldf) + periodic checkpointing + resume.
    run_name = f"{cfg.get('exp_name', 'root_refiner')}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = build_wandb_logger(cfg, run_name=run_name, save_dir=output_dir,
                                api_key=wandb_api_key)
    callbacks = []
    ckpt_cb = build_checkpoint_callback(cfg, output_dir)
    if ckpt_cb is not None:
        callbacks.append(ckpt_cb)
    resume_ckpt = resolve_resume_ckpt(cfg, args.ckpt_path)
    if resume_ckpt:
        log.info("resuming from checkpoint: %s", resume_ckpt)

    # Trainer kwargs from the `trainer` block (LDF style), with CLI/defaults.
    accelerator = trainer_cfg.get("accelerator", "auto")
    devices = args.devices if args.devices is not None else trainer_cfg.get("devices", 1)
    # Multi-device → DDP with find_unused_parameters=True (mirrors train_ldf; the
    # frozen text encoder otherwise risks DDP unused-parameter errors).
    strategy = "auto"
    if _num_devices(devices) > 1:
        from lightning.pytorch.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)
    trainer_kwargs = dict(
        max_steps=max_steps,
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        gradient_clip_val=trainer_cfg.get(
            "gradient_clip_val", train_cfg.get("gradient_clip_val", 1.0)),
        default_root_dir=output_dir,
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 10),
        logger=logger if logger is not None else True,
        callbacks=callbacks,
    )
    # Host-portable precision: a GPU-tuned bf16-mixed config downgrades to fp32 on CPU.
    precision = safe_precision(accelerator, trainer_cfg.get("precision"),
                               cuda_available=torch.cuda.is_available())
    if precision is not None:
        trainer_kwargs["precision"] = precision
    # Step-based validation cadence when configured (LDF style); else epoch.
    # validation_steps counts GLOBAL train steps, so check_val_every_n_epoch must
    # be None — otherwise Lightning reads val_check_interval as a within-epoch
    # batch index and raises when it exceeds the (often smaller) epoch length.
    val_check_interval = (cfg.get("validation") or {}).get("validation_steps")
    if val_loader is not None and val_check_interval:
        trainer_kwargs["val_check_interval"] = val_check_interval
        trainer_kwargs["check_val_every_n_epoch"] = None

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module, train_loader, val_dataloaders=val_loader, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
