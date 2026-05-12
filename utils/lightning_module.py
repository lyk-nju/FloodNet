import os
import time

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_info
from torch.nn.modules.module import _IncompatibleKeys
from torch_ema import ExponentialMovingAverage

from utils.initialize import (
    check_state_dict,
    instantiate,
    print_model_size,
)
from utils.training.module_step import ckpt_step_info

# Set tokenizers parallelism to false to avoid warnings in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BasicLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(
            target=cfg.model.target, cfg=None, hfstyle=False, **cfg.model.params
        )

        # NOTE: ligntning init stage the device is cpu, so no need to move to device
        # EMA only tracks trainable params — frozen backbone (~123M) needs no shadow copy.
        self.ema = ExponentialMovingAverage(
            [p for p in self.model.parameters() if p.requires_grad],
            decay=cfg.model.ema_decay,
        )
        print_model_size(self.model)

        # logging
        self.last_batch_end_time, self.batch_ready_time = None, None
        self.validation_step_outputs = []
        self._skip_next_lightning_load_state_dict = False

        # metric
        self.initialize_metrics()

    def configure_optimizers(self):
        optim_target = self.cfg.optimizer.target
        if len(optim_target.split(".")) == 1:
            optim_target = "torch.optim." + optim_target
        # 只对 requires_grad=True 的参数构建优化器，便于冻结主干仅训练轨迹分支
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = instantiate(
            target=optim_target,
            cfg=None,
            hfstyle=False,
            params=trainable_params,
            **self.cfg.optimizer.params,
        )

        scheduler_target = self.cfg.lr_scheduler.target
        if len(scheduler_target.split(".")) == 1:
            scheduler_target = "torch.optim.lr_scheduler." + scheduler_target
        lr_scheduler = instantiate(
            target=scheduler_target,
            cfg=None,
            hfstyle=False,
            optimizer=optimizer,
            **self.cfg.lr_scheduler.params,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def load_state_dict(self, state_dict, strict=True):
        if self._skip_next_lightning_load_state_dict:
            self._skip_next_lightning_load_state_dict = False
            return _IncompatibleKeys([], [])
        return self.model.load_state_dict(state_dict, strict=strict)

    def on_load_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        if "ema_state" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state"])
            rank_zero_info("init ema from ckpt")
        else:
            self.ema = ExponentialMovingAverage(
                [p for p in self.model.parameters() if p.requires_grad],
                decay=self.cfg.model.ema_decay,
            )
            rank_zero_info("init ema from current model weights")

        # Compare state_dict and parameters
        check_state_dict(
            state_dict=self.model.state_dict(),
            named_parameters=self.model.named_parameters(),
            named_buffers=self.model.named_buffers(),
        )
        self._skip_next_lightning_load_state_dict = True
        # --- DEBUG: hash right after loading ---
        import hashlib
        _h_sd = hashlib.sha256()
        _h_ema = hashlib.sha256()
        _sd = self.model.state_dict()
        for _k, _v in sorted(_sd.items()):
            _h_sd.update(_v.cpu().numpy().tobytes())
        for _s in self.ema.shadow_params:
            _h_ema.update(_s.cpu().numpy().tobytes())
        _dbg_file = os.path.join(
            os.environ.get("FLOODNET_DEBUG_DIR", "/tmp"),
            "eval_state.log",
        )
        with open(_dbg_file, "a") as _f:
            _f.write(
                f"[LOAD step={self.global_step}] "
                f"state_dict_hash={_h_sd.hexdigest()[:24]} "
                f"ema_hash={_h_ema.hexdigest()[:24]}\n"
            )
        # --- END DEBUG ---

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state"] = self.ema.state_dict()
        checkpoint["state_dict"] = self.model.state_dict()
        # Snapshot EMA shadow params as float32 CPU tensors keyed by param name.
        # This lets standalone eval (run_eval.py) reproduce EMA-applied weights
        # exactly, regardless of Lightning's precision-plugin lifecycle ordering.
        trainable_names = [
            name for name, p in self.model.named_parameters() if p.requires_grad
        ]
        checkpoint["ema_applied_trainable"] = {
            name: s.detach().float().cpu().clone()
            for name, s in zip(trainable_names, self.ema.shadow_params)
        }
        # --- DEBUG: hash state_dict + ema + snapshot right at save time ---
        import hashlib
        _h_sd = hashlib.sha256()
        _h_ema = hashlib.sha256()
        _h_snap = hashlib.sha256()
        for _k, _v in sorted(checkpoint["state_dict"].items()):
            _h_sd.update(_v.cpu().numpy().tobytes())
        for _s in checkpoint["ema_state"]["shadow_params"]:
            _h_ema.update(_s.cpu().numpy().tobytes())
        for _v in checkpoint["ema_applied_trainable"].values():
            _h_snap.update(_v.numpy().tobytes())
        _dbg_file = os.path.join(
            os.environ.get("FLOODNET_DEBUG_DIR", "/tmp"),
            "eval_state.log",
        )
        with open(_dbg_file, "a") as _f:
            _f.write(
                f"[SAVE step={self.global_step}] "
                f"sd_hash={_h_sd.hexdigest()[:16]} "
                f"ema_hash={_h_ema.hexdigest()[:16]} "
                f"snap_hash={_h_snap.hexdigest()[:16]}\n"
            )
        # --- END DEBUG ---

    def _step(self, batch, is_training=True):
        out = self.model(batch)
        return out

    def on_train_batch_start(self, batch, batch_idx):
        self.batch_ready_time = time.time()

    def training_step(self, batch, batch_idx):
        net_start_time = time.time()
        # forward
        loss_dict = self._step(batch, is_training=True)
        # logging
        net_end_time = time.time()
        data_time = (
            self.batch_ready_time - self.last_batch_end_time
            if self.last_batch_end_time is not None
            else 0.0
        )
        net_time = net_end_time - net_start_time
        batch_size = self.cfg.data.train_bs
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "data_time", data_time, on_step=True, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "net_time", net_time, on_step=True, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "ckpt_absolute_step",
            float(ckpt_step_info(self).metric_value),
            on_step=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        for key, value in loss_dict.items():
            self.log(
                f"train_loss/{key}",
                value.item(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        return loss_dict["total"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.last_batch_end_time = time.time()
        self.ema.to(self.device)
        self.ema.update()
        # Calculate average difference using vectorized operations
        if self.global_step % 100 == 0:
            self.log("ema_decay", self.ema.decay, sync_dist=False)
            with torch.no_grad():
                # Streaming mean to avoid allocating a ~500 MB concat tensor every 100 steps.
                total_abs, total_n = torch.zeros((), device=self.device), 0
                for p, sp in zip(
                    (p for p in self.model.parameters() if p.requires_grad),
                    self.ema.shadow_params,
                ):
                    total_abs += (p.detach() - sp.to(p.device)).abs().sum()
                    total_n += p.numel()
                avg_diff = total_abs / max(total_n, 1)
                self.log("ema_diff/avg", avg_diff, sync_dist=True)

    # NOTE: lightning handles with torch.no_grad() and model.eval() automatically
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx >= 1:
            _force = getattr(self, "_eval_on_resume", False)
            if (
                not self.trainer.sanity_checking
                and self.global_step > 0
                and (_force or self.global_step % self.cfg.validation.test_steps == 0)
            ):
                self.test_step(batch, batch_idx, dataloader_idx=dataloader_idx - 1)
        else:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            with self.ema.average_parameters(trainable):
                loss_dict = self._step(batch, is_training=False)
            # logging
            batch_size = self.cfg.data.val_bs
            for key, value in loss_dict.items():
                self.log(
                    f"val_loss/{key}",
                    value.item(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
            # metrics
            self.update_metrics(batch)
        return

    def on_validation_epoch_end(self):
        if (
            not self.trainer.sanity_checking
            and self.global_step > 0
            and self.global_step % self.cfg.validation.test_steps == 0
        ):
            self.on_test_epoch_end()
        # metrics
        self.compute_metrics()

    # NOTE: lightning handles with torch.no_grad() and model.eval() automatically
    def test_step(self, batch, batch_idx, dataloader_idx=0, test_loader_idx=None):
        # Lightning requires the exact name `dataloader_idx` when multiple
        # test dataloaders are passed. Keep `test_loader_idx` as a compatibility
        # alias for internal callers and older helper code.
        if test_loader_idx is None:
            test_loader_idx = dataloader_idx
        self.update_test(batch, batch_idx=batch_idx, test_loader_idx=test_loader_idx)
        return

    def on_test_epoch_end(self):
        # Only rank 0 does rendering and wandb logging
        if self.trainer.global_rank == 0:
            self.process_test_results()

    def initialize_metrics(self):
        pass

    def update_metrics(self, batch):
        pass

    def compute_metrics(self):
        pass

    def update_test(self, batch, batch_idx=None, test_loader_idx=0):
        pass

    def process_test_results(self):
        pass
