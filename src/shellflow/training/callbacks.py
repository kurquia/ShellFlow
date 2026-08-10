"""
Custom PyTorch Lightning callbacks for ShellFlow training.

Provides:
- GradientNormCallback: Logs gradient norms
- ConfigLoggingCallback: Logs full config to W&B at start
- ModelCompileCallback: torch.compile at training start
- BestMetricCallback: Tracks and logs the best validation metric
- TrainingStatsCallback: Logs epoch timing / throughput stats
- get_default_callbacks: Assembles the standard callback set
"""

from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback


class GradientNormCallback(Callback):
    """
    Callback to log gradient norms during training.

    Useful for debugging training instability.
    """

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        # Compute total gradient norm (batched — single fused kernel)
        grads = [p.grad for p in pl_module.parameters() if p.grad is not None]
        if grads:
            norms = torch._foreach_norm(grads, 2.0)
            total_norm = torch.stack(norms).norm(2.0).item()
        else:
            total_norm = 0.0

        pl_module.log("train/grad_norm", total_norm, prog_bar=False, sync_dist=True)


class ConfigLoggingCallback(Callback):
    """
    Callback to log complete configuration to W&B at training start.

    Ensures all hyperparameters are captured for reproducibility.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Log config to W&B if logger is available
        if trainer.logger is not None:
            # Log as hyperparameters
            trainer.logger.log_hyperparams(self.config)


class ModelCompileCallback(Callback):
    """
    Callback to compile the model with torch.compile at training start.

    This is done as a callback to ensure proper device placement first.
    """

    def __init__(
        self,
        compile_model: bool = False,
        compile_mode: str = "default",
        compile_fullgraph: bool = False,
        compile_dynamic: bool = None,
    ):
        super().__init__()
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.compile_fullgraph = compile_fullgraph
        self.compile_dynamic = compile_dynamic
        self._compiled = False

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self.compile_model or self._compiled:
            return

        # Check if torch.compile is available (PyTorch 2.0+)
        if not hasattr(torch, "compile"):
            print("Warning: torch.compile not available. Skipping model compilation.")
            return

        print(
            f"Compiling model with mode='{self.compile_mode}', dynamic={self.compile_dynamic}..."
        )

        # Compile the inner model (not the Lightning module)
        if hasattr(pl_module, "model"):
            pl_module.model = torch.compile(
                pl_module.model,
                mode=self.compile_mode,
                fullgraph=self.compile_fullgraph,
                dynamic=self.compile_dynamic,
            )
            self._compiled = True
            print("Model compiled successfully.")


class BestMetricCallback(Callback):
    """
    Callback to track and log best validation metrics.

    More flexible than ModelCheckpoint for custom metric tracking.
    """

    def __init__(
        self,
        monitor: str = "val/loss",
        mode: str = "min",
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0

    def state_dict(self) -> dict:
        return {"best_value": self.best_value, "best_epoch": self.best_epoch}

    def load_state_dict(self, state_dict: dict) -> None:
        self.best_value = state_dict.get("best_value", self.best_value)
        self.best_epoch = state_dict.get("best_epoch", self.best_epoch)

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        # Get current metric value
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        current = current.item() if torch.is_tensor(current) else current

        # Check if this is the best
        is_best = False
        if self.mode == "min" and current < self.best_value:
            is_best = True
        elif self.mode == "max" and current > self.best_value:
            is_best = True

        if is_best:
            self.best_value = current
            self.best_epoch = trainer.current_epoch
            # No sync_dist=True: `current` was already sync'd via the
            # val/loss `sync_dist=True` log in on_validation_epoch_end, so
            # best_value is deterministic across ranks. Adding another
            # all_reduce here doubles the collective surface without
            # changing the value, and risks asymmetry if any rank takes a
            # different early-return path above (e.g., callback_metrics
            # divergence). Local-only log is correct and sufficient.
            pl_module.log(f"best/{self.monitor}", self.best_value, sync_dist=False)
            pl_module.log("best/epoch", self.best_epoch, sync_dist=False)


class TrainingStatsCallback(Callback):
    """
    Callback to log training statistics like throughput and time.
    """

    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        self.epoch_events = 0

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        import time

        self.epoch_start_time = time.time()
        self.epoch_events = 0

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch is not None and "features" in batch:
            self.epoch_events += batch["features"].size(0)

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        import time

        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            events_per_sec = self.epoch_events / epoch_time if epoch_time > 0 else 0

            # sync_dist=False: these are informational per-rank timing stats.
            # With sync_dist=True under DDP, RichProgressBar.on_train_epoch_end
            # fires compute() for these metrics on rank 0 ONLY (its
            # _update_metrics short-circuits on disabled ranks 1-3), which
            # calls torch.distributed.barrier + torch.distributed.all_reduce
            # (= 2× ncclAllReduce) on rank 0 alone. When the
            # sample-plot callback's _update_gen_kde_buffers
            # subsequently calls dist.all_gather (= ncclAllGather) on ranks
            # 1-3, the NCCL collective types mismatch (ncclAllReduce vs
            # ncclAllGather) → deadlock. Before the gen-KDE refresh existed,
            # the next collective
            # ranks 1-3 hit was Higgs gen's dist.barrier (= ncclAllReduce),
            # which matched rank 0's allreduces by SeqNum and unblocked. The
            # all_gather broke that latent invariant. Reporting these as
            # rank-0-local is acceptable (they're informational, not training
            # signal).
            pl_module.log("train/epoch_time_sec", epoch_time, sync_dist=False)
            pl_module.log("train/events_per_sec", events_per_sec, sync_dist=False)
            pl_module.log("train/epoch_events", self.epoch_events, sync_dist=False)


def get_default_callbacks(
    config: dict,
    monitor: str = "val/loss",
    mode: str = "min",
    checkpoint_dir: str = "checkpoints",
    save_top_k: int = 3,
    save_every_n_epochs: int = 5,
    enable_early_stopping: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0001,
    compile_model: bool = False,
    compile_mode: str = "default",
    compile_dynamic: bool = None,
) -> list[Callback]:
    """
    Create a standard set of callbacks for training.

    Args:
        config: Full experiment config dict to log
        monitor: Metric to monitor for checkpointing
        mode: 'min' or 'max' for the monitored metric
        checkpoint_dir: Directory to save checkpoints
        save_top_k: Keep top k checkpoints
        save_every_n_epochs: Also save every n epochs
        enable_early_stopping: Whether to use early stopping
        early_stopping_patience: Epochs without improvement before stopping
        early_stopping_min_delta: Minimum change to qualify as improvement
        compile_model: Whether to compile the model
        compile_mode: torch.compile mode

    Returns:
        List of callbacks
    """
    from pathlib import Path

    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        RichProgressBar,
    )
    from lightning.pytorch.callbacks import (
        ModelCheckpoint as _ModelCheckpoint,
    )

    class ModelCheckpoint(_ModelCheckpoint):
        """Lustre-safe ModelCheckpoint that tolerates missing files on removal."""

        def _remove_checkpoint(self, trainer, filepath):
            try:
                super()._remove_checkpoint(trainer, filepath)
            except FileNotFoundError:
                pass

    callbacks = []

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Gradient norm logging
    callbacks.append(GradientNormCallback(log_every_n_steps=50))

    # Config logging
    callbacks.append(ConfigLoggingCallback(config))

    # Training stats
    callbacks.append(TrainingStatsCallback())

    # Best metric tracking
    callbacks.append(BestMetricCallback(monitor=monitor, mode=mode))

    # Model compilation
    if compile_model:
        callbacks.append(
            ModelCompileCallback(
                compile_model=True,
                compile_mode=compile_mode,
                compile_dynamic=compile_dynamic,
            )
        )

    # Checkpointing - best model
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks.append(
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best-{epoch:02d}-{" + monitor.replace("/", "_") + ":.4f}",
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=True,
            verbose=True,
        )
    )

    # Checkpointing - periodic
    if save_every_n_epochs > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="epoch-{epoch:02d}",
                every_n_epochs=save_every_n_epochs,
                save_top_k=-1,  # Keep all periodic checkpoints
                verbose=False,
            )
        )

    # Early stopping
    if enable_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                verbose=True,
            )
        )

    # Rich progress bar
    try:
        callbacks.append(RichProgressBar())
    except ImportError:
        pass  # Rich not installed, use default progress bar

    return callbacks
