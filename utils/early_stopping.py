from __future__ import annotations

import copy
from typing import Any

import torch
from ignite.engine import Engine, Events

from utils.reporter import Reporter


class EarlyStopping:
    """Exact EarlyStopping implementation from DA-Replications/src/lstm_dann/EarlyStopper."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.best_state_dict: dict[str, Any] | None = None
        self.early_stop = False

    def step(
        self,
        current_val_rmse: float,
        model: torch.nn.Module,
        epoch: int | None = None,
        reporter: Reporter | None = None,
    ) -> bool:
        if current_val_rmse < self.best_score - self.min_delta:
            self.best_score = current_val_rmse
            if epoch is not None:
                self.best_epoch = epoch
            self.epochs_no_improve = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            if reporter is not None:
                reporter.log_model(model, step=epoch)
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore_best_weights(self, model: torch.nn.Module) -> None:
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


def attach_early_stopping(
    trainer: Engine,
    evaluator: Engine,
    model: torch.nn.Module,
    patience: int = 20,
    min_delta: float = 1e-4,
    metric_key: str = "rmse_loss",
    reporter: Reporter | None = None,
    log_model_on_improve: bool = False,
) -> EarlyStopping:
    """
    Attaches DA-Replications EarlyStopping to an Ignite trainer + validation evaluator.
    Must be registered AFTER the validation evaluator's EPOCH_COMPLETED handler and
    BEFORE ModelCheckpoint's COMPLETED handler so best weights are restored prior to saving.
    """
    stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    @trainer.on(Events.EPOCH_COMPLETED)
    def _check_early_stop(engine: Engine) -> None:
        val_metrics = evaluator.state.metrics
        if metric_key not in val_metrics:
            return
        current_val = float(val_metrics[metric_key])
        epoch = engine.state.epoch
        should_stop = stopper.step(
            current_val_rmse=current_val,
            model=model,
            epoch=epoch,
            reporter=reporter if log_model_on_improve else None,
        )
        if should_stop:
            msg = (
                f"Early stopping at epoch {epoch} "
                f"(best {metric_key}={stopper.best_score:.4f} at epoch {stopper.best_epoch})"
            )
            if reporter is not None:
                reporter.info(msg)
            else:
                print(msg, flush=True)
            stopper.restore_best_weights(model)
            engine.terminate()

    @trainer.on(Events.COMPLETED)
    def _restore_on_complete(_: Engine) -> None:
        stopper.restore_best_weights(model)

    return stopper
