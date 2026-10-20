from __future__ import annotations

import logging
from pathlib import Path

from ignite.engine import Engine, Events
from tqdm import tqdm

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:  # noqa: BLE001 - logging must never break the caller
            self.handleError(record)


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

        console_handler = TqdmLoggingHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file is not None:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False

    return logger


def attach_progress_bar(
    engine: Engine,
    desc: str = "Running",
    show_epochs: bool = False,
    postfix_key: str | None = None,
    leave: bool = False,
) -> None:
    """Attaches a tqdm progress bar to an Ignite Engine."""
    bar_holder: dict[str, tqdm | None] = {"bar": None}

    @engine.on(Events.EPOCH_STARTED)
    def _start_bar(eng: Engine) -> None:
        if show_epochs and eng.state.max_epochs and eng.state.max_epochs > 1:
            label = f"Epoch {eng.state.epoch}/{eng.state.max_epochs} [{desc}]"
        else:
            label = desc
        bar_holder["bar"] = tqdm(
            total=eng.state.epoch_length,
            desc=label,
            leave=leave,
        )

    @engine.on(Events.ITERATION_COMPLETED)
    def _update_bar(eng: Engine) -> None:
        bar = bar_holder["bar"]
        if bar is None:
            return
        bar.update(1)
        if (
            postfix_key
            and isinstance(eng.state.output, dict)
            and postfix_key in eng.state.output
        ):
            val = eng.state.output[postfix_key]
            if hasattr(val, "item"):
                val = val.item()
            if isinstance(val, (int, float)):
                bar.set_postfix({postfix_key: f"{val:.4f}"})

    @engine.on(Events.EPOCH_COMPLETED)
    def _close_bar(_: Engine) -> None:
        bar = bar_holder["bar"]
        if bar is not None:
            bar.close()
            bar_holder["bar"] = None
