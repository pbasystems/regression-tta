from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from utils.logger import get_logger

try:
    import wandb
except ImportError:  # wandb is an optional integration
    wandb = None


def _sanitize_artifact_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-.]", "-", name)


class Reporter:
    def __init__(
        self,
        name: str,
        log_file: str | Path | None = None,
        use_wandb: bool = False,
        wandb_project: str | None = "Regression-TTA",
        wandb_config: dict[str, Any] | None = None,
        wandb_run_name: str | None = None,
    ) -> None:
        self.logger = get_logger(name, log_file=log_file)
        self.use_wandb = use_wandb
        self._run = None
        cwd = os.environ.get("BASE_WORKING_DIR", str(Path.cwd()))
        self.temp_path = os.path.join(cwd, "temp")
        os.makedirs(self.temp_path, exist_ok=True)

        if self.use_wandb:
            if wandb is None:
                self.logger.warning(
                    "wandb is not installed; continuing without wandb logging."
                )
                self.use_wandb = False
            elif not wandb_project:
                self.logger.warning(
                    "wandb_project not set; continuing without wandb logging."
                )
                self.use_wandb = False
            else:
                try:
                    self._run = wandb.init(
                        project=wandb_project, name=wandb_run_name, config=wandb_config
                    )
                except Exception as exc:  # noqa: BLE001 - wandb init is best-effort
                    self.logger.warning(
                        "Failed to initialise wandb run (%s); continuing without wandb logging.",
                        exc,
                    )
                    self.use_wandb = False

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        formatted = " - ".join(
            f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
            for key, value in metrics.items()
        )
        prefix = f"Step {step} - " if step is not None else ""
        self.logger.info("%s%s", prefix, formatted)

        if self.use_wandb and wandb is not None:
            wandb.log(metrics, step=step)

    def log_model(self, model: torch.nn.Module, step: int | None = None) -> None:
        if self.use_wandb and wandb is not None:
            torch.save(
                model.state_dict(), os.path.join(self.temp_path, f"model_{step}.pth")
            )
            artifact = wandb.Artifact(name=f"model_{step}", type="model")
            artifact.add_file(
                local_path=os.path.join(self.temp_path, f"model_{step}.pth"),
                name=f"model_{step}.pth",
            )
            wandb.log_artifact(artifact)

    def log_artifact_files(
        self,
        files: Sequence[str | Path],
        artifact_name: str,
        artifact_type: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Uploads one or more files (*.pt, *.json, *.csv, etc.) as a WandB Artifact."""
        if not (self.use_wandb and wandb is not None):
            return

        safe_name = _sanitize_artifact_name(artifact_name)
        artifact = wandb.Artifact(
            name=safe_name,
            type=artifact_type,
            metadata=metadata or {},
        )
        added_files = []
        for f in files:
            p = Path(f)
            if p.is_file():
                artifact.add_file(local_path=str(p), name=p.name)
                added_files.append(p.name)

        if added_files:
            wandb.log_artifact(artifact)
            self.logger.info(
                "Uploaded WandB artifact '%s' (%s): %s",
                safe_name,
                artifact_type,
                ", ".join(added_files),
            )

    def log_directory_artifacts(
        self,
        directory: str | Path,
        artifact_name: str,
        artifact_type: str = "model",
        patterns: Sequence[str] = ("*.pt", "*.pth", "*.json", "*.csv", "*.yaml"),
        extra_files: Sequence[str | Path] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Collects matching files in a directory (plus any extra files) and logs them to WandB."""
        dir_path = Path(directory)
        collected: list[Path] = []
        for pattern in patterns:
            collected.extend(sorted(dir_path.glob(pattern)))
        for ef in extra_files:
            ep = Path(ef)
            if ep.is_file() and ep not in collected:
                collected.append(ep)
        self.log_artifact_files(
            collected,
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            metadata=metadata,
        )

    def info(self, msg: str, *args: Any) -> None:
        self.logger.info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self.logger.warning(msg, *args)

    def finish(self) -> None:
        if self.use_wandb and wandb is not None:
            wandb.finish()


def get_domain_tag(config: dict[str, Any]) -> str:
    ds_cfg = config.get("dataset", {}).get("config", {})
    src_fd = ds_cfg.get("source_fd", config.get("dataset", {}).get("name", "source"))
    tgt_fd = ds_cfg.get("target_fd", "target")
    return f"{src_fd}-to-{tgt_fd}"


def create_reporter_from_config(
    config: dict[str, Any],
    default_name: str,
    default_stage: str,
) -> Reporter:
    wandb_cfg = config.get("wandb", {})
    use_wandb = wandb_cfg.get("use_wandb", True)
    project = wandb_cfg.get("project", "Regression-TTA")

    domain_tag = get_domain_tag(config)
    default_run_name = f"{domain_tag} | {default_stage}"
    run_name = wandb_cfg.get("run_name", default_run_name)

    return Reporter(
        name=default_name,
        log_file=wandb_cfg.get("log_file", None),
        use_wandb=use_wandb,
        wandb_project=project,
        wandb_config=config,
        wandb_run_name=run_name,
    )
