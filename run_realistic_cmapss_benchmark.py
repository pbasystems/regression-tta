from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from ignite.engine import Events
from torch.utils.data import DataLoader

from adaptation import create_optimizer as create_tta_optimizer
from dataset import get_datasets
from evaluation.evaluator import RegressionEvaluator
from feature_stats import FeatureStatCalculator
from model import create_regressor
from trainer import RegressionTrainer
from tta import TTAEngine
from tune_optuna import compute_unsupervised_subspace_kl
from utils.early_stopping import attach_early_stopping
from utils.reporter import Reporter
from utils.seed import fix_seed

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL_FDS = ["FD001", "FD002", "FD003", "FD004"]

# ---------------------------------------------------------------------------
# ONE FIXED ARCHITECTURE ACROSS ALL TASKS (No per-task architecture changes)
# ---------------------------------------------------------------------------
# Hybrid 'lstm_tta': Fixed LSTM-DANN FeatureExtractor + Original regression-tta Linear Head nn.Linear(f_size, 1)
FIXED_LSTM_TTA_ARCH: dict[str, Any] = {
    "input_size": 24,
    "lstm_hidden_size": 128,
    "f_size": 64,
    "num_layers": 1,
    "lstm_dropout": 0.5,
    "use_bn": True,
}

# Full 'lstm_dann' with fixed architecture across all 4 FDs
FIXED_LSTM_DANN_ARCH: dict[str, Any] = {
    "input_size": 24,
    "lstm_hidden_size": 128,
    "f_size": 64,
    "num_layers": 1,
    "lstm_dropout": 0.5,
    "regressor_dropout": 0.3,
    "regressor_first_hidden_size": 32,
    "regressor_second_hidden_size": 0,
}

# Original 'table' (MLPRegressor) fixed architecture
FIXED_MLP_ARCH: dict[str, Any] = {
    "in_dims": 24 * 30,
    "h_dims": 128,
    "n_rep": 4,
}


def get_fixed_arch(model_type: str) -> dict[str, Any]:
    match model_type:
        case "lstm_tta":
            return copy.deepcopy(FIXED_LSTM_TTA_ARCH)
        case "lstm_dann":
            return copy.deepcopy(FIXED_LSTM_DANN_ARCH)
        case "table":
            return copy.deepcopy(FIXED_MLP_ARCH)
        case _:
            raise ValueError(f"Unsupported model_type: {model_type!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Realistic Deployment CMAPSS Benchmark:\n"
            "  1) Source-Only & Target-Only baselines (with a single fixed architecture)\n"
            "  2) General Source Training Hyperparameters (shared across all tasks)\n"
            "  3) Out-of-the-box Default TTA (SSA) evaluation (no Optuna at test time)\n"
            "  4) Dataset-General TTA Hyperparameters (one shared TTA config for the dataset, not per-task)\n"
            "  5) Online & Offline SSA evaluation across all 12 tasks."
        )
    )
    parser.add_argument(
        "--model-type",
        choices=["lstm_tta", "lstm_dann", "table"],
        default="lstm_tta",
        help=(
            "Fixed model architecture used across all tasks:\n"
            "  - 'lstm_tta' (default): Fixed LSTM_DANN FeatureExtractor + original regression-tta Linear(f_size, 1) regressor\n"
            "  - 'lstm_dann': Fixed LSTM_DANN FeatureExtractor + MLP Regressor\n"
            "  - 'table': Original regression-tta MLPRegressor"
        ),
    )
    parser.add_argument(
        "-o",
        default="result/benchmark_realistic",
        help="Root output directory for models, general hyperparameters, and final CSV tables",
    )
    parser.add_argument(
        "--storage",
        default="sqlite:///result/optuna/optuna_studies.db",
        help="Optuna SQLite storage URL",
    )
    parser.add_argument(
        "--n-general-source-trials",
        type=int,
        default=20,
        help="Number of Optuna trials to find ONE general source-training hyperparameter set (default: 20; 0 to skip)",
    )
    parser.add_argument(
        "--n-general-tta-trials",
        type=int,
        default=30,
        help="Number of Optuna trials to find ONE general TTA hyperparameter set for the dataset (default: 30)",
    )
    parser.add_argument(
        "--calibration-pair",
        nargs=2,
        default=["FD001", "FD003"],
        metavar=("CAL_SRC", "CAL_TGT"),
        help=(
            "Source->Target validation pair (using training splits only, never test splits) "
            "for calibrating general supervised TTA hyperparameters (default: FD001 FD003)"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum training epochs for source models (default: 200, with EarlyStopping)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="EarlyStopping patience on source validation RMSE (default: 20)",
    )
    parser.add_argument(
        "--fds",
        nargs="+",
        default=ALL_FDS,
        choices=ALL_FDS,
        help="Subset of CMAPSS sub-datasets to include (default: FD001 FD002 FD003 FD004)",
    )
    parser.add_argument(
        "--target-split",
        choices=["test", "train"],
        default="test",
        help="Target split used for final adaptation and evaluation (default: test)",
    )
    parser.add_argument(
        "--last-window-only",
        action="store_true",
        help="Evaluate only on the last window per engine in target test split",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wandb-project",
        default="Regression-TTA",
        help="WandB project name (default: Regression-TTA)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    return parser.parse_args()


def build_base_train_config(
    fd: str,
    epochs: int,
    patience: int = 20,
    model_type: str = "lstm_tta",
    train_hp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    arch_cfg = get_fixed_arch(model_type)
    hp = train_hp or {}

    lr = float(hp.get("source_lr", 1.0e-3))
    wd = float(hp.get("source_weight_decay", 1.0e-4))
    batch_size = int(hp.get("source_batch_size", 256))
    norm_mode = str(hp.get("norm_mode", "minmax"))

    if model_type in ("lstm_tta", "lstm_dann") and "lstm_dropout" in hp:
        arch_cfg["lstm_dropout"] = float(hp["lstm_dropout"])

    return {
        "epoch": epochs,
        "early_stopping": {
            "enabled": True,
            "patience": patience,
            "min_delta": 1.0e-4,
        },
        "dataset": {
            "name": "cmapss",
            "config": {
                "source_domain": True,
                "norm_mode": norm_mode,
                "combine_splits_for_stats": True,
                "source_fd": fd,
                "target_fd": fd,
                "source_split": "train",
                "target_split": "train",
                "feature_cols": "all_24",
                "max_rul": 125.0,
                "normalize_rul": False,
                "window_size": 30,
            },
            "train_ratio": 0.9,
            "split_by_engine": True,
        },
        "regressor": {
            "type": model_type,
            "config": arch_cfg,
        },
        "optimizer": {
            "param": "all",
            "name": "Adam",
            "config": {
                "lr": lr,
                "weight_decay": wd,
            },
        },
        "trainer": {"compile_model": None},
        "evaluator": {"compile_model": None},
        "train_dataloader": {"batch_size": batch_size, "shuffle": True},
        "val_dataloader": {"batch_size": 256},
    }


def build_tta_config(
    source_fd: str,
    target_fd: str,
    ckpt_path: Path,
    stat_path: Path,
    model_type: str = "lstm_tta",
    train_cfg: dict[str, Any] | None = None,
    tta_hp: dict[str, Any] | None = None,
    target_split: str = "test",
    last_window_only: bool = False,
) -> dict[str, Any]:
    """
    Builds a TTA configuration.
    If `tta_hp` is None, uses the exact Default TTA settings from the regression-tta paper.
    If `tta_hp` is provided, applies the general dataset-wide TTA hyperparameters.
    """
    arch_cfg = (
        copy.deepcopy(train_cfg["regressor"]["config"])
        if train_cfg
        else get_fixed_arch(model_type)
    )
    norm_mode = (
        train_cfg["dataset"]["config"].get("norm_mode", "minmax")
        if train_cfg
        else "minmax"
    )

    # Default TTA settings from regression-tta paper
    hp = tta_hp or {}
    adapt_param = str(hp.get("adapt_param", "fe"))
    opt_name = str(hp.get("optimizer_name", "Adam"))
    lr = float(hp.get("lr", 1.0e-4))
    wd = float(hp.get("weight_decay", 0.0))
    train_mode = bool(hp.get("train_mode", False))
    weight_bias = float(hp.get("weight_bias", 1.0))
    weight_exp = float(hp.get("weight_exp", 1.0))
    contrib_top_k = int(hp.get("contrib_top_k", 5))
    loss_eps = float(hp.get("loss_eps", 1.0e-8))
    batch_size = int(hp.get("batch_size", 256))

    return {
        "dataset": {
            "name": "cmapss",
            "config": {
                "source_domain": False,
                "norm_mode": norm_mode,
                "combine_splits_for_stats": True,
                "source_fd": source_fd,
                "target_fd": target_fd,
                "source_split": "train",
                "target_split": target_split,
                "feature_cols": "all_24",
                "max_rul": 125.0,
                "normalize_rul": False,
                "window_size": 30,
                "last_window_only": last_window_only,
            },
        },
        "regressor": {
            "source": str(ckpt_path),
            "type": model_type,
            "config": arch_cfg,
        },
        "optimizer": {
            "param": adapt_param,
            "name": opt_name,
            "config": {
                "lr": lr,
                "weight_decay": wd,
            },
        },
        "tta": {
            "type": "pca",
            "config": {
                "train_mode": train_mode,
                "weight_bias": weight_bias,
                "weight_exp": weight_exp,
                "pc_config": {
                    "stat_file": str(stat_path),
                    "contrib_top_k": contrib_top_k,
                },
                "loss_config": {"eps": loss_eps},
                "compile_model": None,
            },
        },
        "evaluator": {"compile_model": None},
        "adapt_dataloader": {"batch_size": batch_size, "shuffle": True},
        "val_dataloader": {"batch_size": 256},
    }


def compute_and_save_feature_stats(
    net: torch.nn.Module,
    train_cfg: dict[str, Any],
    stat_path: Path,
) -> int:
    stat_cfg = copy.deepcopy(train_cfg)
    stat_cfg["dataset"]["train_aug"] = False
    ds = get_datasets(stat_cfg)[0]
    dl = DataLoader(ds, batch_size=256, shuffle=False)

    calc = FeatureStatCalculator(net, compile_model=None)
    calc.run(dl)
    mean, cov, features = calc.compute_stats()
    cov_np = cov.numpy()
    eigvals, eigvecs = np.linalg.eigh(cov_np)
    cov_rank = int(np.linalg.matrix_rank(cov_np))

    stat_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean,
            "basis": torch.from_numpy(eigvecs),
            "eigvals": torch.from_numpy(eigvals),
        },
        str(stat_path),
    )
    torch.save(features, str(stat_path.parent / "raw_features.pt"))
    return cov_rank


def train_or_load_domain_model(
    fd: str,
    out_root: Path,
    subdir_name: str,
    epochs: int,
    patience: int,
    model_type: str,
    train_hp: dict[str, Any] | None,
    seed: int,
    use_wandb: bool,
    wandb_project: str,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    """Trains (or loads if cached) a source domain model with the fixed architecture."""
    domain_dir = out_root / subdir_name / f"{fd}_{model_type}"
    domain_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = domain_dir / f"source_model_{fd}.pt"
    stat_path = domain_dir / "feature_stats.pt"
    cfg_path = domain_dir / "train_config.yaml"
    meta_path = domain_dir / "train_summary.json"

    train_cfg = build_base_train_config(
        fd, epochs=epochs, patience=patience, model_type=model_type, train_hp=train_hp
    )

    if (
        ckpt_path.is_file()
        and stat_path.is_file()
        and meta_path.is_file()
        and cfg_path.is_file()
    ):
        print(f"[{fd} | {subdir_name}] Loading existing model from {ckpt_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f)
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return ckpt_path, stat_path, train_cfg, meta

    print(
        f"\n================ TRAINING [{subdir_name}] ({model_type}) ON {fd} (max {epochs} epochs, patience={patience}) ================"
    )
    fix_seed(seed)
    reporter = Reporter(
        name=f"Realistic.{subdir_name}.{fd}.{model_type}",
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config=train_cfg,
        wandb_run_name=f"{fd} | {model_type} | {subdir_name}",
    )

    net = create_regressor(train_cfg).to(DEVICE)
    opt = getattr(torch.optim, train_cfg["optimizer"]["name"])(
        net.parameters(), **train_cfg["optimizer"]["config"]
    )
    train_ds, val_ds = get_datasets(train_cfg)
    train_dl = DataLoader(train_ds, **train_cfg["train_dataloader"])
    val_dl = DataLoader(val_ds, **train_cfg["val_dataloader"])

    trainer = RegressionTrainer(net, opt, **train_cfg["trainer"])
    val_ev = RegressionEvaluator(net, **train_cfg["evaluator"])

    @trainer.on(Events.EPOCH_COMPLETED)
    def _eval_and_log(engine: Any) -> None:
        val_ev.run(val_dl)
        ep = engine.state.epoch
        reporter.log_metrics(
            {
                "epoch": ep,
                "train/rmse": float(trainer.state.metrics["rmse"]),
                "train/mae": float(trainer.state.metrics["mae"]),
                "val/rmse_loss": float(val_ev.state.metrics["rmse_loss"]),
                "val/mae_loss": float(val_ev.state.metrics["mae_loss"]),
                "val/R2": float(val_ev.state.metrics["R2"]),
            },
            step=ep,
        )

    stopper = attach_early_stopping(
        trainer=trainer,
        evaluator=val_ev,
        model=net,
        patience=patience,
        reporter=reporter,
    )
    trainer.run(train_dl, max_epochs=epochs)
    val_ev.run(val_dl)

    torch.save(net.state_dict(), ckpt_path)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.dump(train_cfg, f, sort_keys=False)

    cov_rank = compute_and_save_feature_stats(net, train_cfg, stat_path)

    meta = {
        "fd": fd,
        "model_type": model_type,
        "subdir": subdir_name,
        "max_epochs": epochs,
        "best_epoch": stopper.best_epoch,
        "patience": patience,
        "cov_rank": cov_rank,
        "train_hp": train_hp or {},
        "train_metrics": {k: float(v) for k, v in trainer.state.metrics.items()},
        "val_metrics": {k: float(v) for k, v in val_ev.state.metrics.items()},
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    reporter.log_directory_artifacts(
        directory=domain_dir,
        artifact_name=f"{subdir_name}-{model_type}-{fd}",
        artifact_type="model",
    )
    reporter.finish()
    return ckpt_path, stat_path, train_cfg, meta


def tune_general_source_hyperparameters(
    calibration_fd: str,
    out_root: Path,
    storage: str,
    n_trials: int,
    epochs: int,
    patience: int,
    model_type: str,
    seed: int,
    use_wandb: bool,
    wandb_project: str,
) -> dict[str, Any]:
    """
    Finds ONE general set of source training hyperparameters (learning rate, weight decay,
    batch size, dropout) using the fixed architecture on a single reference source domain (`calibration_fd`).
    Architecture dimensions (`lstm_hidden_size`, `f_size`, `num_layers`) remain strictly fixed!
    """
    tune_dir = out_root / "general_source_tuning"
    tune_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tune_dir / f"general_source_hp_{model_type}.json"

    if summary_path.is_file():
        print(
            f"[General Source HP | {model_type}] Loading cached general source hyperparameters from {summary_path}"
        )
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)["best_params"]

    print(
        f"\n================ TUNING GENERAL SOURCE HYPERPARAMETERS ({model_type}) ON {calibration_fd} ({n_trials} trials) ================"
    )
    study_name = f"realistic_general_source_hp_{model_type}_{calibration_fd}"

    reporter = Reporter(
        name=f"Realistic.GeneralSourceHP.{model_type}",
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config={
            "calibration_fd": calibration_fd,
            "model_type": model_type,
            "n_trials": n_trials,
            "fixed_arch": get_fixed_arch(model_type),
        },
        wandb_run_name=f"General Source HP Tuning | {model_type} ({calibration_fd})",
    )

    def objective(trial: optuna.Trial) -> float:
        fix_seed(seed)
        hp: dict[str, Any] = {
            "source_lr": trial.suggest_float("source_lr", 3e-4, 5e-3, log=True),
            "source_weight_decay": trial.suggest_float(
                "source_weight_decay", 1e-6, 1e-3, log=True
            ),
            "source_batch_size": trial.suggest_categorical(
                "source_batch_size", [64, 128, 256]
            ),
            "norm_mode": "minmax",
        }
        if model_type in ("lstm_tta", "lstm_dann"):
            hp["lstm_dropout"] = trial.suggest_float("lstm_dropout", 0.1, 0.5)

        cfg = build_base_train_config(
            calibration_fd,
            epochs=epochs,
            patience=patience,
            model_type=model_type,
            train_hp=hp,
        )
        net = create_regressor(cfg).to(DEVICE)
        opt = getattr(torch.optim, cfg["optimizer"]["name"])(
            net.parameters(), **cfg["optimizer"]["config"]
        )
        train_ds, val_ds = get_datasets(cfg)
        train_dl = DataLoader(train_ds, **cfg["train_dataloader"])
        val_dl = DataLoader(val_ds, **cfg["val_dataloader"])

        trainer = RegressionTrainer(net, opt, **cfg["trainer"])
        val_ev = RegressionEvaluator(net, **cfg["evaluator"])

        @trainer.on(Events.EPOCH_COMPLETED)
        def _run_val(_: Any) -> None:
            val_ev.run(val_dl)

        stopper = attach_early_stopping(
            trainer=trainer,
            evaluator=val_ev,
            model=net,
            patience=patience,
        )
        trainer.run(train_dl, max_epochs=cfg["epoch"])
        val_ev.run(val_dl)

        val_rmse = float(val_ev.state.metrics["rmse_loss"])
        trial.set_user_attr("best_epoch", stopper.best_epoch)
        reporter.log_metrics(
            {
                "trial": trial.number,
                "trial/best_epoch": stopper.best_epoch,
                "trial/val_rmse": val_rmse,
            },
            step=trial.number,
        )
        return val_rmse

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )
    completed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    remaining = max(0, n_trials - completed)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    summary = {
        "calibration_fd": calibration_fd,
        "model_type": model_type,
        "study_name": study_name,
        "best_val_rmse": float(study.best_value),
        "best_params": study.best_params,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    study.trials_dataframe().to_csv(
        tune_dir / f"general_source_trials_{model_type}.csv", index=False
    )
    reporter.finish()
    return dict(study.best_params)


def tune_general_tta_hyperparameters(
    cal_source_fd: str,
    cal_target_fd: str,
    objective_mode: str,
    source_models: dict[str, tuple[Path, Path, dict[str, Any], dict[str, Any]]],
    out_root: Path,
    storage: str,
    n_trials: int,
    model_type: str,
    seed: int,
    use_wandb: bool,
    wandb_project: str,
) -> dict[str, Any]:
    """
    Finds ONE general set of TTA hyperparameters for the entire C-MAPSS dataset
    (NOT per-task!) using a single calibration pair (`cal_source_fd -> cal_target_fd`)
    on the target TRAINING split (`target_split='train'`), so the target test splits
    are never touched during hyperparameter tuning.

    Supported `objective_mode`:
      - 'unsupervised_kl': Label-free subspace KL alignment on calibration pair
      - 'calibration_rmse': Offline RMSE on calibration pair (`train_FD001 -> train_FD003`)
    """
    tune_dir = out_root / "general_tta_tuning"
    tune_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tune_dir / f"general_tta_hp_{model_type}_{objective_mode}.json"

    if summary_path.is_file():
        print(
            f"[General TTA HP | {objective_mode}] Loading cached general TTA hyperparameters from {summary_path}"
        )
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)["best_params"]

    print(
        f"\n================ TUNING GENERAL DATASET-WIDE TTA HYPERPARAMETERS ({model_type} | {objective_mode}) "
        f"ON CALIBRATION PAIR {cal_source_fd}->train_{cal_target_fd} ({n_trials} trials) ================"
    )
    study_name = f"realistic_general_tta_{model_type}_{objective_mode}_{cal_source_fd}_to_{cal_target_fd}"

    src_ckpt, src_stat, src_train_cfg, _ = source_models[cal_source_fd]
    stat_dict = torch.load(src_stat, map_location="cpu")
    max_pc_dim = int(stat_dict["eigvals"].shape[0])

    # Pre-load calibration dataset (using target_split='train' so test splits remain unseen)
    base_cal_cfg = build_tta_config(
        source_fd=cal_source_fd,
        target_fd=cal_target_fd,
        ckpt_path=src_ckpt,
        stat_path=src_stat,
        model_type=model_type,
        train_cfg=src_train_cfg,
        tta_hp=None,
        target_split="train",
        last_window_only=False,
    )
    _, cal_ds = get_datasets(base_cal_cfg)

    has_bn = model_type in ("table", "lstm_tta")
    adapt_param_choices = ["fe", "fe_bn", "all"] if has_bn else ["fe", "all"]

    reporter = Reporter(
        name=f"Realistic.GeneralTTA.{model_type}.{objective_mode}",
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config={
            "calibration_pair": f"{cal_source_fd}->train_{cal_target_fd}",
            "objective_mode": objective_mode,
            "model_type": model_type,
            "n_trials": n_trials,
        },
        wandb_run_name=f"General TTA HP Tuning [{objective_mode}] | {model_type}",
    )

    def objective(trial: optuna.Trial) -> float:
        fix_seed(seed)
        tta_hp = {
            "contrib_top_k": trial.suggest_int("contrib_top_k", 2, min(32, max_pc_dim)),
            "lr": trial.suggest_float("lr", 1e-5, 2e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
            "optimizer_name": trial.suggest_categorical(
                "optimizer_name", ["Adam", "AdamW", "SGD"]
            ),
            "adapt_param": trial.suggest_categorical(
                "adapt_param", adapt_param_choices
            ),
            "weight_bias": trial.suggest_float("weight_bias", 0.1, 4.0),
            "weight_exp": trial.suggest_float("weight_exp", 0.1, 2.5),
            "loss_eps": trial.suggest_float("loss_eps", 1e-8, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
            "train_mode": trial.suggest_categorical("train_mode", [False, True]),
        }

        cfg = build_tta_config(
            source_fd=cal_source_fd,
            target_fd=cal_target_fd,
            ckpt_path=src_ckpt,
            stat_path=src_stat,
            model_type=model_type,
            train_cfg=src_train_cfg,
            tta_hp=tta_hp,
            target_split="train",
            last_window_only=False,
        )
        cal_dl = DataLoader(cal_ds, **cfg["adapt_dataloader"])

        reg = create_regressor(cfg).to(DEVICE)
        reg.load_state_dict(torch.load(src_ckpt, map_location=DEVICE))
        opt = create_tta_optimizer(reg, cfg)
        engine = TTAEngine(reg, opt, **cfg["tta"]["config"])
        ev = RegressionEvaluator(reg, **cfg["evaluator"])

        engine.run(cal_dl)
        ev.run(cal_dl)

        unsup_kl = compute_unsupervised_subspace_kl(
            reg,
            DataLoader(cal_ds, **cfg["val_dataloader"]),
            stat_file=str(src_stat),
        )
        online_rmse = float(engine.state.metrics["rmse_loss"])
        offline_rmse = float(ev.state.metrics["rmse_loss"])

        trial.set_user_attr("unsup_kl", unsup_kl)
        trial.set_user_attr("online_rmse", online_rmse)
        trial.set_user_attr("offline_rmse", offline_rmse)

        reporter.log_metrics(
            {
                "trial": trial.number,
                "trial/unsup_kl": unsup_kl,
                "trial/online_rmse": online_rmse,
                "trial/offline_rmse": offline_rmse,
            },
            step=trial.number,
        )

        if objective_mode == "unsupervised_kl":
            return unsup_kl
        return offline_rmse

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )
    completed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    remaining = max(0, n_trials - completed)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    summary = {
        "calibration_pair": f"{cal_source_fd}->train_{cal_target_fd}",
        "objective_mode": objective_mode,
        "model_type": model_type,
        "study_name": study_name,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    study.trials_dataframe().to_csv(
        tune_dir / f"general_tta_trials_{model_type}_{objective_mode}.csv", index=False
    )
    reporter.finish()
    return dict(study.best_params)


def evaluate_model_without_tta(
    tta_cfg: dict[str, Any],
    ckpt_path: Path,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluates a checkpoint directly on the target dataset without any adaptation."""
    fix_seed(seed)
    _, val_ds = get_datasets(tta_cfg)
    val_dl = DataLoader(val_ds, **tta_cfg["val_dataloader"])

    net = create_regressor(tta_cfg).to(DEVICE)
    net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    ev = RegressionEvaluator(net, **tta_cfg["evaluator"])
    ev.run(val_dl)
    return {k: float(v) for k, v in ev.state.metrics.items()}


def evaluate_ssa_tta(
    tta_cfg: dict[str, Any],
    ckpt_path: Path,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Runs SSA Test-Time Adaptation (single pass of simple gradient steps, no Optuna)
    and returns both Online (streaming) and Offline (post-adaptation) metrics.
    """
    fix_seed(seed)
    _, val_ds = get_datasets(tta_cfg)
    adapt_dl = DataLoader(val_ds, **tta_cfg["adapt_dataloader"])
    eval_dl = DataLoader(val_ds, **tta_cfg["val_dataloader"])

    net = create_regressor(tta_cfg).to(DEVICE)
    net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    opt = create_tta_optimizer(net, tta_cfg)
    engine = TTAEngine(net, opt, **tta_cfg["tta"]["config"])
    ev = RegressionEvaluator(net, **tta_cfg["evaluator"])

    engine.run(adapt_dl)
    ev.run(eval_dl)
    return {
        "online": {k: float(v) for k, v in engine.state.metrics.items()},
        "offline": {k: float(v) for k, v in ev.state.metrics.items()},
    }


def save_realistic_benchmark_tables(
    rows: list[dict[str, Any]],
    general_hyperparams: dict[str, Any],
    out_root: Path,
) -> tuple[Path, Path, Path]:
    df_all = pd.DataFrame(rows)

    rmse_cols = [
        "Task",
        "Source_Only_RMSE",
        "Target_Only_RMSE",
        "General_Source_Unadapted_RMSE",
        "Default_TTA_Online_RMSE",
        "Default_TTA_Offline_RMSE",
        "General_TTA_UnsupKL_Online_RMSE",
        "General_TTA_UnsupKL_Offline_RMSE",
        "General_TTA_Calibrated_Online_RMSE",
        "General_TTA_Calibrated_Offline_RMSE",
    ]
    mae_cols = [
        "Task",
        "Source_Only_MAE",
        "Target_Only_MAE",
        "General_Source_Unadapted_MAE",
        "Default_TTA_Online_MAE",
        "Default_TTA_Offline_MAE",
        "General_TTA_UnsupKL_Online_MAE",
        "General_TTA_UnsupKL_Offline_MAE",
        "General_TTA_Calibrated_Online_MAE",
        "General_TTA_Calibrated_Offline_MAE",
    ]

    existing_rmse_cols = [c for c in rmse_cols if c in df_all.columns]
    existing_mae_cols = [c for c in mae_cols if c in df_all.columns]

    rmse_csv = out_root / "summary_rmse_table.csv"
    mae_csv = out_root / "summary_mae_table.csv"
    full_csv = out_root / "summary_all_metrics.csv"
    hp_json = out_root / "general_hyperparameters.json"

    df_all[existing_rmse_cols].to_csv(rmse_csv, index=False)
    df_all[existing_mae_cols].to_csv(mae_csv, index=False)
    df_all.to_csv(full_csv, index=False)

    with hp_json.open("w", encoding="utf-8") as f:
        json.dump(general_hyperparams, f, indent=4, ensure_ascii=False)

    return rmse_csv, mae_csv, full_csv


def main() -> None:
    args = parse_args()
    out_root = Path(args.o)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.storage.startswith("sqlite:///"):
        db_file = Path(args.storage.replace("sqlite:///", "", 1))
        db_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        db_file = None

    use_wandb = not args.no_wandb
    fds = args.fds
    model_type = args.model_type
    cal_src, cal_tgt = args.calibration_pair

    general_hyperparams: dict[str, Any] = {
        "model_type": model_type,
        "fixed_architecture": get_fixed_arch(model_type),
        "general_source_train_hp": {},
        "default_tta_hp": {
            "adapt_param": "fe",
            "optimizer_name": "Adam",
            "lr": 1.0e-4,
            "weight_decay": 0.0,
            "train_mode": False,
            "weight_bias": 1.0,
            "weight_exp": 1.0,
            "contrib_top_k": 5,
            "loss_eps": 1.0e-8,
            "batch_size": 256,
        },
        "general_tta_unsup_kl_hp": {},
        "general_tta_calibrated_hp": {},
    }

    # -------------------------------------------------------------------------
    # STEP 1: Train (or Load) Untuned Base Models for Source-Only & Target-Only
    #         using the single fixed architecture.
    # -------------------------------------------------------------------------
    base_models: dict[str, tuple[Path, Path, dict[str, Any], dict[str, Any]]] = {}
    for fd in fds:
        base_models[fd] = train_or_load_domain_model(
            fd=fd,
            out_root=out_root,
            subdir_name="01_base_domain_models",
            epochs=args.epochs,
            patience=args.patience,
            model_type=model_type,
            train_hp=None,
            seed=args.seed,
            use_wandb=use_wandb,
            wandb_project=args.wandb_project,
        )

    # Compute Target-Only (Oracle) Baselines on each FD
    target_only_metrics: dict[str, dict[str, float]] = {}
    for target_fd in fds:
        t_ckpt, t_stat, t_train_cfg, _ = base_models[target_fd]
        t_eval_cfg = build_tta_config(
            source_fd=target_fd,
            target_fd=target_fd,
            ckpt_path=t_ckpt,
            stat_path=t_stat,
            model_type=model_type,
            train_cfg=t_train_cfg,
            tta_hp=None,
            target_split=args.target_split,
            last_window_only=args.last_window_only,
        )
        target_only_metrics[target_fd] = evaluate_model_without_tta(
            t_eval_cfg, t_ckpt, seed=args.seed
        )
        print(
            f"[Target-Only Oracle {target_fd}->{target_fd} ({model_type})] "
            f"RMSE={target_only_metrics[target_fd]['rmse_loss']:.2f}, "
            f"MAE={target_only_metrics[target_fd]['mae_loss']:.2f}"
        )

    # -------------------------------------------------------------------------
    # STEP 2: Find ONE Set of General Source Training Hyperparameters
    #         (using fixed architecture) & Train All 4 Source Models With It
    # -------------------------------------------------------------------------
    if args.n_general_source_trials > 0:
        general_src_hp = tune_general_source_hyperparameters(
            calibration_fd=cal_src,
            out_root=out_root,
            storage=args.storage,
            n_trials=args.n_general_source_trials,
            epochs=args.epochs,
            patience=args.patience,
            model_type=model_type,
            seed=args.seed,
            use_wandb=use_wandb,
            wandb_project=args.wandb_project,
        )
        general_hyperparams["general_source_train_hp"] = general_src_hp

        general_source_models: dict[
            str, tuple[Path, Path, dict[str, Any], dict[str, Any]]
        ] = {}
        for fd in fds:
            general_source_models[fd] = train_or_load_domain_model(
                fd=fd,
                out_root=out_root,
                subdir_name="02_general_tuned_source_models",
                epochs=args.epochs,
                patience=args.patience,
                model_type=model_type,
                train_hp=general_src_hp,
                seed=args.seed,
                use_wandb=use_wandb,
                wandb_project=args.wandb_project,
            )
    else:
        general_source_models = base_models

    # -------------------------------------------------------------------------
    # STEP 3: Find ONE Set of Dataset-General TTA Hyperparameters
    #         (Calibrated on `cal_src -> train_cal_tgt`, NOT per-task!)
    # -------------------------------------------------------------------------
    general_tta_unsup_hp = tune_general_tta_hyperparameters(
        cal_source_fd=cal_src,
        cal_target_fd=cal_tgt,
        objective_mode="unsupervised_kl",
        source_models=general_source_models,
        out_root=out_root,
        storage=args.storage,
        n_trials=args.n_general_tta_trials,
        model_type=model_type,
        seed=args.seed,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
    )
    general_hyperparams["general_tta_unsup_kl_hp"] = general_tta_unsup_hp

    general_tta_cal_hp = tune_general_tta_hyperparameters(
        cal_source_fd=cal_src,
        cal_target_fd=cal_tgt,
        objective_mode="calibration_rmse",
        source_models=general_source_models,
        out_root=out_root,
        storage=args.storage,
        n_trials=args.n_general_tta_trials,
        model_type=model_type,
        seed=args.seed,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
    )
    general_hyperparams["general_tta_calibrated_hp"] = general_tta_cal_hp

    # -------------------------------------------------------------------------
    # STEP 4: Evaluate Across All 12 Cross-Domain Tasks (Zero Optuna at Test Time!)
    #         Each task runs a single pass of simple SSA gradient steps.
    # -------------------------------------------------------------------------
    table_rows: list[dict[str, Any]] = []

    for source_fd in fds:
        for target_fd in fds:
            if source_fd == target_fd:
                continue

            task_name = f"{source_fd}->{target_fd}"
            print(
                "\n=================================================================="
            )
            print(f"   REALISTIC DEPLOYMENT TASK: {task_name} (model={model_type})")
            print("==================================================================")

            # A) Base Source-Only & Target-Only
            base_ckpt, base_stat, base_train_cfg, _ = base_models[source_fd]
            base_eval_cfg = build_tta_config(
                source_fd=source_fd,
                target_fd=target_fd,
                ckpt_path=base_ckpt,
                stat_path=base_stat,
                model_type=model_type,
                train_cfg=base_train_cfg,
                tta_hp=None,
                target_split=args.target_split,
                last_window_only=args.last_window_only,
            )
            source_only = evaluate_model_without_tta(
                base_eval_cfg, base_ckpt, seed=args.seed
            )
            target_only = target_only_metrics[target_fd]

            # B) General-Tuned Source Model (Unadapted)
            gen_ckpt, gen_stat, gen_train_cfg, _ = general_source_models[source_fd]
            default_tta_cfg = build_tta_config(
                source_fd=source_fd,
                target_fd=target_fd,
                ckpt_path=gen_ckpt,
                stat_path=gen_stat,
                model_type=model_type,
                train_cfg=gen_train_cfg,
                tta_hp=None,  # Default regression-tta settings
                target_split=args.target_split,
                last_window_only=args.last_window_only,
            )
            gen_source_unadapted = evaluate_model_without_tta(
                default_tta_cfg, gen_ckpt, seed=args.seed
            )

            # C) Experiment 1: Default TTA Settings from regression-tta (Online & Offline SSA)
            default_tta_metrics = evaluate_ssa_tta(
                default_tta_cfg, gen_ckpt, seed=args.seed
            )

            # D) Experiment 2: General Dataset-Wide TTA (Unsupervised KL Calibrated) — Online & Offline SSA
            unsup_tta_cfg = build_tta_config(
                source_fd=source_fd,
                target_fd=target_fd,
                ckpt_path=gen_ckpt,
                stat_path=gen_stat,
                model_type=model_type,
                train_cfg=gen_train_cfg,
                tta_hp=general_tta_unsup_hp,
                target_split=args.target_split,
                last_window_only=args.last_window_only,
            )
            gen_unsup_tta_metrics = evaluate_ssa_tta(
                unsup_tta_cfg, gen_ckpt, seed=args.seed
            )

            # E) Experiment 3: General Dataset-Wide TTA (Calibration-Pair Tuned) — Online & Offline SSA
            cal_tta_cfg = build_tta_config(
                source_fd=source_fd,
                target_fd=target_fd,
                ckpt_path=gen_ckpt,
                stat_path=gen_stat,
                model_type=model_type,
                train_cfg=gen_train_cfg,
                tta_hp=general_tta_cal_hp,
                target_split=args.target_split,
                last_window_only=args.last_window_only,
            )
            gen_cal_tta_metrics = evaluate_ssa_tta(
                cal_tta_cfg, gen_ckpt, seed=args.seed
            )

            row: dict[str, Any] = {
                "Task": task_name,
                "Model_Type": model_type,
                "Source_FD": source_fd,
                "Target_FD": target_fd,
                # 1. Bounds
                "Source_Only_RMSE": round(source_only["rmse_loss"], 4),
                "Source_Only_MAE": round(source_only["mae_loss"], 4),
                "Source_Only_R2": round(source_only["R2"], 4),
                "Target_Only_RMSE": round(target_only["rmse_loss"], 4),
                "Target_Only_MAE": round(target_only["mae_loss"], 4),
                "Target_Only_R2": round(target_only["R2"], 4),
                # 2. General Source Tuned (Unadapted)
                "General_Source_Unadapted_RMSE": round(
                    gen_source_unadapted["rmse_loss"], 4
                ),
                "General_Source_Unadapted_MAE": round(
                    gen_source_unadapted["mae_loss"], 4
                ),
                # 3. Default TTA Settings (Online & Offline SSA)
                "Default_TTA_Online_RMSE": round(
                    default_tta_metrics["online"]["rmse_loss"], 4
                ),
                "Default_TTA_Online_MAE": round(
                    default_tta_metrics["online"]["mae_loss"], 4
                ),
                "Default_TTA_Offline_RMSE": round(
                    default_tta_metrics["offline"]["rmse_loss"], 4
                ),
                "Default_TTA_Offline_MAE": round(
                    default_tta_metrics["offline"]["mae_loss"], 4
                ),
                "Default_TTA_Offline_R2": round(
                    default_tta_metrics["offline"]["R2"], 4
                ),
                # 4. General Dataset-Wide TTA (Unsupervised KL) — Online & Offline SSA
                "General_TTA_UnsupKL_Online_RMSE": round(
                    gen_unsup_tta_metrics["online"]["rmse_loss"], 4
                ),
                "General_TTA_UnsupKL_Online_MAE": round(
                    gen_unsup_tta_metrics["online"]["mae_loss"], 4
                ),
                "General_TTA_UnsupKL_Offline_RMSE": round(
                    gen_unsup_tta_metrics["offline"]["rmse_loss"], 4
                ),
                "General_TTA_UnsupKL_Offline_MAE": round(
                    gen_unsup_tta_metrics["offline"]["mae_loss"], 4
                ),
                "General_TTA_UnsupKL_Offline_R2": round(
                    gen_unsup_tta_metrics["offline"]["R2"], 4
                ),
                # 5. General Dataset-Wide TTA (Calibration-Pair) — Online & Offline SSA
                "General_TTA_Calibrated_Online_RMSE": round(
                    gen_cal_tta_metrics["online"]["rmse_loss"], 4
                ),
                "General_TTA_Calibrated_Online_MAE": round(
                    gen_cal_tta_metrics["online"]["mae_loss"], 4
                ),
                "General_TTA_Calibrated_Offline_RMSE": round(
                    gen_cal_tta_metrics["offline"]["rmse_loss"], 4
                ),
                "General_TTA_Calibrated_Offline_MAE": round(
                    gen_cal_tta_metrics["offline"]["mae_loss"], 4
                ),
                "General_TTA_Calibrated_Offline_R2": round(
                    gen_cal_tta_metrics["offline"]["R2"], 4
                ),
            }
            table_rows.append(row)

            rmse_csv, mae_csv, full_csv = save_realistic_benchmark_tables(
                table_rows, general_hyperparams, out_root
            )
            print(f"[Saved incremental realistic benchmark tables to {out_root}]")

    rmse_csv, mae_csv, full_csv = save_realistic_benchmark_tables(
        table_rows, general_hyperparams, out_root
    )
    df_rmse = pd.read_csv(rmse_csv)
    print("\n================ FINAL REALISTIC BENCHMARK RMSE TABLE ================")
    print(df_rmse.to_string(index=False))
    print(f"\nCSV tables saved to:\n  - {rmse_csv}\n  - {mae_csv}\n  - {full_csv}")

    if use_wandb:
        final_reporter = Reporter(
            name=f"Realistic.FinalSummary.{model_type}",
            use_wandb=True,
            wandb_project=args.wandb_project,
            wandb_config=vars(args),
            wandb_run_name=f"CMAPSS Realistic Benchmark Summary ({model_type})",
        )
        extra_files = [db_file] if (db_file is not None and db_file.is_file()) else []
        final_reporter.log_directory_artifacts(
            directory=out_root,
            artifact_name=f"cmapss-realistic-benchmark-{model_type}",
            artifact_type="benchmark_results",
            patterns=("*.csv", "*.json", "*.yaml"),
            extra_files=extra_files,
        )
        final_reporter.finish()


if __name__ == "__main__":
    main()
