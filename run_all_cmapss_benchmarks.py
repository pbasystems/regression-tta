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
from torch.utils.data import DataLoader

from adaptation import create_optimizer as create_tta_optimizer
from dataset import get_datasets
from evaluation.evaluator import RegressionEvaluator
from feature_stats import FeatureStatCalculator
from model import create_regressor
from trainer import RegressionTrainer
from tta import TTAEngine
from tune_optuna import run_tta_trial
from utils.reporter import Reporter
from utils.seed import fix_seed

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL_FDS = ["FD001", "FD002", "FD003", "FD004"]

# Default LSTM-DANN architecture per source FD (matching DA-Replications/configs/lstm_dann)
DEFAULT_ARCH_BY_SOURCE: dict[str, dict[str, Any]] = {
    "FD001": {
        "input_size": 24,
        "lstm_hidden_size": 128,
        "f_size": 64,
        "num_layers": 1,
        "lstm_dropout": 0.5,
        "regressor_dropout": 0.3,
        "regressor_first_hidden_size": 32,
        "regressor_second_hidden_size": 0,
    },
    "FD002": {
        "input_size": 24,
        "lstm_hidden_size": 64,
        "f_size": 64,
        "num_layers": 1,
        "lstm_dropout": 0.1,
        "regressor_dropout": 0.0,
        "regressor_first_hidden_size": 32,
        "regressor_second_hidden_size": 0,
    },
    "FD003": {
        "input_size": 24,
        "lstm_hidden_size": 64,
        "f_size": 64,
        "num_layers": 2,
        "lstm_dropout": 0.3,
        "regressor_dropout": 0.1,
        "regressor_first_hidden_size": 32,
        "regressor_second_hidden_size": 32,
    },
    "FD004": {
        "input_size": 24,
        "lstm_hidden_size": 100,
        "f_size": 30,
        "num_layers": 1,
        "lstm_dropout": 0.5,
        "regressor_dropout": 0.0,
        "regressor_first_hidden_size": 20,
        "regressor_second_hidden_size": 0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full CMAPSS cross-domain benchmark (all 12 FD00X->FD00Y tasks) "
            "across Source-Only, Target-Only, and Optuna-tuned objectives, and export CSV tables."
        )
    )
    parser.add_argument(
        "-o",
        default="result/benchmark_all",
        help="Root output directory for models, Optuna studies, best hyperparameters, and final CSV tables",
    )
    parser.add_argument(
        "--storage",
        default="sqlite:///result/optuna/optuna_studies.db",
        help="Optuna SQLite storage URL (default: sqlite:///result/optuna/optuna_studies.db)",
    )
    parser.add_argument(
        "--n-tta-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per task for each TTA objective (default: 50)",
    )
    parser.add_argument(
        "--n-source-trials",
        type=int,
        default=20,
        help="Number of Optuna trials per domain for source_val_rmse tuning (default: 20; set 0 to skip source tuning)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs for base and source-tuned models (default: 50)",
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
        help="Target split used for adaptation and evaluation (default: test)",
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
    arch_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "epoch": epochs,
        "dataset": {
            "name": "cmapss",
            "config": {
                "source_domain": True,
                "norm_mode": "minmax",
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
            "type": "lstm_dann",
            "config": copy.deepcopy(arch_cfg or DEFAULT_ARCH_BY_SOURCE[fd]),
        },
        "optimizer": {
            "param": "all",
            "name": "Adam",
            "config": {
                "lr": 1.0e-3,
                "weight_decay": 1.0e-4,
            },
        },
        "trainer": {"compile_model": None},
        "evaluator": {"compile_model": None},
        "train_dataloader": {"batch_size": 256, "shuffle": True},
        "val_dataloader": {"batch_size": 256},
    }


def build_base_tta_config(
    source_fd: str,
    target_fd: str,
    ckpt_path: Path,
    stat_path: Path,
    arch_cfg: dict[str, Any],
    target_split: str = "test",
    last_window_only: bool = False,
) -> dict[str, Any]:
    return {
        "dataset": {
            "name": "cmapss",
            "config": {
                "source_domain": False,
                "norm_mode": "minmax",
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
            "type": "lstm_dann",
            "config": copy.deepcopy(arch_cfg),
        },
        "optimizer": {
            "param": "fe",
            "name": "Adam",
            "config": {
                "lr": 1.0e-4,
                "weight_decay": 0.0,
            },
        },
        "tta": {
            "type": "pca",
            "config": {
                "train_mode": False,
                "weight_bias": 1.0,
                "weight_exp": 1.0,
                "pc_config": {
                    "stat_file": str(stat_path),
                    "contrib_top_k": 5,
                },
                "loss_config": {"eps": 1.0e-8},
                "compile_model": None,
            },
        },
        "evaluator": {"compile_model": None},
        "adapt_dataloader": {"batch_size": 256, "shuffle": True},
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
    epochs: int,
    seed: int,
    use_wandb: bool,
    wandb_project: str,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    """Trains (or loads if already cached) the base model and feature_stats.pt for a single FD domain."""
    domain_dir = out_root / "domain_models" / fd
    domain_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = domain_dir / f"source_model_{fd}.pt"
    stat_path = domain_dir / "feature_stats.pt"
    meta_path = domain_dir / "train_summary.json"
    train_cfg = build_base_train_config(fd, epochs=epochs)

    if ckpt_path.is_file() and stat_path.is_file() and meta_path.is_file():
        print(f"[{fd}] Loading existing base model from {ckpt_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return ckpt_path, stat_path, train_cfg, meta

    print(f"\n================ TRAINING BASE MODEL ON {fd} ({epochs} epochs) ================")
    print(
        f"\n================ TRAINING BASE MODEL ON {fd} ({epochs} epochs) ================"
    )
    fix_seed(seed)
    reporter = Reporter(
        name=f"Benchmark.Train.{fd}",
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config=train_cfg,
        wandb_run_name=f"{fd} | Base Domain Model Train",
    )

    net = create_regressor(train_cfg).to(DEVICE)
    opt = getattr(torch.optim, train_cfg["optimizer"]["name"])(
        net.parameters(), **train_cfg["optimizer"]["config"]
    )
    train_ds, val_ds = get_datasets(train_cfg)
    train_dl = DataLoader(train_ds, **train_cfg["train_dataloader"])
    val_dl = DataLoader(val_ds, **train_cfg["val_dataloader"])

    trainer = RegressionTrainer(net, opt, **train_cfg["trainer"])
    trainer.run(train_dl, max_epochs=epochs)

    val_ev = RegressionEvaluator(net, **train_cfg["evaluator"])
    val_ev.run(val_dl)

    torch.save(net.state_dict(), ckpt_path)
    cov_rank = compute_and_save_feature_stats(net, train_cfg, stat_path)

    meta = {
        "fd": fd,
        "epochs": epochs,
        "cov_rank": cov_rank,
        "train_metrics": {k: float(v) for k, v in trainer.state.metrics.items()},
        "val_metrics": {k: float(v) for k, v in val_ev.state.metrics.items()},
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    reporter.log_metrics(
        {
            "val/rmse_loss": meta["val_metrics"]["rmse_loss"],
            "val/mae_loss": meta["val_metrics"]["mae_loss"],
            "val/R2": meta["val_metrics"]["R2"],
            "cov_rank": cov_rank,
        }
    )
    reporter.log_directory_artifacts(
        directory=domain_dir,
        artifact_name=f"base-domain-model-{fd}",
        artifact_type="model",
    )
    reporter.finish()

    return ckpt_path, stat_path, train_cfg, meta


def tune_source_domain_optuna(
    fd: str,
    out_root: Path,
    storage: str,
    n_trials: int,
    epochs: int,
    seed: int,
    use_wandb: bool,
    wandb_project: str,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    """Tunes source model architecture & optimizer on FD validation split (source_val_rmse objective)."""
    tuned_dir = out_root / "source_tuned_models" / fd
    tuned_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = tuned_dir / f"best_source_model_{fd}.pt"
    stat_path = tuned_dir / "feature_stats.pt"
    cfg_path = tuned_dir / "best_train_config.yaml"
    summary_path = tuned_dir / "source_optuna_summary.json"

    if (
        ckpt_path.is_file()
        and stat_path.is_file()
        and cfg_path.is_file()
        and summary_path.is_file()
    ):
        print(f"[{fd}] Loading existing source-tuned model from {ckpt_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            best_train_cfg = yaml.safe_load(f)
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        return ckpt_path, stat_path, best_train_cfg, summary

    print(
        f"\n================ OPTUNA SOURCE TUNING ON {fd} ({n_trials} trials) ================"
    )
    study_name = f"benchmark_source_val_rmse_{fd}"
    base_train_cfg = build_base_train_config(fd, epochs=epochs)

    reporter = Reporter(
        name=f"Benchmark.SourceOptuna.{fd}",
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config={"fd": fd, "n_trials": n_trials, "epochs": epochs},
        wandb_run_name=f"{fd} | Optuna [source_val_rmse] ({n_trials} trials)",
    )

    best_holder: dict[str, Any] = {
        "score": float("inf"),
        "config": None,
        "state_dict": None,
        "val_metrics": None,
    }

    def objective(trial: optuna.Trial) -> float:
        fix_seed(seed)
        cfg = copy.deepcopy(base_train_cfg)

        lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])
        f_size = trial.suggest_categorical("f_size", [32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 1, 2)
        reg_first = trial.suggest_categorical(
            "regressor_first_hidden_size", [16, 32, 64]
        )
        reg_second = trial.suggest_categorical(
            "regressor_second_hidden_size", [0, 16, 32]
        )
        lstm_dropout = trial.suggest_float("lstm_dropout", 0.1, 0.5)
        reg_dropout = trial.suggest_float("regressor_dropout", 0.0, 0.4)
        source_lr = trial.suggest_float("source_lr", 3e-4, 5e-3, log=True)
        source_wd = trial.suggest_float("source_weight_decay", 1e-6, 1e-3, log=True)
        source_batch_size = trial.suggest_categorical(
            "source_batch_size", [64, 128, 256]
        )

        cfg["regressor"]["config"].update(
            {
                "lstm_hidden_size": lstm_hidden_size,
                "f_size": f_size,
                "num_layers": num_layers,
                "regressor_first_hidden_size": reg_first,
                "regressor_second_hidden_size": reg_second,
                "lstm_dropout": lstm_dropout,
                "regressor_dropout": reg_dropout,
            }
        )
        cfg["optimizer"]["config"]["lr"] = source_lr
        cfg["optimizer"]["config"]["weight_decay"] = source_wd
        cfg["train_dataloader"]["batch_size"] = source_batch_size

        net = create_regressor(cfg).to(DEVICE)
        opt = getattr(torch.optim, cfg["optimizer"]["name"])(
            net.parameters(), **cfg["optimizer"]["config"]
        )
        train_ds, val_ds = get_datasets(cfg)
        train_dl = DataLoader(train_ds, **cfg["train_dataloader"])
        val_dl = DataLoader(val_ds, **cfg["val_dataloader"])

        trainer = RegressionTrainer(net, opt, **cfg["trainer"])
        trainer.run(train_dl, max_epochs=cfg["epoch"])

        val_ev = RegressionEvaluator(net, **cfg["evaluator"])
        val_ev.run(val_dl)
        val_metrics = {k: float(v) for k, v in val_ev.state.metrics.items()}
        val_rmse = val_metrics["rmse_loss"]

        trial.set_user_attr("val_rmse", val_rmse)
        trial.set_user_attr("val_mae", val_metrics["mae_loss"])
        trial.set_user_attr("val_R2", val_metrics["R2"])

        reporter.log_metrics(
            {
                "trial": trial.number,
                "trial/source_val_rmse": val_rmse,
                "trial/source_val_mae": val_metrics["mae_loss"],
                "trial/source_val_R2": val_metrics["R2"],
            },
            step=trial.number,
        )

        if val_rmse < best_holder["score"]:
            best_holder["score"] = val_rmse
            best_holder["config"] = cfg
            best_holder["state_dict"] = copy.deepcopy(net.state_dict())
            best_holder["val_metrics"] = val_metrics

        return val_rmse

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )
    remaining = max(
        0,
        n_trials
        - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
    )
    if remaining > 0 or best_holder["config"] is None:
        study.optimize(objective, n_trials=max(1, remaining))

    best_cfg = best_holder["config"]
    torch.save(best_holder["state_dict"], ckpt_path)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.dump(best_cfg, f, sort_keys=False)

    best_net = create_regressor(best_cfg).to(DEVICE)
    best_net.load_state_dict(best_holder["state_dict"])
    cov_rank = compute_and_save_feature_stats(best_net, best_cfg, stat_path)

    study.trials_dataframe().to_csv(tuned_dir / "optuna_trials.csv", index=False)
    summary = {
        "fd": fd,
        "study_name": study_name,
        "best_trial": study.best_trial.number,
        "best_source_val_rmse": float(study.best_value),
        "best_params": study.best_params,
        "val_metrics": best_holder["val_metrics"],
        "cov_rank": cov_rank,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    reporter.log_directory_artifacts(
        directory=tuned_dir,
        artifact_name=f"source-tuned-{fd}",
        artifact_type="model",
    )
    reporter.finish()
    return ckpt_path, stat_path, best_cfg, summary


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


def evaluate_model_with_default_tta(
    tta_cfg: dict[str, Any],
    ckpt_path: Path,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Runs default TTA configuration on the target dataset and returns online & offline metrics."""
    fix_seed(seed)
    _, val_ds = get_datasets(tta_cfg)
    val_dl = DataLoader(val_ds, **tta_cfg["adapt_dataloader"])

    net = create_regressor(tta_cfg).to(DEVICE)
    net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    opt = create_tta_optimizer(net, tta_cfg)
    engine = TTAEngine(net, opt, **tta_cfg["tta"]["config"])
    ev = RegressionEvaluator(net, **tta_cfg["evaluator"])

    engine.run(val_dl)
    ev.run(val_dl)
    return {
        "online": {k: float(v) for k, v in engine.state.metrics.items()},
        "offline": {k: float(v) for k, v in ev.state.metrics.items()},
    }


def tune_tta_for_task_objective(
    source_fd: str,
    target_fd: str,
    objective_name: str,
    base_tta_cfg: dict[str, Any],
    ckpt_path: Path,
    out_root: Path,
    storage: str,
    n_trials: int,
    seed: int,
    use_wandb: bool,
    wandb_project: str,
) -> dict[str, Any]:
    """Runs Optuna TTA hyperparameter tuning for a single (source_fd -> target_fd, objective_name) pair."""
    task_tag = f"{source_fd}_to_{target_fd}"
    study_dir = out_root / "tasks" / task_tag / objective_name
    study_dir.mkdir(parents=True, exist_ok=True)
    summary_path = study_dir / "study_summary.json"
    best_cfg_path = study_dir / "best_tta_config.yaml"
    best_model_path = study_dir / "best_adapted_regressor.pt"

    if summary_path.is_file() and best_cfg_path.is_file() and best_model_path.is_file():
        print(
            f"[{task_tag} | {objective_name}] Loading cached study summary from {summary_path}"
        )
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    print(
        f"\n--- Tuning Task {source_fd} -> {target_fd} | Objective: {objective_name} ({n_trials} trials) ---"
    )
    study_name = f"benchmark_{source_fd}_to_{target_fd}_{objective_name}"
    stat_file = base_tta_cfg["tta"]["config"]["pc_config"]["stat_file"]
    stat_dict = torch.load(stat_file, map_location="cpu")
    max_pc_dim = int(stat_dict["eigvals"].shape[0])

    _, val_ds = get_datasets(base_tta_cfg)

    reporter = Reporter(
        name=f"Benchmark.TTA.{task_tag}.{objective_name}",
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config={
            "task": f"{source_fd}->{target_fd}",
            "objective": objective_name,
            "n_trials": n_trials,
            "base_tta_config": base_tta_cfg,
        },
        wandb_run_name=f"{source_fd}-to-{target_fd} | Optuna [{objective_name}] ({n_trials} trials)",
    )

    best_state: dict[str, Any] = {
        "score": float("inf"),
        "config": None,
        "metrics": None,
        "state_dict": None,
    }

    def objective(trial: optuna.Trial) -> float:
        score, cfg, metrics, reg = run_tta_trial(
            trial=trial,
            base_config=base_tta_cfg,
            val_ds=val_ds,
            ckpt_path=ckpt_path,
            max_pc_dim=max_pc_dim,
            seed=seed,
            objective_type=objective_name,
            reporter=reporter,
        )
        if score < best_state["score"]:
            best_state["score"] = score
            best_state["config"] = cfg
            best_state["metrics"] = metrics
            best_state["state_dict"] = copy.deepcopy(reg.state_dict())
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )
    completed_count = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    remaining = max(0, n_trials - completed_count)
    if remaining > 0 or best_state["metrics"] is None:
        study.optimize(objective, n_trials=max(1, remaining))

    if best_state["config"] is not None:
        with best_cfg_path.open("w", encoding="utf-8") as f:
            yaml.dump(best_state["config"], f, sort_keys=False)
    if best_state["state_dict"] is not None:
        torch.save(best_state["state_dict"], best_model_path)

    study.trials_dataframe().to_csv(study_dir / "optuna_trials.csv", index=False)

    summary = {
        "task": f"{source_fd}->{target_fd}",
        "source_fd": source_fd,
        "target_fd": target_fd,
        "objective": objective_name,
        "study_name": study_name,
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_objective_value": float(study.best_value),
        "best_params": study.best_params,
        "best_metrics": best_state["metrics"],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    reporter.log_directory_artifacts(
        directory=study_dir,
        artifact_name=f"optuna-{objective_name}-{source_fd}-to-{target_fd}",
        artifact_type="optuna_study",
    )
    reporter.finish()
    return summary


def save_benchmark_tables(
    rows: list[dict[str, Any]],
    all_hyperparams: dict[str, Any],
    out_root: Path,
) -> tuple[Path, Path, Path]:
    df_all = pd.DataFrame(rows)

    rmse_cols = [
        "Task",
        "Source_Only_RMSE",
        "Target_Only_RMSE",
        "Source_Val_Tuned_Unadapted_RMSE",
        "Source_Val_Tuned_Default_TTA_Offline_RMSE",
        "TTA_Unsupervised_KL_Online_RMSE",
        "TTA_Unsupervised_KL_Offline_RMSE",
        "TTA_Online_RMSE_Objective_Online_RMSE",
        "TTA_Online_RMSE_Objective_Offline_RMSE",
        "TTA_Offline_RMSE_Objective_Offline_RMSE",
    ]
    mae_cols = [
        "Task",
        "Source_Only_MAE",
        "Target_Only_MAE",
        "Source_Val_Tuned_Unadapted_MAE",
        "Source_Val_Tuned_Default_TTA_Offline_MAE",
        "TTA_Unsupervised_KL_Online_MAE",
        "TTA_Unsupervised_KL_Offline_MAE",
        "TTA_Online_RMSE_Objective_Online_MAE",
        "TTA_Online_RMSE_Objective_Offline_MAE",
        "TTA_Offline_RMSE_Objective_Offline_MAE",
    ]

    existing_rmse_cols = [c for c in rmse_cols if c in df_all.columns]
    existing_mae_cols = [c for c in mae_cols if c in df_all.columns]

    rmse_csv = out_root / "summary_rmse_table.csv"
    mae_csv = out_root / "summary_mae_table.csv"
    full_csv = out_root / "summary_all_metrics.csv"
    hp_json = out_root / "best_hyperparameters_all_tasks.json"

    df_all[existing_rmse_cols].to_csv(rmse_csv, index=False)
    df_all[existing_mae_cols].to_csv(mae_csv, index=False)
    df_all.to_csv(full_csv, index=False)

    with hp_json.open("w", encoding="utf-8") as f:
        json.dump(all_hyperparams, f, indent=4, ensure_ascii=False)

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

    # -------------------------------------------------------------------------
    # STEP 1: Train (or Load) Base Models & Feature Stats for Each FD (FD001..FD004)
    #         Also Tune Source-Validation Models (source_val_rmse objective)
    # -------------------------------------------------------------------------
    base_models: dict[str, tuple[Path, Path, dict[str, Any], dict[str, Any]]] = {}
    source_tuned_models: dict[
        str, tuple[Path, Path, dict[str, Any], dict[str, Any]]
    ] = {}
    all_hyperparams: dict[str, Any] = {"source_val_rmse": {}, "tasks": {}}

    for fd in fds:
        base_models[fd] = train_or_load_domain_model(
            fd=fd,
            out_root=out_root,
            epochs=args.epochs,
            seed=args.seed,
            use_wandb=use_wandb,
            wandb_project=args.wandb_project,
        )
        if args.n_source_trials > 0:
            source_tuned_models[fd] = tune_source_domain_optuna(
                fd=fd,
                out_root=out_root,
                storage=args.storage,
                n_trials=args.n_source_trials,
                epochs=args.epochs,
                seed=args.seed,
                use_wandb=use_wandb,
                wandb_project=args.wandb_project,
            )
            all_hyperparams["source_val_rmse"][fd] = source_tuned_models[fd][3][
                "best_params"
            ]

    # -------------------------------------------------------------------------
    # STEP 2: Compute Target-Only Baselines (Trained on FD_Y -> Tested on FD_Y)
    # -------------------------------------------------------------------------
    target_only_metrics: dict[str, dict[str, float]] = {}
    for target_fd in fds:
        t_ckpt, t_stat, t_train_cfg, _ = base_models[target_fd]
        t_eval_cfg = build_base_tta_config(
            source_fd=target_fd,
            target_fd=target_fd,
            ckpt_path=t_ckpt,
            stat_path=t_stat,
            arch_cfg=t_train_cfg["regressor"]["config"],
            target_split=args.target_split,
            last_window_only=args.last_window_only,
        )
        target_only_metrics[target_fd] = evaluate_model_without_tta(
            t_eval_cfg, t_ckpt, seed=args.seed
        )
        print(
            f"[Target-Only Baseline {target_fd}->{target_fd}] "
            f"RMSE={target_only_metrics[target_fd]['rmse_loss']:.2f}, "
            f"MAE={target_only_metrics[target_fd]['mae_loss']:.2f}"
        )

    # -------------------------------------------------------------------------
    # STEP 3: Loop Over All Cross-Domain Tasks (FD00X -> FD00Y, X != Y)
    # -------------------------------------------------------------------------
    table_rows: list[dict[str, Any]] = []
    tta_objectives = ["unsupervised_kl", "online_rmse", "offline_rmse"]

    for source_fd in fds:
        for target_fd in fds:
            if source_fd == target_fd:
                continue

            task_name = f"{source_fd}->{target_fd}"
            print(
                f"\n=================================================================="
            )
            print(f"                  BENCHMARK TASK: {task_name}")
            print(f"==================================================================")

            src_ckpt, src_stat, src_train_cfg, _ = base_models[source_fd]
            base_tta_cfg = build_base_tta_config(
                source_fd=source_fd,
                target_fd=target_fd,
                ckpt_path=src_ckpt,
                stat_path=src_stat,
                arch_cfg=src_train_cfg["regressor"]["config"],
                target_split=args.target_split,
                last_window_only=args.last_window_only,
            )

            # 1) Source-Only (Trained on Source_FD, tested directly on Target_FD without adaptation)
            source_only = evaluate_model_without_tta(
                base_tta_cfg, src_ckpt, seed=args.seed
            )

            # 2) Target-Only (Trained on Target_FD, tested on Target_FD)
            target_only = target_only_metrics[target_fd]

            row: dict[str, Any] = {
                "Task": task_name,
                "Source_FD": source_fd,
                "Target_FD": target_fd,
                "Source_Only_RMSE": round(source_only["rmse_loss"], 4),
                "Source_Only_MAE": round(source_only["mae_loss"], 4),
                "Source_Only_R2": round(source_only["R2"], 4),
                "Target_Only_RMSE": round(target_only["rmse_loss"], 4),
                "Target_Only_MAE": round(target_only["mae_loss"], 4),
                "Target_Only_R2": round(target_only["R2"], 4),
            }

            # 3) Source-Val-Tuned Objective (Tuned purely on Source_FD validation split)
            if source_fd in source_tuned_models:
                st_ckpt, st_stat, st_train_cfg, _ = source_tuned_models[source_fd]
                st_tta_cfg = build_base_tta_config(
                    source_fd=source_fd,
                    target_fd=target_fd,
                    ckpt_path=st_ckpt,
                    stat_path=st_stat,
                    arch_cfg=st_train_cfg["regressor"]["config"],
                    target_split=args.target_split,
                    last_window_only=args.last_window_only,
                )
                st_unadapted = evaluate_model_without_tta(
                    st_tta_cfg, st_ckpt, seed=args.seed
                )
                st_default_tta = evaluate_model_with_default_tta(
                    st_tta_cfg, st_ckpt, seed=args.seed
                )

                row["Source_Val_Tuned_Unadapted_RMSE"] = round(
                    st_unadapted["rmse_loss"], 4
                )
                row["Source_Val_Tuned_Unadapted_MAE"] = round(
                    st_unadapted["mae_loss"], 4
                )
                row["Source_Val_Tuned_Default_TTA_Online_RMSE"] = round(
                    st_default_tta["online"]["rmse_loss"], 4
                )
                row["Source_Val_Tuned_Default_TTA_Offline_RMSE"] = round(
                    st_default_tta["offline"]["rmse_loss"], 4
                )
                row["Source_Val_Tuned_Default_TTA_Offline_MAE"] = round(
                    st_default_tta["offline"]["mae_loss"], 4
                )

            # 4) TTA Optuna Objectives (unsupervised_kl, online_rmse, offline_rmse)
            all_hyperparams["tasks"][task_name] = {}
            for obj_name in tta_objectives:
                summary = tune_tta_for_task_objective(
                    source_fd=source_fd,
                    target_fd=target_fd,
                    objective_name=obj_name,
                    base_tta_cfg=base_tta_cfg,
                    ckpt_path=src_ckpt,
                    out_root=out_root,
                    storage=args.storage,
                    n_trials=args.n_tta_trials,
                    seed=args.seed,
                    use_wandb=use_wandb,
                    wandb_project=args.wandb_project,
                )
                all_hyperparams["tasks"][task_name][obj_name] = {
                    "best_objective_value": summary["best_objective_value"],
                    "best_params": summary["best_params"],
                }
                bm = summary["best_metrics"]

                match obj_name:
                    case "unsupervised_kl":
                        prefix = "TTA_Unsupervised_KL"
                    case "online_rmse":
                        prefix = "TTA_Online_RMSE_Objective"
                    case "offline_rmse":
                        prefix = "TTA_Offline_RMSE_Objective"
                    case _:
                        prefix = obj_name

                row[f"{prefix}_Online_RMSE"] = round(
                    float(bm["online"]["rmse_loss"]), 4
                )
                row[f"{prefix}_Online_MAE"] = round(float(bm["online"]["mae_loss"]), 4)
                row[f"{prefix}_Offline_RMSE"] = round(
                    float(bm["offline"]["rmse_loss"]), 4
                )
                row[f"{prefix}_Offline_MAE"] = round(
                    float(bm["offline"]["mae_loss"]), 4
                )
                row[f"{prefix}_Offline_r"] = round(float(bm["offline"]["r"]), 4)

            table_rows.append(row)

            # Incrementally save CSV tables after every task completes
            rmse_csv, mae_csv, full_csv = save_benchmark_tables(
                table_rows, all_hyperparams, out_root
            )
            print(f"[Saved incremental benchmark tables to {out_root}]")

    # -------------------------------------------------------------------------
    # STEP 4: Final Summary & WandB Table Upload
    # -------------------------------------------------------------------------
    rmse_csv, mae_csv, full_csv = save_benchmark_tables(
        table_rows, all_hyperparams, out_root
    )
    df_rmse = pd.read_csv(rmse_csv)
    print("\n================ FINAL BENCHMARK RMSE TABLE ================")
    print(df_rmse.to_string(index=False))
    print(f"\nCSV tables saved to:\n  - {rmse_csv}\n  - {mae_csv}\n  - {full_csv}")

    if use_wandb:
        final_reporter = Reporter(
            name="Benchmark.FinalSummary",
            use_wandb=True,
            wandb_project=args.wandb_project,
            wandb_config=vars(args),
            wandb_run_name="CMAPSS All-12-Tasks Benchmark Summary",
        )
        extra_files = [db_file] if (db_file is not None and db_file.is_file()) else []
        final_reporter.log_directory_artifacts(
            directory=out_root,
            artifact_name="cmapss-all-tasks-benchmark-summary",
            artifact_type="benchmark_results",
            patterns=("*.csv", "*.json", "*.yaml"),
            extra_files=extra_files,
        )
        final_reporter.finish()


if __name__ == "__main__":
    main()
