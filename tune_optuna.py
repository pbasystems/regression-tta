from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from pprint import pprint
from typing import Any

import optuna
import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from adaptation import create_optimizer as create_tta_optimizer
from dataset import get_datasets
from evaluation.evaluator import RegressionEvaluator
from feature_stats import resolve_checkpoint_path
from model import create_regressor
from trainer import RegressionTrainer
from tta import TTAEngine
from utils.loss import diagonal_gaussian_kl_loss
from utils.pca_basis import get_pca_basis
from utils.reporter import Reporter, get_domain_tag
from utils.seed import fix_seed

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Objective / Multi-Scenario Optuna Search for Regression-TTA"
    )
    parser.add_argument(
        "-c",
        default="configs/tta/cmapss.yaml",
        help="Base TTA YAML config file",
    )
    parser.add_argument(
        "--train-config",
        default="configs/train_source/cmapss.yaml",
        help="Base source-training YAML config file",
    )
    parser.add_argument(
        "-o",
        default=None,
        help="Output directory (defaults to result/optuna/<domain>_<objective>)",
    )
    parser.add_argument(
        "--storage",
        default="sqlite:///result/optuna/optuna_studies.db",
        help="Optuna SQLite storage URL (default: sqlite:///result/optuna/optuna_studies.db)",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Optional custom Optuna study name",
    )
    parser.add_argument(
        "--objective",
        choices=[
            "source_val_rmse",
            "unsupervised_kl",
            "online_rmse",
            "offline_rmse",
            "full_offline_rmse",
        ],
        default="source_val_rmse",
        help=(
            "1) 'source_val_rmse': Tune source training using ONLY source validation RMSE (zero target leakage). "
            "2) 'unsupervised_kl': Tune TTA using ONLY label-free subspace KL divergence on target (no target labels). "
            "3) 'online_rmse': Tune TTA for streaming online RMSE. "
            "4) 'offline_rmse': Tune TTA for post-adaptation offline RMSE (oracle upper bound). "
            "5) 'full_offline_rmse': Jointly tune source + TTA."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=15,
        help="Number of Optuna trials to run",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging during hyperparameter search",
    )
    return parser.parse_args()


@torch.no_grad()
def compute_unsupervised_subspace_kl(
    net: torch.nn.Module,
    dataloader: DataLoader,
    stat_file: str,
    eval_top_k: int = 10,
    eval_eps: float = 1e-6,
) -> float:
    """Computes label-free, un-gameable weighted symmetric KL divergence in the fixed top-k source PCA subspace."""
    net.eval()
    mean, basis, var = get_pca_basis(stat_file=stat_file, contrib_top_k=eval_top_k)
    mean = mean.to(DEVICE)
    basis = basis.to(DEVICE)
    var = var.clamp(min=eval_eps).to(DEVICE)

    dim_weight = torch.abs(net.regressor.weight @ basis).flatten() + 1.0
    dim_weight = dim_weight / dim_weight.sum()

    feats = []
    for x, _ in dataloader:
        x = x.to(DEVICE)
        feats.append(net.feature(x))
    features = torch.cat(feats, dim=0)
    f_pc = (features - mean) @ basis
    f_pc_mean = f_pc.mean(dim=0)
    f_pc_var = f_pc.var(dim=0).clamp(min=eval_eps)
    zeros = torch.zeros_like(f_pc_mean)

    kl_fwd = diagonal_gaussian_kl_loss(
        f_pc_mean, f_pc_var, zeros, var, eps=0.0, dim_reduction="none"
    )
    kl_rev = diagonal_gaussian_kl_loss(
        zeros, var, f_pc_mean, f_pc_var, eps=0.0, dim_reduction="none"
    )
    sym_kl = (kl_fwd + kl_rev) @ dim_weight
    return float(sym_kl.item())


def run_source_val_trial(
    trial: optuna.Trial,
    base_train_cfg: dict[str, Any],
    base_tta_cfg: dict[str, Any],
    out_dir: Path,
    seed: int,
    reporter: Reporter | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any], torch.nn.Module]:
    """Tunes source model using ONLY Source Validation RMSE (strictly zero target domain leakage)."""
    fix_seed(seed)
    train_cfg = copy.deepcopy(base_train_cfg)
    tta_cfg = copy.deepcopy(base_tta_cfg)

    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])
    f_size = trial.suggest_categorical("f_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 2)
    reg_first = trial.suggest_categorical("regressor_first_hidden_size", [16, 32, 64])
    reg_second = trial.suggest_categorical("regressor_second_hidden_size", [0, 16, 32])
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.1, 0.5)
    reg_dropout = trial.suggest_float("regressor_dropout", 0.0, 0.4)
    source_lr = trial.suggest_float("source_lr", 3e-4, 5e-3, log=True)
    source_wd = trial.suggest_float("source_weight_decay", 1e-6, 1e-3, log=True)
    source_batch_size = trial.suggest_categorical("source_batch_size", [64, 128, 256])

    for cfg in (train_cfg, tta_cfg):
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

    train_cfg["optimizer"]["config"]["lr"] = source_lr
    train_cfg["optimizer"]["config"]["weight_decay"] = source_wd
    train_cfg["train_dataloader"]["batch_size"] = source_batch_size

    net = create_regressor(train_cfg).to(DEVICE)
    source_opt = getattr(torch.optim, train_cfg["optimizer"]["name"])(
        net.parameters(), **train_cfg["optimizer"]["config"]
    )
    train_ds, val_ds = get_datasets(train_cfg)
    train_dl = DataLoader(train_ds, **train_cfg["train_dataloader"])
    val_dl = DataLoader(val_ds, **train_cfg["val_dataloader"])

    trainer = RegressionTrainer(net, source_opt, **train_cfg["trainer"])
    trainer.run(train_dl, max_epochs=train_cfg["epoch"])

    # Evaluate on Source Validation split (the ONLY optimization objective here)
    val_ev = RegressionEvaluator(net, **train_cfg["evaluator"])
    val_ev.run(val_dl)
    source_val_metrics = dict(val_ev.state.metrics)
    source_val_rmse = float(source_val_metrics["rmse_loss"])

    # Also evaluate how this source model transfers to Target (unadapted) for logging/visibility
    _, target_ds = get_datasets(tta_cfg)
    target_dl = DataLoader(target_ds, **tta_cfg["val_dataloader"])
    target_ev = RegressionEvaluator(net, **tta_cfg["evaluator"])
    target_ev.run(target_dl)
    target_unadapted_metrics = dict(target_ev.state.metrics)

    metrics = {
        "source_train": dict(trainer.state.metrics),
        "source_val": source_val_metrics,
        "target_unadapted": target_unadapted_metrics,
    }

    trial.set_user_attr("source_val_rmse", source_val_rmse)
    trial.set_user_attr("source_val_mae", float(source_val_metrics["mae_loss"]))
    trial.set_user_attr("source_val_R2", float(source_val_metrics["R2"]))
    trial.set_user_attr(
        "target_unadapted_rmse", float(target_unadapted_metrics["rmse_loss"])
    )
    trial.set_user_attr(
        "target_unadapted_mae", float(target_unadapted_metrics["mae_loss"])
    )

    if reporter is not None:
        reporter.log_metrics(
            {
                "trial": trial.number,
                "trial/source_val_rmse": source_val_rmse,
                "trial/source_val_mae": float(source_val_metrics["mae_loss"]),
                "trial/source_val_R2": float(source_val_metrics["R2"]),
                "trial/target_unadapted_rmse": float(
                    target_unadapted_metrics["rmse_loss"]
                ),
                "trial/target_unadapted_mae": float(
                    target_unadapted_metrics["mae_loss"]
                ),
            },
            step=trial.number,
        )

    return source_val_rmse, train_cfg, metrics, net


def run_tta_trial(
    trial: optuna.Trial,
    base_config: dict[str, Any],
    val_ds: torch.utils.data.Dataset,
    ckpt_path: Path,
    max_pc_dim: int,
    seed: int,
    objective_type: str,
    reporter: Reporter | None = None,
) -> tuple[float, dict[str, Any], dict[str, Any], torch.nn.Module]:
    fix_seed(seed)
    cfg = copy.deepcopy(base_config)

    contrib_top_k = trial.suggest_int("contrib_top_k", 2, min(32, max_pc_dim))
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical(
        "optimizer_name", ["Adam", "AdamW", "SGD"]
    )
    adapt_param = trial.suggest_categorical("adapt_param", ["fe", "all"])
    weight_bias = trial.suggest_float("weight_bias", 0.0, 5.0)
    weight_exp = trial.suggest_float("weight_exp", 0.0, 3.0)
    loss_eps = trial.suggest_float("loss_eps", 1e-8, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    train_mode = trial.suggest_categorical("train_mode", [False, True])

    cfg["optimizer"]["param"] = adapt_param
    cfg["optimizer"]["name"] = optimizer_name
    cfg["optimizer"]["config"] = {
        "lr": lr,
        "weight_decay": weight_decay,
    }
    cfg["tta"]["config"]["train_mode"] = train_mode
    cfg["tta"]["config"]["weight_bias"] = weight_bias
    cfg["tta"]["config"]["weight_exp"] = weight_exp
    cfg["tta"]["config"]["pc_config"]["contrib_top_k"] = contrib_top_k
    cfg["tta"]["config"]["loss_config"]["eps"] = loss_eps
    cfg["adapt_dataloader"]["batch_size"] = batch_size

    val_dl = DataLoader(val_ds, **cfg["adapt_dataloader"])

    regressor = create_regressor(cfg).to(DEVICE)
    regressor.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    opt = create_tta_optimizer(regressor, cfg)
    engine = TTAEngine(regressor, opt, **cfg["tta"]["config"])
    reg_evaluator = RegressionEvaluator(regressor, **cfg["evaluator"])

    engine.run(val_dl)
    reg_evaluator.run(val_dl)

    stat_file = cfg["tta"]["config"]["pc_config"]["stat_file"]
    unsup_kl = compute_unsupervised_subspace_kl(
        regressor,
        DataLoader(val_ds, **cfg["val_dataloader"]),
        stat_file=stat_file,
    )
    model_dist = float(reg_evaluator.state.metrics["model_dist"])
    unsup_score = unsup_kl + 0.1 * model_dist

    metrics = {
        "iteration": engine.state.iteration,
        "unsupervised_kl": unsup_kl,
        "unsupervised_score": unsup_score,
        "online": dict(engine.state.metrics),
        "offline": dict(reg_evaluator.state.metrics),
    }

    match objective_type:
        case "unsupervised_kl":
            score = unsup_score
        case "online_rmse":
            score = float(metrics["online"]["rmse_loss"])
        case "offline_rmse" | "full_offline_rmse":
            score = float(metrics["offline"]["rmse_loss"])
        case _:
            raise ValueError(f"Invalid objective_type: {objective_type}")

    if not torch.isfinite(torch.tensor(score)):
        raise optuna.TrialPruned("Non-finite objective value")

    trial.set_user_attr("unsupervised_kl", unsup_kl)
    trial.set_user_attr("online_rmse", float(metrics["online"]["rmse_loss"]))
    trial.set_user_attr("online_mae", float(metrics["online"]["mae_loss"]))
    trial.set_user_attr("offline_rmse", float(metrics["offline"]["rmse_loss"]))
    trial.set_user_attr("offline_mae", float(metrics["offline"]["mae_loss"]))
    trial.set_user_attr("offline_R2", float(metrics["offline"]["R2"]))
    trial.set_user_attr("offline_r", float(metrics["offline"]["r"]))

    if reporter is not None:
        reporter.log_metrics(
            {
                "trial": trial.number,
                "trial/objective": score,
                "trial/unsupervised_kl": unsup_kl,
                "trial/contrib_top_k": contrib_top_k,
                "trial/lr": lr,
                "trial/batch_size": batch_size,
                "trial/online_rmse": float(metrics["online"]["rmse_loss"]),
                "trial/online_mae": float(metrics["online"]["mae_loss"]),
                "trial/offline_rmse": float(metrics["offline"]["rmse_loss"]),
                "trial/offline_mae": float(metrics["offline"]["mae_loss"]),
            },
            step=trial.number,
        )

    return score, cfg, metrics, regressor


def main() -> None:
    args = parse_args()

    with open(args.c, "r", encoding="utf-8") as f:
        base_tta_config = yaml.safe_load(f)

    with open(args.train_config, "r", encoding="utf-8") as f:
        base_train_config = yaml.safe_load(f)

    domain_tag = get_domain_tag(base_tta_config)
    out_dir = Path(args.o or f"result/optuna/{domain_tag}_{args.objective}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.storage.startswith("sqlite:///"):
        db_file = Path(args.storage.replace("sqlite:///", "", 1))
        db_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        db_file = None

    study_name = args.study_name or f"regression_tta_{domain_tag}_{args.objective}"
    use_wandb = (
        False
        if args.no_wandb
        else base_tta_config.get("wandb", {}).get("use_wandb", True)
    )
    project = base_tta_config.get("wandb", {}).get("project", "Regression-TTA")

    reporter = Reporter(
        name=f"RegressionTTA.Optuna.{domain_tag}.{args.objective}",
        use_wandb=use_wandb,
        wandb_project=project,
        wandb_config={
            "objective": args.objective,
            "n_trials": args.n_trials,
            "storage": args.storage,
            "study_name": study_name,
            "base_tta_config": base_tta_config,
        },
        wandb_run_name=f"{domain_tag} | Optuna [{args.objective}] ({args.n_trials} trials)",
    )

    best_state: dict[str, Any] = {
        "score": float("inf"),
        "config": None,
        "metrics": None,
        "state_dict": None,
    }

    if args.objective == "source_val_rmse":

        def objective(trial: optuna.Trial) -> float:
            score, cfg, metrics, net = run_source_val_trial(
                trial=trial,
                base_train_cfg=base_train_config,
                base_tta_cfg=base_tta_config,
                out_dir=out_dir,
                seed=args.seed,
                reporter=reporter,
            )
            if score < best_state["score"]:
                best_state["score"] = score
                best_state["config"] = cfg
                best_state["metrics"] = metrics
                best_state["state_dict"] = copy.deepcopy(net.state_dict())
            return score

    else:
        ckpt_path = resolve_checkpoint_path(base_tta_config["regressor"]["source"])
        stat_file = base_tta_config["tta"]["config"]["pc_config"]["stat_file"]
        stat_dict = torch.load(stat_file, map_location="cpu")
        max_pc_dim = int(stat_dict["eigvals"].shape[0])

        _, val_ds = get_datasets(base_tta_config)

        fix_seed(args.seed)
        base_net = create_regressor(base_tta_config).to(DEVICE)
        base_net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        base_ev = RegressionEvaluator(base_net, **base_tta_config["evaluator"])
        base_ev.run(DataLoader(val_ds, **base_tta_config["val_dataloader"]))
        unadapted_metrics = dict(base_ev.state.metrics)
        reporter.info("Unadapted Baseline on %s: %s", domain_tag, unadapted_metrics)

        def objective(trial: optuna.Trial) -> float:
            score, cfg, metrics, reg = run_tta_trial(
                trial=trial,
                base_config=base_tta_config,
                val_ds=val_ds,
                ckpt_path=ckpt_path,
                max_pc_dim=max_pc_dim,
                seed=args.seed,
                objective_type=args.objective,
                reporter=reporter,
            )
            if score < best_state["score"]:
                best_state["score"] = score
                best_state["config"] = cfg
                best_state["metrics"] = {"unadapted": unadapted_metrics, **metrics}
                best_state["state_dict"] = copy.deepcopy(reg.state_dict())
            return score

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )
    reporter.info(
        "Optuna study '%s' stored at %s. View live with: optuna-dashboard %s",
        study_name,
        args.storage,
        args.storage,
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("\n================ OPTUNA STUDY COMPLETE ================")
    print(f"Study: {study_name}")
    print(
        f"Best Trial #{study.best_trial.number} ({args.objective}): {study.best_value:.4f}"
    )
    pprint(study.best_params)
    pprint(best_state["metrics"])

    best_cfg_name = (
        "best_train_config.yaml"
        if args.objective == "source_val_rmse"
        else "best_tta_config.yaml"
    )
    best_cfg_path = out_dir / best_cfg_name
    if best_state["config"] is not None:
        with best_cfg_path.open("w", encoding="utf-8") as f:
            yaml.dump(best_state["config"], f, sort_keys=False)

    best_model_name = (
        "best_source_model.pt"
        if args.objective == "source_val_rmse"
        else "best_adapted_regressor.pt"
    )
    best_model_path = out_dir / best_model_name
    if best_state["state_dict"] is not None:
        torch.save(best_state["state_dict"], best_model_path)

    trials_df = study.trials_dataframe()
    trials_csv_path = out_dir / "optuna_trials.csv"
    trials_df.to_csv(trials_csv_path, index=False)

    summary = {
        "domain": domain_tag,
        "study_name": study_name,
        "storage": args.storage,
        "objective": args.objective,
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_metrics": best_state["metrics"],
    }
    summary_path = out_dir / "study_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    flat_best: dict[str, Any] = {
        "best/trial": study.best_trial.number,
        f"best/{args.objective}": float(study.best_value),
    }
    for pk, pv in study.best_params.items():
        flat_best[f"best_param/{pk}"] = pv
    reporter.log_metrics(flat_best)

    extra_files = [db_file] if (db_file is not None and db_file.is_file()) else []
    reporter.log_directory_artifacts(
        directory=out_dir,
        artifact_name=f"optuna-{args.objective}-{domain_tag}",
        artifact_type="optuna_study",
        patterns=("*.pt", "*.pth", "*.json", "*.csv", "*.yaml", "*.db"),
        extra_files=extra_files,
    )
    reporter.finish()


if __name__ == "__main__":
    main()
