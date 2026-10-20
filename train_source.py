import copy
import json
import sys
from pathlib import Path
from pprint import pprint
from typing import Any

import torch
import yaml
from dotenv import load_dotenv
from ignite.engine.events import Events
from ignite.handlers.checkpoint import ModelCheckpoint
from ignite.handlers.param_scheduler import (
    CosineAnnealingScheduler,
    create_lr_scheduler_with_warmup,
)
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dataset import get_datasets
from evaluation.evaluator import RegressionEvaluator
from handlers import EvaluationAccumulator, EvaluationRunner
from model import Regressor, create_regressor
from trainer import RegressionTrainer
from utils.early_stopping import attach_early_stopping
from utils.reporter import create_reporter_from_config, get_domain_tag
from utils.seed import fix_seed

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", required=True, help="config")
    parser.add_argument("-o", required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pprint(vars(args))
    main(args)


def main(args):
    with open(args.c, "r", encoding="utf-8") as f:
        if args.c.endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    pprint(config)
    sys.stdout.flush()

    Path(args.o).mkdir(parents=True, exist_ok=True)
    with Path(args.o, "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    fix_seed(args.seed)

    reporter = create_reporter_from_config(
        config,
        default_name=f"RegressionTTA.TrainSource.{args.seed}",
        default_stage=f"Source-Train | Seed {args.seed}",
    )

    net = create_regressor(config).to(DEVICE)

    opt = create_optimizer(net, config)

    train_ds, val_ds = get_datasets(config)
    train_dl = DataLoader(train_ds, **config["train_dataloader"])
    val_dl = DataLoader(val_ds, **config["val_dataloader"])

    evaluator = RegressionEvaluator(net, **config["evaluator"])

    val_ev_logger = EvaluationAccumulator()
    val_ev_runner = EvaluationRunner(
        evaluator, val_dl, "val_" + config["dataset"]["name"], val_ev_logger
    )

    trainer_cfg = copy.deepcopy(config["trainer"])
    es_cfg = config.get("early_stopping", trainer_cfg.pop("early_stopping", None))

    trainer = RegressionTrainer(net, opt, **trainer_cfg)

    if "scheduler" in config["optimizer"]:
        scheduler = create_scheduler(opt, config, len(train_dl))
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    train_ev_logger = EvaluationAccumulator()
    train_ev_runner = EvaluationRunner(
        trainer,
        train_dl,
        "train_" + config["dataset"]["name"],
        train_ev_logger,
        run_evaluator=False,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, train_ev_runner)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, val_ev_runner)

    def _log_epoch_to_reporter(engine):
        epoch = engine.state.epoch
        log_dict: dict[str, Any] = {"epoch": epoch}
        for k, v in trainer.state.metrics.items():
            log_dict[f"train/{k}"] = float(v) if isinstance(v, (int, float)) else v
        for k, v in evaluator.state.metrics.items():
            log_dict[f"val/{k}"] = float(v) if isinstance(v, (int, float)) else v
        reporter.log_metrics(log_dict, step=epoch)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_epoch_to_reporter)

    # Attach early stopping (restores best weights before ModelCheckpoint saves)
    if es_cfg is None or es_cfg.get("enabled", True):
        patience = (es_cfg or {}).get("patience", 20)
        min_delta = (es_cfg or {}).get("min_delta", 1e-4)
        attach_early_stopping(
            trainer=trainer,
            evaluator=evaluator,
            model=net,
            patience=patience,
            min_delta=min_delta,
            reporter=reporter,
        )

    trainer.add_event_handler(
        Events.COMPLETED,
        ModelCheckpoint(args.o, "source", require_empty=False),
        {"model": net},
    )

    p = Path(args.o)
    trainer.add_event_handler(
        Events.COMPLETED,
        lambda _: val_ev_logger.get_dataframe().to_csv(
            str(p / "val_metrics.csv"), index=False
        ),
    )
    trainer.add_event_handler(
        Events.COMPLETED,
        lambda _: train_ev_logger.get_dataframe().to_csv(
            str(p / "train_metrics.csv"), index=False
        ),
    )
    trainer.run(train_dl, max_epochs=config["epoch"])

    domain_tag = get_domain_tag(config)
    reporter.log_directory_artifacts(
        directory=args.o,
        artifact_name=f"source-{domain_tag}",
        artifact_type="model",
    )
    reporter.finish()
    print("done")


def create_optimizer(net: Regressor, config: dict[str, Any]) -> Optimizer:
    match p := config["optimizer"]["param"]:
        case "all":
            param = net.parameters()
        case _:
            raise ValueError(f"Invalid param: {p!r}")

    print("param:", p)
    opt = eval(f"torch.optim.{config['optimizer']['name']}")(
        param, **config["optimizer"]["config"]
    )
    return opt


def create_scheduler(opt: Optimizer, config: dict[str, Any], iter_per_epoch: int):
    scheduler_config = copy.deepcopy(config["optimizer"]["scheduler"])

    match scheduler_config["type"]:
        case "cos":
            scheduler_config["config"]["cos"]["cycle_size"] = (
                config["epoch"] * iter_per_epoch
            )
            scheduler = CosineAnnealingScheduler(
                opt, **scheduler_config["config"]["cos"]
            )
        case "warmup_cos":
            wd = scheduler_config["config"]["warmup"]["warmup_duration"]
            scheduler_config["config"]["cos"]["cycle_size"] = (
                config["epoch"] - wd + 1
            ) * iter_per_epoch
            cos_sc = CosineAnnealingScheduler(opt, **scheduler_config["config"]["cos"])

            scheduler_config["config"]["warmup"]["warmup_duration"] *= iter_per_epoch
            scheduler = create_lr_scheduler_with_warmup(
                cos_sc, **scheduler_config["config"]["warmup"]
            )

        case _:
            raise ValueError(f"Invalid scheduler type: {scheduler_config!r}")

    return scheduler


if __name__ == "__main__":
    parse_args()
