import itertools
import json
from pathlib import Path
from pprint import pprint
from typing import Any

import torch
import yaml
from dotenv import load_dotenv
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from torch.utils.data import DataLoader

from dataset import get_datasets
from evaluation.evaluator import RegressionEvaluator
from feature_stats import resolve_checkpoint_path
from model import Regressor, create_regressor, extract_bn_layers
from tta import TTAEngine
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
    parser.add_argument("--save", action="store_true", help="save model")

    args = parser.parse_args()
    pprint(vars(args))
    main(args)


def main(args):
    fix_seed(args.seed)

    with open(args.c, "r", encoding="utf-8") as f:
        if args.c.endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    pprint(config)

    Path(args.o).mkdir(parents=True, exist_ok=True)
    with Path(args.o, "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f)

    reporter = create_reporter_from_config(
        config,
        default_name=f"RegressionTTA.Adaptation.{args.seed}",
        default_stage=f"TTA | Seed {args.seed}",
    )

    regressor = create_regressor(config).to(DEVICE)
    ckpt_path = resolve_checkpoint_path(config["regressor"]["source"])
    regressor.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print(f"load {ckpt_path}")

    _, val_ds = get_datasets(config)
    val_dl = DataLoader(val_ds, **config["adapt_dataloader"])

    reg_evaluator = RegressionEvaluator(regressor, **config["evaluator"])
    reg_evaluator.run(val_dl)
    unadapted_metrics = dict(reg_evaluator.state.metrics)

    opt = create_optimizer(regressor, config)
    engine = TTAEngine(regressor, opt, **config["tta"]["config"])

    engine.add_event_handler(
        Events.COMPLETED,
        ModelCheckpoint(args.o, "adapted", require_empty=False),
        {"regressor": regressor},
    )

    engine.run(val_dl)
    reg_evaluator.run(val_dl)

    metrics = {
        "iteration": engine.state.iteration,
        "unadapted": unadapted_metrics,
        "online": engine.state.metrics,
        "offline": reg_evaluator.state.metrics,
    }
    pprint(metrics)

    with Path(args.o, "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    flat_wandb_metrics: dict[str, Any] = {"iteration": engine.state.iteration}
    for stage_name in ("unadapted", "online", "offline"):
        for k, v in metrics[stage_name].items():
            flat_wandb_metrics[f"{stage_name}/{k}"] = (
                float(v) if isinstance(v, (int, float)) else v
            )
    reporter.log_metrics(flat_wandb_metrics)

    extra_pt_files: list[Path] = []
    if ckpt_path.is_file():
        extra_pt_files.append(ckpt_path)
    stat_file = (
        config.get("tta", {}).get("config", {}).get("pc_config", {}).get("stat_file")
    )
    if stat_file and Path(stat_file).is_file():
        extra_pt_files.append(Path(stat_file))
        raw_feat_candidate = Path(stat_file).parent / "raw_features.pt"
        if raw_feat_candidate.is_file():
            extra_pt_files.append(raw_feat_candidate)

    domain_tag = get_domain_tag(config)
    reporter.log_directory_artifacts(
        directory=args.o,
        artifact_name=f"tta-{domain_tag}",
        artifact_type="model",
        extra_files=extra_pt_files,
    )
    reporter.finish()


def create_optimizer(net: Regressor, config: dict[str, Any]) -> torch.optim.Optimizer:
    match config["optimizer"]["param"]:
        case "all":
            params = net.parameters()
        case "fe":
            params = net.get_feature_extractor().parameters()
        case "fe_bn":
            bn_layers = extract_bn_layers(net.get_feature_extractor())
            params = itertools.chain.from_iterable(l.parameters() for l in bn_layers)
        case _ as p:
            raise ValueError(f"Invalid param: {p!r}")

    opt = eval(f"torch.optim.{config['optimizer']['name']}")(
        params, **config["optimizer"]["config"]
    )
    return opt


if __name__ == "__main__":
    parse_args()
