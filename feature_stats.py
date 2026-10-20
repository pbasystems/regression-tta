import json
from dataclasses import InitVar, dataclass
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from ignite.engine import Engine
from torch import Tensor
from torch.utils.data import DataLoader

from dataset import get_datasets
from model import Regressor, create_regressor
from utils.logger import attach_progress_bar
from utils.reporter import create_reporter_from_config, get_domain_tag

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint_path(source_path: str) -> Path:
    p = Path(source_path)
    if p.is_file():
        return p
    search_dir = p if p.is_dir() else p.parent
    candidates = sorted(
        search_dir.glob("source_model_*.pt"), key=lambda x: x.stat().st_mtime
    )
    if candidates:
        return candidates[-1]
    return p


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", required=True, help="config file")
    parser.add_argument("-o", required=True, help="output directory")
    parser.add_argument("--save_feature", action="store_true")
    parser.add_argument("--validation", action="store_true")

    args = parser.parse_args()
    pprint(vars(args))
    main(args)


def main(args):
    Path(args.o).mkdir(parents=True, exist_ok=True)

    with open(args.c, "r", encoding="utf-8") as f:
        if args.c.endswith(".json"):
            config = json.load(f)
        else:
            config = yaml.safe_load(f)
    config["dataset"]["train_aug"] = False
    pprint(config)

    reporter = create_reporter_from_config(
        config,
        default_name="RegressionTTA.FeatureStats",
        default_stage="Feature-Stats",
    )

    ds = get_datasets(config)[1 if args.validation else 0]
    dl = DataLoader(ds, **config["dataloader"])

    regressor = create_regressor(config).to(DEVICE)
    ckpt_path = resolve_checkpoint_path(config["regressor"]["source"])
    regressor.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print(f"load regressor: {ckpt_path}")

    calc = FeatureStatCalculator(regressor, **config["calculator"])
    print("run calculator", flush=True)
    calc.run(dl)

    mean, cov, features = calc.compute_stats()
    cov_np = cov.numpy()
    eigvals, eigvecs = np.linalg.eigh(cov_np)

    cov_rank = int(np.linalg.matrix_rank(cov_np))
    print(f"rank of cov: {cov_rank}")

    stat_path = Path(args.o, "feature_stats.pt")
    torch.save(
        {
            "mean": mean,
            "basis": torch.from_numpy(eigvecs),
            "eigvals": torch.from_numpy(eigvals),
        },
        str(stat_path),
    )

    files_to_upload = [stat_path]
    if ckpt_path.is_file():
        files_to_upload.append(ckpt_path)

    if args.save_feature:
        raw_feat_path = Path(args.o, "raw_features.pt")
        torch.save(features, str(raw_feat_path))
        files_to_upload.append(raw_feat_path)
        print("feature saved")

    reporter.log_metrics({"cov_rank": cov_rank, "num_samples": int(features.shape[0])})
    domain_tag = get_domain_tag(config)
    reporter.log_artifact_files(
        files=files_to_upload,
        artifact_name=f"feature-stats-{domain_tag}",
        artifact_type="feature_stats",
    )
    reporter.finish()


@dataclass
class FeatureStatCalculator(Engine):
    regressor: Regressor
    compile_model: InitVar[dict | None]

    def __post_init__(self, compile_model: dict | None):
        super().__init__(self.inference)

        attach_progress_bar(
            self,
            desc="feature stats",
            show_epochs=False,
            leave=False,
        )

        if compile_model is None:
            self.regressor_feature = self.regressor.feature
        else:
            try:
                self.regressor_feature = torch.compile(
                    self.regressor.feature, **compile_model
                )
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")
                self.regressor_feature = self.regressor.feature

        self.reset()

    def reset(self):
        self._features = []

    @torch.no_grad()
    def inference(self, engine: Engine, batch: tuple[Tensor, Tensor]):
        self.regressor.eval()

        x, _ = batch
        x = x.to(DEVICE)

        feat = self.regressor_feature(x)  # (B,D)
        self._features.append(feat.cpu())

    def compute_stats(self, ddof: int = 1) -> tuple[Tensor, Tensor, Tensor]:
        assert len(self._features) >= 1, "no data accumulated"

        features = torch.cat(self._features)
        mean = features.mean(dim=0)
        features_c = features - mean  # (B,D)
        cov = features_c.T @ features_c / (features.shape[0] - ddof)

        return mean, cov, features


if __name__ == "__main__":
    parse_args()
