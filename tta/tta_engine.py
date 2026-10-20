from dataclasses import InitVar, dataclass

import torch
from ignite.contrib.metrics.regression.r2_score import R2Score
from ignite.engine import Engine
from ignite.metrics import MeanAbsoluteError, RootMeanSquaredError
from torch import Tensor

from evaluation.metrics import ModelDistanceMetric, PearsonCorrelation
from model import Regressor
from utils.logger import attach_progress_bar
from utils.loss import diagonal_gaussian_kl_loss
from utils.pca_basis import get_pca_basis

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TTAEngine(Engine):
    net: Regressor
    opt: torch.optim.Optimizer
    train_mode: bool
    pc_config: InitVar[dict]
    loss_config: InitVar[dict]
    weight_bias: InitVar[float]
    weight_exp: InitVar[float]
    compile_model: InitVar[dict | None]

    @torch.no_grad()
    def __post_init__(
        self,
        pc_config: dict,
        loss_config: dict,
        weight_bias: float,
        weight_exp: float,
        compile_model: dict | None,
    ):
        super().__init__(self.update)

        y_ot = lambda d: (d["y_pred"], d["y"])
        RootMeanSquaredError(y_ot).attach(self, "rmse_loss")
        MeanAbsoluteError(y_ot).attach(self, "mae_loss")
        R2Score(y_ot).attach(self, "R2")
        PearsonCorrelation(y_ot).attach(self, "r")

        ModelDistanceMetric(self.net).attach(self, "model_dist")

        attach_progress_bar(
            self,
            desc="TTA adaptation",
            show_epochs=False,
            postfix_key="kl_loss",
            leave=False,
        )

        mean, basis, var = get_pca_basis(**pc_config)
        self.mean = mean.to(DEVICE)
        self.basis = basis.to(DEVICE)
        self.var = var.to(DEVICE)

        self.loss_fn = lambda m1, v1, m2, v2: diagonal_gaussian_kl_loss(
            m1, v1, m2, v2, dim_reduction="none", **loss_config
        )

        self.dim_weight = (
            torch.abs(self.net.regressor.weight @ self.basis).flatten() + weight_bias
        )
        self.dim_weight = self.dim_weight.pow(weight_exp)

        print(f"dim_weight: {self.dim_weight}")

        self.feature_extractor = self.net.feature
        if compile_model is not None:
            try:
                self.feature_extractor = torch.compile(
                    self.net.feature, **compile_model
                )
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")

    def update(self, engine: Engine, batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        if self.train_mode:
            self.net.train()
        else:
            self.net.eval()
        self.net.zero_grad()

        x, y = batch
        x = x.to(DEVICE)

        feature = self.feature_extractor(x)
        y_pred = self.net.predict_from_feature(feature)

        # (B,D) @ (D,d) -> (B,d)
        f_pc = (feature - self.mean) @ self.basis
        f_pc_mean = f_pc.mean(dim=0)  # (d)
        f_pc_var = f_pc.var(dim=0)  # (d)

        zeros = torch.zeros_like(f_pc_mean)
        kl_loss = self.loss_fn(f_pc_mean, f_pc_var, zeros, self.var) + self.loss_fn(
            zeros, self.var, f_pc_mean, f_pc_var
        )

        kl_loss = kl_loss @ self.dim_weight

        kl_loss.backward()
        self.opt.step()

        return {
            "y_pred": y_pred,
            "y": y.to(DEVICE).float().flatten(),
            "feat_pc": f_pc,
            "kl_loss": kl_loss.detach(),
        }
