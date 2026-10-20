from dataclasses import InitVar, dataclass

import torch
import torch.nn.functional as F
from ignite.contrib.metrics.regression.r2_score import R2Score
from ignite.engine import Engine
from ignite.metrics import MeanAbsoluteError, RootMeanSquaredError
from torch import Tensor

from evaluation.metrics import OptimizerLastState, PearsonCorrelation
from model import Regressor
from utils.logger import attach_progress_bar

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RegressionTrainer(Engine):
    net: Regressor
    opt: torch.optim.Optimizer
    compile_model: InitVar[dict | None]

    def __post_init__(self, compile_model: dict | None):
        super().__init__(self.update)

        y_ot = lambda d: (d["y_pred"], d["y"])
        RootMeanSquaredError(y_ot).attach(self, "rmse")
        MeanAbsoluteError(y_ot).attach(self, "mae")
        R2Score(y_ot).attach(self, "R2")
        PearsonCorrelation(y_ot).attach(self, "r")

        OptimizerLastState(self.opt, "lr").attach(self, "lr")

        attach_progress_bar(
            self,
            desc="regression",
            show_epochs=True,
            postfix_key="loss",
            leave=False,
        )

        if compile_model is not None:
            try:
                self.net = torch.compile(self.net, **compile_model)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")

    def update(self, engine: Engine, batch: tuple[Tensor, Tensor]) -> dict[str, Tensor]:
        self.net.train()
        self.opt.zero_grad()

        x, y = batch
        x = x.to(DEVICE)
        y = y.float().flatten().to(DEVICE)

        y_pred = self.net(x)

        loss = F.mse_loss(y_pred, y)

        loss.backward()
        self.opt.step()

        return {
            "y_pred": y_pred,
            "y": y,
            "loss": loss.detach(),
        }
