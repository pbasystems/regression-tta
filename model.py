from collections.abc import Iterator
from typing import Any

import timm
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.feature_extraction import create_feature_extractor


class Regressor(nn.Module):
    regressor: nn.Module

    def feature(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def predict_from_feature(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def get_regressor(self) -> nn.Module:
        raise NotImplementedError

    def get_feature_extractor(self) -> nn.Module:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        z = self.feature(x)
        y_pred = self.predict_from_feature(z)
        return y_pred


class CNNRegressor(Regressor):
    def __init__(self, backbone: str, pretrained: bool, in_channels: int):
        super().__init__()

        match backbone:
            case "resnet26":
                base_net = timm.create_model("resnet26", pretrained=pretrained)
                if in_channels != 3:
                    base_net.conv1 = nn.Conv2d(
                        in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                    )
                self.feature_extractor = create_feature_extractor(
                    base_net, {"global_pool": "feature"}
                )

            case "resnet50":
                weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
                base_net = resnet50(weights=weights)
                if in_channels != 3:
                    base_net.conv1 = nn.Conv2d(
                        in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                    )
                self.feature_extractor = create_feature_extractor(
                    base_net, {"avgpool": "feature"}
                )

            case _:
                raise ValueError(f"Invalid backbone: {backbone!r}")

        self.regressor = nn.Linear(2048, 1)

    def feature(self, x: Tensor) -> Tensor:
        z: Tensor = self.feature_extractor(x)["feature"]
        return z.flatten(start_dim=1)

    def predict_from_feature(self, z: Tensor) -> Tensor:
        y_pred: Tensor = self.regressor(z)
        return y_pred.flatten()

    def get_regressor(self) -> nn.Module:
        return self.regressor

    def get_feature_extractor(self) -> nn.Module:
        return self.feature_extractor


class MLPRegressor(Regressor):
    def __init__(self, in_dims: int, h_dims: int, n_rep: int):
        super().__init__()

        self.fe = nn.Sequential(
            nn.Linear(in_dims, h_dims, bias=False),
            nn.BatchNorm1d(h_dims),
            nn.ReLU(),
            *(
                nn.Sequential(
                    nn.Linear(h_dims, h_dims, bias=False),
                    nn.BatchNorm1d(h_dims),
                    nn.ReLU(),
                )
                for _ in range(n_rep)
            ),
        )

        self.regressor = nn.Linear(h_dims, 1)

    def feature(self, x: Tensor) -> Tensor:
        if x.ndim >= 3:
            x = x.flatten(start_dim=1)
        return self.fe(x)

    def predict_from_feature(self, z: Tensor) -> Tensor:
        y_pred: Tensor = self.regressor(z)
        return y_pred.flatten()

    def get_regressor(self) -> nn.Linear:
        return self.regressor

    def get_feature_extractor(self) -> nn.Module:
        return self.fe


class CNN1DFeatureExtractor(nn.Module):
    """1D CNN Feature Extractor for raw vibration waveforms (e.g. PHM 2012 (2560, 2))."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        f_size: int = 128,
        n_blocks: int = 4,
        kernel_size: int = 15,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        ch_in = in_channels
        ch_out = base_channels

        for i in range(n_blocks):
            k = max(3, kernel_size - 2 * i * 2)
            pad = k // 2
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        ch_in, ch_out, kernel_size=k, stride=2, padding=pad, bias=False
                    ),
                    nn.BatchNorm1d(ch_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
                )
            )
            ch_in = ch_out
            ch_out = min(ch_out * 2, 256)

        self.conv_blocks = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(ch_in, f_size, bias=False),
            nn.BatchNorm1d(f_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Accept (B, L, C) or (B, C, L)
        if x.ndim == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)
        h = self.conv_blocks(x)
        h = self.global_pool(h)
        return self.proj(h)


class CNN1DRegressor(Regressor):
    """1D CNN Feature Extractor + Linear Regressor Head for raw time-series/vibration signals."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        f_size: int = 128,
        n_blocks: int = 4,
        kernel_size: int = 15,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_extractor = CNN1DFeatureExtractor(
            in_channels=in_channels,
            base_channels=base_channels,
            f_size=f_size,
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.regressor = nn.Linear(f_size, 1)

    def feature(self, x: Tensor) -> Tensor:
        return self.feature_extractor(x)

    def predict_from_feature(self, z: Tensor) -> Tensor:
        y_pred: Tensor = self.regressor(z)
        return y_pred.flatten()

    def get_regressor(self) -> nn.Linear:
        return self.regressor

    def get_feature_extractor(self) -> nn.Module:
        return self.feature_extractor


class LSTMDANNFeatureExtractor(nn.Module):
    """Exact FeatureExtractor from DA-Replications/src/lstm_dann/Model/FeatureExtractor."""

    def __init__(
        self,
        input_size: int = 24,
        lstm_hidden_size: int = 128,
        f_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Linear(lstm_hidden_size, f_size)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, sequence_length(T_w), input_size)
        _, (hn, _) = self.lstm(x)
        return self.projection(self.dropout(self.activation(hn[-1])))


class LSTMDANNRegressorHead(nn.Module):
    """Exact Regressor from DA-Replications/src/lstm_dann/Model/Regressor."""

    def __init__(
        self,
        feature_size: int = 64,
        regressor_first_hidden_size: int = 32,
        regressor_second_hidden_size: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        if regressor_second_hidden_size == 0:\n            self.net = nn.Sequential(
                nn.Linear(feature_size, regressor_first_hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(regressor_first_hidden_size, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(feature_size, regressor_first_hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(regressor_first_hidden_size, regressor_second_hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(regressor_second_hidden_size, 1),
            )

    @property
    def weight(self) -> Tensor:
        """Effective linear weight (1, feature_size) for SSA dimension weighting."""
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        w = linear_layers[-1].weight
        for layer in reversed(linear_layers[:-1]):
            w = w @ layer.weight
        return w

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LSTMDANNRegressor(Regressor):
    """LSTM-DANN FeatureExtractor + Regressor adapted for regression-tta."""

    def __init__(
        self,
        input_size: int = 24,
        lstm_hidden_size: int = 128,
        f_size: int = 64,
        num_layers: int = 1,
        lstm_dropout: float = 0.5,
        regressor_dropout: float = 0.3,
        regressor_first_hidden_size: int = 32,
        regressor_second_hidden_size: int = 0,
    ):
        super().__init__()
        self.feature_extractor = LSTMDANNFeatureExtractor(
            input_size=input_size,
            lstm_hidden_size=lstm_hidden_size,
            f_size=f_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
        )
        self.regressor = LSTMDANNRegressorHead(
            feature_size=f_size,
            regressor_first_hidden_size=regressor_first_hidden_size,
            regressor_second_hidden_size=regressor_second_hidden_size,
            dropout=regressor_dropout,
        )

    def feature(self, x: Tensor) -> Tensor:
        return self.feature_extractor(x)

    def predict_from_feature(self, z: Tensor) -> Tensor:
        y_pred: Tensor = self.regressor(z)
        return y_pred.flatten()

    def get_regressor(self) -> nn.Module:
        return self.regressor

    def get_feature_extractor(self) -> nn.Module:
        return self.feature_extractor


class LSTMTTARegressor(Regressor):
    """
    Hybrid Architecture:
    - Fixed FeatureExtractor from LSTM_DANN (LSTM -> ReLU -> Dropout -> Linear(f_size) [+ optional BatchNorm1d])
    - Original Linear Regressor Head from regression-tta: nn.Linear(f_size, 1)
    """

    def __init__(
        self,
        input_size: int = 24,
        lstm_hidden_size: int = 128,
        f_size: int = 64,
        num_layers: int = 1,
        lstm_dropout: float = 0.5,
        use_bn: bool = True,
    ):
        super().__init__()
        base_fe = LSTMDANNFeatureExtractor(
            input_size=input_size,
            lstm_hidden_size=lstm_hidden_size,
            f_size=f_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
        )
        if use_bn:
            self.feature_extractor: nn.Module = nn.Sequential(
                base_fe,
                nn.BatchNorm1d(f_size),
            )
        else:
            self.feature_extractor = base_fe

        # Original regression-tta linear head: z -> y_pred
        self.regressor = nn.Linear(f_size, 1)

    def feature(self, x: Tensor) -> Tensor:
        return self.feature_extractor(x)

    def predict_from_feature(self, z: Tensor) -> Tensor:
        y_pred: Tensor = self.regressor(z)
        return y_pred.flatten()

    def get_regressor(self) -> nn.Linear:
        return self.regressor

    def get_feature_extractor(self) -> nn.Module:
        return self.feature_extractor


def create_regressor(config: dict[str, Any]) -> Regressor:
    match config["regressor"]["type"]:
        case "image":
            net = CNNRegressor(**config["regressor"]["config"])

        case "table":
            net = MLPRegressor(**config["regressor"]["config"])

        case "cnn1d":
            net = CNN1DRegressor(**config["regressor"]["config"])

        case "lstm_dann":
            net = LSTMDANNRegressor(**config["regressor"]["config"])

        case "lstm_tta":
            net = LSTMTTARegressor(**config["regressor"]["config"])

        case _ as t:
            raise ValueError(f"Invalid type: {t!r}")

    return net


def extract_bn_layers(mod: nn.Module) -> Iterator[_BatchNorm]:
    for m in mod.children():
        if isinstance(m, _BatchNorm):
            yield m
        else:
            yield from extract_bn_layers(m)
