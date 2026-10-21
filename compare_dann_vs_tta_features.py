from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from adaptation import create_optimizer as create_tta_optimizer
from dataset import get_datasets
from evaluation.evaluator import RegressionEvaluator
from model import LSTMDANNFeatureExtractor, LSTMDANNRegressorHead, create_regressor
from run_realistic_cmapss_benchmark import (
    build_tta_config,
    train_or_load_domain_model,
)
from tta import TTAEngine
from utils.early_stopping import EarlyStopping
from utils.seed import fix_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Exact Original LSTM_DANN Model & Adversarial Trainer (from DA-Replications)
# ---------------------------------------------------------------------------
class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


class OriginalLSTMDANN(torch.nn.Module):
    """Exact LSTM_DANN architecture from C:\\Users\\User\\Documents\\DA-Replications\\src\\lstm_dann\\Model."""

    def __init__(
        self,
        input_size: int = 24,
        lstm_hidden_size: int = 128,
        f_size: int = 64,
        num_layers: int = 1,
        lstm_dropout: float = 0.5,
        regressor_dropout: float = 0.3,
        classifier_dropout: float = 0.3,
        alpha: float = 0.8,
        regressor_first_hidden_size: int = 32,
        regressor_second_hidden_size: int = 0,
        classifier_first_hidden_size: int = 32,
    ):
        super().__init__()
        self.alpha = alpha
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
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(f_size, classifier_first_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=classifier_dropout),
            torch.nn.Linear(classifier_first_hidden_size, 1),
            torch.nn.Sigmoid(),
        )

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        reg_out = self.regressor(features)
        rev_features = _GradientReversalFn.apply(features, self.alpha)
        cls_out = self.classifier(rev_features)
        return reg_out, cls_out


def train_or_load_original_lstm_dann(
    source_fd: str,
    target_fd: str,
    out_dir: Path,
    external_ckpt: str | None = None,
    epochs: int = 200,
    patience: int = 20,
    seed: int = 42,
) -> tuple[OriginalLSTMDANN, Path]:
    """
    Loads a saved LSTM_DANN checkpoint (either from DA-Replications or cached in `out_dir`),
    or trains the exact original LSTM_DANN (FD001 -> FD002 config from DA-Replications)
    and saves `original_lstm_dann_FD001_to_FD002.pt`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = (
        Path(external_ckpt)
        if external_ckpt is not None
        else (out_dir / f"original_lstm_dann_{source_fd}_to_{target_fd}.pt")
    )

    model = OriginalLSTMDANN(
        input_size=24,
        lstm_hidden_size=128,
        f_size=64,
        num_layers=1,
        lstm_dropout=0.5,
        regressor_dropout=0.3,
        classifier_dropout=0.3,
        alpha=0.8,
        regressor_first_hidden_size=32,
        regressor_second_hidden_size=0,
        classifier_first_hidden_size=32,
    ).to(DEVICE)

    if ckpt_path.is_file():
        print(f"[LSTM-DANN] Loading saved checkpoint from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        return model, ckpt_path

    print(
        f"\n================ TRAINING ORIGINAL LSTM-DANN ({source_fd} -> {target_fd}) ================"
    )
    fix_seed(seed)

    src_cfg = {
        "dataset": {
            "name": "cmapss",
            "config": {
                "source_domain": True,
                "norm_mode": "minmax",
                "combine_splits_for_stats": True,
                "source_fd": source_fd,
                "target_fd": target_fd,
                "source_split": "train",
                "target_split": "train",
                "feature_cols": "all_24",
                "max_rul": 125.0,
                "normalize_rul": False,
                "window_size": 30,
            },
            "train_ratio": 0.9,
            "split_by_engine": True,
        }
    }
    tgt_cfg = copy.deepcopy(src_cfg)
    tgt_cfg["dataset"]["config"]["source_domain"] = False

    src_train_ds, src_val_ds = get_datasets(src_cfg)
    tgt_train_ds, _ = get_datasets(tgt_cfg)

    src_train_dl = DataLoader(src_train_ds, batch_size=256, shuffle=True)
    src_val_dl = DataLoader(src_val_ds, batch_size=256, shuffle=False)
    tgt_train_dl = DataLoader(tgt_train_ds, batch_size=256, shuffle=True)

    lr_source_reg = 0.01
    lr_domain_class = 0.01
    l2_reg = 0.01

    reg_opt = torch.optim.SGD(
        [
            {"params": model.feature_extractor.parameters(), "lr": lr_source_reg},
            {
                "params": model.regressor.parameters(),
                "lr": lr_source_reg,
                "weight_decay": l2_reg,
            },
        ]
    )
    dom_opt = torch.optim.SGD(
        [
            {"params": model.feature_extractor.parameters(), "lr": lr_domain_class},
            {
                "params": model.classifier.parameters(),
                "lr": lr_domain_class,
                "weight_decay": l2_reg,
            },
        ]
    )
    reg_sched = MultiStepLR(reg_opt, milestones=[100], gamma=0.1)
    dom_sched = MultiStepLR(dom_opt, milestones=[100], gamma=0.1)

    l1_loss = torch.nn.L1Loss()
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    stopper = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        model.train()
        for x_s, y_s in src_train_dl:
            x_s = x_s.to(DEVICE)
            y_s = y_s.to(DEVICE).unsqueeze(1)
            reg_opt.zero_grad()
            y_pred, _ = model(x_s)
            loss_r = l1_loss(y_pred, y_s)
            loss_r.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            reg_opt.step()

        for (x_s, _), (x_t, _) in zip(src_train_dl, tgt_train_dl):
            x_s = x_s.to(DEVICE)
            x_t = x_t.to(DEVICE)
            dom_opt.zero_grad()
            _, d_s = model(x_s)
            _, d_t = model(x_t)
            loss_d = bce_loss(d_s, torch.zeros_like(d_s)) + bce_loss(
                d_t, torch.ones_like(d_t)
            )
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            dom_opt.step()

        model.eval()
        val_rmse_sum = 0.0
        with torch.no_grad():
            for x_v, y_v in src_val_dl:
                x_v = x_v.to(DEVICE)
                y_v = y_v.to(DEVICE).unsqueeze(1)
                y_vp, _ = model(x_v)
                val_rmse_sum += torch.sqrt(mse_loss(y_vp, y_v)).item()
        val_rmse = val_rmse_sum / max(len(src_val_dl), 1)

        if stopper.step(val_rmse, model, epoch=epoch + 1):
            print(
                f"[LSTM-DANN] Early stopping at epoch {epoch + 1} (best epoch {stopper.best_epoch}, val_rmse={stopper.best_score:.2f})"
            )
            break
        reg_sched.step()
        dom_sched.step()

    stopper.restore_best_weights(model)
    torch.save(model.state_dict(), ckpt_path)
    print(f"[LSTM-DANN] Saved trained model to {ckpt_path}")
    model.eval()
    return model, ckpt_path


# ---------------------------------------------------------------------------
# 2. Train & Adapt SSA Model Using General Calibrated Offline Config (Save Model)
# ---------------------------------------------------------------------------
def train_and_adapt_general_calibrated_tta(
    source_fd: str,
    target_fd: str,
    realistic_root: Path,
    out_dir: Path,
    model_type: str = "lstm_tta",
    seed: int = 42,
) -> tuple[torch.nn.Module, torch.nn.Module, Path]:
    """
    Returns:
      - source_net: Unadapted source model (encodes FD001 source features in SSA's feature space)
      - adapted_net: General Calibrated Offline adapted SSA model (encodes FD002 target features aligned to FD001)
      - adapted_ckpt_path: Saved checkpoint path
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    hp_file = realistic_root / "general_hyperparameters.json"
    if hp_file.is_file():
        with hp_file.open("r", encoding="utf-8") as f:
            hp_data = json.load(f)
        src_hp = hp_data.get("general_source_train_hp", {})
        cal_tta_hp = hp_data.get("general_tta_calibrated_hp", {})
    else:
        src_hp = {
            "source_lr": 0.00038197033573826074,
            "source_weight_decay": 1.1849527803101846e-05,
            "source_batch_size": 256,
            "lstm_dropout": 0.3263862285466911,
        }
        cal_tta_hp = {
            "contrib_top_k": 7,
            "lr": 0.0016981589331430675,
            "weight_decay": 3.687809169328372e-06,
            "optimizer_name": "Adam",
            "adapt_param": "fe_bn",
            "weight_bias": 2.189800819810747,
            "weight_exp": 1.531488723604991,
            "loss_eps": 2.6642180122269597e-07,
            "batch_size": 64,
            "train_mode": True,
        }

    src_ckpt, src_stat, src_train_cfg, _ = train_or_load_domain_model(
        fd=source_fd,
        out_root=realistic_root,
        subdir_name="02_general_tuned_source_models",
        epochs=200,
        patience=20,
        model_type=model_type,
        train_hp=src_hp,
        seed=seed,
        use_wandb=False,
        wandb_project="Regression-TTA",
    )

    tta_cfg = build_tta_config(
        source_fd=source_fd,
        target_fd=target_fd,
        ckpt_path=src_ckpt,
        stat_path=src_stat,
        model_type=model_type,
        train_cfg=src_train_cfg,
        tta_hp=cal_tta_hp,
        target_split="test",
        last_window_only=False,
    )

    source_net = create_regressor(tta_cfg).to(DEVICE)
    source_net.load_state_dict(torch.load(src_ckpt, map_location=DEVICE))
    source_net.eval()

    adapted_ckpt_path = (
        out_dir
        / f"adapted_general_calibrated_{model_type}_{source_fd}_to_{target_fd}.pt"
    )
    adapted_net = create_regressor(tta_cfg).to(DEVICE)

    if adapted_ckpt_path.is_file():
        print(
            f"[General Calibrated TTA] Loading saved adapted model from {adapted_ckpt_path}"
        )
        adapted_net.load_state_dict(torch.load(adapted_ckpt_path, map_location=DEVICE))
        adapted_net.eval()
        return source_net, adapted_net, adapted_ckpt_path

    print(
        f"\n================ ADAPTING GENERAL CALIBRATED TTA ({source_fd} -> {target_fd}) ================"
    )
    fix_seed(seed)
    _, val_ds = get_datasets(tta_cfg)
    adapt_dl = DataLoader(val_ds, **tta_cfg["adapt_dataloader"])
    eval_dl = DataLoader(val_ds, **tta_cfg["val_dataloader"])

    adapted_net.load_state_dict(torch.load(src_ckpt, map_location=DEVICE))
    opt = create_tta_optimizer(adapted_net, tta_cfg)
    engine = TTAEngine(adapted_net, opt, **tta_cfg["tta"]["config"])
    ev = RegressionEvaluator(adapted_net, **tta_cfg["evaluator"])

    engine.run(adapt_dl)
    ev.run(eval_dl)
    print(
        f"[General Calibrated TTA] Adapted Offline RMSE={ev.state.metrics['rmse_loss']:.2f}, "
        f"MAE={ev.state.metrics['mae_loss']:.2f}"
    )

    torch.save(adapted_net.state_dict(), adapted_ckpt_path)
    print(f"[General Calibrated TTA] Saved adapted model to {adapted_ckpt_path}")
    adapted_net.eval()
    return source_net, adapted_net, adapted_ckpt_path


# ---------------------------------------------------------------------------
# 3. Feature Extraction on Source (FD001) and Target (FD002) Test Sets
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> np.ndarray:
    """Extracts (N, 64) feature vectors from `model.feature(x)`."""
    model.eval()
    feats = []
    for x, _ in dataloader:
        x = x.to(DEVICE)
        feats.append(model.feature(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# 4. Side-by-Side Domain Scatter Plot (Left: LSTM-DANN, Right: SSA)
#    Blue = Source (FD001), Red/Orange = Target (FD002)
# ---------------------------------------------------------------------------
def plot_side_by_side_domain_scatter(
    dann_src_feats: np.ndarray,
    dann_tgt_feats: np.ndarray,
    ssa_src_feats: np.ndarray,
    ssa_tgt_feats: np.ndarray,
    source_fd: str,
    target_fd: str,
    out_dir: Path,
    max_points_per_domain: int = 1500,
    seed: int = 42,
) -> list[Path]:
    """
    Generates the exact clean side-by-side scatter style requested:
      - Left:  LSTM-DANN (FD001 in blue vs. FD002 in red-orange)
      - Right: SSA (General Calibrated Offline) (FD001 in blue vs. FD002 in red-orange)
    Saves both a t-SNE figure (matching the DANN paper visual) and a PCA figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    def _subsample_pair(
        f_src: np.ndarray, f_tgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(f_src) > max_points_per_domain:
            idx_s = rng.choice(len(f_src), size=max_points_per_domain, replace=False)
            f_src = f_src[idx_s]
        if len(f_tgt) > max_points_per_domain:
            idx_t = rng.choice(len(f_tgt), size=max_points_per_domain, replace=False)
            f_tgt = f_tgt[idx_t]
        return f_src, f_tgt

    dann_s, dann_t = _subsample_pair(dann_src_feats, dann_tgt_feats)
    ssa_s, ssa_t = _subsample_pair(ssa_src_feats, ssa_tgt_feats)

    # Compute joint 2D t-SNE embeddings for (FD001 + FD002) in each model's space
    print("[t-SNE] Computing 2D t-SNE embedding for LSTM-DANN (FD001 + FD002)...")
    dann_joint = np.vstack([dann_s, dann_t])
    dann_tsne = TSNE(
        n_components=2, perplexity=30, random_state=seed, init="pca"
    ).fit_transform(dann_joint)
    dann_s_2d = dann_tsne[: len(dann_s)]
    dann_t_2d = dann_tsne[len(dann_s) :]

    print("[t-SNE] Computing 2D t-SNE embedding for SSA (FD001 + FD002)...")
    ssa_joint = np.vstack([ssa_s, ssa_t])
    ssa_tsne = TSNE(
        n_components=2, perplexity=30, random_state=seed, init="pca"
    ).fit_transform(ssa_joint)
    ssa_s_2d = ssa_tsne[: len(ssa_s)]
    ssa_t_2d = ssa_tsne[len(ssa_s) :]

    # Colors matching your reference image: Periwinkle Blue for FD001, Red-Orange for FD002
    color_src = "#5c6bc0"  # Blue
    color_tgt = "#d84315"  # Red-Orange

    saved: list[Path] = []

    # -----------------------------------------------------------------------
    # Figure 1: Clean frameless t-SNE (exact style of your reference image)
    # -----------------------------------------------------------------------
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 6.2), dpi=220)
    fig.patch.set_facecolor("white")

    # Left: LSTM-DANN
    ax_left.scatter(
        dann_s_2d[:, 0],
        dann_s_2d[:, 1],
        c=color_src,
        s=9,
        alpha=0.80,
        edgecolors="none",
        label=source_fd,
    )
    ax_left.scatter(
        dann_t_2d[:, 0],
        dann_t_2d[:, 1],
        c=color_tgt,
        s=9,
        alpha=0.80,
        edgecolors="none",
        label=target_fd,
    )
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    for spine in ax_left.spines.values():
        spine.set_visible(False)
    ax_left.set_xlabel(
        "LSTM-DANN", fontsize=14, fontweight="medium", color="#555555", labelpad=14
    )

    # Right: SSA (General Calibrated Offline)
    ax_right.scatter(
        ssa_s_2d[:, 0],
        ssa_s_2d[:, 1],
        c=color_src,
        s=9,
        alpha=0.80,
        edgecolors="none",
        label=source_fd,
    )
    ax_right.scatter(
        ssa_t_2d[:, 0],
        ssa_t_2d[:, 1],
        c=color_tgt,
        s=9,
        alpha=0.80,
        edgecolors="none",
        label=target_fd,
    )
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    for spine in ax_right.spines.values():
        spine.set_visible(False)
    ax_right.set_xlabel(
        "SSA (General Calibrated Offline)",
        fontsize=14,
        fontweight="medium",
        color="#555555",
        labelpad=14,
    )

    # Shared legend at top
    handles, labels = ax_left.get_legend_handles_labels()
    fig.legend(
        handles,
        [f"Source ({source_fd})", f"Target ({target_fd})"],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=12,
        markerscale=2.2,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    tsne_path = out_dir / f"tsne_lstm_dann_vs_ssa_{source_fd}_to_{target_fd}.png"
    fig.savefig(tsne_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved.append(tsne_path)

    # -----------------------------------------------------------------------
    # Figure 2: Clean frameless 2D PCA (Left: LSTM-DANN, Right: SSA)
    # -----------------------------------------------------------------------
    dann_pca = PCA(n_components=2, random_state=seed).fit_transform(dann_joint)
    dann_s_pca = dann_pca[: len(dann_s)]
    dann_t_pca = dann_pca[len(dann_s) :]

    ssa_pca = PCA(n_components=2, random_state=seed).fit_transform(ssa_joint)
    ssa_s_pca = ssa_pca[: len(ssa_s)]
    ssa_t_pca = ssa_pca[len(ssa_s) :]

    fig_p, (ax_pl, ax_pr) = plt.subplots(1, 2, figsize=(13, 6.2), dpi=220)
    fig_p.patch.set_facecolor("white")

    ax_pl.scatter(
        dann_s_pca[:, 0],
        dann_s_pca[:, 1],
        c=color_src,
        s=9,
        alpha=0.80,
        edgecolors="none",
        label=source_fd,
    )
    ax_pl.scatter(
        dann_t_pca[:, 0],
        dann_t_pca[:, 1],
        c=color_tgt,
        s=9,
        alpha=0.80,
        edgecolors="none",
        label=target_fd,
    )
    ax_pl.set_xticks([])
    ax_pl.set_yticks([])
    for spine in ax_pl.spines.values():
        spine.set_visible(False)
    ax_pl.set_xlabel(
        "LSTM-DANN (PCA)",
        fontsize=14,
        fontweight="medium",
        color="#555555",
        labelpad=14,
    )

    ax_pr.scatter(
        ssa_s_pca[:, 0],
        ssa_s_pca[:, 1],
        c=color_src,
        s=9,
        alpha=0.80,
        edgecolors="none",
        label=source_fd,
    )
    ax_pr.scatter(
        ssa_t_pca[:, 0],
        ssa_t_pca[:, 1],
        c=color_tgt,
        s=9,
        alpha=0.80,
        edgecolors="none",
        label=target_fd,
    )
    ax_pr.set_xticks([])
    ax_pr.set_yticks([])
    for spine in ax_pr.spines.values():
        spine.set_visible(False)
    ax_pr.set_xlabel(
        "SSA (General Calibrated Offline - PCA)",
        fontsize=14,
        fontweight="medium",
        color="#555555",
        labelpad=14,
    )

    fig_p.legend(
        handles,
        [f"Source ({source_fd})", f"Target ({target_fd})"],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=12,
        markerscale=2.2,
    )

    fig_p.tight_layout(rect=[0, 0, 1, 0.93])
    pca_path = out_dir / f"pca_lstm_dann_vs_ssa_{source_fd}_to_{target_fd}.png"
    fig_p.savefig(pca_path, bbox_inches="tight", facecolor="white")
    plt.close(fig_p)
    saved.append(pca_path)

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot side-by-side domain feature scatter (FD001 in Blue vs FD002 in Red-Orange):\n"
            "  Left  = Original LSTM-DANN\n"
            "  Right = SSA (General Calibrated Offline)"
        )
    )
    parser.add_argument("--source-fd", default="FD001")
    parser.add_argument("--target-fd", default="FD002")
    parser.add_argument(
        "--dann-ckpt",
        default=None,
        help="Optional path to a pre-saved LSTM-DANN .pt checkpoint",
    )
    parser.add_argument(
        "--realistic-root",
        default="result/benchmark_realistic",
        help="Path to result/benchmark_realistic directory",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Max windows sampled per domain (FD001 and FD002) for clean t-SNE scatter (default: 2000)",
    )
    parser.add_argument(
        "--last-window-only",
        action="store_true",
        help="Extract features only on the last window per engine in test sets",
    )
    parser.add_argument(
        "-o",
        default="result/feature_comparison_fd001_to_fd002",
        help="Output directory for saved models, feature arrays, and plots",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.o)
    out_dir.mkdir(parents=True, exist_ok=True)
    realistic_root = Path(args.realistic_root)

    # 1. Train or load Original LSTM-DANN (saves original_lstm_dann_FD001_to_FD002.pt)
    dann_model, dann_ckpt = train_or_load_original_lstm_dann(
        source_fd=args.source_fd,
        target_fd=args.target_fd,
        out_dir=out_dir,
        external_ckpt=args.dann_ckpt,
        seed=args.seed,
    )

    # 2. Train & adapt General Calibrated Offline SSA model (saves adapted_general_calibrated_lstm_tta_FD001_to_FD002.pt)
    source_net, ssa_adapted_net, ssa_ckpt = train_and_adapt_general_calibrated_tta(
        source_fd=args.source_fd,
        target_fd=args.target_fd,
        realistic_root=realistic_root,
        out_dir=out_dir,
        model_type="lstm_tta",
        seed=args.seed,
    )

    # 3. Build Source Test (test_FD001) and Target Test (test_FD002) DataLoaders
    src_test_cfg = {
        "dataset": {
            "name": "cmapss",
            "config": {
                "source_domain": True,
                "norm_mode": "minmax",
                "combine_splits_for_stats": True,
                "source_fd": args.source_fd,
                "target_fd": args.target_fd,
                "source_split": "test",
                "target_split": "test",
                "feature_cols": "all_24",
                "max_rul": 125.0,
                "normalize_rul": False,
                "window_size": 30,
                "last_window_only": args.last_window_only,
            },
        }
    }
    tgt_test_cfg = copy.deepcopy(src_test_cfg)
    tgt_test_cfg["dataset"]["config"]["source_domain"] = False

    _, src_test_ds = get_datasets(src_test_cfg)
    _, tgt_test_ds = get_datasets(tgt_test_cfg)

    src_test_dl = DataLoader(src_test_ds, batch_size=256, shuffle=False)
    tgt_test_dl = DataLoader(tgt_test_ds, batch_size=256, shuffle=False)

    # 4. Extract (FD001, FD002) feature vectors for both models:
    #    - Left (LSTM-DANN): encodes both FD001 and FD002 with its joint encoder
    #    - Right (SSA): source encoder encodes FD001 reference features; adapted SSA encoder encodes FD002 aligned features
    dann_src_feats = extract_features(dann_model, src_test_dl)
    dann_tgt_feats = extract_features(dann_model, tgt_test_dl)

    ssa_src_feats = extract_features(source_net, src_test_dl)
    ssa_tgt_feats = extract_features(ssa_adapted_net, tgt_test_dl)

    feat_npz = out_dir / f"domain_features_{args.source_fd}_and_{args.target_fd}.npz"
    np.savez_compressed(
        feat_npz,
        dann_src_feats=dann_src_feats,
        dann_tgt_feats=dann_tgt_feats,
        ssa_src_feats=ssa_src_feats,
        ssa_tgt_feats=ssa_tgt_feats,
    )

    # 5. Plot side-by-side: Left = LSTM-DANN, Right = SSA (FD001 Blue vs FD002 Red-Orange)
    plots = plot_side_by_side_domain_scatter(
        dann_src_feats=dann_src_feats,
        dann_tgt_feats=dann_tgt_feats,
        ssa_src_feats=ssa_src_feats,
        ssa_tgt_feats=ssa_tgt_feats,
        source_fd=args.source_fd,
        target_fd=args.target_fd,
        out_dir=out_dir,
        max_points_per_domain=args.max_points,
        seed=args.seed,
    )

    print(f"\nSaved LSTM-DANN checkpoint: {dann_ckpt}")
    print(f"Saved SSA adapted checkpoint: {ssa_ckpt}")
    print(f"Saved extracted domain features: {feat_npz}")
    for p in plots:
        print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()
