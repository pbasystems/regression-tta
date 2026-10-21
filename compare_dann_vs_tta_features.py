from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
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
    and saves `lstm_dann_model.pt`.
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
        return model, ckpt_path

    print(
        f"\n================ TRAINING ORIGINAL LSTM-DANN ({source_fd} -> {target_fd}) ================"
    )
    fix_seed(seed)

    # Exact dataset setup from DA-Replications/run_experiment.py
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

    # Exact hyperparameters from DA-Replications/configs/lstm_dann/FD001_FD002.yaml
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
        # Stage 1: Source regression step
        for x_s, y_s in src_train_dl:
            x_s = x_s.to(DEVICE)
            y_s = y_s.to(DEVICE).unsqueeze(1)
            reg_opt.zero_grad()
            y_pred, _ = model(x_s)
            loss_r = l1_loss(y_pred, y_s)
            loss_r.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            reg_opt.step()

        # Stage 2: Domain adversarial step
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

        # Validate on Source Val RMSE
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
    Loads (or trains) the general-tuned source model and adapts it on `target_fd`
    using the `general_tta_calibrated_hp` settings, saving the adapted checkpoint.
    Returns: (unadapted_source_model, adapted_tta_model, adapted_ckpt_path)
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

    # Unadapted source model
    source_net = create_regressor(tta_cfg).to(DEVICE)
    source_net.load_state_dict(torch.load(src_ckpt, map_location=DEVICE))
    source_net.eval()

    # Adapted General Calibrated Offline TTA model
    adapted_ckpt_path = (
        out_dir / f"adapted_general_calibrated_{model_type}_{source_fd}_to_{target_fd}.pt"
    )
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
# 3. Feature Extraction & Representation Similarity Metrics (CKA, CCA, RSA)
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features_and_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    is_dann: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (features [N, D], y_pred [N], y_true [N]) in deterministic sample order."""
    model.eval()
    feats_list, preds_list, y_list = [], [], []
    for x, y in dataloader:
        x = x.to(DEVICE)
        if is_dann:
            z = model.feature(x)
            y_hat, _ = model(x)
            y_hat = y_hat.flatten()
        else:
            z = model.feature(x)
            y_hat = model.predict_from_feature(z).flatten()
        feats_list.append(z.cpu().numpy())
        preds_list.append(y_hat.cpu().numpy())
        y_list.append(y.numpy().flatten())

    return (
        np.concatenate(feats_list, axis=0),
        np.concatenate(preds_list, axis=0),
        np.concatenate(y_list, axis=0),
    )


def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Computes Linear Centered Kernel Alignment (CKA) in [0, 1], invariant to orthogonal rotation & isotropic scaling."""
    X_c = X - X.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X_c.T @ Y_c, ord="fro") ** 2
    hsic_xx = np.linalg.norm(X_c.T @ X_c, ord="fro") ** 2
    hsic_yy = np.linalg.norm(Y_c.T @ Y_c, ord="fro") ** 2
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12))


def compute_rbf_cka(X: np.ndarray, Y: np.ndarray, max_samples: int = 2000) -> float:
    """Computes RBF-Kernel CKA on up to `max_samples` points using median distance heuristic."""
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
        Y = Y[idx]

    def _rbf_gram(Z: np.ndarray) -> np.ndarray:
        sq_dists = (
            np.sum(Z**2, axis=1, keepdims=True) + np.sum(Z**2, axis=1) - 2.0 * (Z @ Z.T)
        )
        sq_dists = np.maximum(sq_dists, 0.0)
        median_sq = (
            float(np.median(sq_dists[sq_dists > 0])) if np.any(sq_dists > 0) else 1.0
        )
        K = np.exp(-sq_dists / (2.0 * median_sq + 1e-12))
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    Kx = _rbf_gram(X)
    Ky = _rbf_gram(Y)
    num = float(np.sum(Kx * Ky))
    den = float(np.sqrt(np.sum(Kx * Kx) * np.sum(Ky * Ky)) + 1e-12)
    return num / den


def compute_cca_correlations(
    X: np.ndarray, Y: np.ndarray, n_components: int = 10
) -> np.ndarray:
    """Computes top-k Canonical Correlation Analysis (CCA) correlations between two feature spaces."""
    X_s = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    Y_s = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)
    cca = CCA(n_components=n_components, max_iter=1000)
    X_c, Y_c = cca.fit_transform(X_s, Y_s)
    corrs = [
        float(np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]) for i in range(n_components)
    ]
    return np.array(corrs, dtype=float)


def compute_rsa_spearman(
    X: np.ndarray, Y: np.ndarray, max_samples: int = 1500
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Representational Similarity Analysis (RSA):
    Measures Spearman rank correlation between all pairwise Euclidean distances in X vs Y.
    """
    if len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
        Y = Y[idx]

    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    Y_norm = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)

    dist_x = pdist(X_norm, metric="euclidean")
    dist_y = pdist(Y_norm, metric="euclidean")
    rho, _ = spearmanr(dist_x, dist_y)
    return float(rho), dist_x, dist_y


# ---------------------------------------------------------------------------
# 4. Visualization Suite
# ---------------------------------------------------------------------------
def generate_feature_comparison_plots(
    z_dann: np.ndarray,
    z_tta: np.ndarray,
    z_src_unadapted: np.ndarray,
    y_true: np.ndarray,
    pred_dann: np.ndarray,
    pred_tta: np.ndarray,
    metrics_summary: dict[str, Any],
    cca_corrs: np.ndarray,
    dist_dann: np.ndarray,
    dist_tta: np.ndarray,
    out_dir: Path,
    task_label: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Subsample for t-SNE & scatter readability if large
    rng = np.random.RandomState(42)
    n_total = len(y_true)
    sub_idx = (
        rng.choice(n_total, size=min(2000, n_total), replace=False)
        if n_total > 2000
        else np.arange(n_total)
    )

    # -----------------------------------------------------------------------
    # Plot 1: 2x3 Master Comparison Dashboard (PCA & t-SNE colored by True RUL)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5), dpi=200)

    models_feats = [
        ("Unadapted Source Encoder", z_src_unadapted),
        ("General Calibrated TTA (SSA) Encoder", z_tta),
        ("Original LSTM-DANN Encoder", z_dann),
    ]

    for col_idx, (title, Z) in enumerate(models_feats):
        Z_std = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-8)

        # Row 0: 2D PCA
        pca = PCA(n_components=2, random_state=42)
        Z_pca = pca.fit_transform(Z_std)
        var_exp = pca.explained_variance_ratio_ * 100.0
        ax_p = axes[0, col_idx]
        sc_p = ax_p.scatter(
            Z_pca[sub_idx, 0],
            Z_pca[sub_idx, 1],
            c=y_true[sub_idx],
            cmap="viridis",
            s=12,
            alpha=0.8,
        )
        ax_p.set_title(
            f"{title}\n2D PCA ({var_exp[0]:.1f}% + {var_exp[1]:.1f}% var)",
            fontsize=11,
            fontweight="bold",
        )
        ax_p.set_xlabel("PC 1")
        ax_p.set_ylabel("PC 2")
        ax_p.grid(True, linestyle="--", alpha=0.35)
        plt.colorbar(sc_p, ax=ax_p, label="True RUL (Cycles)")

        # Row 1: 2D t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
        Z_tsne = tsne.fit_transform(Z_std[sub_idx])
        ax_t = axes[1, col_idx]
        sc_t = ax_t.scatter(
            Z_tsne[:, 0],
            Z_tsne[:, 1],
            c=y_true[sub_idx],
            cmap="viridis",
            s=12,
            alpha=0.8,
        )
        ax_t.set_title(f"{title}\n2D t-SNE Manifold", fontsize=11, fontweight="bold")
        ax_t.set_xlabel("t-SNE 1")
        ax_t.set_ylabel("t-SNE 2")
        ax_t.grid(True, linestyle="--", alpha=0.35)
        plt.colorbar(sc_t, ax=ax_t, label="True RUL (Cycles)")

    fig.suptitle(
        f"Target Test Feature Manifold Comparison ({task_label}) Colored by True RUL\n"
        f"Linear CKA (TTA vs DANN) = {metrics_summary['linear_cka_tta_vs_dann']:.3f} | "
        f"RBF CKA = {metrics_summary['rbf_cka_tta_vs_dann']:.3f} | "
        f"RSA Spearman r = {metrics_summary['rsa_spearman_tta_vs_dann']:.3f}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    p1 = out_dir / "01_feature_manifolds_pca_and_tsne.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # -----------------------------------------------------------------------
    # Plot 2: Quantitative Alignment Dashboard (CCA + RSA + PC1 vs RUL + Preds)
    # -----------------------------------------------------------------------
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=200)

    # Panel A: Top-10 CCA Canonical Correlations
    comps = np.arange(1, len(cca_corrs) + 1)
    bars = ax1.bar(
        comps, cca_corrs, color="#2980b9", edgecolor="black", alpha=0.9, width=0.65
    )
    for b in bars:
        h = b.get_height()
        ax1.annotate(
            f"{h:.2f}",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax1.set_ylim(0, 1.12)
    ax1.set_xticks(comps)
    ax1.set_xlabel("Canonical Component Index", fontweight="bold")
    ax1.set_ylabel("Canonical Correlation (r)", fontweight="bold")
    ax1.set_title(
        f"A. Canonical Correlation Analysis (CCA): TTA vs. LSTM-DANN\n"
        f"(Top-3 Mean Correlation = {float(np.mean(cca_corrs[:3])):.3f})",
        fontsize=11,
        fontweight="bold",
    )
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel B: Representational Similarity Analysis (Pairwise Distance Scatter)
    pair_sub = rng.choice(len(dist_dann), size=min(8000, len(dist_dann)), replace=False)
    ax2.hexbin(
        dist_dann[pair_sub],
        dist_tta[pair_sub],
        gridsize=45,
        cmap="Blues",
        mincnt=1,
    )
    ax2.set_xlabel("Pairwise Feature Distance in LSTM-DANN Space", fontweight="bold")
    ax2.set_ylabel(
        "Pairwise Feature Distance in Calibrated TTA Space", fontweight="bold"
    )
    ax2.set_title(
        f"B. Representational Similarity Analysis (Pairwise Geometry)\n"
        f"Spearman Rank Correlation = {metrics_summary['rsa_spearman_tta_vs_dann']:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    ax2.grid(True, linestyle="--", alpha=0.35)

    # Panel C: Dominant Principal Component (PC1) vs True RUL Trajectory
    pc1_dann = (
        PCA(n_components=1)
        .fit_transform((z_dann - z_dann.mean(0)) / (z_dann.std(0) + 1e-8))
        .flatten()
    )
    pc1_tta = (
        PCA(n_components=1)
        .fit_transform((z_tta - z_tta.mean(0)) / (z_tta.std(0) + 1e-8))
        .flatten()
    )
        pc1_dann = -pc1_dann
    if np.corrcoef(pc1_tta, y_true)[0, 1] < 0:
        pc1_tta = -pc1_tta

    pc1_dann_z = (pc1_dann - pc1_dann.mean()) / (pc1_dann.std() + 1e-8)
    pc1_tta_z = (pc1_tta - pc1_tta.mean()) / (pc1_tta.std() + 1e-8)
    r_pc1 = float(np.corrcoef(pc1_dann_z, pc1_tta_z)[0, 1])

    ax3.scatter(
        y_true[sub_idx],
        pc1_tta_z[sub_idx],
        s=10,
        alpha=0.5,
        color="#e67e22",
        label="General Calibrated TTA (PC1)",
    )
    ax3.scatter(
        y_true[sub_idx],
        pc1_dann_z[sub_idx],
        s=10,
        alpha=0.5,
        color="#2ecc71",
        label="Original LSTM-DANN (PC1)",
    )
    ax3.set_xlabel("True RUL (Cycles)", fontweight="bold")
    ax3.set_ylabel("Normalized 1st Principal Component (PC1)", fontweight="bold")
    ax3.set_title(
        f"C. Dominant Feature Axis (PC1) vs. True RUL\n"
        f"Correlation between TTA PC1 & DANN PC1: r = {r_pc1:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    ax3.legend(frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.35)

    # Panel D: RUL Predictions Comparison (TTA vs LSTM-DANN)
    r_pred = float(np.corrcoef(pred_dann, pred_tta)[0, 1])
    sc4 = ax4.scatter(
        pred_dann[sub_idx],
        pred_tta[sub_idx],
        c=y_true[sub_idx],
        cmap="viridis",
        s=14,
        alpha=0.75,
    )
    lims = [
        min(float(pred_dann.min()), float(pred_tta.min()), 0.0),
        max(float(pred_dann.max()), float(pred_tta.max()), 130.0),
    ]
    ax4.plot(lims, lims, "r--", linewidth=1.2, label="y = x (Identical Prediction)")
    ax4.set_xlabel(
        f"Original LSTM-DANN Predicted RUL (RMSE={metrics_summary['dann_test_rmse']:.2f})",
        fontweight="bold",
    )
    ax4.set_ylabel(
        f"General Calibrated TTA Predicted RUL (RMSE={metrics_summary['tta_test_rmse']:.2f})",
        fontweight="bold",
    )
    ax4.set_title(
        f"D. Prediction Agreement on Target Test Set\n"
        f"Pearson r(y_pred_DANN, y_pred_TTA) = {r_pred:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    ax4.legend(frameon=True, loc="upper left")
    ax4.grid(True, linestyle="--", alpha=0.35)
    plt.colorbar(sc4, ax=ax4, label="True RUL (Cycles)")

    fig2.suptitle(
        f"Feature Space Equivalence Analysis ({task_label}): General Calibrated TTA vs. Original LSTM-DANN",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig2.tight_layout()
    p2 = out_dir / "02_feature_similarity_metrics_dashboard.png"
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)
    saved.append(p2)

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare feature representations encoded by Original LSTM-DANN (joint adversarial training) "
            "vs. General Calibrated Offline TTA (source-only training + test-time subspace alignment) "
            "on C-MAPSS target test set (default: FD001 -> FD002)."
        )
    )
    parser.add_argument("--source-fd", default="FD001")
    parser.add_argument("--target-fd", default="FD002")
    parser.add_argument(
        "--dann-ckpt",
        default=None,
        help="Optional path to a pre-saved LSTM-DANN .pt checkpoint from DA-Replications",
    )
    parser.add_argument(
        "--realistic-root",
        default="result/benchmark_realistic",
        help="Path to result/benchmark_realistic directory",
    )
    parser.add_argument(
        "--last-window-only",
        action="store_true",
        help="Extract features only on the last window per engine (100/259 engines) instead of all test windows",
    )
    parser.add_argument(
        "-o",
        default="result/feature_comparison_fd001_to_fd002",
        help="Output directory for saved models, feature arrays, metrics JSON, and plots",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.o)
    out_dir.mkdir(parents=True, exist_ok=True)
    realistic_root = Path(args.realistic_root)

    # 1. Train or load Original LSTM-DANN (and save checkpoint)
    dann_model, dann_ckpt = train_or_load_original_lstm_dann(
        source_fd=args.source_fd,
        target_fd=args.target_fd,
        out_dir=out_dir,
        external_ckpt=args.dann_ckpt,
        seed=args.seed,
    )

    # 2. Train & adapt General Calibrated Offline TTA model (and save checkpoint)
    source_net, tta_net, tta_ckpt = train_and_adapt_general_calibrated_tta(
        source_fd=args.source_fd,
        target_fd=args.target_fd,
        realistic_root=realistic_root,
        out_dir=out_dir,
        model_type="lstm_tta",
        seed=args.seed,
    )

    # 3. Build deterministic Target Test DataLoader (shuffle=False)
    eval_cfg = {
        "dataset": {
            "name": "cmapss",
            "config": {
                "source_domain": False,
                "norm_mode": "minmax",
                "combine_splits_for_stats": True,
                "source_fd": args.source_fd,
                "target_fd": args.target_fd,
                "source_split": "train",
                "target_split": "test",
                "feature_cols": "all_24",
                "max_rul": 125.0,
                "normalize_rul": False,
                "window_size": 30,
                "last_window_only": args.last_window_only,
            },
        }
    }
    _, test_ds = get_datasets(eval_cfg)
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False)

    # 4. Extract feature vectors (N, 64) from all three encoders on the exact same test windows
    z_dann, pred_dann, y_true = extract_features_and_predictions(
        dann_model, test_dl, is_dann=True
    )
    z_tta, pred_tta, _ = extract_features_and_predictions(
        tta_net, test_dl, is_dann=False
    )
    z_src, pred_src, _ = extract_features_and_predictions(
        source_net, test_dl, is_dann=False
    )

    # Save raw extracted feature vectors
    feat_npz = out_dir / "extracted_target_test_features.npz"
    np.savez_compressed(
        feat_npz,
        z_dann=z_dann,
        z_tta=z_tta,
        z_source_unadapted=z_src,
        pred_dann=pred_dann,
        pred_tta=pred_tta,
        pred_source_unadapted=pred_src,
        y_true=y_true,
    )

    # 5. Compute Representation Similarity Metrics (CKA, CCA, RSA)
    lin_cka_tta_dann = compute_linear_cka(z_tta, z_dann)
    rbf_cka_tta_dann = compute_rbf_cka(z_tta, z_dann)
    lin_cka_src_dann = compute_linear_cka(z_src, z_dann)
    lin_cka_src_tta = compute_linear_cka(z_src, z_tta)

    cca_corrs = compute_cca_correlations(z_tta, z_dann, n_components=10)
    rsa_rho, dist_dann, dist_tta = compute_rsa_spearman(z_dann, z_tta)

    dann_rmse = float(np.sqrt(np.mean((pred_dann - y_true) ** 2)))
    tta_rmse = float(np.sqrt(np.mean((pred_tta - y_true) ** 2)))
    src_rmse = float(np.sqrt(np.mean((pred_src - y_true) ** 2)))

    metrics_summary = {
        "task": f"{args.source_fd}->{args.target_fd}",
        "n_test_samples": len(y_true),
        "feature_dim": int(z_tta.shape[1]),
        "dann_checkpoint": str(dann_ckpt),
        "tta_checkpoint": str(tta_ckpt),
        "unadapted_source_test_rmse": round(src_rmse, 4),
        "dann_test_rmse": round(dann_rmse, 4),
        "tta_test_rmse": round(tta_rmse, 4),
        "linear_cka_tta_vs_dann": round(lin_cka_tta_dann, 4),
        "rbf_cka_tta_vs_dann": round(rbf_cka_tta_dann, 4),
        "linear_cka_unadapted_vs_dann": round(lin_cka_src_dann, 4),
        "linear_cka_unadapted_vs_tta": round(lin_cka_src_tta, 4),
        "rsa_spearman_tta_vs_dann": round(rsa_rho, 4),
        "top_10_cca_correlations": [round(float(c), 4) for c in cca_corrs],
        "top_3_cca_mean": round(float(np.mean(cca_corrs[:3])), 4),
        "prediction_pearson_r_tta_vs_dann": round(
            float(np.corrcoef(pred_dann, pred_tta)[0, 1]), 4
        ),
    }

    summary_json = out_dir / "feature_similarity_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)

    # 6. Generate Plots
    plots = generate_feature_comparison_plots(
        z_dann=z_dann,
        z_tta=z_tta,
        z_src_unadapted=z_src,
        y_true=y_true,
        pred_dann=pred_dann,
        pred_tta=pred_tta,
        metrics_summary=metrics_summary,
        cca_corrs=cca_corrs,
        dist_dann=dist_dann,
        dist_tta=dist_tta,
        out_dir=out_dir,
        task_label=f"{args.source_fd} -> {args.target_fd}",
    )

    print("\n================ FEATURE COMPARISON SUMMARY ================")
    print(json.dumps(metrics_summary, indent=2))
    print(f"\nSaved extracted features to: {feat_npz}")
    print(f"Saved similarity metrics to: {summary_json}")
    for p in plots:
        print(f"Saved plot: {p}")


if __name__ == "__main__":
    main()
