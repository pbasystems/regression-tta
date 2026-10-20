from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Values extracted from the LSTM-DANN paper table (Mean +/- Std across 10 trials)
LSTM_DANN_PAPER_RMSE: dict[str, tuple[float, float]] = {
    "FD001->FD002": (48.62, 6.83),
    "FD001->FD003": (45.87, 3.58),
    "FD001->FD004": (43.82, 4.15),
    "FD002->FD001": (28.10, 5.03),
    "FD002->FD003": (37.46, 1.54),
    "FD002->FD004": (31.85, 1.65),
    "FD003->FD001": (31.74, 9.37),
    "FD003->FD002": (44.62, 1.21),
    "FD003->FD004": (47.94, 5.78),
    "FD004->FD001": (31.54, 2.42),
    "FD004->FD002": (24.93, 1.82),
    "FD004->FD003": (27.84, 2.69),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate separate RMSE & MAE plots for each evaluated method "
            "(with Source-Only, LSTM-DANN Paper, and Target-Only references) "
            "plus a combined all-methods comparison plot."
        )
    )
    parser.add_argument(
        "--csv",
        default="result/result/benchmark_all/summary_all_metrics.csv",
        help="Path to summary_all_metrics.csv",
    )
    parser.add_argument(
        "--paper-csv",
        default="result/result/benchmark_all/lstm_dann_paper_results.csv",
        help="Path to lstm_dann_paper_results.csv",
    )
    parser.add_argument(
        "-o",
        default="result/result/benchmark_all/plots",
        help="Output directory for generated plot images",
    )
    return parser.parse_args()


def _annotate_bars(
    ax: plt.Axes,
    bars,
    fmt: str = "{:.1f}",
    fontsize: int = 6.5,
    rotation: int = 0,
) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 2.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
        )


def _draw_grouped_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    spec: list[tuple[str, str, str]],
    title: str,
    ylabel: str,
    show_xlabel: bool = True,
    annotate: bool = True,
    annotate_rotation: int = 0,
    annotate_fontsize: int = 6.5,
) -> None:
    tasks = df["Task"].tolist()
    x = np.arange(len(tasks))
    n_series = len(spec)
    total_group_width = 0.84 if n_series > 5 else 0.78
    bar_width = total_group_width / n_series
    offsets = np.linspace(
        -total_group_width / 2.0 + bar_width / 2.0,
        total_group_width / 2.0 - bar_width / 2.0,
        n_series,
    )

    for offset, (col, label, color) in zip(offsets, spec):
        vals = df[col].to_numpy(dtype=float)
        mean_val = float(np.mean(vals))
        bars = ax.bar(
            x + offset,
            vals,
            width=bar_width * 0.90,
            label=f"{label} (Mean: {mean_val:.2f})",
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.92,
        )
        if annotate:
            _annotate_bars(
                ax,
                bars,
                fmt="{:.1f}",
                fontsize=annotate_fontsize,
                rotation=annotate_rotation,
            )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=11)
    if show_xlabel:
        ax.set_xlabel(
            "Cross-Domain Task (Source -> Target)", fontsize=11, fontweight="bold"
        )
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    max_y = float(df[[c for c, _, _ in spec]].to_numpy().max())
    ax.set_ylim(0, max_y * (1.22 if annotate_rotation else 1.16))
    ncol = 2 if n_series > 4 else 1
    ax.legend(frameon=True, fontsize=9, loc="upper left", ncol=ncol)


def save_method_plots(
    df: pd.DataFrame,
    slug: str,
    method_title: str,
    rmse_col: str,
    mae_col: str,
    method_color: str,
    out_dir: Path,
) -> list[Path]:
    rmse_spec = [
        ("Source_Only_RMSE", "Source Only (Lower Bound)", "#e74c3c"),
        (rmse_col, method_title, method_color),
        ("LSTM_DANN_Paper_RMSE", "LSTM-DANN Paper", "#f39c12"),
        ("Target_Only_RMSE", "Target Only (Oracle Upper Bound)", "#2ecc71"),
    ]
    mae_spec = [
        ("Source_Only_MAE", "Source Only (Lower Bound)", "#e74c3c"),
        (mae_col, method_title, method_color),
        ("Target_Only_MAE", "Target Only (Oracle Upper Bound)", "#2ecc71"),
    ]

    saved: list[Path] = []

    # 1. Combined 2-row figure (RMSE top, MAE bottom)
    combined_path = out_dir / f"{slug}_rmse_and_mae.png"
    fig, (ax_rmse, ax_mae) = plt.subplots(
        2, 1, figsize=(16, 11.5), dpi=200, sharex=True
    )
    _draw_grouped_bars(
        ax_rmse,
        df,
        rmse_spec,
        title=f"RMSE: {method_title} vs. Bounds & LSTM-DANN Paper",
        ylabel="RMSE (Cycles)",
        show_xlabel=False,
    )
    _draw_grouped_bars(
        ax_mae,
        df,
        mae_spec,
        title=f"MAE: {method_title} vs. Bounds",
        ylabel="MAE (Cycles)",
        show_xlabel=True,
    )
    fig.tight_layout()
    fig.savefig(combined_path, bbox_inches="tight")
    plt.close(fig)
    saved.append(combined_path)

    # 2. Standalone RMSE figure
    rmse_path = out_dir / f"{slug}_rmse.png"
    fig_r, ax_r = plt.subplots(figsize=(16, 6.2), dpi=200)
    _draw_grouped_bars(
        ax_r,
        df,
        rmse_spec,
        title=f"RMSE: {method_title} vs. Bounds & LSTM-DANN Paper",
        ylabel="RMSE (Cycles)",
        show_xlabel=True,
    )
    fig_r.tight_layout()
    fig_r.savefig(rmse_path, bbox_inches="tight")
    plt.close(fig_r)
    saved.append(rmse_path)

    # 3. Standalone MAE figure
    mae_path = out_dir / f"{slug}_mae.png"
    fig_m, ax_m = plt.subplots(figsize=(16, 6.2), dpi=200)
    _draw_grouped_bars(
        ax_m,
        df,
        mae_spec,
        title=f"MAE: {method_title} vs. Bounds",
        ylabel="MAE (Cycles)",
        show_xlabel=True,
    )
    fig_m.tight_layout()
    fig_m.savefig(mae_path, bbox_inches="tight")
    plt.close(fig_m)
    saved.append(mae_path)

    return saved


def save_all_methods_combined_plots(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    all_rmse_spec = [
        ("Source_Only_RMSE", "Source Only (Lower Bound)", "#e74c3c"),
        ("Source_Val_Tuned_Unadapted_RMSE", "Source-Val Tuned (Unadapted)", "#9b59b6"),
        (
            "Source_Val_Tuned_Default_TTA_Offline_RMSE",
            "Source-Val Tuned + Default TTA",
            "#3498db",
        ),
        (
            "TTA_Unsupervised_KL_Offline_RMSE",
            "TTA (unsupervised_kl) Offline",
            "#1abc9c",
        ),
        (
            "TTA_Online_RMSE_Objective_Online_RMSE",
            "TTA (online_rmse) Online",
            "#e67e22",
        ),
        (
            "TTA_Online_RMSE_Objective_Offline_RMSE",
            "TTA (online_rmse) Offline",
            "#d35400",
        ),
        (
            "TTA_Offline_RMSE_Objective_Offline_RMSE",
            "TTA (offline_rmse) Offline",
            "#2980b9",
        ),
        ("LSTM_DANN_Paper_RMSE", "LSTM-DANN Paper", "#f1c40f"),
        ("Target_Only_RMSE", "Target Only (Oracle Upper Bound)", "#2ecc71"),
    ]

    all_mae_spec = [
        ("Source_Only_MAE", "Source Only (Lower Bound)", "#e74c3c"),
        ("Source_Val_Tuned_Unadapted_MAE", "Source-Val Tuned (Unadapted)", "#9b59b6"),
        (
            "Source_Val_Tuned_Default_TTA_Offline_MAE",
            "Source-Val Tuned + Default TTA",
            "#3498db",
        ),
        (
            "TTA_Unsupervised_KL_Offline_MAE",
            "TTA (unsupervised_kl) Offline",
            "#1abc9c",
        ),
        (
            "TTA_Online_RMSE_Objective_Online_MAE",
            "TTA (online_rmse) Online",
            "#e67e22",
        ),
        (
            "TTA_Online_RMSE_Objective_Offline_MAE",
            "TTA (online_rmse) Offline",
            "#d35400",
        ),
        (
            "TTA_Offline_RMSE_Objective_Offline_MAE",
            "TTA (offline_rmse) Offline",
            "#2980b9",
        ),
        ("Target_Only_MAE", "Target Only (Oracle Upper Bound)", "#2ecc71"),
    ]

    saved: list[Path] = []

    # 1. Combined 2-row figure (All methods RMSE + MAE)
    combined_all_path = out_dir / "all_methods_combined_rmse_and_mae.png"
    fig, (ax_rmse, ax_mae) = plt.subplots(
        2, 1, figsize=(20, 13.5), dpi=200, sharex=True
    )
    _draw_grouped_bars(
        ax_rmse,
        df,
        all_rmse_spec,
        title="All Methods Comparison: RMSE Across CMAPSS Cross-Domain Tasks",
        ylabel="RMSE (Cycles)",
        show_xlabel=False,
        annotate=True,
        annotate_rotation=90,
        annotate_fontsize=5.5,
    )
    _draw_grouped_bars(
        ax_mae,
        df,
        all_mae_spec,
        title="All Methods Comparison: MAE Across CMAPSS Cross-Domain Tasks",
        ylabel="MAE (Cycles)",
        show_xlabel=True,
        annotate=True,
        annotate_rotation=90,
        annotate_fontsize=5.5,
    )
    fig.tight_layout()
    fig.savefig(combined_all_path, bbox_inches="tight")
    plt.close(fig)
    saved.append(combined_all_path)

    # 2. Standalone All-Methods RMSE
    rmse_all_path = out_dir / "all_methods_combined_rmse.png"
    fig_r, ax_r = plt.subplots(figsize=(20, 7.2), dpi=200)
    _draw_grouped_bars(
        ax_r,
        df,
        all_rmse_spec,
        title="All Methods Comparison: RMSE Across CMAPSS Cross-Domain Tasks",
        ylabel="RMSE (Cycles)",
        show_xlabel=True,
        annotate=True,
        annotate_rotation=90,
        annotate_fontsize=6,
    )
    fig_r.tight_layout()
    fig_r.savefig(rmse_all_path, bbox_inches="tight")
    plt.close(fig_r)
    saved.append(rmse_all_path)

    # 3. Standalone All-Methods MAE
    mae_all_path = out_dir / "all_methods_combined_mae.png"
    fig_m, ax_m = plt.subplots(figsize=(20, 7.2), dpi=200)
    _draw_grouped_bars(
        ax_m,
        df,
        all_mae_spec,
        title="All Methods Comparison: MAE Across CMAPSS Cross-Domain Tasks",
        ylabel="MAE (Cycles)",
        show_xlabel=True,
        annotate=True,
        annotate_rotation=90,
        annotate_fontsize=6,
    )
    fig_m.tight_layout()
    fig_m.savefig(mae_all_path, bbox_inches="tight")
    plt.close(fig_m)
    saved.append(mae_all_path)

    return saved


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    paper_csv = Path(args.paper_csv)
    if paper_csv.is_file():
        paper_df = pd.read_csv(paper_csv)[
            ["Task", "LSTM_DANN_Paper_RMSE", "LSTM_DANN_Paper_RMSE_Std"]
        ]
        df = df.merge(paper_df, on="Task", how="left")
    else:
        df["LSTM_DANN_Paper_RMSE"] = df["Task"].map(
            lambda t: LSTM_DANN_PAPER_RMSE[t][0]
        )
        df["LSTM_DANN_Paper_RMSE_Std"] = df["Task"].map(
            lambda t: LSTM_DANN_PAPER_RMSE[t][1]
        )

    out_dir = Path(args.o)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        (
            "01_source_val_tuned_unadapted",
            "Source-Val Tuned (Unadapted)",
            "Source_Val_Tuned_Unadapted_RMSE",
            "Source_Val_Tuned_Unadapted_MAE",
            "#9b59b6",
        ),
        (
            "02_source_val_tuned_default_tta",
            "Source-Val Tuned + Default TTA",
            "Source_Val_Tuned_Default_TTA_Offline_RMSE",
            "Source_Val_Tuned_Default_TTA_Offline_MAE",
            "#3498db",
        ),
        (
            "03_tta_unsupervised_kl_offline",
            "TTA (unsupervised_kl) Offline",
            "TTA_Unsupervised_KL_Offline_RMSE",
            "TTA_Unsupervised_KL_Offline_MAE",
            "#1abc9c",
        ),
        (
            "04_tta_online_rmse_online",
            "TTA (online_rmse) Online",
            "TTA_Online_RMSE_Objective_Online_RMSE",
            "TTA_Online_RMSE_Objective_Online_MAE",
            "#e67e22",
        ),
        (
            "05_tta_online_rmse_offline",
            "TTA (online_rmse) Offline",
            "TTA_Online_RMSE_Objective_Offline_RMSE",
            "TTA_Online_RMSE_Objective_Offline_MAE",
            "#d35400",
        ),
        (
            "06_tta_offline_rmse_offline",
            "TTA (offline_rmse) Offline",
            "TTA_Offline_RMSE_Objective_Offline_RMSE",
            "TTA_Offline_RMSE_Objective_Offline_MAE",
            "#2980b9",
        ),
    ]

    all_saved: list[Path] = []
    for slug, title, r_col, m_col, color in methods:
        all_saved.extend(
            save_method_plots(df, slug, title, r_col, m_col, color, out_dir)
        )

    all_saved.extend(save_all_methods_combined_plots(df, out_dir))

    for p in all_saved:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
