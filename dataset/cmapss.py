from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split

from dataset.dataset_config import CMAPSS_PATH

OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
ALL_FEATURE_COLS = OP_COLS + SENSOR_COLS
ALL_COLUMNS = ["unit", "cycle"] + ALL_FEATURE_COLS

DEFAULT_14_SENSOR_COLS = [
    "s2",
    "s3",
    "s4",
    "s7",
    "s8",
    "s9",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s17",
    "s20",
    "s21",
]


def _load_cmapss_split(
    data_dir: Path,
    fd_name: str,
    split: str = "train",
    max_rul: float | None = 125.0,
    normalize_rul: bool = False,
    units: Sequence[int] | None = None,
) -> pd.DataFrame:
    file_path = data_dir / f"{split}_{fd_name}.txt"
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=ALL_COLUMNS)

    max_cycles = df.groupby("unit")["cycle"].transform("max")
    cycles_from_end = max_cycles - df["cycle"]

    if split == "train":
        df["rul"] = cycles_from_end.astype(np.float32)
    elif split == "test":
        rul_path = data_dir / f"RUL_{fd_name}.txt"
        rul_end = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul_end"])
        rul_end.index = rul_end.index + 1
        end_rul = df["unit"].map(rul_end["rul_end"])
        df["rul"] = (cycles_from_end + end_rul).astype(np.float32)
    else:
        raise ValueError(f"Invalid split: {split!r}. Must be 'train' or 'test'.")

    if units is not None:
        df = df[df["unit"].isin(units)].reset_index(drop=True)

    if max_rul is not None:
        df["rul"] = np.minimum(df["rul"], float(max_rul))
        if normalize_rul:
            df["rul"] = df["rul"] / float(max_rul)

    return df


def _compute_combined_minmax(
    data_dir: Path,
    fd_name: str,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_df = pd.read_csv(
        data_dir / f"train_{fd_name}.txt", sep=r"\s+", header=None, names=ALL_COLUMNS
    )
    test_df = pd.read_csv(
        data_dir / f"test_{fd_name}.txt", sep=r"\s+", header=None, names=ALL_COLUMNS
    )
    values = pd.concat(
        [train_df[feature_cols], test_df[feature_cols]], axis=0
    ).to_numpy(dtype=np.float32)
    return values.min(axis=0), values.max(axis=0)


def _apply_minmax(
    values: np.ndarray,
    f_min: np.ndarray,
    f_max: np.ndarray,
) -> np.ndarray:
    span = f_max - f_min
    span = np.where(span == 0, 1.0, span)
    return (values - f_min) / span


def _build_windows_lstm_dann(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact window construction from DA-Replications/src/lstm_dann/Dataset."""
    if window_size <= 1:
        x = df[feature_cols].to_numpy(dtype=np.float32)
        y = df["rul"].to_numpy(dtype=np.float32)
        u = df["unit"].to_numpy(dtype=np.int64)
        return x, y, u

    w = window_size
    windows, targets, unit_numbers = [], [], []

    for unit_number, group in df.groupby("unit", sort=True):
        group = group.sort_values("cycle")
        features = group[feature_cols].to_numpy(dtype=np.float32)
        rul = group["rul"].to_numpy(dtype=np.float32)

        if len(features) <= w:
            pad_len = w + 1 - len(features)
            features = np.vstack(
                [np.zeros((pad_len, features.shape[1]), dtype=np.float32), features]
            )
            rul = np.concatenate([np.full(pad_len, rul[0], dtype=np.float32), rul])

        n_steps = len(features) - w
        for t in range(w, w + n_steps):
            windows.append(features[t - w : t])
            targets.append(rul[t])
            unit_numbers.append(unit_number)

    return (
        np.stack(windows),
        np.array(targets, dtype=np.float32),
        np.array(unit_numbers, dtype=np.int64),
    )


def split_by_engine(
    dataset: "CMAPSSDataset",
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Splits a CMAPSSDataset at the engine/unit level (from DA-Replications)."""
    unique_units = np.unique(dataset.unit_numbers)
    rng = np.random.RandomState(seed)
    shuffled_units = rng.permutation(unique_units)

    n_val = round(len(unique_units) * val_ratio)
    val_units = set(shuffled_units[:n_val])
    train_units = set(shuffled_units[n_val:])

    train_indices = [i for i, u in enumerate(dataset.unit_numbers) if u in train_units]
    val_indices = [i for i, u in enumerate(dataset.unit_numbers) if u in val_units]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


class CMAPSSDataset(Dataset):
    def __init__(
        self,
        source_domain: bool = True,
        standardize: bool = False,
        norm_mode: str = "minmax",
        combine_splits_for_stats: bool = True,
        source_fd: str = "FD001",
        target_fd: str = "FD002",
        source_split: str = "train",
        target_split: str = "train",
        source_units: list[int] | None = None,
        target_units: list[int] | None = None,
        feature_cols: list[str] | str | None = "all_24",
        max_rul: float | None = 125.0,
        normalize_rul: bool = False,
        window_size: int = 30,
        last_window_only: bool = False,
        data_dir: str | None = None,
    ):
        root = Path(data_dir or CMAPSS_PATH)
        if feature_cols == "all_24" or feature_cols is None:
            cols = ALL_FEATURE_COLS
        elif feature_cols == "sensors_14":
            cols = DEFAULT_14_SENSOR_COLS
        else:
            cols = list(feature_cols)

        source_df = _load_cmapss_split(
            root,
            fd_name=source_fd,
            split=source_split,
            max_rul=max_rul,
            normalize_rul=normalize_rul,
            units=source_units,
        )
        target_df = _load_cmapss_split(
            root,
            fd_name=target_fd,
            split=target_split,
            max_rul=max_rul,
            normalize_rul=normalize_rul,
            units=target_units,
        )

        if norm_mode == "minmax":
            if combine_splits_for_stats:
                s_min, s_max = _compute_combined_minmax(root, source_fd, cols)
                t_min, t_max = _compute_combined_minmax(root, target_fd, cols)
            else:
                s_vals = source_df[cols].to_numpy(dtype=np.float32)
                s_min, s_max = s_vals.min(axis=0), s_vals.max(axis=0)
                t_vals = target_df[cols].to_numpy(dtype=np.float32)
                t_min, t_max = t_vals.min(axis=0), t_vals.max(axis=0)

            source_df[cols] = _apply_minmax(
                source_df[cols].to_numpy(dtype=np.float32), s_min, s_max
            )
            target_df[cols] = _apply_minmax(
                target_df[cols].to_numpy(dtype=np.float32), t_min, t_max
            )
        elif norm_mode == "standardize" or standardize:
            source_mean = source_df[cols].mean()
            source_std = source_df[cols].std(ddof=0)

            source_df[cols] = (source_df[cols] - source_mean) / (source_std + 1e-8)
            target_df[cols] = (target_df[cols] - source_mean) / (source_std + 1e-8)

        active_df = source_df if source_domain else target_df
        self.data, self.labels, self.unit_numbers = _build_windows_lstm_dann(
            active_df, cols, window_size
        )

        if last_window_only:
            _, last_pos_reversed = np.unique(self.unit_numbers[::-1], return_index=True)
            last_indices = sorted(len(self.unit_numbers) - 1 - last_pos_reversed)
            self.data = self.data[last_indices]
            self.labels = self.labels[last_indices]
            self.unit_numbers = self.unit_numbers[last_indices]

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.data[i]), torch.tensor(self.labels[i])


def get_cmapss(config: dict) -> tuple[Dataset, Dataset]:
    ds = CMAPSSDataset(**config["dataset"]["config"])

    if r := config["dataset"].get("train_ratio"):
        if config["dataset"].get("split_by_engine", True):
            train_ds, val_ds = split_by_engine(ds, val_ratio=1.0 - r, seed=42)
        else:
            train_num = int(len(ds) * r)
            val_num = len(ds) - train_num
            train_ds, val_ds = random_split(ds, [train_num, val_num])
    else:
        train_ds = ds
        val_ds = ds

    return train_ds, val_ds
