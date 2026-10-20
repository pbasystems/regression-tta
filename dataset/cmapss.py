from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, random_split

from dataset.dataset_config import CMAPSS_PATH

ALL_COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

DEFAULT_SENSOR_COLS = [
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

    if split == "train":
        max_cycles = df.groupby("unit")["cycle"].transform("max")
        df["rul"] = (max_cycles - df["cycle"]).astype(np.float32)
    elif split == "test":
        rul_path = data_dir / f"RUL_{fd_name}.txt"
        rul_end = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul_end"])
        rul_end["unit"] = np.arange(1, len(rul_end) + 1)
        df = df.merge(rul_end, on="unit", how="left")
        max_cycles = df.groupby("unit")["cycle"].transform("max")
        df["rul"] = (max_cycles - df["cycle"] + df["rul_end"]).astype(np.float32)
        df = df.drop(columns=["rul_end"])
    else:
        raise ValueError(f"Invalid split: {split!r}. Must be 'train' or 'test'.")

    if units is not None:
        df = df[df["unit"].isin(units)].reset_index(drop=True)

    if max_rul is not None:
        df["rul"] = np.minimum(df["rul"], float(max_rul))
        if normalize_rul:
            df["rul"] = df["rul"] / float(max_rul)

    return df


def _build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if window_size <= 1:
        x = df[feature_cols].to_numpy(dtype=np.float32)
        y = df["rul"].to_numpy(dtype=np.float32)
        return x, y

    x_list = []
    y_list = []
    for _, group in df.groupby("unit", sort=False):
        feats = group[feature_cols].to_numpy(dtype=np.float32)
        ruls = group["rul"].to_numpy(dtype=np.float32)
        if len(feats) < window_size:
            pad_len = window_size - len(feats)
            feats = np.pad(feats, ((pad_len, 0), (0, 0)), mode="edge")
            ruls = np.pad(ruls, (pad_len, 0), mode="edge")
        for i in range(len(feats) - window_size + 1):
            x_list.append(feats[i : i + window_size])
            y_list.append(ruls[i + window_size - 1])

    return np.stack(x_list, axis=0), np.array(y_list, dtype=np.float32)


class CMAPSSDataset(Dataset):
    def __init__(
        self,
        source_domain: bool = True,
        standardize: bool = True,
        source_fd: str = "FD001",
        target_fd: str = "FD003",
        source_split: str = "train",
        target_split: str = "train",
        source_units: list[int] | None = None,
        target_units: list[int] | None = None,
        feature_cols: list[str] | None = None,
        max_rul: float | None = 125.0,
        normalize_rul: bool = False,
        window_size: int = 1,
        data_dir: str | None = None,
    ):
        root = Path(data_dir or CMAPSS_PATH)
        cols = feature_cols if feature_cols is not None else DEFAULT_SENSOR_COLS

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

        if standardize:
            source_mean = source_df[cols].mean()
            source_std = source_df[cols].std(ddof=0)

            source_df[cols] = (source_df[cols] - source_mean) / (source_std + 1e-8)
            target_df[cols] = (target_df[cols] - source_mean) / (source_std + 1e-8)

        active_df = source_df if source_domain else target_df
        self.data, self.labels = _build_windows(active_df, cols, window_size)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, i: int) -> tuple[np.ndarray, float]:
        return self.data[i], float(self.labels[i])


def get_cmapss(config: dict) -> tuple[Dataset, Dataset]:
    ds = CMAPSSDataset(**config["dataset"]["config"])

    if r := config["dataset"].get("train_ratio"):
        train_num = int(len(ds) * r)
        val_num = len(ds) - train_num
        train_ds, val_ds = random_split(ds, [train_num, val_num])
    else:
        train_ds = ds
        val_ds = ds

    return train_ds, val_ds
