"""
氯离子时序数据集：按 (sample_id, depth_mm) 分组，滑动窗口构造 (seq_len, features) -> (pred_len,)。
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("engine.dataset")

DEFAULT_FEATURE_COLS = ["time_days", "depth_mm", "w_c_ratio", "concentration"]


class ChlorideDataset(Dataset):
    """按 (sample_id, depth_mm) 分组的时序滑动窗口数据集，特征归一化后返回 Tensor。"""

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 5,
        pred_len: int = 1,
        feature_cols: Optional[list[str]] = None,
        target_col: str = "concentration",
        scaler: Optional[MinMaxScaler] = None,
    ) -> None:
        """构建时序样本。

        Args:
            df: 必须含 sample_id, depth_mm, time_days 及 feature_cols 中各列。
            seq_len: 输入序列长度（时间步）。
            pred_len: 预测长度（输出时间步数）。
            feature_cols: 特征列名；None 时用 DEFAULT_FEATURE_COLS。
            target_col: 目标列名，必须在 feature_cols 中。
            scaler: 已拟合的 MinMaxScaler；None 时在本数据集上 fit。
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_cols = feature_cols or list(DEFAULT_FEATURE_COLS)
        self.target_col = target_col
        if target_col not in self.feature_cols:
            raise ValueError(f"target_col '{target_col}' must be in feature_cols {self.feature_cols}")
        self._target_idx = self.feature_cols.index(target_col)

        samples: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for (_sid, depth), grp in df.groupby(["sample_id", "depth_mm"]):
            grp = grp.sort_values("time_days")
            feat = grp[self.feature_cols].values.astype(np.float32)
            n = len(feat) - seq_len - pred_len + 1
            if n <= 0:
                continue
            for i in range(n):
                samples.append(feat[i : i + seq_len])
                targets.append(feat[i + seq_len : i + seq_len + pred_len, self._target_idx])

        if not samples:
            self.X = np.zeros((0, seq_len, len(self.feature_cols)), dtype=np.float32)
            self.Y = np.zeros((0, pred_len), dtype=np.float32)
            self.scaler = MinMaxScaler()
            logger.warning("ChlorideDataset: no samples produced (empty or too short series)")
        else:
            self.X = np.array(samples, dtype=np.float32)
            self.Y = np.array(targets, dtype=np.float32)
            n, s, f = self.X.shape
            if scaler is None:
                self.scaler = MinMaxScaler()
                flat = self.X.reshape(-1, f)
                self.X = self.scaler.fit_transform(flat).reshape(n, s, f).astype(np.float32)
            else:
                self.scaler = scaler
                flat = self.X.reshape(-1, f)
                self.X = self.scaler.transform(flat).reshape(n, s, f).astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])


def prepare_data(
    df: pd.DataFrame,
    seq_len: int = 5,
    pred_len: int = 1,
    test_ratio: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, MinMaxScaler]:
    """按 sample_id 划分训练/测试，构建 DataLoader 与统一 MinMaxScaler。

    Args:
        df: 含 sample_id, time_days, depth_mm 及特征列。
        seq_len: 输入序列长度。
        pred_len: 预测步数。
        test_ratio: 测试集所占比例（按试件数）。
        batch_size: 批大小。
        seed: 划分试件时的随机种子。

    Returns:
        (train_loader, test_loader, scaler)，scaler 在训练集上拟合。
    """
    rng = np.random.default_rng(seed)
    sample_ids = df["sample_id"].unique()
    n_test = max(1, int(len(sample_ids) * test_ratio))
    rng.shuffle(sample_ids)
    test_ids = set(sample_ids[:n_test])
    train_ids = set(sample_ids[n_test:])
    if not train_ids:
        train_ids, test_ids = set(sample_ids[: len(sample_ids) - n_test]), set(sample_ids[-n_test:])

    train_df = df[df["sample_id"].isin(train_ids)]
    test_df = df[df["sample_id"].isin(test_ids)]
    train_ds = ChlorideDataset(train_df, seq_len=seq_len, pred_len=pred_len)
    test_ds = ChlorideDataset(test_df, seq_len=seq_len, pred_len=pred_len, scaler=train_ds.scaler)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    logger.info(
        "prepare_data: train samples=%s, test samples=%s",
        len(train_ds),
        len(test_ds),
    )
    return train_loader, test_loader, train_ds.scaler
