"""
训练流程：加载数据、构建模型、训练循环、评估指标。
支持内存训练（不写磁盘），用于 Streamlit 演示。
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from engine.dataset import prepare_data
from engine.fick import generate_mock_data
from engine.models import MODEL_REGISTRY

logger = logging.getLogger("engine.trainer")

INPUT_SIZE = 4


@dataclass
class TrainConfig:
    """训练配置。"""

    data_path: str  # CSV 文件路径或 "mock"
    model_type: str  # "attention_lstm" | "simple_lstm" | "gru" | "bp"
    seq_len: int = 5
    pred_len: int = 1
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    epochs: int = 200
    learning_rate: float = 0.001
    batch_size: int = 32
    test_ratio: float = 0.2
    save_dir: str = "saved_models"


@dataclass
class TrainResult:
    """训练结果（写盘版本）。"""

    model_path: str
    scaler_path: str
    metrics: dict
    loss_history: list
    training_time_seconds: float


def _load_data(config: TrainConfig) -> pd.DataFrame:
    """根据 data_path 加载 DataFrame。"""
    if config.data_path.strip().lower() == "mock":
        logger.info("Using mock data from generate_mock_data()")
        return generate_mock_data(n_samples=8, noise_std=0.015, seed=42)
    path = Path(config.data_path)
    if not path.is_file():
        raise FileNotFoundError(f"data_path not found: {config.data_path}")
    return pd.read_csv(path)


def _build_model(config: TrainConfig) -> nn.Module:
    """根据 model_type 构建模型。"""
    if config.model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {config.model_type}, choose from {list(MODEL_REGISTRY.keys())}")
    cls = MODEL_REGISTRY[config.model_type]
    if config.model_type == "bp":
        return cls(input_size=INPUT_SIZE, seq_len=config.seq_len, pred_len=config.pred_len)
    return cls(
        input_size=INPUT_SIZE,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        pred_len=config.pred_len,
        dropout=config.dropout,
    )


def _compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """计算 MAE, RMSE, MAPE, R²。"""
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mask = np.abs(y_true) > 1e-9
    mape = float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100) if mask.any() else 0.0
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-10))
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def _get_meta(config: TrainConfig) -> dict:
    """构建模型 meta 字典。"""
    return {
        "model_type": config.model_type,
        "seq_len": config.seq_len,
        "pred_len": config.pred_len,
        "input_size": INPUT_SIZE,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
    }


def train_in_memory(
    config: TrainConfig,
    df: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> tuple[dict, object, dict, dict, list, float]:
    """在内存中训练，不写磁盘。用于 Streamlit 等无持久存储环境。

    Args:
        config: 训练配置。
        df: 可选，直接传入 DataFrame；为 None 时根据 config.data_path 加载。
        progress_callback: 可选，每 epoch 结束时调用 (current_epoch, total_epochs, current_loss)。

    Returns:
        (state_dict, scaler, meta, metrics, loss_history, training_time_seconds)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if df is None:
        df = _load_data(config)
    train_loader, test_loader, scaler = prepare_data(
        df,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        seed=42,
    )
    model = _build_model(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    loss_history: list = []
    t0 = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        scheduler.step()
        if progress_callback is not None:
            progress_callback(epoch, config.epochs, avg_loss)

    elapsed = time.perf_counter() - t0

    model.eval()
    all_pred: list = []
    all_true: list = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            all_pred.append(pred)
            all_true.append(yb.numpy())
    y_pred = np.concatenate(all_pred, axis=0).flatten()
    y_true = np.concatenate(all_true, axis=0).flatten()
    metrics = _compute_metrics(y_pred, y_true)
    meta = _get_meta(config)

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    logger.info("train_in_memory finished: metrics=%s", metrics)
    return state_dict, scaler, meta, metrics, loss_history, elapsed


def train(
    config: TrainConfig,
    df: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> TrainResult:
    """执行训练并保存到磁盘（可选使用）。

    Args:
        config: 训练配置。
        df: 可选，直接传入 DataFrame。
        progress_callback: 可选，每 epoch 结束时调用。

    Returns:
        TrainResult 含 model_path, scaler_path, metrics, loss_history, training_time_seconds。
    """
    state_dict, scaler, meta, metrics, loss_history, elapsed = train_in_memory(
        config, df=df, progress_callback=progress_callback
    )
    import pickle
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "model.pth"
    scaler_path = save_dir / "scaler.pkl"
    meta_path = save_dir / "model_meta.pkl"
    torch.save(state_dict, model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    return TrainResult(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        metrics=metrics,
        loss_history=loss_history,
        training_time_seconds=elapsed,
    )
