"""
氯离子浓度预测模型：Attention-LSTM、LSTM、GRU、BP。
"""

from typing import Type

import torch
import torch.nn as nn


class AttentionLSTM(nn.Module):
    """带 Self-Attention 的 LSTM。"""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        pred_len: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input (B, T, F), output (B, pred_len)."""
        lstm_out, _ = self.lstm(x)
        attn_w = torch.softmax(self.attention(lstm_out), dim=1)
        self._last_attn_weights = attn_w  # for testing: (B, T, 1)
        context = torch.sum(attn_w * lstm_out, dim=1)
        return self.fc(context)


class SimpleLSTM(nn.Module):
    """标准 LSTM。"""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        pred_len: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input (B, T, F), output (B, pred_len)."""
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    """GRU 模型。"""

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_layers: int = 2,
        pred_len: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input (B, T, F), output (B, pred_len)."""
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class BPNet(nn.Module):
    """全连接 BP 网络，输入展平为 (B, input_size * seq_len)。"""

    def __init__(
        self,
        input_size: int = 4,
        seq_len: int = 5,
        pred_len: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        flat_size = input_size * seq_len
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input (B, seq_len, input_size), output (B, pred_len)."""
        return self.net(x)


MODEL_REGISTRY: dict[str, Type[nn.Module]] = {
    "attention_lstm": AttentionLSTM,
    "simple_lstm": SimpleLSTM,
    "gru": GRUModel,
    "bp": BPNet,
}
