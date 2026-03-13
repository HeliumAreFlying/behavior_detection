"""
行为正确性识别模型：LSTM + 多任务输出 (correct/incorrect + reason)
"""

import torch
import torch.nn as nn


# 与 data_generator 一致的 reason 类别
REASON_NAMES = [
    "ate_x2_then_food",
    "ate_food_no_x2",
    "in_progress",
    "self_collision",
    "snake_collision",
    "x2_wasted",
    "timeout",
]
REASON_TO_IDX = {r: i for i, r in enumerate(REASON_NAMES)}
NUM_REASONS = len(REASON_NAMES)


class BehaviorCorrectnessModel(nn.Module):
    """
    输入: (batch, seq_len, input_dim) 蛇头等时序特征
    输出: correct (batch, 2), reason (batch, num_reasons)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc_correct = nn.Linear(hidden_dim, 2)
        self.fc_reason = nn.Linear(hidden_dim, NUM_REASONS)
        self.fc_endpoint = nn.Linear(hidden_dim, 1)  # 端点检测：是否应在此刻输出结论

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, input_dim)
        lengths: (batch,) 有效长度，None 时假设无 padding
        """
        if lengths is not None:
            # pack 以便 LSTM 正确处理变长序列
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # 取每个序列最后有效时刻
            last_idx = (lengths - 1).clamp(min=0).long()
            last_out = out[torch.arange(out.size(0), device=out.device), last_idx]
        else:
            out, _ = self.lstm(x)
            last_out = out[:, -1, :]

        last_out = self.drop(last_out)
        correct_logits = self.fc_correct(last_out)
        reason_logits = self.fc_reason(last_out)
        endpoint_logits = self.fc_endpoint(last_out)  # (batch, 1)
        return correct_logits, reason_logits, endpoint_logits
