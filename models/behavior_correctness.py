"""
行为正确性识别模型：LSTM + 注意力 + 多任务输出 (correct/incorrect + reason)
"""

import math
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

    改进: 双向 LSTM + 自注意力 (残差连接)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        lstm_hidden = hidden_dim * 2 if bidirectional else hidden_dim

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(dropout)
        self.fc_correct = nn.Linear(lstm_hidden, 2)
        self.fc_reason = nn.Linear(lstm_hidden, NUM_REASONS)
        if use_attention:
            self.attn_scale = 1.0 / math.sqrt(lstm_hidden)

    def _self_attention(self, out: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        """Last-position query attention + residual. out: (B, T, D) -> (B, D)"""
        B, T, D = out.shape
        scale = self.attn_scale if self.use_attention else 1.0
        scores = torch.bmm(out, out.transpose(1, 2)) * scale  # (B, T, T)
        if lengths is not None:
            mask = torch.arange(T, device=out.device)[None, :] >= lengths[:, None]
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(weights, out)  # (B, T, D)
        return ctx[:, -1, :]  # (B, D)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, input_dim)
        lengths: (batch,) 有效长度，None 时假设无 padding
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            last_idx = (lengths - 1).clamp(min=0).long()
            last_out = out[torch.arange(out.size(0), device=out.device), last_idx]
        else:
            out, _ = self.lstm(x)
            last_out = out[:, -1, :]

        if self.use_attention:
            attn_out = self._self_attention(out, lengths)
            last_out = last_out + attn_out  # 残差

        last_out = self.drop(last_out)
        correct_logits = self.fc_correct(last_out)
        reason_logits = self.fc_reason(last_out)
        return correct_logits, reason_logits
