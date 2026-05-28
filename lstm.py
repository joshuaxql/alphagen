"""
AlphaGen LSTM 策略网络

架构:
  - 2层 LSTM (hidden=128, dropout=0.1) 做序列编码
  - Policy head (MLP 2×64) 输出 token logits
  - Value head (MLP 2×64) 输出标量 value
"""

import torch
import torch.nn as nn

from tokens import VOCAB_SIZE


class AlphaGenNet(nn.Module):
    """LSTM 策略网络，用于生成 alpha 因子表达式。"""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        head_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, vocab_size),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, token_idx, hidden=None):
        """
        单步前向：
          token_idx: (batch,) 当前 token 索引
          hidden: (h, c) LSTM 隐状态
        返回: logits (batch, vocab_size), value (batch,), new_hidden
        """
        x = self.embedding(token_idx).unsqueeze(1)  # (batch, 1, embed)
        out, hidden = self.lstm(x, hidden)  # out: (batch, 1, hidden)
        h = out.squeeze(1)  # (batch, hidden)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value, hidden

    def forward_sequence(self, token_ids, lengths):
        """
        批量前向（PPO 更新时用）：
          token_ids: (batch, max_len) 输入序列（已 padding）
          lengths: (batch,) 每条序列实际长度
        返回: all_logits (batch, max_len, vocab_size), all_values (batch, max_len)
        """
        x = self.embedding(token_ids)  # (batch, max_len, embed)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        logits = self.policy_head(unpacked)
        values = self.value_head(unpacked).squeeze(-1)
        return logits, values

    def init_hidden(self, batch_size=1, device=None):
        """初始化 LSTM 隐状态。"""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)
