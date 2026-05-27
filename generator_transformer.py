"""
Alpha Generator — Transformer 策略网络 + PPO 训练

架构设计：
- Transformer Encoder 替代 LSTM，更好地捕获长程依赖
- 位置编码：正弦余弦编码，适应RPN序列的顺序特性
- Causal Mask：确保生成时只能看到之前的token
- 同样的 Policy/Value Head 设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


class PositionalEncoding(nn.Module):
    """
    正弦余弦位置编码
    为RPN序列添加位置信息，因为表达式中token的顺序很重要
    """

    def __init__(self, d_model: int, max_len: int = 25, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数位置用sin，奇数位置用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 形状: (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder 块
    包含自注意力层和前馈网络
    """

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1
    ):
        super().__init__()

        # 自注意力层
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数
        self.activation = nn.ReLU()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (batch_size, seq_len, d_model)
            src_mask: 注意力掩码，防止看到未来信息
            src_key_padding_mask: 填充掩码
        """
        # 自注意力 + 残差连接
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络 + 残差连接
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class AlphaGenTransformer(nn.Module):
    """
    Transformer 版本的 Alpha 因子生成网络

    关键设计：
    1. Causal Mask：生成过程中只能看到之前的token，防止信息泄露
    2. 位置编码：RPN序列的顺序很重要，需要位置信息
    3. 更大的 embedding 维度：32→64，更好地表示token语义
    4. 多层 Transformer Encoder：捕获复杂的依赖关系
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,  # 增大embedding维度，更好地表示token
        nhead: int = 4,  # 注意力头数
        num_layers: int = 3,  # Transformer层数
        dim_feedforward: int = 256,  # 前馈网络维度
        dropout: float = 0.1,
        max_seq_len: int = 25,  # 最大序列长度
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 位置编码
        self.pos_encoder = PositionalEncoding(
            embed_dim, max_len=max_seq_len, dropout=dropout
        )

        # Transformer Encoder 层堆叠
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(
                TransformerEncoderBlock(
                    d_model=embed_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
            )
        self.transformer_encoder = nn.ModuleList(encoder_layers)

        # 策略头（输出动作logits）
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, vocab_size),
        )

        # 价值头（输出状态价值）
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化网络权重"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

        # 初始化策略头和价值头
        for module in [self.policy_head, self.value_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        生成因果掩码（上三角掩码）
        确保位置i只能注意到位置j<=i，防止看到未来信息

        Args:
            sz: 序列长度
        Returns:
            形状为 (sz, sz) 的布尔掩码，True表示需要忽略的位置
        """
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask

    def forward(
        self,
        token_ids: torch.Tensor,
        hidden: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # 保持接口兼容，但Transformer不需要
    ) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """
        单步前向传播（用于推理和生成）

        Args:
            token_ids: (batch_size,) 当前token的索引
            hidden: 保持接口兼容，Transformer中未使用
        Returns:
            logits: (batch_size, vocab_size)
            value: (batch_size,)
            hidden: None (Transformer不需要隐藏状态)
        """
        batch_size = token_ids.size(0)

        # 如果输入是单个token，需要扩展维度
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(1)  # (batch_size, 1)

        seq_len = token_ids.size(1)

        # 1. 嵌入 + 位置编码
        x = self.embedding(token_ids) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)

        # 2. 生成因果掩码（防止看到未来信息）
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(x.device)

        # 3. Transformer 编码
        for layer in self.transformer_encoder:
            x = layer(x, src_mask=causal_mask)

        # 4. 取最后一个时间步的输出
        # 对于单步生成，seq_len=1，直接取第一个位置
        if seq_len == 1:
            final_output = x[:, 0, :]
        else:
            final_output = x[:, -1, :]  # 取最后一个时间步

        # 5. 计算策略logits和状态价值
        logits = self.policy_head(final_output)
        value = self.value_head(final_output).squeeze(-1)

        return logits, value, None

    def forward_sequence(
        self, token_ids: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量前向传播（用于PPO更新）

        Args:
            token_ids: (batch_size, max_len) 填充后的token序列
            lengths: (batch_size,) 每条序列的实际长度
        Returns:
            all_logits: (batch_size, max_len, vocab_size)
            all_values: (batch_size, max_len)
        """
        batch_size, max_len = token_ids.shape

        # 1. 嵌入 + 位置编码
        x = self.embedding(token_ids) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)

        # 2. 创建填充掩码（标记哪些位置是填充的）
        padding_mask = torch.arange(max_len, device=lengths.device).expand(
            batch_size, max_len
        ) >= lengths.unsqueeze(1)

        # 3. 生成因果掩码
        causal_mask = self._generate_square_subsequent_mask(max_len).to(x.device)

        # 4. Transformer 编码
        for layer in self.transformer_encoder:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=padding_mask)

        # 5. 计算所有时间步的logits和values
        all_logits = self.policy_head(x)
        all_values = self.value_head(x).squeeze(-1)

        return all_logits, all_values

    def init_hidden(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> None:
        """
        保持接口兼容，Transformer不需要初始化隐藏状态
        """
        return None
