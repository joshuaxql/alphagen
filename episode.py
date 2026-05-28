"""
Episode 数据结构

存储 RL 训练过程中的单条轨迹数据。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Episode:
    """单条 RL 轨迹数据。"""

    token_ids: List[int] = field(default_factory=list)  # 完整序列含 BEG
    actions: List[int] = field(default_factory=list)  # tokens[1:]
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    reward: float = 0.0
