"""
Alpha Generator — 兼容性包装器

此文件保留用于向后兼容，实际实现已分离到：
- lstm.py: AlphaGenNet (LSTM 策略网络)
- transformer.py: AlphaGenTransformer (Transformer 策略网络)
- episode.py: Episode 数据结构
- ppo.py: PPOAgent
- grpo.py: GRPOAgent
"""

# 向后兼容导入
from lstm import AlphaGenNet
from transformer import AlphaGenTransformer
from episode import Episode
from ppo import PPOAgent
from grpo import GRPOAgent

__all__ = ["AlphaGenNet", "AlphaGenTransformer", "Episode", "PPOAgent", "GRPOAgent"]
