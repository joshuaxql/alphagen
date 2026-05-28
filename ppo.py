"""
PPO Agent

Proximal Policy Optimization (PPO) 实现，用于训练 alpha 因子生成器。
使用 Monte Carlo 风格回报计算优势估计。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List

from lstm import AlphaGenNet
from episode import Episode


class PPOAgent:
    """PPO 强化学习 Agent。"""

    def __init__(
        self,
        net: AlphaGenNet,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
        gamma: float = 0.99,
    ):
        self.net = net.to(device)
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma

    @torch.no_grad()
    def select_action(self, token_idx: int, valid_mask: np.ndarray, hidden):
        """
        单步动作选择。
        返回: action_idx, log_prob, value, new_hidden
        """
        self.net.eval()
        t = torch.tensor([token_idx], device=self.device)
        logits, value, hidden = self.net(t, hidden)

        mask_t = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
        logits[0, ~mask_t] = float("-inf")

        dist = Categorical(logits=logits[0])
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item(), hidden

    def _compute_episode_advantages(
        self, ep_rewards: np.ndarray, ep_values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """为单个 episode 计算 returns 与 advantages（Monte Carlo 模式）。"""
        n = len(ep_values)
        if n == 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty

        # Monte Carlo: episode 中每个 step 共享最终 reward
        final_reward = float(ep_rewards[-1])
        returns_ep = np.full(n, final_reward, dtype=np.float32)
        advantages_ep = returns_ep - ep_values

        if n > 1:
            adv_std = advantages_ep.std()
            if adv_std > 1e-8:
                advantages_ep = (advantages_ep - advantages_ep.mean()) / adv_std

        return returns_ep.astype(np.float32), advantages_ep.astype(np.float32)

    def update(self, episodes: List[Episode]):
        """PPO 多 epoch 更新（使用 Monte Carlo 风格回报）。"""
        if not episodes:
            return {}

        self.net.train()

        # ── 按 episode 分组收集数据 ──
        # 每步的 reward: 最后一步 = episode.reward, 其余 = 0
        all_input_ids: list[list[int]] = []
        all_actions: list[int] = []
        all_old_logprobs: list[float] = []
        all_masks_np: list[np.ndarray] = []
        all_rewards: list[float] = []
        episode_boundaries: list[tuple[int, int]] = []  # (start, end) per episode

        offset = 0
        for ep in episodes:
            n_steps = len(ep.actions)
            if n_steps == 0:
                continue
            for i in range(n_steps):
                all_input_ids.append(ep.token_ids[: i + 1])
                all_actions.append(ep.actions[i])
                all_old_logprobs.append(ep.log_probs[i])
                all_masks_np.append(ep.masks[i])
            # reward: 中间步=0, 最后一步=episode reward
            all_rewards.extend([0.0] * (n_steps - 1) + [ep.reward])
            episode_boundaries.append((offset, offset + n_steps))
            offset += n_steps

        n_total = len(all_actions)
        if n_total == 0:
            return {}

        # ── 为批量 LSTM 做 padding ──
        max_seq = max(len(s) for s in all_input_ids)
        padded = torch.zeros(n_total, max_seq, dtype=torch.long, device=self.device)
        lengths = torch.zeros(n_total, dtype=torch.long, device=self.device)
        for i, seq in enumerate(all_input_ids):
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            lengths[i] = len(seq)

        # ── 前向传播获取所有 step 的 value（no_grad 用于 advantage 计算）──
        with torch.no_grad():
            _, all_values_out = self.net.forward_sequence(padded, lengths)
            last_idx_full = (lengths - 1).long().to(self.device)
            batch_idx_full = torch.arange(n_total, device=self.device)
            step_values = all_values_out[batch_idx_full, last_idx_full].cpu().numpy()

        # ── 按 episode 计算 returns / advantages ──
        all_advantages = np.zeros(n_total, dtype=np.float32)
        all_returns_np = np.zeros(n_total, dtype=np.float32)

        for start, end in episode_boundaries:
            ep_rewards = np.asarray(all_rewards[start:end], dtype=np.float32)
            ep_values = np.asarray(step_values[start:end], dtype=np.float32)
            returns_ep, advantages_ep = self._compute_episode_advantages(
                ep_rewards, ep_values
            )
            all_advantages[start:end] = advantages_ep
            all_returns_np[start:end] = returns_ep

        # ── 转换为 tensor ──
        advantages_t = torch.tensor(
            all_advantages, dtype=torch.float32, device=self.device
        )
        returns_t = torch.tensor(
            all_returns_np, dtype=torch.float32, device=self.device
        )
        actions_t = torch.tensor(all_actions, dtype=torch.long, device=self.device)
        old_lp = torch.tensor(all_old_logprobs, dtype=torch.float32, device=self.device)
        masks_t = torch.tensor(
            np.stack(all_masks_np), dtype=torch.bool, device=self.device
        )

        stats = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
        n_updates = 0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n_total)
            for start in range(0, n_total, self.batch_size):
                idx = perm[start : start + self.batch_size]
                b_pad = padded[idx]
                b_len = lengths[idx]
                b_act = actions_t[idx]
                b_olp = old_lp[idx]
                b_ret = returns_t[idx]
                b_adv = advantages_t[idx]
                b_mask = masks_t[idx]

                # 前向：取每条序列最后一个有效位置的输出
                all_logits, all_values = self.net.forward_sequence(b_pad, b_len)
                last_idx = (b_len - 1).long().to(self.device)
                batch_idx = torch.arange(len(idx), device=self.device)
                logits = all_logits[batch_idx, last_idx]
                values = all_values[batch_idx, last_idx]

                logits = logits.masked_fill(~b_mask, float("-inf"))
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(b_act)
                entropy = dist.entropy()

                # PPO clipped objective
                ratio = torch.exp(new_lp - b_olp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, b_ret)
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()
                n_updates += 1

        if n_updates > 0:
            for k in stats:
                stats[k] /= n_updates
        return stats

    def state_dict(self) -> dict:
        """返回包含网络参数、优化器状态和超参数的字典。"""
        return {
            "net_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "clip_eps": self.clip_eps,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "ppo_epochs": self.ppo_epochs,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
        }

    def load_state_dict(self, state: dict) -> None:
        """从字典恢复网络参数、优化器状态和超参数。"""
        self.net.load_state_dict(state["net_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.clip_eps = state["clip_eps"]
        self.entropy_coef = state["entropy_coef"]
        self.value_coef = state["value_coef"]
        self.max_grad_norm = state["max_grad_norm"]
        self.ppo_epochs = state["ppo_epochs"]
        self.batch_size = state["batch_size"]
        self.gamma = state["gamma"]
