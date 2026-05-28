"""
GRPO Agent

Group Relative Policy Optimization (GRPO) 实现。
基于 DeepSeek 论文，使用组内归一化奖励替代 value 函数。
"""

import copy
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from typing import List

from lstm import AlphaGenNet
from episode import Episode
from ppo import PPOAgent


class GRPOAgent(PPOAgent):
    """Group Relative Policy Optimization agent.

    基本思路：
    - 在采样阶段按顺序收集 episode（与现有 pipeline 兼容）
    - 将若干 episode 分为一组（group_size），在组内做奖励归一化
    - 不使用 value 函数；每个 token 的 advantage 直接设为组内归一化后的最终 reward
    - 使用参考模型 `ref_net` 对策略做 KL 正则化（loss 中加上 kl_coef * KL(pi_new || pi_ref)）

    该类继承自 PPOAgent，以重用网络/采样/优化等工具，仅重写 update、state_dict 与 load_state_dict。
    """

    def __init__(
        self,
        net: AlphaGenNet,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
        group_size: int = 64,
        kl_coef: float = 0.04,
        ref_net: AlphaGenNet | None = None,
    ):
        # value_coef=0.0: GRPO 不训练 value head
        super().__init__(
            net,
            lr=lr,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            value_coef=0.0,
            max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs,
            batch_size=batch_size,
            device=device,
        )

        self.group_size = int(group_size)
        self.kl_coef = float(kl_coef)

        # 参考模型：若未提供则在初始化时复制当前网络并冻结
        if ref_net is None:
            self.ref_net = copy.deepcopy(self.net).to(self.device)
        else:
            self.ref_net = ref_net.to(self.device)
        self.ref_net.eval()
        for p in self.ref_net.parameters():
            p.requires_grad = False

    def update(self, episodes: List[Episode]):
        """基于组内相对奖励的 GRPO 更新。"""
        if not episodes:
            return {}

        self.net.train()

        # 按组计算每个 episode 的归一化奖励（组内 mean/std）
        n_eps = len(episodes)
        ep_norm_rewards = [0.0] * n_eps
        eps_arr = episodes
        for i in range(0, n_eps, self.group_size):
            group = eps_arr[i : i + self.group_size]
            r = np.array([float(e.reward) for e in group], dtype=np.float32)
            if r.size == 0:
                continue
            mu = r.mean()
            sigma = r.std()
            if sigma < 1e-8:
                tilde = np.zeros_like(r)
            else:
                tilde = (r - mu) / sigma
            for j, _ in enumerate(group):
                ep_norm_rewards[i + j] = float(tilde[j])

        # ── 将 episodes 展平为 per-step 数据（与 PPO 兼容） ──
        all_input_ids: list[list[int]] = []
        all_actions: list[int] = []
        all_old_logprobs: list[float] = []
        all_masks_np: list[np.ndarray] = []
        all_advantages: list[float] = []

        episode_boundaries: list[tuple[int, int]] = []
        offset = 0
        valid_episode_indices = []  # 记录有效episode的原始索引
        for ei, ep in enumerate(episodes):
            n_steps = len(ep.actions)
            if n_steps == 0:
                continue
            valid_episode_indices.append(ei)
            for i in range(n_steps):
                all_input_ids.append(ep.token_ids[: i + 1])
                all_actions.append(ep.actions[i])
                all_old_logprobs.append(ep.log_probs[i])
                all_masks_np.append(ep.masks[i])
                # 每个 token 的 advantage 为该 episode 在组内归一化后的 final reward
                # 注意：这里需要使用原始索引ei来获取对应的归一化奖励
                all_advantages.append(ep_norm_rewards[ei])
            episode_boundaries.append((offset, offset + n_steps))
            offset += n_steps

        n_total = len(all_actions)
        if n_total == 0:
            return {}

        # padding
        max_seq = max(len(s) for s in all_input_ids)
        padded = torch.zeros(n_total, max_seq, dtype=torch.long, device=self.device)
        lengths = torch.zeros(n_total, dtype=torch.long, device=self.device)
        for i, seq in enumerate(all_input_ids):
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            lengths[i] = len(seq)

        actions_t = torch.tensor(all_actions, dtype=torch.long, device=self.device)
        old_lp = torch.tensor(all_old_logprobs, dtype=torch.float32, device=self.device)
        masks_t = torch.tensor(np.stack(all_masks_np), dtype=torch.bool, device=self.device)
        advantages_t = torch.tensor(all_advantages, dtype=torch.float32, device=self.device)

        stats = {"policy_loss": 0, "entropy": 0, "kl": 0}
        n_updates = 0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n_total)
            for start in range(0, n_total, self.batch_size):
                idx = perm[start : start + self.batch_size]
                b_pad = padded[idx]
                b_len = lengths[idx]
                b_act = actions_t[idx]
                b_olp = old_lp[idx]
                b_adv = advantages_t[idx]
                b_mask = masks_t[idx]

                # 前向：取每条序列最后一个有效位置的输出
                all_logits, _ = self.net.forward_sequence(b_pad, b_len)
                last_idx = (b_len - 1).long().to(self.device)
                batch_idx = torch.arange(len(idx), device=self.device)
                logits = all_logits[batch_idx, last_idx]

                logits = logits.masked_fill(~b_mask, float("-inf"))
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(b_act)
                entropy = dist.entropy()

                # GRPO clipped objective（与 PPO 类似，但 advantage 来自组内归一化）
                ratio = torch.exp(new_lp - b_olp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # 参考模型的 KL 正则化（使用论文中的无偏估计器）
                with torch.no_grad():
                    ref_logits_all, _ = self.ref_net.forward_sequence(b_pad, b_len)
                ref_logits = ref_logits_all[batch_idx, last_idx]
                ref_logits = ref_logits.masked_fill(~b_mask, float("-inf"))
                dist_ref = Categorical(logits=ref_logits)
                
                # 使用论文公式(4)的无偏估计器: KL = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1
                log_ratio_ref = dist_ref.log_prob(b_act) - dist.log_prob(b_act)
                ratio_ref = torch.exp(log_ratio_ref)
                kl = (ratio_ref - log_ratio_ref - 1).mean()

                # 修正：熵系数应该是正数（鼓励探索）
                loss = policy_loss + self.kl_coef * kl + self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["kl"] += kl.item()
                n_updates += 1

        if n_updates > 0:
            for k in stats:
                stats[k] /= n_updates
        return stats

    def state_dict(self) -> dict:
        """返回包含网络参数、优化器状态和超参数的字典。"""
        base = super().state_dict()
        base.update({
            "group_size": int(self.group_size),
            "kl_coef": float(self.kl_coef),
            "ref_net_state_dict": self.ref_net.state_dict(),
        })
        return base

    def load_state_dict(self, state: dict) -> None:
        """从字典恢复网络参数、优化器状态和超参数。"""
        super().load_state_dict(state)
        self.group_size = int(state.get("group_size", self.group_size))
        self.kl_coef = float(state.get("kl_coef", self.kl_coef))
        ref_state = state.get("ref_net_state_dict")
        if ref_state is not None:
            try:
                self.ref_net.load_state_dict(ref_state)
            except Exception:
                # 如果 ref_net 结构不匹配，忽略加载
                pass
