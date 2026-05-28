"""
训练管线

流程：
  1. 加载数据，构建 StockData 和 target
  2. 初始化 AlphaCombinationModel 和 PPO Agent
  3. 循环：生成 episode → 评估 alpha → 更新组合模型 → PPO 更新
  4. 保存最终 alpha 池
"""

# 在导入任何可能触发 OpenMP 的库之前设置环境变量，
# 避免 Windows 下 libiomp5md.dll 重复初始化导致的崩溃。
# 注意：这是一个常见的开发/调试绕过；若用于生产应通过确保只链接单一 OpenMP 运行时来解决。
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


import os
import json
import time
import inspect
import numpy as np
import torch
import argparse

from tokens import (
    VOCAB,
    VOCAB_SIZE,
    TOKEN_TO_IDX,
    TIME_DELTAS,
    Token,
    TokenType,
    BEG_TOKEN,
)
from expression import (
    parse_rpn_to_tree,
    strip_special_tokens,
    tree_to_formula,
    parse_formula,
)
from calculator import StockData, calc_rank_ic
from combination import AlphaCombinationModel
from masking import RPNBuilder
from ppo import PPOAgent
from grpo import GRPOAgent
from episode import Episode
from data import Data
from common import ADJUST_PREV
from reporting import (
    CombinationEvaluator,
    WarmupDataContext,
    compute_benchmark_result,
    evaluate_weighted_alpha_values,
    estimate_required_warmup_days,
    plot_backtest_comparison,
    plot_equity_curves,
    plot_training_history,
    save_json,
    save_training_history,
)
from backtest import Backtester, load_pool_alpha_from_file

BEG_IDX = TOKEN_TO_IDX[Token(TokenType.BEG, BEG_TOKEN)]


def set_random_seed(seed: int) -> None:
    """设置随机种子，便于实验复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    """捕获 numpy / torch 的随机状态，用于断点续训。"""
    return {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict | None) -> None:
    """恢复随机状态。"""
    if not state:
        return
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(tuple(numpy_state))
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


# ============================================================
# 0. 单 episode 生成
# ============================================================
def collect_episode(
    agent: PPOAgent,
    builder: RPNBuilder,
    device: str = "cpu",
) -> Episode:
    """生成一条 RPN 序列。"""
    builder.reset()
    ep = Episode()

    # 起始 BEG
    builder.step(BEG_IDX)
    ep.token_ids.append(BEG_IDX)

    hidden = agent.net.init_hidden(batch_size=1, device=device)

    while not builder.done:
        valid_mask = builder.get_valid_mask()

        if not valid_mask.any():
            break

        action, log_prob, value, hidden = agent.select_action(
            ep.token_ids[-1], valid_mask, hidden
        )

        ep.actions.append(action)
        ep.log_probs.append(log_prob)
        ep.values.append(value)
        ep.masks.append(valid_mask.copy())

        builder.step(action)
        ep.token_ids.append(action)

    return ep


# ============================================================
# 1. 评估 episode 并计算 reward
# ============================================================
def evaluate_episode(
    ep: Episode,
    combination: AlphaCombinationModel,
    reward_mode: str = "multi",
    # 多目标奖励函数参数
    reward_ic_weight: float = 10.0,
    reward_icir_weight: float = 3.0,
    reward_rank_ic_weight: float = 2.0,
    reward_balance_bonus: float = 0.05,
    reward_redundancy_threshold: float = 0.7,
    reward_redundancy_coef: float = 0.5,
    reward_reject_low_ic: float = -0.2,
    reward_reject_redundant: float = -0.15,
    reward_reject_no_improve: float = -0.1,
) -> dict:
    """
    解析 episode 的 token 序列，评估 alpha，加入组合模型。
    返回结构化结果：
      - reward: PPO 使用的奖励
      - accepted: 新 alpha 是否被接纳进组合
      - formula: 表达式字符串（若可解析）
      - candidate_ic: 该 alpha 单独对 target 的 IC
      - ic_delta: 接纳后带来的组合 IC 增量
      - status: 接纳/拒绝原因
    """

    def _result(
        reward: float,
        accepted: bool = False,
        formula: str | None = None,
        candidate_ic: float | None = None,
        ic_delta: float | None = None,
        status: str = "invalid",
    ) -> dict:
        return {
            "reward": float(reward),
            "accepted": bool(accepted),
            "formula": formula,
            "candidate_ic": candidate_ic,
            "ic_delta": ic_delta,
            "status": status,
        }

    tokens = [VOCAB[i] for i in ep.token_ids]
    stripped = strip_special_tokens(tokens)

    if len(stripped) == 0:
        return _result(-1.0, status="invalid_empty")

    tree = parse_rpn_to_tree(stripped)
    if tree is None:
        return _result(-1.0, status="invalid_parse")

    formula = tree_to_formula(tree)

    try:
        alpha_val = combination.calculator.evaluate(tree)
    except Exception:
        return _result(-1.0, formula=formula, status="invalid_eval_exception")

    if alpha_val is None or np.all(np.isnan(alpha_val)):
        return _result(-1.0, formula=formula, status="invalid_all_nan")

    valid_ratio = np.mean(~np.isnan(alpha_val))
    if valid_ratio < 0.1:
        return _result(-1.0, formula=formula, status="invalid_sparse")

    old_ic = combination.get_combination_ic()
    new_ic = combination.add_alpha(
        tree,
        alpha_values=alpha_val,
        already_normed=True,
        baseline_ic=old_ic,
    )

    if combination.last_add_accepted:
        ic_delta = new_ic - old_ic
        if reward_mode == "simple":
            reward = reward_ic_weight * ic_delta
        elif reward_mode == "multi":
            icir_delta = combination.last_add_icir_delta

            # 计算候选 alpha 的 Rank IC（Spearman 相关系数）
            candidate_rank_ic = calc_rank_ic(alpha_val, combination.target)

            # 正负平衡奖励：如果新因子IC方向与多数池内因子相反，给奖励
            pos_count = sum(1 for ic in combination.ic_vector if ic > 0)
            neg_count = len(combination.ic_vector) - pos_count
            balance_bonus = 0.0
            candidate_ic_sign = combination.last_add_candidate_ic
            if candidate_ic_sign > 0 and pos_count < neg_count:
                balance_bonus = reward_balance_bonus
            elif candidate_ic_sign < 0 and neg_count < pos_count:
                balance_bonus = reward_balance_bonus

            # 冗余惩罚：与已有因子相关性越高，惩罚越大
            redundancy_penalty = 0.0
            if combination.pool_size > 1:
                # 取最后一个因子（刚加入的）与池中其他因子的最大互相关
                last_idx = combination.pool_size - 1
                max_mutual = (
                    max(abs(combination.ic_matrix[last_idx, j]) for j in range(last_idx))
                    if last_idx > 0
                    else 0.0
                )
                if max_mutual > reward_redundancy_threshold:
                    redundancy_penalty = (max_mutual - reward_redundancy_threshold) * reward_redundancy_coef

            reward = (
                reward_ic_weight * ic_delta
                + reward_icir_weight * icir_delta
                + reward_rank_ic_weight * candidate_rank_ic
                + balance_bonus
                - redundancy_penalty
            )
        else:
            raise ValueError(f"Unknown reward_mode: {reward_mode}")

        return _result(
            reward=reward,
            accepted=True,
            formula=formula,
            candidate_ic=combination.last_add_candidate_ic,
            ic_delta=ic_delta,
            status=combination.last_add_status,
        )

    rejection_penalty = {
        "rejected_low_ic": reward_reject_low_ic,
        "rejected_redundant": reward_reject_redundant,
        "rejected_no_improve": reward_reject_no_improve,
    }
    return _result(
        reward=rejection_penalty.get(combination.last_add_status, -1.0),
        accepted=False,
        formula=formula,
        candidate_ic=combination.last_add_candidate_ic,
        ic_delta=combination.last_add_ic_delta,
        status=combination.last_add_status,
    )


# ============================================================
# 2. 主训练函数
# ============================================================
def train(
    stock_data: StockData,
    target: np.ndarray,
    val_stock_data: StockData | None = None,
    val_target: np.ndarray | None = None,
    val_data_context: WarmupDataContext | None = None,
    num_iterations: int = 100,
    episodes_per_iter: int = 64,
    max_pool_size: int = 20,
    max_seq_len: int = 20,
    horizon: int = 20,
    lr: float = 3e-4,
    device: str = "cpu",
    save_dir: str = "checkpoints",
    gamma: float = 0.99,
    reward_mode: str = "multi",
    # 多目标奖励函数参数
    reward_ic_weight: float = 10.0,
    reward_icir_weight: float = 3.0,
    reward_rank_ic_weight: float = 2.0,
    reward_balance_bonus: float = 0.05,
    reward_redundancy_threshold: float = 0.7,
    reward_redundancy_coef: float = 0.5,
    reward_reject_low_ic: float = -0.2,
    reward_reject_redundant: float = -0.15,
    reward_reject_no_improve: float = -0.1,
    model_type: str = "rnn",
    rl_algo: str = "ppo",
    grpo_group_size: int = 64,
    grpo_kl_coef: float = 0.04,
    # Transformer 专用参数
    tf_embed_dim: int = 64,
    tf_nhead: int = 4,
    tf_num_layers: int = 3,
    tf_dim_feedforward: int = 256,
    tf_dropout: float = 0.1,
    resume: bool = False,
    resume_tag: str = "latest",
):
    """Algorithm 2 完整训练循环。"""
    os.makedirs(save_dir, exist_ok=True)

    # 初始化组合模型
    combination = AlphaCombinationModel(stock_data, target, max_pool_size=max_pool_size)

    # 根据 model_type 初始化网络
    if model_type == "transformer":
        from transformer import AlphaGenTransformer

        net = AlphaGenTransformer(
            vocab_size=VOCAB_SIZE,
            embed_dim=tf_embed_dim,
            nhead=tf_nhead,
            num_layers=tf_num_layers,
            dim_feedforward=tf_dim_feedforward,
            dropout=tf_dropout,
            max_seq_len=max_seq_len,
        )
        print(
            f"模型: Transformer (embed_dim={tf_embed_dim}, nhead={tf_nhead}, "
            f"layers={tf_num_layers}, ff_dim={tf_dim_feedforward}, dropout={tf_dropout})"
        )
    else:
        from lstm import AlphaGenNet

        net = AlphaGenNet(vocab_size=VOCAB_SIZE)
        print(f"模型: LSTM (embed_dim=32, hidden_dim=128, num_layers=2)")

    # 根据选择的 RL 算法实例化 Agent
    if rl_algo == "grpo":
        agent = GRPOAgent(
            net,
            lr=lr,
            device=device,
            group_size=grpo_group_size,
            kl_coef=grpo_kl_coef,
        )
        print(f"RL 算法: GRPO (group_size={grpo_group_size}, kl_coef={grpo_kl_coef})")
    else:
        agent = PPOAgent(
            net,
            lr=lr,
            device=device,
            gamma=gamma,
        )
        print("RL 算法: PPO")

    builder = RPNBuilder(max_len=max_seq_len)
    val_evaluator = (
        CombinationEvaluator(val_stock_data, val_target)
        if val_stock_data is not None
        and val_target is not None
        and val_data_context is None
        else None
    )
    preloaded_val_warmup = None
    if val_data_context is not None:
        preloaded_val_warmup = val_data_context.preload_generator_warmup(
            max_seq_len=max_seq_len,
            time_deltas=TIME_DELTAS,
        )
    best_score = -np.inf
    history = []
    start_iteration = 0

    if resume:
        resume_state = load_training_state(save_dir, tag=resume_tag, device=device)
        if resume_state is None:
            print(f"断点续训: 未找到 {resume_tag} checkpoint，改为从头开始。")
        else:
            agent_state = resume_state.get("agent_state")
            if agent_state is not None:
                agent.load_state_dict(agent_state)
            else:
                net_state_dict = resume_state.get("net_state_dict")
                if net_state_dict is not None:
                    net.load_state_dict(net_state_dict)

            combination_state = resume_state.get("combination_state")
            if combination_state is not None:
                restore_combination_state(combination, combination_state)

            restore_rng_state(resume_state.get("rng_state"))

            metadata = resume_state.get("metadata") or load_checkpoint_metadata(
                save_dir
            )
            start_iteration = int(metadata.get("iteration", 0) or 0)
            loaded_best_score = metadata.get("best_score")
            if loaded_best_score is not None and np.isfinite(loaded_best_score):
                best_score = float(loaded_best_score)
            history = list(metadata.get("history", []))
            print(
                f"断点续训: 已恢复至第 {start_iteration} 轮, "
                f"当前池大小 {combination.pool_size}, "
                f"best_score={best_score if np.isfinite(best_score) else 'None'}"
            )

    print(f"开始训练: {num_iterations} 轮, 每轮 {episodes_per_iter} episode")
    print(f"设备: {device}, 股票数: {stock_data.n_stocks}, 交易日: {stock_data.n_days}")
    print("股票过滤: 默认排除北交所(BJ)和ST股票")
    print(f"奖励函数: {reward_mode}")
    if val_data_context is not None:
        print(
            f"验证集: 股票数 {val_data_context.stock_data.n_stocks}, "
            f"交易日 {len(val_data_context.official_trade_dates())}, "
            f"正式起点 {val_data_context.start_date_str}"
        )
        print(
            f"验证预热: 预加载 {preloaded_val_warmup} 个交易日, "
            f"扩展起点 {val_data_context.loaded_start_date_str}"
        )
    elif val_stock_data is not None:
        print(
            f"验证集: 股票数 {val_stock_data.n_stocks}, 交易日 {val_stock_data.n_days}"
        )
    print("=" * 70)

    for iteration in range(start_iteration, num_iterations):
        t0 = time.time()
        episodes = []
        rewards = []
        valid_count = 0
        accepted_alphas = []

        # 收集 episodes
        for _ in range(episodes_per_iter):
            ep = collect_episode(agent, builder, device=device)

            if len(ep.actions) == 0:
                continue

            episode_result = evaluate_episode(
                ep,
                combination,
                reward_mode=reward_mode,
                # 多目标奖励参数
                reward_ic_weight=reward_ic_weight,
                reward_icir_weight=reward_icir_weight,
                reward_rank_ic_weight=reward_rank_ic_weight,
                reward_balance_bonus=reward_balance_bonus,
                reward_redundancy_threshold=reward_redundancy_threshold,
                reward_redundancy_coef=reward_redundancy_coef,
                reward_reject_low_ic=reward_reject_low_ic,
                reward_reject_redundant=reward_reject_redundant,
                reward_reject_no_improve=reward_reject_no_improve,
            )
            reward = episode_result["reward"]
            ep.reward = reward
            episodes.append(ep)
            rewards.append(reward)

            if reward > -0.5:
                valid_count += 1
            if episode_result["accepted"]:
                accepted_alphas.append(
                    {
                        "formula": episode_result["formula"],
                        "candidate_ic": float(episode_result["candidate_ic"]),
                        "ic_delta": float(episode_result["ic_delta"]),
                    }
                )

        # PPO 更新
        stats = agent.update(episodes)

        # 日志
        elapsed = time.time() - t0
        avg_reward = np.mean(rewards) if rewards else 0
        avg_len = np.mean([len(ep.token_ids) for ep in episodes]) if episodes else 0
        train_metrics = evaluate_weighted_alpha_values(
            combination.alpha_values,
            combination.weights,
            target,
            already_normed_target=False,
        )
        val_metrics = (
            val_data_context.evaluate(combination.alpha_exprs, combination.weights)
            if val_data_context is not None
            else (
                val_evaluator.evaluate(combination.alpha_exprs, combination.weights)
                if val_evaluator is not None
                else None
            )
        )
        selection_score = (
            val_metrics["ic"] if val_metrics is not None else train_metrics["ic"]
        )
        if not np.isfinite(selection_score):
            selection_score = -np.inf
        val_log = ""
        if val_metrics is not None:
            val_log = (
                f"val_loss={val_metrics['loss']:.4f}  "
                f"val_ic={val_metrics['ic']:+.4f}  "
                f"val_icir={val_metrics['icir']:+.4f}  "
            )

        row = {
            "iteration": iteration + 1,
            "pool_size": combination.pool_size,
            "accepted_alpha_count": int(len(accepted_alphas)),
            "accepted_alphas": accepted_alphas,
            "train_loss": train_metrics["loss"],
            "train_ic": train_metrics["ic"],
            "train_icir": train_metrics["icir"],
            "val_loss": None if val_metrics is None else val_metrics["loss"],
            "val_ic": None if val_metrics is None else val_metrics["ic"],
            "val_icir": None if val_metrics is None else val_metrics["icir"],
            "avg_reward": float(avg_reward),
            "valid_episodes": int(valid_count),
            "num_episodes": int(len(episodes)),
            "avg_seq_len": float(avg_len),
            "policy_loss": float(stats.get("policy_loss", 0.0)),
            "value_loss": float(stats.get("value_loss", 0.0)),
            "entropy": float(stats.get("entropy", 0.0)),
            "elapsed_sec": float(elapsed),
        }
        history.append(row)

        print(
            f"[{iteration+1:3d}/{num_iterations}] "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"train_ic={train_metrics['ic']:+.4f}  "
            f"train_icir={train_metrics['icir']:+.4f}  "
            f"{val_log}"
            f"avg_r={avg_reward:+.4f}  "
            f"accepted={len(accepted_alphas)}  "
            f"pool={combination.pool_size}  "
            f"valid={valid_count}/{len(episodes)}  "
            f"avg_len={avg_len:.1f}  "
            f"p_loss={stats.get('policy_loss', 0):.4f}  "
            f"v_loss={stats.get('value_loss', 0):.4f}  "
            f"entropy={stats.get('entropy', 0):.3f}  "
            f"time={elapsed:.1f}s"
        )
        if accepted_alphas:
            for idx, info in enumerate(accepted_alphas, start=1):
                print(
                    f"    accepted[{idx}] "
                    f"candidate_ic={info['candidate_ic']:+.4f}  "
                    f"ic_delta={info['ic_delta']:+.4f}  "
                    f"{info['formula']}"
                )

        # 保存最佳
        if selection_score > best_score and combination.pool_size > 0:
            best_score = selection_score
            save_checkpoint(
                combination,
                net,
                save_dir,
                tag="best",
                agent=agent,
                iteration=iteration + 1,
                best_score=best_score,
                history=history,
            )

        save_checkpoint(
            combination,
            net,
            save_dir,
            tag="latest",
            agent=agent,
            iteration=iteration + 1,
            best_score=best_score,
            history=history,
        )

        # 每 20 轮保存
        if (iteration + 1) % 20 == 0:
            save_checkpoint(
                combination,
                net,
                save_dir,
                tag=f"iter{iteration+1}",
                agent=agent,
                iteration=iteration + 1,
                best_score=best_score,
                history=history,
            )

    # 最终保存
    save_checkpoint(
        combination,
        net,
        save_dir,
        tag="final",
        agent=agent,
        iteration=len(history),
        best_score=best_score,
        history=history,
    )
    save_training_history(history, save_dir)
    plot_training_history(history, os.path.join(save_dir, "training_metrics.png"))

    final_train_metrics = evaluate_weighted_alpha_values(
        combination.alpha_values,
        combination.weights,
        target,
        already_normed_target=False,
    )
    final_val_metrics = (
        val_data_context.evaluate(combination.alpha_exprs, combination.weights)
        if val_data_context is not None
        else (
            val_evaluator.evaluate(combination.alpha_exprs, combination.weights)
            if val_evaluator is not None
            else None
        )
    )
    summary = {
        "best_selection_score": float(best_score) if np.isfinite(best_score) else None,
        "reward_mode": reward_mode,
        "model_type": model_type,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "history_length": len(history),
    }
    save_json(summary, os.path.join(save_dir, "training_summary.json"))

    print("=" * 70)
    print(
        f"训练完成. 最终 train_IC = {final_train_metrics['ic']:+.4f}, "
        f"train_ICIR = {final_train_metrics['icir']:+.4f}"
    )
    if final_val_metrics is not None:
        print(
            f"最终 val_IC = {final_val_metrics['ic']:+.4f}, "
            f"val_ICIR = {final_val_metrics['icir']:+.4f}"
        )
    combination.summary()

    return combination, net, history


def build_combination_checkpoint_state(combination) -> dict:
    """构建可序列化的 alpha 池状态。"""
    weights = np.asarray(getattr(combination, "weights", []), dtype=np.float64)
    ic_vector = np.asarray(getattr(combination, "ic_vector", []), dtype=np.float64)
    ic_matrix = np.asarray(
        getattr(combination, "ic_matrix", np.array([]).reshape(0, 0)),
        dtype=np.float64,
    )
    return {
        "alpha_formulas": [
            tree_to_formula(expr) for expr in getattr(combination, "alpha_exprs", [])
        ],
        "weights": weights.tolist(),
        "ic_vector": ic_vector.tolist(),
        "ic_matrix": ic_matrix.tolist(),
    }


def _coerce_checkpoint_metadata(
    iteration: int | None,
    best_score: float | None,
    history: list[dict] | None,
) -> tuple[int, float | None, list[dict]]:
    """规范化 checkpoint 元数据。

    若调用方没有显式传递元数据，则尝试从调用栈中的 `metadata` 字典读取，
    以兼容现有测试与轻量脚本调用。
    """
    if iteration is None and best_score is None and history is None:
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        caller = caller.f_back if caller is not None else None
        candidate = caller.f_locals.get("metadata") if caller is not None else None
        if isinstance(candidate, dict):
            iteration = candidate.get("iteration")
            best_score = candidate.get("best_score")
            history = candidate.get("history")

    history = [] if history is None else list(history)
    if iteration is None:
        iteration = len(history)
    if best_score is not None and np.isfinite(best_score):
        best_score = float(best_score)
    else:
        best_score = None
    return int(iteration), best_score, history


def _tagged_metadata_path(save_dir: str, tag: str) -> str:
    return os.path.join(save_dir, f"checkpoint_metadata_{tag}.json")


def _tagged_training_state_path(save_dir: str, tag: str) -> str:
    return os.path.join(save_dir, f"training_state_{tag}.pt")


def save_checkpoint(
    combination,
    net,
    save_dir,
    tag="latest",
    *,
    agent: PPOAgent | None = None,
    iteration: int | None = None,
    best_score: float | None = None,
    history: list[dict] | None = None,
):
    os.makedirs(save_dir, exist_ok=True)
    iteration, best_score, history = _coerce_checkpoint_metadata(
        iteration, best_score, history
    )

    # 保存网络
    torch.save(net.state_dict(), os.path.join(save_dir, f"net_{tag}.pt"))

    # 保存 alpha 池信息
    pool_info = []
    for i, expr in enumerate(combination.alpha_exprs):
        pool_info.append(
            {
                "formula": tree_to_formula(expr),
                "weight": float(combination.weights[i]),
                "ic": float(combination.ic_vector[i]),
            }
        )
    with open(os.path.join(save_dir, f"pool_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(pool_info, f, ensure_ascii=False, indent=2)

    metadata = {
        "tag": tag,
        "iteration": int(iteration),
        "best_score": best_score,
        "history": history,
        "pool_size": int(combination.pool_size),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(metadata, _tagged_metadata_path(save_dir, tag))
    if tag == "latest":
        save_json(metadata, os.path.join(save_dir, "checkpoint_metadata.json"))

    training_state = {
        "tag": tag,
        "metadata": metadata,
        "net_state_dict": net.state_dict(),
        "agent_state": None if agent is None else agent.state_dict(),
        "combination_state": build_combination_checkpoint_state(combination),
        "rng_state": capture_rng_state(),
    }
    torch.save(training_state, _tagged_training_state_path(save_dir, tag))
    if tag == "latest":
        torch.save(training_state, os.path.join(save_dir, "training_state.pt"))


def restore_combination_state(combination, checkpoint_state: dict) -> None:
    """从 checkpoint 恢复 alpha 池状态。

    参数:
        combination: AlphaCombinationModel 实例
        checkpoint_state: 包含 alpha_formulas、weights、ic_vector、ic_matrix 的字典
    """
    formula_strings = checkpoint_state["alpha_formulas"]
    alpha_exprs = []
    alpha_values = []
    for formula_string in formula_strings:
        tree = parse_formula(formula_string)
        if tree is None:
            raise ValueError(f"无法从 checkpoint 恢复公式: {formula_string}")
        alpha_val = combination.calculator.evaluate(tree)
        if alpha_val is None or np.all(np.isnan(alpha_val)):
            raise ValueError(f"无法重新计算 checkpoint 中的公式: {formula_string}")
        alpha_exprs.append(tree)
        alpha_values.append(alpha_val)

    combination.alpha_exprs = alpha_exprs
    combination.alpha_values = alpha_values
    combination.weights = np.array(checkpoint_state["weights"], dtype=np.float64)
    combination.ic_vector = np.array(checkpoint_state["ic_vector"], dtype=np.float64)
    ic_matrix_data = checkpoint_state["ic_matrix"]
    if len(ic_matrix_data) > 0:
        combination.ic_matrix = np.array(ic_matrix_data, dtype=np.float64)
    else:
        combination.ic_matrix = np.array([]).reshape(0, 0)
    if len(combination.alpha_exprs) != len(combination.weights):
        raise ValueError("checkpoint 中 alpha 数量与权重数量不一致")


def load_checkpoint_metadata(path: str) -> dict:
    """读取 checkpoint 元数据；path 可为目录或 JSON 文件。"""
    metadata_path = (
        os.path.join(path, "checkpoint_metadata.json") if os.path.isdir(path) else path
    )
    if not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_resume_iteration(path: str) -> int:
    """返回断点续训应开始的 iteration（0-indexed）。"""
    metadata = load_checkpoint_metadata(path)
    return int(metadata.get("iteration", 0) or 0)


def load_training_state(
    save_dir: str,
    tag: str = "latest",
    device: str = "cpu",
) -> dict | None:
    """加载包含 agent / pool / metadata 的训练状态。"""
    state_path = (
        os.path.join(save_dir, "training_state.pt")
        if tag == "latest"
        else _tagged_training_state_path(save_dir, tag)
    )
    if not os.path.isfile(state_path):
        return None
    return torch.load(state_path, map_location=device, weights_only=False)


# ============================================================
# 4. 配置持久化
# ============================================================
def save_config(args: argparse.Namespace, save_dir: str) -> str:
    """将 argparse 配置保存为 JSON 文件."""
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)
    return config_path


def load_config(config_path: str) -> dict:
    """从 JSON 文件加载配置."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 5. 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Alpha 因子生成训练管线")
    # ── 数据参数 ──
    parser.add_argument(
        "--train_start", default="20240101", help="训练集起始日期 (YYYYMMDD)"
    )
    parser.add_argument(
        "--train_end", default="20251231", help="训练集结束日期 (YYYYMMDD)"
    )
    parser.add_argument(
        "--val_start", default="20260101", help="验证集起始日期 (YYYYMMDD)"
    )
    parser.add_argument(
        "--val_end", default="20260501", help="验证集结束日期 (YYYYMMDD)"
    )
    # ── 训练参数 ──
    parser.add_argument("--iterations", type=int, default=50, help="训练轮数")
    parser.add_argument("--episodes", type=int, default=64, help="每轮 episode 数")
    parser.add_argument("--pool_size", type=int, default=15, help="Alpha 池最大容量")
    parser.add_argument("--horizon", type=int, default=3, help="目标收益前瞻天数")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    # ── 回测参数 ──
    parser.add_argument("--n_hold", type=int, default=20, help="持仓股票数")
    parser.add_argument("--n_swap", type=int, default=3, help="每期换仓数")
    parser.add_argument("--commission", type=float, default=0.001, help="交易佣金率")
    parser.add_argument("--benchmark_code", default="000300.SH", help="基准指数代码")
    # ── 输出与设备 ──
    parser.add_argument("--save_dir", default="outputs", help="输出目录")
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="折扣因子 gamma",
    )
    parser.add_argument(
        "--reward_mode",
        choices=["simple", "multi"],
        default="multi",
        help="奖励函数: simple=仅IC增量, multi=多目标奖励",
    )
    # ── 多目标奖励函数参数 ──
    parser.add_argument(
        "--reward_ic_weight",
        type=float,
        default=10.0,
        help="[multi] IC 增量权重",
    )
    parser.add_argument(
        "--reward_icir_weight",
        type=float,
        default=3.0,
        help="[multi] ICIR 增量权重",
    )
    parser.add_argument(
        "--reward_rank_ic_weight",
        type=float,
        default=2.0,
        help="[multi] Rank IC 权重",
    )
    parser.add_argument(
        "--reward_balance_bonus",
        type=float,
        default=0.05,
        help="[multi] 正负平衡奖励",
    )
    parser.add_argument(
        "--reward_redundancy_threshold",
        type=float,
        default=0.7,
        help="[multi] 冗余惩罚阈值（互相关）",
    )
    parser.add_argument(
        "--reward_redundancy_coef",
        type=float,
        default=0.5,
        help="[multi] 冗余惩罚系数",
    )
    parser.add_argument(
        "--reward_reject_low_ic",
        type=float,
        default=-0.2,
        help="[multi] 拒绝低IC的惩罚",
    )
    parser.add_argument(
        "--reward_reject_redundant",
        type=float,
        default=-0.15,
        help="[multi] 拒绝冗余因子的惩罚",
    )
    parser.add_argument(
        "--reward_reject_no_improve",
        type=float,
        default=-0.1,
        help="[multi] 拒绝无提升因子的惩罚",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备 (cpu/cuda)",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 save_dir 中最近一次 latest checkpoint 断点续训",
    )
    parser.add_argument(
        "--resume_tag",
        default="latest",
        help="指定恢复使用的 checkpoint 标签",
    )
    # ── 模型选择 ──
    parser.add_argument(
        "--model",
        choices=["rnn", "transformer"],
        default="rnn",
        help="策略网络类型: rnn (LSTM, 默认) 或 transformer",
    )
    parser.add_argument(
        "--rl_algo",
        choices=["ppo", "grpo"],
        default="ppo",
        help="强化学习算法: ppo 或 grpo",
    )
    parser.add_argument(
        "--grpo_group_size",
        type=int,
        default=64,
        help="GRPO: 每个组的采样数量 G",
    )
    parser.add_argument(
        "--grpo_kl_coef",
        type=float,
        default=0.04,
        help="GRPO: KL 正则化系数 beta",
    )
    # ── Transformer 专用参数 ──
    parser.add_argument(
        "--tf_embed_dim", type=int, default=64, help="[Transformer] embedding 维度"
    )
    parser.add_argument(
        "--tf_nhead", type=int, default=4, help="[Transformer] 注意力头数"
    )
    parser.add_argument(
        "--tf_num_layers", type=int, default=3, help="[Transformer] Encoder 层数"
    )
    parser.add_argument(
        "--tf_dim_feedforward", type=int, default=256, help="[Transformer] 前馈网络维度"
    )
    parser.add_argument(
        "--tf_dropout", type=float, default=0.1, help="[Transformer] Dropout 概率"
    )
    args = parser.parse_args()
    set_random_seed(args.seed)
    save_config(args, args.save_dir)

    print("加载训练集数据...")
    reader = Data()
    train_df = reader.daily(
        start_date=args.train_start,
        end_date=args.train_end,
        bj=False,
        st=False,
        adjust=ADJUST_PREV,
    )

    train_df = train_df[
        ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "vwap"]
    ]
    print(f"训练集数据行数: {len(train_df)}")

    train_sd = StockData(train_df)
    train_target = train_sd.get_target(horizon=args.horizon)

    print("加载验证集数据...")
    val_context = WarmupDataContext(
        reader=reader,
        start_date=args.val_start,
        end_date=args.val_end,
        horizon=args.horizon,
        adjust=ADJUST_PREV,
        bj=False,
        st=False,
    )
    print("验证集数据将在训练开始前按生成器最大窗口一次性预热加载。")

    combination, net, _ = train(
        stock_data=train_sd,
        target=train_target,
        val_data_context=val_context,
        num_iterations=args.iterations,
        episodes_per_iter=args.episodes,
        max_pool_size=args.pool_size,
        horizon=args.horizon,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        gamma=args.gamma,
        reward_mode=args.reward_mode,
        # 多目标奖励函数参数
        reward_ic_weight=args.reward_ic_weight,
        reward_icir_weight=args.reward_icir_weight,
        reward_rank_ic_weight=args.reward_rank_ic_weight,
        reward_balance_bonus=args.reward_balance_bonus,
        reward_redundancy_threshold=args.reward_redundancy_threshold,
        reward_redundancy_coef=args.reward_redundancy_coef,
        reward_reject_low_ic=args.reward_reject_low_ic,
        reward_reject_redundant=args.reward_reject_redundant,
        reward_reject_no_improve=args.reward_reject_no_improve,
        model_type=args.model,
        rl_algo=args.rl_algo,
        grpo_group_size=args.grpo_group_size,
        grpo_kl_coef=args.grpo_kl_coef,
        tf_embed_dim=args.tf_embed_dim,
        tf_nhead=args.tf_nhead,
        tf_num_layers=args.tf_num_layers,
        tf_dim_feedforward=args.tf_dim_feedforward,
        tf_dropout=args.tf_dropout,
        resume=args.resume,
        resume_tag=args.resume_tag,
    )

    pool_file = os.path.join(args.save_dir, "pool_best.json")
    if os.path.isfile(pool_file):
        with open(pool_file, "r", encoding="utf-8") as f:
            pool_info = json.load(f)
        pool_exprs = []
        for item in pool_info:
            tree = parse_formula(item["formula"])
            if tree is not None:
                pool_exprs.append(tree)
        warmup_days = estimate_required_warmup_days(pool_exprs)
        val_context.ensure_warmup_days(warmup_days)
        print(
            f"\n验证 warm-up: {warmup_days} 个交易日, "
            f"扩展起点 = {val_context.loaded_start_date_str}, "
            f"正式统计起点 = {args.val_start}"
        )

        pool_state = load_pool_alpha_from_file(
            pool_file,
            val_context.stock_data,
            target=val_context.target,
            verbose=True,
            official_start_index=val_context.official_start_index,
        )
        combo_alpha = pool_state["combo_alpha"][val_context.official_start_index :]
        factor_metrics = pool_state["factor_metrics"]
        print(
            f"\n最佳验证池因子表现: IC={factor_metrics['ic']:+.4f}, "
            f"ICIR={factor_metrics['icir']:+.4f}, Loss={factor_metrics['loss']:.4f}"
        )

        bt = Backtester(
            n_hold=args.n_hold,
            n_swap=args.n_swap,
            commission=args.commission,
        )
        official_trade_dates = val_context.official_trade_dates()
        strategy_result = bt.run(
            combo_alpha,
            val_context.official_open_prices(),
            official_trade_dates,
            val_context.stock_data.stock_codes,
        )
        benchmark_result = compute_benchmark_result(
            reader.market(args.benchmark_code),
            official_trade_dates,
            start_offset=strategy_result["trade_date_offset"],
        )

        print(f"\n回测结果（验证集，对比 {args.benchmark_code}）")
        print(
            f"  策略: 年化={strategy_result['annual_return']:+.2%}  "
            f"夏普={strategy_result['sharpe_ratio']:.3f}  "
            f"回撤={strategy_result['max_drawdown']:.2%}"
        )
        print(
            f"  基准: 年化={benchmark_result['annual_return']:+.2%}  "
            f"夏普={benchmark_result['sharpe_ratio']:.3f}  "
            f"回撤={benchmark_result['max_drawdown']:.2%}"
        )

        save_json(
            {
                "factor_metrics": factor_metrics,
                "strategy_backtest": strategy_result,
                "benchmark_backtest": benchmark_result,
                "benchmark_code": args.benchmark_code,
                "warmup_days": warmup_days,
                "warmup_start": val_context.loaded_start_date_str,
                "official_val_start": args.val_start,
            },
            os.path.join(args.save_dir, "validation_report.json"),
        )
        plot_backtest_comparison(
            strategy_result,
            benchmark_result,
            os.path.join(args.save_dir, "backtest_vs_benchmark.png"),
            benchmark_label=args.benchmark_code,
        )
        plot_equity_curves(
            strategy_result,
            benchmark_result,
            os.path.join(args.save_dir, "equity_curve_vs_benchmark.png"),
            benchmark_label=args.benchmark_code,
        )


if __name__ == "__main__":
    main()
