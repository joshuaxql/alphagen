"""
训练管线

流程：
  1. 加载数据，构建 StockData 和 target
  2. 初始化 AlphaCombinationModel 和 PPO Agent
  3. 循环：生成 episode → 评估 alpha → 更新组合模型 → PPO 更新
  4. 保存最终 alpha 池
"""

import os
import json
import time
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
    SEP_TOKEN,
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
from generator import PPOAgent, Episode
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
SEP_IDX = TOKEN_TO_IDX[Token(TokenType.SEP, SEP_TOKEN)]


# ============================================================
# 0. 课程学习管理器
# ============================================================
class CurriculumManager:
    """课程学习：从简单表达式逐步过渡到复杂表达式。

    分阶段训练，逐步增加表达式复杂度上限：
      阶段1: 只允许时序一元操作符 (TS-U)，如 Mean($close, 20)
      阶段2: 允许时序操作符 (TS-U + TS-B)，如 Corr($close, $vol, 20)
      阶段3: 允许所有操作符，限制序列长度
      阶段4: 完全放开

    参数:
        total_iterations: 总训练轮数
        max_seq_len: 最大序列长度
        stage_ratios: 各阶段占比，默认 [0.25, 0.25, 0.25, 0.25]
    """

    # 各阶段允许的操作符类别
    STAGE_ALLOWED_OPS = [
        {"TS-U"},                    # 阶段1: 只有时序一元
        {"TS-U", "TS-B"},            # 阶段2: 时序一元+二元
        {"TS-U", "TS-B", "CS-U"},    # 阶段3: 加上截面一元
        None,                         # 阶段4: 全部允许
    ]

    # 各阶段的最大序列长度占比
    STAGE_SEQ_LEN_RATIO = [0.4, 0.6, 0.8, 1.0]

    def __init__(
        self,
        total_iterations: int,
        max_seq_len: int = 20,
        stage_ratios: list[float] | None = None,
    ):
        self.total_iterations = total_iterations
        self.max_seq_len = max_seq_len
        ratios = stage_ratios or [0.25, 0.25, 0.25, 0.25]
        # 计算各阶段的结束 iteration
        self.stage_boundaries = []
        cumulative = 0
        for r in ratios:
            cumulative += r
            self.stage_boundaries.append(int(cumulative * total_iterations))

    def get_stage(self, iteration: int) -> int:
        """返回当前阶段索引 (0-3)"""
        for i, boundary in enumerate(self.stage_boundaries):
            if iteration < boundary:
                return i
        return len(self.stage_boundaries) - 1

    def get_allowed_op_categories(self, iteration: int) -> set[str] | None:
        """返回当前允许的操作符类别集合，None 表示全部允许"""
        stage = self.get_stage(iteration)
        return self.STAGE_ALLOWED_OPS[stage]

    def get_max_seq_len(self, iteration: int) -> int:
        """返回当前阶段的最大序列长度"""
        stage = self.get_stage(iteration)
        ratio = self.STAGE_SEQ_LEN_RATIO[stage]
        return max(4, int(self.max_seq_len * ratio))

    def describe(self, iteration: int) -> str:
        """返回当前阶段的描述"""
        stage = self.get_stage(iteration)
        allowed = self.STAGE_ALLOWED_OPS[stage]
        seq_len = self.get_max_seq_len(iteration)
        if allowed is None:
            return f"阶段{stage+1}: 全部操作符, max_len={seq_len}"
        return f"阶段{stage+1}: {allowed}, max_len={seq_len}"


# ============================================================
# 1. 单 episode 生成
# ============================================================
def collect_episode(
    agent: PPOAgent,
    builder: RPNBuilder,
    device: str = "cpu",
    allowed_op_categories: set[str] | None = None,
) -> Episode:
    """生成一条 RPN 序列。

    参数:
        allowed_op_categories: 允许的操作符类别，None 表示全部允许。
    """
    builder.reset()
    ep = Episode()

    # 起始 BEG
    builder.step(BEG_IDX)
    ep.token_ids.append(BEG_IDX)

    hidden = agent.net.init_hidden(batch_size=1, device=device)

    while not builder.done:
        valid_mask = builder.get_valid_mask(
            allowed_op_categories=allowed_op_categories
        )

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
# 2. 评估 episode 并计算 reward
# ============================================================
def evaluate_episode(
    ep: Episode,
    combination: AlphaCombinationModel,
) -> dict:
    """
    解析 episode 的 token 序列，评估 alpha，加入组合模型。
    返回结构化结果：
      - reward: PPO 使用的奖励（多目标）
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
        icir_delta = combination.last_add_icir_delta

        # 计算候选 alpha 的 Rank IC（Spearman 相关系数）
        candidate_rank_ic = calc_rank_ic(alpha_val, combination.target)

        # 正负平衡奖励：如果新因子IC方向与多数池内因子相反，给奖励
        pos_count = sum(1 for ic in combination.ic_vector if ic > 0)
        neg_count = len(combination.ic_vector) - pos_count
        balance_bonus = 0.0
        candidate_ic_sign = combination.last_add_candidate_ic
        if candidate_ic_sign > 0 and pos_count < neg_count:
            balance_bonus = 0.05
        elif candidate_ic_sign < 0 and neg_count < pos_count:
            balance_bonus = 0.05

        # 冗余惩罚：与已有因子相关性越高，惩罚越大
        redundancy_penalty = 0.0
        if combination.pool_size > 1:
            # 取最后一个因子（刚加入的）与池中其他因子的最大互相关
            last_idx = combination.pool_size - 1
            max_mutual = max(
                abs(combination.ic_matrix[last_idx, j])
                for j in range(last_idx)
            ) if last_idx > 0 else 0.0
            if max_mutual > 0.7:
                redundancy_penalty = (max_mutual - 0.7) * 0.5

        # 多目标奖励
        reward = (
            10.0 * ic_delta        # IC增量（主目标）
            + 3.0 * icir_delta     # ICIR增量（稳定性）
            + 2.0 * candidate_rank_ic  # Rank IC（鲁棒性）
            + balance_bonus        # 正负平衡
            - redundancy_penalty   # 冗余惩罚
        )

        return _result(
            reward=reward,
            accepted=True,
            formula=formula,
            candidate_ic=combination.last_add_candidate_ic,
            ic_delta=ic_delta,
            status=combination.last_add_status,
        )

    rejection_penalty = {
        "rejected_low_ic": -0.2,
        "rejected_redundant": -0.15,
        "rejected_no_improve": -0.1,
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
# 3. 主训练函数
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
    gae_lambda: float = 0.95,
    enable_curriculum: bool = True,
    model_type: str = "rnn",
    # Transformer 专用参数
    tf_embed_dim: int = 64,
    tf_nhead: int = 4,
    tf_num_layers: int = 3,
    tf_dim_feedforward: int = 256,
    tf_dropout: float = 0.1,
):
    """Algorithm 2 完整训练循环。"""
    os.makedirs(save_dir, exist_ok=True)

    # 初始化组合模型
    combination = AlphaCombinationModel(stock_data, target, max_pool_size=max_pool_size)

    # 根据 model_type 初始化网络
    if model_type == "transformer":
        from generator_transformer import AlphaGenTransformer
        net = AlphaGenTransformer(
            vocab_size=VOCAB_SIZE,
            embed_dim=tf_embed_dim,
            nhead=tf_nhead,
            num_layers=tf_num_layers,
            dim_feedforward=tf_dim_feedforward,
            dropout=tf_dropout,
            max_seq_len=max_seq_len,
        )
        print(f"模型: Transformer (embed_dim={tf_embed_dim}, nhead={tf_nhead}, "
              f"layers={tf_num_layers}, ff_dim={tf_dim_feedforward}, dropout={tf_dropout})")
    else:
        from generator import AlphaGenNet
        net = AlphaGenNet(vocab_size=VOCAB_SIZE)
        print(f"模型: LSTM (embed_dim=32, hidden_dim=128, num_layers=2)")

    agent = PPOAgent(net, lr=lr, device=device, gamma=gamma, gae_lambda=gae_lambda)

    # 课程学习管理器
    curriculum = CurriculumManager(
        total_iterations=num_iterations, max_seq_len=max_seq_len
    ) if enable_curriculum else None

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

    print(f"开始训练: {num_iterations} 轮, 每轮 {episodes_per_iter} episode")
    print(f"设备: {device}, 股票数: {stock_data.n_stocks}, 交易日: {stock_data.n_days}")
    print("股票过滤: 默认排除北交所(BJ)和ST股票")
    if curriculum is not None:
        print(f"课程学习: 已启用 ({curriculum.describe(0)} → {curriculum.describe(num_iterations-1)})")
    else:
        print("课程学习: 未启用")
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

    for iteration in range(num_iterations):
        t0 = time.time()
        episodes = []
        rewards = []
        valid_count = 0
        accepted_alphas = []

        # 课程学习：根据当前 iteration 调整参数
        allowed_ops = None
        if curriculum is not None:
            allowed_ops = curriculum.get_allowed_op_categories(iteration)
            builder.max_len = curriculum.get_max_seq_len(iteration)

        # 收集 episodes
        for _ in range(episodes_per_iter):
            ep = collect_episode(
                agent, builder, device=device,
                allowed_op_categories=allowed_ops,
            )

            if len(ep.actions) == 0:
                continue

            episode_result = evaluate_episode(ep, combination)
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
            + (f"  [{curriculum.describe(iteration)}]" if curriculum else "")
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
            save_checkpoint(combination, net, save_dir, tag="best")

        # 每 20 轮保存
        if (iteration + 1) % 20 == 0:
            save_checkpoint(combination, net, save_dir, tag=f"iter{iteration+1}")

    # 最终保存
    save_checkpoint(combination, net, save_dir, tag="final")
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


def save_checkpoint(combination, net, save_dir, tag="latest"):
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


# ============================================================
# 4. 入口
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Alpha 因子生成训练管线")
    # ── 数据参数 ──
    parser.add_argument("--train_start", default="20190101", help="训练集起始日期 (YYYYMMDD)")
    parser.add_argument("--train_end", default="20241231", help="训练集结束日期 (YYYYMMDD)")
    parser.add_argument("--val_start", default="20250101", help="验证集起始日期 (YYYYMMDD)")
    parser.add_argument("--val_end", default="20251231", help="验证集结束日期 (YYYYMMDD)")
    # ── 训练参数 ──
    parser.add_argument("--iterations", type=int, default=20, help="训练轮数")
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
    parser.add_argument("--gamma", type=float, default=0.99, help="GAE discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--enable_curriculum", action="store_true", default=True, help="启用课程学习")
    parser.add_argument("--no_curriculum", action="store_false", dest="enable_curriculum", help="禁用课程学习")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备 (cpu/cuda)")
    # ── 模型选择 ──
    parser.add_argument(
        "--model", choices=["rnn", "transformer"], default="rnn",
        help="策略网络类型: rnn (LSTM, 默认) 或 transformer"
    )
    # ── Transformer 专用参数 ──
    parser.add_argument("--tf_embed_dim", type=int, default=64, help="[Transformer] embedding 维度")
    parser.add_argument("--tf_nhead", type=int, default=4, help="[Transformer] 注意力头数")
    parser.add_argument("--tf_num_layers", type=int, default=3, help="[Transformer] Encoder 层数")
    parser.add_argument("--tf_dim_feedforward", type=int, default=256, help="[Transformer] 前馈网络维度")
    parser.add_argument("--tf_dropout", type=float, default=0.1, help="[Transformer] Dropout 概率")
    args = parser.parse_args()

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
        gae_lambda=args.gae_lambda,
        enable_curriculum=args.enable_curriculum,
        model_type=args.model,
        tf_embed_dim=args.tf_embed_dim,
        tf_nhead=args.tf_nhead,
        tf_num_layers=args.tf_num_layers,
        tf_dim_feedforward=args.tf_dim_feedforward,
        tf_dropout=args.tf_dropout,
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
