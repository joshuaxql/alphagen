"""
回测引擎 — Top-k / Drop-n 交易策略

要求:
  1. 每天对所有股票计算 alpha 分数
  2. 第一天等权买入 top-n 只
  3. 之后每天：卖出持仓中得分最低的 k 只，买入全市场得分最高的 k 只
  4. 输出：年化收益率、夏普比率、最大回撤
"""

import numpy as np
import os
import json
from calculator import StockData, AlphaCalculator
from combination import AlphaCombinationModel
from expression import parse_formula
from reporting import (
    compute_benchmark_result,
    compute_factor_metrics,
    plot_backtest_comparison,
    plot_equity_curves,
    save_json,
    estimate_required_warmup_days,
    WarmupDataContext,
)


class Backtester:
    """Top-k/Drop-n 回测。"""

    RESULT_START_OFFSET = 2

    def __init__(
        self,
        n_hold: int = 20,
        n_swap: int = 3,
        commission: float = 0.001,
    ):
        self.n_hold = n_hold
        self.n_swap = n_swap
        self.commission = commission

    def run(
        self,
        alpha_values: np.ndarray,
        open_prices: np.ndarray,
        trade_dates: list,
        stock_codes: list,
    ) -> dict:
        """
        时序：第 t 天收盘算出 alpha → 第 t+1 天开盘交易 → 持仓收益 = open[t+2]/open[t+1]-1

        参数:
            alpha_values: (n_days, n_stocks) alpha 预测分数（第 t 天收盘后可得）
            open_prices:  (n_days, n_stocks) 开盘价（用于计算交易价格和收益）
            trade_dates:  交易日列表
            stock_codes:  股票代码列表

        返回: 回测结果 dict
        """
        n_days, n_stocks = alpha_values.shape
        assert open_prices.shape == (n_days, n_stocks)

        portfolio_values = [1.0]
        holdings = set()  # 持仓股票索引集合
        daily_pnl = []

        # t 是信号日（收盘后得到 alpha[t]）
        # 交易在 t+1 开盘执行，收益 = open[t+2]/open[t+1] - 1
        # 因此 t 最大到 n_days-3（需要 open[t+1] 和 open[t+2]）
        for t in range(n_days - 2):
            scores = alpha_values[t].copy()

            # 跳过 alpha 全 nan 的日子
            valid = ~np.isnan(scores)
            if valid.sum() < self.n_hold:
                portfolio_values.append(portfolio_values[-1])
                daily_pnl.append(0.0)
                continue

            # 将 nan 分数设为 -inf（排到最后）
            scores[~valid] = -np.inf

            trade_cost = 0.0

            if len(holdings) == 0:
                # 首次建仓：选 top n_hold，在 t+1 开盘买入
                top_idx = np.argsort(scores)[-self.n_hold :]
                holdings = set(top_idx.tolist())
                n_buys = len(holdings)
                trade_cost = n_buys * self.commission / max(n_buys, 1)
            else:
                # 调仓：卖出持仓中得分最低的 k 只
                hold_scores = [(idx, scores[idx]) for idx in holdings]
                hold_scores.sort(key=lambda x: x[1])
                to_sell = [idx for idx, _ in hold_scores[: self.n_swap]]

                # 买入全市场得分最高的 k 只（排除已持仓）
                ranking = np.argsort(scores)[::-1]
                to_buy = []
                for idx in ranking:
                    if idx not in holdings and len(to_buy) < self.n_swap:
                        to_buy.append(idx)
                    if len(to_buy) >= self.n_swap:
                        break

                # 执行调仓
                for idx in to_sell:
                    holdings.discard(idx)
                for idx in to_buy:
                    holdings.add(idx)

                n_trades = len(to_sell) + len(to_buy)
                trade_cost = n_trades * self.commission / max(len(holdings), 1)

            # 持仓收益：open[t+2] / open[t+1] - 1
            hold_list = list(holdings)
            hold_rets = []
            for idx in hold_list:
                o1 = open_prices[t + 1, idx]
                o2 = open_prices[t + 2, idx]
                if np.isnan(o1) or np.isnan(o2) or abs(o1) < 1e-10:
                    hold_rets.append(0.0)
                else:
                    hold_rets.append(o2 / o1 - 1)

            avg_ret = np.mean(hold_rets) if hold_rets else 0.0
            net_ret = avg_ret - trade_cost
            portfolio_values.append(portfolio_values[-1] * (1 + net_ret))
            daily_pnl.append(net_ret)

        # 计算指标
        pv = np.array(portfolio_values[1:])
        pnl = np.array(daily_pnl)

        total_return = pv[-1] - 1 if len(pv) > 0 else 0
        n_years = len(pnl) / 242  # 约 242 个交易日/年
        annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

        pnl_clean = pnl[~np.isnan(pnl)]
        if len(pnl_clean) > 1 and pnl_clean.std() > 1e-12:
            sharpe = pnl_clean.mean() / pnl_clean.std() * np.sqrt(242)
        else:
            sharpe = 0.0

        cum_max = np.maximum.accumulate(pv)
        drawdown = (cum_max - pv) / np.where(cum_max > 0, cum_max, 1)
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "n_days": len(pnl),
            "daily_returns": pnl.tolist(),
            "portfolio_values": pv.tolist(),
            "trade_dates": trade_dates[
                self.RESULT_START_OFFSET : self.RESULT_START_OFFSET + len(pv)
            ],
            "initial_date": (
                trade_dates[self.RESULT_START_OFFSET - 1]
                if len(trade_dates) >= self.RESULT_START_OFFSET and len(pv) > 0
                else None
            ),
            "trade_date_offset": self.RESULT_START_OFFSET,
        }


def load_pool_alpha_from_file(
    pool_file: str,
    stock_data: StockData,
    target: np.ndarray | None = None,
    verbose: bool = False,
    official_start_index: int = 0,
) -> dict:
    with open(pool_file, "r", encoding="utf-8") as f:
        pool_info = json.load(f)

    calc = AlphaCalculator(stock_data)
    alpha_values_list = []
    weights_list = []
    formulas = []
    exprs = []
    loaded = 0

    for i, info in enumerate(pool_info):
        formula = info["formula"]
        saved_weight = info.get("weight", 1.0)
        tree = parse_formula(formula)
        if tree is None:
            if verbose:
                print(f"  [跳过] alpha {i}: 无法解析公式 '{formula}'")
            continue

        raw_val = calc.evaluate(tree)
        if raw_val is None or np.all(np.isnan(raw_val)):
            if verbose:
                print(f"  [跳过] alpha {i}: 计算结果无效 '{formula}'")
            continue

        alpha_values_list.append(raw_val)
        weights_list.append(saved_weight)
        formulas.append(formula)
        exprs.append(tree)
        loaded += 1

        if verbose and target is not None:
            val_metrics = compute_factor_metrics(
                raw_val[official_start_index:],
                target[official_start_index:],
            )
            print(
                f"  alpha {i}: val_IC={val_metrics['ic']:+.4f}  "
                f"W={saved_weight:+.4f}  {formula}"
            )

    weights = np.array(weights_list, dtype=np.float64)
    combo_alpha = np.zeros((stock_data.n_days, stock_data.n_stocks), dtype=np.float64)
    for val, w in zip(alpha_values_list, weights):
        combo_alpha += w * val

    factor_metrics = (
        compute_factor_metrics(
            combo_alpha[official_start_index:],
            target[official_start_index:],
        )
        if target is not None
        else None
    )
    return {
        "alpha_values": alpha_values_list,
        "weights": weights,
        "formulas": formulas,
        "exprs": exprs,
        "loaded": loaded,
        "combo_alpha": combo_alpha,
        "factor_metrics": factor_metrics,
    }


def compute_combination_alpha(
    combination: AlphaCombinationModel,
    stock_data: StockData,
) -> np.ndarray:
    """用组合模型权重计算最终 alpha 值。"""
    if combination.pool_size == 0:
        return np.zeros((stock_data.n_days, stock_data.n_stocks))

    combo = np.zeros((stock_data.n_days, stock_data.n_stocks))
    for i, val in enumerate(combination.alpha_values):
        combo += combination.weights[i] * val
    return combo


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--val_start", default="20260101")
    parser.add_argument("--val_end", default="20260510")
    parser.add_argument("--pool_file", default="outputs/pool_best.json")
    parser.add_argument("--n_hold", type=int, default=20)
    parser.add_argument("--n_swap", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--benchmark_code", default="000300.SH")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    from data import Data
    from common import ADJUST_PREV

    print("加载验证集数据...")
    reader = Data()
    warmup_context = WarmupDataContext(
        reader=reader,
        start_date=args.val_start,
        end_date=args.val_end,
        horizon=args.horizon,
        adjust=ADJUST_PREV,
        bj=False,
        st=False,
    )
    warmup_context.ensure_warmup_days(0)
    print("股票过滤: 默认排除北交所(BJ)和ST股票")

    print(f"加载 alpha 池: {args.pool_file}")
    with open(args.pool_file, "r", encoding="utf-8") as f:
        pool_info = json.load(f)
    pool_exprs = []
    for item in pool_info:
        tree = parse_formula(item["formula"])
        if tree is not None:
            pool_exprs.append(tree)
    warmup_days = estimate_required_warmup_days(pool_exprs)
    warmup_context.ensure_warmup_days(warmup_days)
    print(
        f"warm-up: {warmup_days} 个交易日, "
        f"扩展起点 = {warmup_context.loaded_start_date_str}, "
        f"正式统计起点 = {args.val_start}"
    )

    sd = warmup_context.stock_data
    target = warmup_context.target
    pool_state = load_pool_alpha_from_file(
        args.pool_file,
        sd,
        target=target,
        verbose=True,
        official_start_index=warmup_context.official_start_index,
    )

    print(f"\n成功加载 {pool_state['loaded']} 个 alpha")
    if pool_state["loaded"] == 0:
        print("没有可用的 alpha，退出")
        return

    combo_alpha = pool_state["combo_alpha"][warmup_context.official_start_index :]
    factor_metrics = pool_state["factor_metrics"]
    print(
        f"\n组合alpha验证期: IC = {factor_metrics['ic']:+.4f}, "
        f"ICIR = {factor_metrics['icir']:+.4f}, "
        f"Loss = {factor_metrics['loss']:.4f}"
    )
    print(f"Alpha池: {pool_state['loaded']} 个alpha")

    # 回测
    bt = Backtester(
        n_hold=args.n_hold,
        n_swap=args.n_swap,
        commission=args.commission,
    )
    official_trade_dates = warmup_context.official_trade_dates()
    result = bt.run(
        combo_alpha,
        warmup_context.official_open_prices(),
        official_trade_dates,
        sd.stock_codes,
    )
    benchmark_result = compute_benchmark_result(
        reader.market(args.benchmark_code),
        official_trade_dates,
        start_offset=result["trade_date_offset"],
    )

    print(f"\n{'='*50}")
    print(f"回测结果 ({result['n_days']} 个交易日, 对比 {args.benchmark_code})")
    print(f"{'='*50}")
    print(
        f"  策略总收益率:   {result['total_return']:+.2%}\n"
        f"  策略年化收益率: {result['annual_return']:+.2%}\n"
        f"  策略夏普比率:   {result['sharpe_ratio']:.3f}\n"
        f"  策略最大回撤:   {result['max_drawdown']:.2%}"
    )
    print(
        f"  基准总收益率:   {benchmark_result['total_return']:+.2%}\n"
        f"  基准年化收益率: {benchmark_result['annual_return']:+.2%}\n"
        f"  基准夏普比率:   {benchmark_result['sharpe_ratio']:.3f}\n"
        f"  基准最大回撤:   {benchmark_result['max_drawdown']:.2%}"
    )

    output_dir = args.output_dir or os.path.dirname(args.pool_file) or "."
    save_json(
        {
            "factor_metrics": factor_metrics,
            "strategy_backtest": result,
            "benchmark_backtest": benchmark_result,
            "benchmark_code": args.benchmark_code,
        },
        os.path.join(output_dir, "backtest_report.json"),
    )
    plot_backtest_comparison(
        result,
        benchmark_result,
        os.path.join(output_dir, "backtest_vs_benchmark.png"),
        benchmark_label=args.benchmark_code,
    )
    plot_equity_curves(
        result,
        benchmark_result,
        os.path.join(output_dir, "equity_curve_vs_benchmark.png"),
        benchmark_label=args.benchmark_code,
    )


def run_backtest(
    combination: AlphaCombinationModel,
    stock_data: StockData,
    n_hold: int = 20,
    n_swap: int = 3,
    commission: float = 0.001,
) -> dict:
    """便捷函数：直接从 combination 对象回测。"""
    alpha = compute_combination_alpha(combination, stock_data)
    bt = Backtester(n_hold=n_hold, n_swap=n_swap, commission=commission)
    result = bt.run(
        alpha, stock_data.data["open"], stock_data.trade_dates, stock_data.stock_codes
    )

    print(f"\n{'='*50}")
    print(f"回测结果 ({result['n_days']} 个交易日)")
    print(f"{'='*50}")
    print(f"  总收益率:   {result['total_return']:+.2%}")
    print(f"  年化收益率: {result['annual_return']:+.2%}")
    print(f"  夏普比率:   {result['sharpe_ratio']:.3f}")
    print(f"  最大回撤:   {result['max_drawdown']:.2%}")

    return result


if __name__ == "__main__":
    main()
