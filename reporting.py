"""
训练/因子/回测评估与可视化工具
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from calculator import AlphaCalculator, calc_daily_ic, normalize_alpha
from expression import (
    BinaryOpNode,
    ConstantNode,
    ExprNode,
    FeatureNode,
    TSBinaryOpNode,
    TSUnaryOpNode,
    UnaryOpNode,
    tree_to_formula,
)

TRADING_DAYS_PER_YEAR = 242
EMA_WARMUP_MULTIPLIER = 3


def calc_icir(
    alpha_values: np.ndarray,
    target: np.ndarray,
    already_normed: bool = False,
) -> float:
    """ICIR = mean(daily_ic) / std(daily_ic)。"""
    daily_ic = calc_daily_ic(alpha_values, target, already_normed=already_normed)
    valid = daily_ic[~np.isnan(daily_ic)]
    if len(valid) < 2:
        return 0.0
    std = np.std(valid, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(np.mean(valid) / std)


def calc_factor_loss(
    alpha_values: np.ndarray,
    target: np.ndarray,
    already_normed_alpha: bool = False,
    already_normed_target: bool = False,
) -> float:
    """用标准化后的 alpha 与 target 的 MSE 作为因子损失。"""
    norm_alpha = alpha_values if already_normed_alpha else normalize_alpha(alpha_values)
    norm_target = target if already_normed_target else normalize_alpha(target)

    mask = ~(np.isnan(norm_alpha) | np.isnan(norm_target))
    if not np.any(mask):
        return 0.0

    diff = norm_alpha[mask] - norm_target[mask]
    return float(np.mean(diff * diff))


def compute_factor_metrics(
    alpha_values: np.ndarray,
    target: np.ndarray,
    already_normed_alpha: bool = False,
    already_normed_target: bool = False,
) -> Dict[str, float]:
    daily_ic = calc_daily_ic(alpha_values, target, already_normed=already_normed_alpha)
    valid = daily_ic[~np.isnan(daily_ic)]
    ic = float(np.mean(valid)) if len(valid) > 0 else 0.0
    icir = calc_icir(alpha_values, target, already_normed=already_normed_alpha)
    loss = calc_factor_loss(
        alpha_values,
        target,
        already_normed_alpha=already_normed_alpha,
        already_normed_target=already_normed_target,
    )
    return {
        "loss": loss,
        "ic": ic,
        "icir": icir,
        "n_daily_ic": int(len(valid)),
    }


def combine_alpha_values(
    alpha_values: List[np.ndarray],
    weights: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    combo = np.zeros(shape, dtype=np.float64)
    for i, val in enumerate(alpha_values):
        combo += weights[i] * val
    return combo


def evaluate_weighted_alpha_values(
    alpha_values: List[np.ndarray],
    weights: np.ndarray,
    target: np.ndarray,
    already_normed_target: bool = False,
) -> Dict[str, float]:
    combo = combine_alpha_values(alpha_values, weights, target.shape)
    return compute_factor_metrics(
        combo,
        target,
        already_normed_alpha=False,
        already_normed_target=already_normed_target,
    )


def estimate_expr_lookback(expr: ExprNode) -> int:
    """估算表达式首次得到有效值所需的最少观测长度。"""
    if isinstance(expr, (FeatureNode, ConstantNode)):
        return 1

    if isinstance(expr, UnaryOpNode):
        return estimate_expr_lookback(expr.operand)

    if isinstance(expr, BinaryOpNode):
        return max(
            estimate_expr_lookback(expr.left),
            estimate_expr_lookback(expr.right),
        )

    if isinstance(expr, TSUnaryOpNode):
        base = estimate_expr_lookback(expr.operand)
        t = int(expr.time_delta)
        name = expr.operator.name

        if name in ("Ref", "Delta"):
            return base + t
        if name == "EMA":
            return base + EMA_WARMUP_MULTIPLIER * t - 1
        return base + t - 1

    if isinstance(expr, TSBinaryOpNode):
        base = max(
            estimate_expr_lookback(expr.left),
            estimate_expr_lookback(expr.right),
        )
        return base + int(expr.time_delta) - 1

    raise TypeError(f"Unsupported expression node: {type(expr)}")


def estimate_required_warmup_days(exprs: List[ExprNode]) -> int:
    if not exprs:
        return 0
    return max(estimate_expr_lookback(expr) - 1 for expr in exprs)


def estimate_max_generated_expr_warmup_days(
    max_seq_len: int,
    time_deltas: List[int],
) -> int:
    """
    为生成器可产生的表达式给出一个保守 warm-up 上界。

    最深的时序嵌套链形如：
      feature -> TS(feature, t) -> TS(TS(feature, t), t) -> ...
    一个特征占 1 个 token，每多嵌套一层时序算子至少新增
    `time_delta + operator` 两个 token，因此最大嵌套层数约为
    `(max_seq_len - 1) // 2`。
    """
    if max_seq_len <= 1 or not time_deltas:
        return 0
    max_time_delta = max(int(td) for td in time_deltas)
    max_ts_depth = max((int(max_seq_len) - 1) // 2, 0)
    return max_ts_depth * max_time_delta * EMA_WARMUP_MULTIPLIER


class CombinationEvaluator:
    """在指定数据集上评估一组表达式/权重。"""

    def __init__(self, stock_data, target: np.ndarray):
        self.stock_data = stock_data
        self.target = target
        self.calculator = AlphaCalculator(stock_data)
        self._cache: Dict[str, Optional[np.ndarray]] = {}

    def evaluate_expr(self, expr: ExprNode) -> Optional[np.ndarray]:
        formula = tree_to_formula(expr)
        if formula not in self._cache:
            self._cache[formula] = self.calculator.evaluate(expr)
        return self._cache[formula]

    def build_combination_alpha(
        self,
        exprs: List[ExprNode],
        weights: np.ndarray,
    ) -> np.ndarray:
        combo = np.zeros(
            (self.stock_data.n_days, self.stock_data.n_stocks), dtype=np.float64
        )
        for i, expr in enumerate(exprs):
            val = self.evaluate_expr(expr)
            if val is None:
                continue
            combo += weights[i] * val
        return combo

    def evaluate(
        self,
        exprs: List[ExprNode],
        weights: np.ndarray,
        start_index: int = 0,
    ) -> Dict[str, float]:
        combo = self.build_combination_alpha(exprs, weights)
        return compute_factor_metrics(
            combo[start_index:],
            self.target[start_index:],
            already_normed_alpha=False,
            already_normed_target=False,
        )


class WarmupDataContext:
    """为验证/回测按需加载 warm-up 历史，但正式统计从 official_start 开始。"""

    def __init__(
        self,
        reader,
        start_date: str,
        end_date: str,
        horizon: int,
        adjust,
        bj: bool = False,
        st: bool = False,
    ):
        self.reader = reader
        self.start_date_str = start_date
        self.end_date_str = end_date
        self.start_date = datetime.strptime(start_date, "%Y%m%d").date()
        self.horizon = horizon
        self.adjust = adjust
        self.bj = bj
        self.st = st

        self.trade_cal = reader.trade_cal()
        self.loaded_warmup_days = -1
        self.loaded_start_date_str = None
        self.df_len = 0
        self.stock_data = None
        self.target = None
        self.official_start_index = 0
        self.evaluator = None

    def _compute_extended_start_date(self, warmup_days: int) -> str:
        open_dates = self.trade_cal["cal_date"].tolist()
        official_idx = next(
            (i for i, d in enumerate(open_dates) if d >= self.start_date),
            None,
        )
        if official_idx is None:
            raise ValueError(
                f"start_date {self.start_date_str} is outside trade calendar"
            )
        start_idx = max(0, official_idx - warmup_days)
        return open_dates[start_idx].strftime("%Y%m%d")

    def ensure_warmup_days(self, warmup_days: int) -> None:
        warmup_days = max(int(warmup_days), 0)
        if self.stock_data is not None and warmup_days <= self.loaded_warmup_days:
            return

        extended_start = self._compute_extended_start_date(warmup_days)
        df = self.reader.daily(
            start_date=extended_start,
            end_date=self.end_date_str,
            bj=self.bj,
            st=self.st,
            adjust=self.adjust,
        )
        df = df[
            ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "vwap"]
        ]
        self.df_len = len(df)
        from calculator import StockData

        self.stock_data = StockData(df)
        self.target = self.stock_data.get_target(horizon=self.horizon)
        self.official_start_index = next(
            (
                i
                for i, d in enumerate(self.stock_data.trade_dates)
                if d >= self.start_date
            ),
            0,
        )
        keep_mask = np.any(
            np.isfinite(self.stock_data.data["open"][self.official_start_index :]),
            axis=0,
        )
        if not np.any(keep_mask):
            raise ValueError("No stocks with valid official-period data after warm-up.")
        if not np.all(keep_mask):
            for feat, values in self.stock_data.data.items():
                self.stock_data.data[feat] = values[:, keep_mask]
            self.stock_data.stock_codes = [
                code
                for code, keep in zip(self.stock_data.stock_codes, keep_mask)
                if keep
            ]
            self.stock_data.n_stocks = len(self.stock_data.stock_codes)
            self.target = self.target[:, keep_mask]
        self.evaluator = CombinationEvaluator(self.stock_data, self.target)
        self.loaded_warmup_days = warmup_days
        self.loaded_start_date_str = extended_start

    def ensure_warmup_for_exprs(self, exprs: List[ExprNode]) -> int:
        warmup_days = estimate_required_warmup_days(exprs)
        self.ensure_warmup_days(warmup_days)
        return warmup_days

    def preload_generator_warmup(
        self,
        max_seq_len: int,
        time_deltas: List[int],
    ) -> int:
        warmup_days = estimate_max_generated_expr_warmup_days(
            max_seq_len=max_seq_len,
            time_deltas=time_deltas,
        )
        self.ensure_warmup_days(warmup_days)
        return warmup_days

    def evaluate(self, exprs: List[ExprNode], weights: np.ndarray) -> Dict[str, float]:
        self.ensure_warmup_for_exprs(exprs)
        return self.evaluator.evaluate(
            exprs, weights, start_index=self.official_start_index
        )

    def build_official_combination_alpha(
        self,
        exprs: List[ExprNode],
        weights: np.ndarray,
    ) -> np.ndarray:
        self.ensure_warmup_for_exprs(exprs)
        combo = self.evaluator.build_combination_alpha(exprs, weights)
        return combo[self.official_start_index :]

    def official_trade_dates(self) -> List:
        return self.stock_data.trade_dates[self.official_start_index :]

    def official_open_prices(self) -> np.ndarray:
        return self.stock_data.data["open"][self.official_start_index :]

    def official_target(self) -> np.ndarray:
        return self.target[self.official_start_index :]


def compute_return_metrics(
    daily_returns: np.ndarray,
    trade_dates: List,
    initial_date=None,
) -> Dict[str, object]:
    pnl = np.asarray(daily_returns, dtype=np.float64)
    pv = np.cumprod(1.0 + pnl) if len(pnl) > 0 else np.array([], dtype=np.float64)
    total_return = float(pv[-1] - 1.0) if len(pv) > 0 else 0.0

    n_years = len(pnl) / TRADING_DAYS_PER_YEAR
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    pnl_clean = pnl[~np.isnan(pnl)]
    if len(pnl_clean) > 1 and pnl_clean.std() > 1e-12:
        sharpe = pnl_clean.mean() / pnl_clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = 0.0

    if len(pv) > 0:
        cum_max = np.maximum.accumulate(pv)
        drawdown = (cum_max - pv) / np.where(cum_max > 0, cum_max, 1.0)
        max_drawdown = float(drawdown.max())
    else:
        max_drawdown = 0.0

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_drawdown,
        "n_days": int(len(pnl)),
        "daily_returns": pnl.tolist(),
        "portfolio_values": pv.tolist(),
        "trade_dates": list(trade_dates),
        "initial_date": initial_date,
    }


def compute_benchmark_result(
    market_df,
    full_trade_dates: List,
    start_offset: int = 2,
    price_col: str = "open",
) -> Dict[str, object]:
    start_offset = int(start_offset)
    if start_offset < 0:
        raise ValueError("start_offset must be non-negative")
    market = market_df.sort_values("trade_date")
    price_map = dict(zip(market["trade_date"], market[price_col]))

    trade_dates = []
    daily_returns = []
    initial_date = (
        full_trade_dates[start_offset - 1]
        if len(full_trade_dates) >= start_offset and start_offset > 0
        else None
    )
    for idx in range(start_offset, len(full_trade_dates)):
        prev_date = full_trade_dates[idx - 1]
        curr_date = full_trade_dates[idx]
        p1 = price_map.get(prev_date, np.nan)
        p2 = price_map.get(curr_date, np.nan)

        if np.isnan(p1) or np.isnan(p2) or abs(p1) < 1e-10:
            ret = 0.0
        else:
            ret = float(p2 / p1 - 1.0)

        trade_dates.append(curr_date)
        daily_returns.append(ret)

    return compute_return_metrics(
        np.array(daily_returns, dtype=np.float64),
        trade_dates,
        initial_date=initial_date,
    )


def build_plot_series(
    trade_dates: List, portfolio_values: List[float], initial_date=None
):
    dates = list(trade_dates)
    values = list(portfolio_values)
    if initial_date is not None:
        dates = [initial_date] + dates
        values = [1.0] + values
    return dates, values


def save_training_history(history: List[Dict[str, object]], save_dir: str) -> None:
    if not history:
        return

    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "training_history.json")
    csv_path = os.path.join(save_dir, "training_history.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    columns = list(history[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(columns) + "\n")
        for row in history:
            values = []
            for col in columns:
                val = row.get(col, "")
                values.append(str(val))
            f.write(",".join(values) + "\n")


def save_json(payload: Dict[str, object], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _default(o):
        if hasattr(o, "isoformat"):
            return o.isoformat()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(
            f"Object of type {o.__class__.__name__} is not JSON serializable"
        )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_default)


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    return plt, PercentFormatter


def plot_training_history(history: List[Dict[str, object]], output_path: str) -> None:
    if not history:
        return

    plt, _ = _import_pyplot()

    iterations = [row["iteration"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    train_ic = [row["train_ic"] for row in history]
    train_icir = [row["train_icir"] for row in history]
    val_loss = [row.get("val_loss") for row in history]
    val_ic = [row.get("val_ic") for row in history]
    val_icir = [row.get("val_icir") for row in history]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    colors = {"train": "#0B6E4F", "val": "#C84C09"}

    axes[0].plot(
        iterations, train_loss, label="Train", color=colors["train"], linewidth=2
    )
    if any(v is not None for v in val_loss):
        axes[0].plot(
            iterations, val_loss, label="Validation", color=colors["val"], linewidth=2
        )
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Factor Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        iterations, train_ic, label="Train", color=colors["train"], linewidth=2
    )
    if any(v is not None for v in val_ic):
        axes[1].plot(
            iterations, val_ic, label="Validation", color=colors["val"], linewidth=2
        )
    axes[1].set_ylabel("IC")
    axes[1].set_title("IC")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(
        iterations, train_icir, label="Train", color=colors["train"], linewidth=2
    )
    if any(v is not None for v in val_icir):
        axes[2].plot(
            iterations, val_icir, label="Validation", color=colors["val"], linewidth=2
        )
    axes[2].set_ylabel("ICIR")
    axes[2].set_xlabel("Iteration")
    axes[2].set_title("ICIR")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_backtest_comparison(
    strategy_result: Dict[str, object],
    benchmark_result: Dict[str, object],
    output_path: str,
    benchmark_label: str = "CSI300",
) -> None:
    plt, PercentFormatter = _import_pyplot()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    labels = ["Strategy", benchmark_label]
    colors = ["#1565C0", "#EF6C00"]
    metrics = [
        ("annual_return", "Annual Return", True),
        ("sharpe_ratio", "Sharpe Ratio", False),
        ("max_drawdown", "Max Drawdown", True),
    ]

    for ax, (key, title, is_pct) in zip(axes, metrics):
        values = [strategy_result[key], benchmark_result[key]]
        bars = ax.bar(labels, values, color=colors, width=0.55)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if is_pct:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        for bar, value in zip(bars, values):
            text = f"{value:+.2%}" if is_pct else f"{value:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curves(
    strategy_result: Dict[str, object],
    benchmark_result: Dict[str, object],
    output_path: str,
    benchmark_label: str = "CSI300",
) -> None:
    plt, _ = _import_pyplot()

    fig, ax = plt.subplots(figsize=(12, 5))
    strategy_dates, strategy_values = build_plot_series(
        strategy_result["trade_dates"],
        strategy_result["portfolio_values"],
        initial_date=strategy_result.get("initial_date"),
    )
    benchmark_dates, benchmark_values = build_plot_series(
        benchmark_result["trade_dates"],
        benchmark_result["portfolio_values"],
        initial_date=benchmark_result.get("initial_date"),
    )
    ax.plot(
        strategy_dates,
        strategy_values,
        label="Strategy",
        color="#1565C0",
        linewidth=2,
    )
    ax.plot(
        benchmark_dates,
        benchmark_values,
        label=benchmark_label,
        color="#EF6C00",
        linewidth=2,
    )
    ax.set_title("Equity Curve Comparison")
    ax.set_xlabel("Trade Date")
    ax.set_ylabel("Net Value")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
