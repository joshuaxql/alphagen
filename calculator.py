"""
Alpha因子计算引擎
- 将表达式树应用到股票DataFrame上，计算alpha值
- 计算IC（信息系数）

数据格式约定：
  输入DataFrame列: ts_code, trade_date, open, high, low, close, vol, vwap
  内部转换为: dict[feature_name] -> np.ndarray, shape=(n_days, n_stocks)
  即每个特征是一个 (交易日数 x 股票数) 的矩阵
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Optional
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    prange = range
from expression import (
    ExprNode,
    FeatureNode,
    ConstantNode,
    UnaryOpNode,
    BinaryOpNode,
    TSUnaryOpNode,
    TSBinaryOpNode,
    parse_rpn_to_tree,
    strip_special_tokens,
    tree_to_formula,
)
from tokens import Token


# ============================================================
# 1. 数据预处理：DataFrame -> 矩阵字典
# ============================================================
class StockData:
    """
    将长表格式的股票数据转换为矩阵格式。

    属性:
        data: dict[feature_name] -> ndarray(n_days, n_stocks)
        stock_codes: 股票代码列表
        trade_dates: 交易日列表
        n_days: 交易日数
        n_stocks: 股票数
    """

    def __init__(self, df: pd.DataFrame):
        """
        参数:
            df: 包含 ts_code, trade_date, open, high, low, close, vol, vwap 列
        """
        feature_names = ["open", "close", "high", "low", "vol", "vwap"]
        self.data: Dict[str, np.ndarray] = {}

        pivot0 = df.pivot(
            index="trade_date", columns="ts_code", values=feature_names[0]
        )
        pivot0 = pivot0.sort_index()  # 按日期排序

        self.trade_dates = pivot0.index.tolist()
        self.stock_codes = pivot0.columns.tolist()
        self.n_days = len(self.trade_dates)
        self.n_stocks = len(self.stock_codes)

        self.data[feature_names[0]] = pivot0.values.astype(np.float64)

        for feat in feature_names[1:]:
            pivot_f = df.pivot(index="trade_date", columns="ts_code", values=feat)
            pivot_f = pivot_f.reindex(index=self.trade_dates, columns=self.stock_codes)
            self.data[feat] = pivot_f.values.astype(np.float64)

    def get_target(self, horizon: int = 20) -> np.ndarray:
        """
        计算预测目标：未来horizon天的收益率
        target[t, i] = close[t+horizon, i] / close[t, i] - 1

        返回: ndarray(n_days, n_stocks)，最后horizon天为nan
        """
        close = self.data["close"]
        target = np.full_like(close, np.nan)
        if self.n_days > horizon:
            target[:-horizon] = close[horizon:] / close[:-horizon] - 1
        return target


# ============================================================
# 2. 操作符计算函数
# ============================================================


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """安全除法，分母为0时返回0"""
    result = np.where(np.abs(b) < 1e-10, 0.0, a / np.where(np.abs(b) < 1e-10, 1.0, b))
    return result


_CLIP_BOUND = 1e10  # 中间结果绝对值上限


def _as_f64_array(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


@njit(cache=True, parallel=True)
def _clip_and_zscore_numba(x: np.ndarray) -> np.ndarray:
    n_days, n_stocks = x.shape
    result = np.empty((n_days, n_stocks), dtype=np.float64)
    result[:] = np.nan

    for t in prange(n_days):
        count = 0
        mean = 0.0
        for s in range(n_stocks):
            val = x[t, s]
            if np.isfinite(val):
                mean += val
                count += 1

        if count < 2:
            continue

        mean /= count

        sq = 0.0
        for s in range(n_stocks):
            val = x[t, s]
            if np.isfinite(val):
                diff = val - mean
                sq += diff * diff

        sigma = np.sqrt(sq / (count - 1))
        if not np.isfinite(sigma) or sigma < 1e-10:
            for s in range(n_stocks):
                if np.isfinite(x[t, s]):
                    result[t, s] = 0.0
            continue

        lower = mean - 3.0 * sigma
        upper = mean + 3.0 * sigma

        clipped_mean = 0.0
        for s in range(n_stocks):
            val = x[t, s]
            if np.isfinite(val):
                if val < lower:
                    val = lower
                elif val > upper:
                    val = upper
                clipped_mean += val

        clipped_mean /= count

        clipped_sq = 0.0
        for s in range(n_stocks):
            val = x[t, s]
            if np.isfinite(val):
                if val < lower:
                    val = lower
                elif val > upper:
                    val = upper
                diff = val - clipped_mean
                clipped_sq += diff * diff

        sigma2 = np.sqrt(clipped_sq / (count - 1))
        if not np.isfinite(sigma2) or sigma2 < 1e-10:
            for s in range(n_stocks):
                if np.isfinite(x[t, s]):
                    result[t, s] = 0.0
            continue

        for s in range(n_stocks):
            val = x[t, s]
            if np.isfinite(val):
                if val < lower:
                    val = lower
                elif val > upper:
                    val = upper
                result[t, s] = (val - clipped_mean) / sigma2

    return result


@njit(cache=True, parallel=True)
def _rolling_sum_numba(x: np.ndarray, window: int) -> np.ndarray:
    n_days, n_stocks = x.shape
    result = np.empty((n_days, n_stocks), dtype=np.float64)
    result[:] = np.nan
    if n_days < window:
        return result

    for s in prange(n_stocks):
        running_sum = 0.0
        for t in range(n_days):
            val = x[t, s]
            if np.isfinite(val):
                running_sum += val
            if t >= window:
                old = x[t - window, s]
                if np.isfinite(old):
                    running_sum -= old
            if t >= window - 1:
                result[t, s] = running_sum

    return result


@njit(cache=True, parallel=True)
def _rolling_mean_numba(x: np.ndarray, window: int) -> np.ndarray:
    n_days, n_stocks = x.shape
    result = np.empty((n_days, n_stocks), dtype=np.float64)
    result[:] = np.nan
    if n_days < window:
        return result

    for s in prange(n_stocks):
        running_sum = 0.0
        running_count = 0
        for t in range(n_days):
            val = x[t, s]
            if np.isfinite(val):
                running_sum += val
                running_count += 1
            if t >= window:
                old = x[t - window, s]
                if np.isfinite(old):
                    running_sum -= old
                    running_count -= 1
            if t >= window - 1 and running_count > 0:
                result[t, s] = running_sum / running_count

    return result


@njit(cache=True, parallel=True)
def _rolling_var_numba(x: np.ndarray, window: int, ddof: int) -> np.ndarray:
    n_days, n_stocks = x.shape
    result = np.empty((n_days, n_stocks), dtype=np.float64)
    result[:] = np.nan
    if n_days < window:
        return result

    for s in prange(n_stocks):
        running_sum = 0.0
        running_sq_sum = 0.0
        running_count = 0
        for t in range(n_days):
            val = x[t, s]
            if np.isfinite(val):
                running_sum += val
                running_sq_sum += val * val
                running_count += 1
            if t >= window:
                old = x[t - window, s]
                if np.isfinite(old):
                    running_sum -= old
                    running_sq_sum -= old * old
                    running_count -= 1
            if t >= window - 1 and running_count > ddof:
                mean = running_sum / running_count
                var = (running_sq_sum - running_sum * mean) / (running_count - ddof)
                if var < 0.0 and var > -1e-12:
                    var = 0.0
                result[t, s] = var

    return result


@njit(cache=True, parallel=True)
def _rolling_cov_numba(
    x: np.ndarray, y: np.ndarray, window: int, ddof: int
) -> np.ndarray:
    n_days, n_stocks = x.shape
    result = np.empty((n_days, n_stocks), dtype=np.float64)
    result[:] = np.nan
    if n_days < window:
        return result

    for s in prange(n_stocks):
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        count = 0
        for t in range(n_days):
            xv = x[t, s]
            yv = y[t, s]
            if np.isfinite(xv) and np.isfinite(yv):
                sum_x += xv
                sum_y += yv
                sum_xy += xv * yv
                count += 1
            if t >= window:
                old_x = x[t - window, s]
                old_y = y[t - window, s]
                if np.isfinite(old_x) and np.isfinite(old_y):
                    sum_x -= old_x
                    sum_y -= old_y
                    sum_xy -= old_x * old_y
                    count -= 1
            if t >= window - 1 and count > ddof:
                result[t, s] = (sum_xy - (sum_x * sum_y) / count) / (count - ddof)

    return result


@njit(cache=True, parallel=True)
def _rolling_corr_numba(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    n_days, n_stocks = x.shape
    result = np.empty((n_days, n_stocks), dtype=np.float64)
    result[:] = np.nan
    if n_days < window:
        return result

    for s in prange(n_stocks):
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0
        sum_y2 = 0.0
        count = 0
        for t in range(n_days):
            xv = x[t, s]
            yv = y[t, s]
            if np.isfinite(xv) and np.isfinite(yv):
                sum_x += xv
                sum_y += yv
                sum_xy += xv * yv
                sum_x2 += xv * xv
                sum_y2 += yv * yv
                count += 1
            if t >= window:
                old_x = x[t - window, s]
                old_y = y[t - window, s]
                if np.isfinite(old_x) and np.isfinite(old_y):
                    sum_x -= old_x
                    sum_y -= old_y
                    sum_xy -= old_x * old_y
                    sum_x2 -= old_x * old_x
                    sum_y2 -= old_y * old_y
                    count -= 1
            if t >= window - 1 and count > 2:
                mean_x = sum_x / count
                mean_y = sum_y / count
                cov = sum_xy / count - mean_x * mean_y
                var_x = sum_x2 / count - mean_x * mean_x
                var_y = sum_y2 / count - mean_y * mean_y
                if var_x < 0.0 and var_x > -1e-12:
                    var_x = 0.0
                if var_y < 0.0 and var_y > -1e-12:
                    var_y = 0.0
                denom = np.sqrt(var_x * var_y)
                if denom > 1e-10:
                    result[t, s] = cov / denom
                else:
                    result[t, s] = 0.0

    return result


@njit(cache=True, parallel=True)
def _ema_numba(x: np.ndarray, alpha_val: float) -> np.ndarray:
    n_days, n_stocks = x.shape
    result = np.empty((n_days, n_stocks), dtype=np.float64)
    if n_days == 0:
        return result

    for s in prange(n_stocks):
        result[0, s] = x[0, s]
        for t in range(1, n_days):
            curr = x[t, s]
            prev = result[t - 1, s]
            curr_nan = np.isnan(curr)
            prev_nan = np.isnan(prev)
            if curr_nan:
                result[t, s] = prev
            elif prev_nan:
                result[t, s] = curr
            else:
                result[t, s] = alpha_val * curr + (1.0 - alpha_val) * prev

    return result


@njit(cache=True, parallel=True)
def _calc_daily_ic_numba(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    n_days, n_stocks = values.shape
    daily_ic = np.empty(n_days, dtype=np.float64)
    daily_ic[:] = np.nan

    for t in prange(n_days):
        count = 0
        mean_a = 0.0
        mean_y = 0.0
        for s in range(n_stocks):
            a = values[t, s]
            y = targets[t, s]
            if np.isfinite(a) and np.isfinite(y):
                mean_a += a
                mean_y += y
                count += 1

        if count < 3:
            continue

        mean_a /= count
        mean_y /= count

        num = 0.0
        den_a = 0.0
        den_y = 0.0
        for s in range(n_stocks):
            a = values[t, s]
            y = targets[t, s]
            if np.isfinite(a) and np.isfinite(y):
                da = a - mean_a
                dy = y - mean_y
                num += da * dy
                den_a += da * da
                den_y += dy * dy

        if den_a < 1e-10 or den_y < 1e-10:
            daily_ic[t] = 0.0
        else:
            daily_ic[t] = num / np.sqrt(den_a * den_y)

    return daily_ic


def _clip_extreme(x: np.ndarray) -> np.ndarray:
    """将中间结果裁剪到 [-_CLIP_BOUND, _CLIP_BOUND]，inf 置为 nan。"""
    x = np.where(np.isinf(x), np.nan, x)
    return np.clip(x, -_CLIP_BOUND, _CLIP_BOUND)


def _clip_and_zscore(x: np.ndarray) -> np.ndarray:
    """最终输出：逐行 3-sigma 去极值 + z-score 归一化。"""
    x = _as_f64_array(x)
    if NUMBA_AVAILABLE:
        return _clip_and_zscore_numba(x)
    result = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(x.shape[0]):
        row = x[t]
        mask = np.isfinite(row)
        if mask.sum() < 2:
            continue
        valid = row[mask]
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if not np.isfinite(sigma) or sigma < 1e-10:
            result[t, mask] = 0.0
            continue
        lower, upper = mu - 3 * sigma, mu + 3 * sigma
        clipped = np.clip(valid, lower, upper)
        mu2 = np.mean(clipped)
        sigma2 = np.std(clipped, ddof=1)
        if not np.isfinite(sigma2) or sigma2 < 1e-10:
            result[t, mask] = 0.0
            continue
        result[t, mask] = (clipped - mu2) / sigma2
    return result


def _safe_log(x: np.ndarray) -> np.ndarray:
    """安全对数，非正数返回nan"""
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(x > 0, np.log(x), np.nan)
    return result


def _cumsum_rolling(x, window):
    """用 cumsum 计算滚动和与计数，O(n) 复杂度。返回 (rolling_sum, rolling_count)。"""
    valid = ~np.isnan(x)
    x_zero = np.where(valid, x, 0.0)
    cs = np.cumsum(x_zero, axis=0)
    cn = np.cumsum(valid.astype(np.float64), axis=0)
    n_days = x.shape[0]
    rs = cs[window - 1 :].copy()
    rc = cn[window - 1 :].copy()
    if window > 1:
        rs[1:] -= cs[: n_days - window]
        rc[1:] -= cn[: n_days - window]
    return rs, rc


def _rolling_mean_fast(x, window):
    x = _as_f64_array(x)
    if NUMBA_AVAILABLE:
        return _rolling_mean_numba(x, window)
    n = x.shape[0]
    result = np.full_like(x, np.nan, dtype=np.float64)
    if n < window:
        return result
    rs, rc = _cumsum_rolling(x, window)
    safe_c = np.where(rc > 0, rc, 1.0)
    result[window - 1 :] = np.where(rc > 0, rs / safe_c, np.nan)
    return result


def _rolling_sum_fast(x, window):
    x = _as_f64_array(x)
    if NUMBA_AVAILABLE:
        return _rolling_sum_numba(x, window)
    n = x.shape[0]
    result = np.full_like(x, np.nan, dtype=np.float64)
    if n < window:
        return result
    rs, _ = _cumsum_rolling(x, window)
    result[window - 1 :] = rs
    return result


def _rolling_var_fast(x, window, ddof=1):
    x = _as_f64_array(x)
    if NUMBA_AVAILABLE:
        return _rolling_var_numba(x, window, ddof)
    n = x.shape[0]
    result = np.full_like(x, np.nan, dtype=np.float64)
    if n < window:
        return result
    valid = ~np.isnan(x)
    x_zero = np.where(valid, x, 0.0)
    cs_x2 = np.cumsum(x_zero**2, axis=0)
    r_x2 = cs_x2[window - 1 :].copy()
    if window > 1:
        r_x2[1:] -= cs_x2[: n - window]
    rs, rc = _cumsum_rolling(x, window)
    ok = rc > ddof
    safe_c = np.where(ok, rc, 1.0)
    mean = rs / safe_c
    var = (r_x2 - rs * mean) / np.where(ok, rc - ddof, 1.0)
    result[window - 1 :] = np.where(ok, var, np.nan)
    return result


def _rolling_std_fast(x, window, ddof=1):
    v = _rolling_var_fast(x, window, ddof)
    with np.errstate(invalid="ignore"):
        return np.sqrt(v)


def _rolling_extrema_fast(x, window, func):
    """用 sliding_window_view 计算 rolling max/min/median。"""
    from numpy.lib.stride_tricks import sliding_window_view

    n = x.shape[0]
    result = np.full_like(x, np.nan, dtype=np.float64)
    if n < window:
        return result
    sw = sliding_window_view(x, window, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result[window - 1 :] = func(sw, axis=-1)
    return result


def _rolling_mad_fast(x, window):
    from numpy.lib.stride_tricks import sliding_window_view

    n = x.shape[0]
    result = np.full_like(x, np.nan, dtype=np.float64)
    if n < window:
        return result
    sw = sliding_window_view(x, window, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(sw, axis=-1, keepdims=True)
        result[window - 1 :] = np.nanmean(np.abs(sw - m), axis=-1)
    return result


def _rolling_wma_fast(x, window):
    from numpy.lib.stride_tricks import sliding_window_view

    n = x.shape[0]
    result = np.full_like(x, np.nan, dtype=np.float64)
    if n < window:
        return result
    weights = np.arange(1, window + 1, dtype=np.float64)
    weights /= weights.sum()
    sw = sliding_window_view(x, window, axis=0)
    valid = ~np.isnan(sw)
    weighted = np.where(valid, sw, 0.0) * weights.reshape(1, 1, -1)
    weight_sum = np.sum(valid * weights.reshape(1, 1, -1), axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        result[window - 1 :] = np.where(
            weight_sum > 1e-10, weighted.sum(axis=-1) / weight_sum, np.nan
        )
    return result


def _rolling_cov_fast(x, y, window, ddof=1):
    x = _as_f64_array(x)
    y = _as_f64_array(y)
    if NUMBA_AVAILABLE:
        return _rolling_cov_numba(x, y, window, ddof)
    n = x.shape[0]
    result = np.full_like(x, np.nan, dtype=np.float64)
    if n < window:
        return result
    joint = ~(np.isnan(x) | np.isnan(y))
    xf = np.where(joint, x, 0.0)
    yf = np.where(joint, y, 0.0)
    cs_xy = np.cumsum(xf * yf, axis=0)
    cs_x = np.cumsum(xf, axis=0)
    cs_y = np.cumsum(yf, axis=0)
    cs_n = np.cumsum(joint.astype(np.float64), axis=0)

    def rd(cs):
        r = cs[window - 1 :].copy()
        if window > 1:
            r[1:] -= cs[: n - window]
        return r

    rxy, rx, ry, rn = rd(cs_xy), rd(cs_x), rd(cs_y), rd(cs_n)
    ok = rn > ddof
    safe_n = np.where(ok, rn, 1.0)
    cov = (rxy - rx * ry / safe_n) / np.where(ok, rn - ddof, 1.0)
    result[window - 1 :] = np.where(ok, cov, np.nan)
    return result


def _rolling_corr_fast(x, y, window):
    x = _as_f64_array(x)
    y = _as_f64_array(y)
    if NUMBA_AVAILABLE:
        return _rolling_corr_numba(x, y, window)
    n = x.shape[0]
    result = np.full_like(x, np.nan, dtype=np.float64)
    if n < window:
        return result
    joint = ~(np.isnan(x) | np.isnan(y))
    xf = np.where(joint, x, 0.0)
    yf = np.where(joint, y, 0.0)
    cs_xy = np.cumsum(xf * yf, axis=0)
    cs_x = np.cumsum(xf, axis=0)
    cs_y = np.cumsum(yf, axis=0)
    cs_x2 = np.cumsum(xf**2, axis=0)
    cs_y2 = np.cumsum(yf**2, axis=0)
    cs_n = np.cumsum(joint.astype(np.float64), axis=0)

    def rd(cs):
        r = cs[window - 1 :].copy()
        if window > 1:
            r[1:] -= cs[: n - window]
        return r

    rxy, rx, ry = rd(cs_xy), rd(cs_x), rd(cs_y)
    rx2, ry2, rn = rd(cs_x2), rd(cs_y2), rd(cs_n)
    ok = rn > 2
    safe_n = np.where(ok, rn, 1.0)
    mx, my = rx / safe_n, ry / safe_n
    cov = rxy / safe_n - mx * my
    vx = rx2 / safe_n - mx**2
    vy = ry2 / safe_n - my**2
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.sqrt(np.maximum(vx * vy, 0.0))
        corr = np.where((denom > 1e-10) & ok, cov / denom, 0.0)
    result[window - 1 :] = np.where(ok, corr, np.nan)
    return result


# ============================================================
# 3. 表达式树求值器
# ============================================================
class AlphaCalculator:
    """
    在StockData上计算表达式树的值。

    所有中间结果和最终结果的shape都是 (n_days, n_stocks)。
    """

    def __init__(self, stock_data: StockData):
        self.stock_data = stock_data

    def evaluate(self, node: ExprNode) -> Optional[np.ndarray]:
        """
        递归计算表达式树的值。
        返回: ndarray(n_days, n_stocks) 或 None（计算失败）
        """
        try:
            result = self._eval(node)
            # 最终输出做一次归一化，保证数值范围可控
            return _clip_and_zscore(result)
        except Exception:
            return None

    def _eval(self, node: ExprNode) -> np.ndarray:
        sd = self.stock_data

        # --- 叶节点 ---
        if isinstance(node, FeatureNode):
            return sd.data[node.feature_name].copy()

        if isinstance(node, ConstantNode):
            return np.full((sd.n_days, sd.n_stocks), node.value, dtype=np.float64)

        # --- 截面一元 ---
        if isinstance(node, UnaryOpNode):
            x = self._eval(node.operand)
            name = node.operator.name

            if name == "Abs":
                return np.abs(x)
            elif name == "Log":
                return _safe_log(x)

        # --- 截面二元 ---
        if isinstance(node, BinaryOpNode):
            left = self._eval(node.left)
            right = self._eval(node.right)
            name = node.operator.name

            if name == "Add":
                return _clip_extreme(left + right)
            elif name == "Sub":
                return _clip_extreme(left - right)
            elif name == "Mul":
                return _clip_extreme(left * right)
            elif name == "Div":
                return _clip_extreme(_safe_div(left, right))
            elif name == "Greater":
                return np.maximum(left, right)
            elif name == "Less":
                return np.minimum(left, right)

        # --- 时序一元 ---
        if isinstance(node, TSUnaryOpNode):
            x = self._eval(node.operand)
            t = node.time_delta
            name = node.operator.name

            if name == "Ref":
                # x在t天前的值
                result = np.full_like(x, np.nan)
                if t < sd.n_days:
                    result[t:] = x[:-t] if t > 0 else x
                return result

            elif name == "Mean":
                return _rolling_mean_fast(x, t)

            elif name == "Med":
                return _rolling_extrema_fast(x, t, np.nanmedian)

            elif name == "Sum":
                return _rolling_sum_fast(x, t)

            elif name == "Std":
                return _rolling_std_fast(x, t, ddof=1)

            elif name == "Var":
                return _rolling_var_fast(x, t, ddof=1)

            elif name == "Max":
                return _rolling_extrema_fast(x, t, np.nanmax)

            elif name == "Min":
                return _rolling_extrema_fast(x, t, np.nanmin)

            elif name == "Mad":
                return _rolling_mad_fast(x, t)

            elif name == "Delta":
                # x - Ref(x, t)
                result = np.full_like(x, np.nan)
                if t < sd.n_days:
                    result[t:] = x[t:] - x[:-t]
                return result

            elif name == "WMA":
                return _rolling_wma_fast(x, t)

            elif name == "EMA":
                # 指数移动平均
                alpha_val = 2.0 / (t + 1)
                if NUMBA_AVAILABLE:
                    return _ema_numba(_as_f64_array(x), alpha_val)
                result = np.full_like(x, np.nan)
                result[0] = x[0]
                for i in range(1, sd.n_days):
                    prev = result[i - 1]
                    curr = x[i]
                    curr_nan = np.isnan(curr)
                    prev_nan = np.isnan(prev)
                    # 当前有效且前值有效：正常EMA
                    # 当前有效但前值NaN：用当前值初始化
                    # 当前NaN但前值有效：沿用前值（停牌不更新）
                    # 两者都NaN：保持NaN
                    result[i] = np.where(
                        curr_nan,
                        prev,
                        np.where(
                            prev_nan, curr, alpha_val * curr + (1 - alpha_val) * prev
                        ),
                    )
                return result

        # --- 时序二元 ---
        if isinstance(node, TSBinaryOpNode):
            left = self._eval(node.left)
            right = self._eval(node.right)
            t = node.time_delta
            name = node.operator.name

            if name == "Cov":
                return _rolling_cov_fast(left, right, t, ddof=1)

            elif name == "Corr":
                return _rolling_corr_fast(left, right, t)

        raise ValueError(f"Unknown node type: {type(node)}")


# ============================================================
# 4. IC 计算
# ============================================================
def calc_daily_ic(
    alpha_values: np.ndarray, target: np.ndarray, already_normed: bool = False
) -> np.ndarray:
    """
    计算每天的IC（Pearson相关系数）。
    alpha_values: (n_days, n_stocks)
    target:       (n_days, n_stocks)
    already_normed: 为兼容旧接口保留；IC 直接在原值上计算，不再额外归一化
    返回: (n_days,) 每天的IC值
    """
    values = _as_f64_array(alpha_values)
    targets = _as_f64_array(target)
    if NUMBA_AVAILABLE:
        return _calc_daily_ic_numba(values, targets)
    n_days = values.shape[0]
    daily_ic = np.full(n_days, np.nan)

    for t in range(n_days):
        a = values[t]
        y = targets[t]

        mask = np.isfinite(a) & np.isfinite(y)
        if mask.sum() < 3:
            continue

        a_valid = a[mask]
        y_valid = y[mask]

        da = a_valid - np.mean(a_valid)
        dy = y_valid - np.mean(y_valid)
        den_a = np.dot(da, da)
        den_y = np.dot(dy, dy)

        if den_a < 1e-10 or den_y < 1e-10:
            daily_ic[t] = 0.0
        else:
            daily_ic[t] = np.dot(da, dy) / np.sqrt(den_a * den_y)

    return daily_ic


def calc_mean_ic(
    alpha_values: np.ndarray, target: np.ndarray, already_normed: bool = False
) -> float:
    """
    计算平均IC：σ_y(f) = E_t[σ(f(X_t), y_t)]
    """
    daily_ic = calc_daily_ic(alpha_values, target, already_normed=already_normed)
    valid = daily_ic[~np.isnan(daily_ic)]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(valid))


def calc_rank_ic(alpha_values: np.ndarray, target: np.ndarray) -> float:
    """
    计算Rank IC（Spearman相关系数，论文公式9）
    先对alpha值和target分别排序，再计算Pearson相关
    """
    from scipy.stats import rankdata

    n_days = alpha_values.shape[0]
    daily_ric = np.full(n_days, np.nan)

    for t in range(n_days):
        a = alpha_values[t]
        y = target[t]

        mask = ~(np.isnan(a) | np.isnan(y))
        if mask.sum() < 3:
            continue

        a_rank = rankdata(a[mask])
        y_rank = rankdata(y[mask])

        da = a_rank - np.mean(a_rank)
        dy = y_rank - np.mean(y_rank)

        den = np.sqrt(np.dot(da, da) * np.dot(dy, dy))
        if den < 1e-10:
            daily_ric[t] = 0.0
        else:
            daily_ric[t] = np.dot(da, dy) / den

    valid = daily_ric[~np.isnan(daily_ric)]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(valid))


def calc_mutual_ic(alpha1: np.ndarray, alpha2: np.ndarray) -> float:
    """
    计算两个alpha之间的互相关IC（mutual IC）。
    用于衡量alpha之间的相似性。
    """
    return calc_mean_ic(alpha1, alpha2)


# ============================================================
# 5. 归一化
# ============================================================
def normalize_alpha(alpha_values: np.ndarray) -> np.ndarray:
    """
    每天截面上对 alpha 值做 3-sigma 去极值 + z-score 归一化。
    """
    return _clip_and_zscore(alpha_values)


# ============================================================
# 6. 便捷函数：从token序列直接计算alpha
# ============================================================
def evaluate_tokens(tokens: list, stock_data: StockData) -> Optional[np.ndarray]:
    """从token序列直接计算alpha值"""
    stripped = strip_special_tokens(tokens)
    tree = parse_rpn_to_tree(stripped)
    if tree is None:
        return None

    calc = AlphaCalculator(stock_data)
    result = calc.evaluate(tree)
    return result


# ============================================================
# 7. 测试
# ============================================================
if __name__ == "__main__":
    # 通过data.py阅读数据
    from data import Data
    from common import ADJUST_PREV

    reader = Data()
    df = reader.daily(start_date="20250101", end_date="20260501", adjust=ADJUST_PREV)
    df = df[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "vwap"]]
    print("读取数据成功")
    print(df.head())

    # 构建StockData
    sd = StockData(df)
    target = sd.get_target(horizon=20)

    print(f"\n数据shape: ({sd.n_days}, {sd.n_stocks})")
    print(f"目标非nan数: {(~np.isnan(target)).sum()}")

    # 测试一个简单的alpha: Mean($close, 20)
    from tokens import OP_MEAN, OP_ADD, OP_DIV, TokenType
    from expression import tree_to_formula

    tokens_test = [
        Token(TokenType.FEATURE, "close"),
        Token(TokenType.TIME_DELTA, 20),
        Token(TokenType.OPERATOR, OP_MEAN),
    ]

    alpha_vals = evaluate_tokens(tokens_test, sd)
    print(f"\nAlpha: Mean($close, 20)")
    print(f"  Alpha shape: {alpha_vals.shape}")
    print(f"  非nan数: {(~np.isnan(alpha_vals)).sum()}")

    # 计算IC
    ic = calc_mean_ic(alpha_vals, target)
    ric = calc_rank_ic(alpha_vals, target)
    print(f"  Mean IC: {ic:.4f}")
    print(f"  Rank IC: {ric:.4f}")

    # 测试归一化
    norm_vals = normalize_alpha(alpha_vals)
    print(f"\n归一化后:")
    print(f"  第20天均值: {np.nanmean(norm_vals[20]):.6f} (应接近0)")
    print(f"  第20天长度: {np.sqrt(np.nansum(norm_vals[20]**2)):.6f} (应接近1)")

    # 测试更复杂的alpha: Div($close, Mean($close, 20))
    from tokens import OP_DIV

    tokens_test2 = [
        Token(TokenType.FEATURE, "close"),
        Token(TokenType.FEATURE, "close"),
        Token(TokenType.TIME_DELTA, 20),
        Token(TokenType.OPERATOR, OP_MEAN),
        Token(TokenType.OPERATOR, OP_DIV),
    ]

    tree2 = parse_rpn_to_tree(strip_special_tokens(tokens_test2))
    print(f"\nAlpha: {tree_to_formula(tree2)}")

    alpha_vals2 = evaluate_tokens(tokens_test2, sd)
    ic2 = calc_mean_ic(alpha_vals2, target)
    ric2 = calc_rank_ic(alpha_vals2, target)
    print(f"  Mean IC: {ic2:.4f}")
    print(f"  Rank IC: {ric2:.4f}")
