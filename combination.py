"""
Alpha组合模型

核心思路：
  - 维护一个alpha池 F = {f1, f2, ..., fk}，每个alpha有权重 w_i
  - 组合模型输出 z = Σ w_i * f_i(X)
  - 用MSE损失（只需IC和互相关矩阵）
  - 新alpha加入后，梯度下降优化权重，淘汰最弱alpha
"""

import numpy as np
from typing import List, Optional
from calculator import (
    StockData,
    AlphaCalculator,
    normalize_alpha,
    calc_mean_ic,
    calc_daily_ic,
)
from expression import (
    ExprNode,
    parse_rpn_to_tree,
    strip_special_tokens,
    tree_to_formula,
)
from tokens import Token


class AlphaCombinationModel:
    """
    Alpha组合模型

    维护：
      - alpha_exprs: 表达式树列表
      - alpha_values: 归一化后的alpha值矩阵列表，每个 shape=(n_days, n_stocks)
      - weights: 各alpha的权重
      - ic_vector: 每个alpha与target的IC，即 σ_y(f_i)
      - ic_matrix: alpha之间的互相关矩阵，即 σ(f_i, f_j)
    """

    def __init__(
        self, stock_data: StockData, target: np.ndarray, max_pool_size: int = 20
    ):
        """
        参数:
            stock_data: 股票数据
            target: 预测目标 (n_days, n_stocks)
            max_pool_size: alpha池最大容量（论文中 k ∈ {10,20,50,100}）
        """
        self.stock_data = stock_data
        self.calculator = AlphaCalculator(stock_data)

        # target 保持原始收益率，仅在计算 IC 时做中心化/缩放
        self.target = np.where(np.isfinite(target), target.astype(np.float64), np.nan)

        self.max_pool_size = max_pool_size

        # alpha池
        self.alpha_exprs: List[ExprNode] = []
        self.alpha_values: List[np.ndarray] = []  # 归一化后的值
        self.weights: np.ndarray = np.array([], dtype=np.float64)

        # IC缓存（Theorem 3.1的关键，避免重复计算）
        self.ic_vector: np.ndarray = np.array([], dtype=np.float64)  # (k,)
        self.ic_matrix: np.ndarray = np.array([]).reshape(0, 0)  # (k, k)
        self.last_add_status: str = "idle"
        self.last_add_accepted: bool = False
        self.last_add_ic_delta: float = 0.0
        self.last_add_icir_delta: float = 0.0
        self.last_add_candidate_ic: float = 0.0

    @property
    def pool_size(self) -> int:
        return len(self.alpha_exprs)

    def _set_last_add_result(
        self,
        status: str,
        accepted: bool,
        ic_delta: float = 0.0,
        icir_delta: float = 0.0,
        candidate_ic: float = 0.0,
    ) -> None:
        self.last_add_status = status
        self.last_add_accepted = accepted
        self.last_add_ic_delta = float(ic_delta)
        self.last_add_icir_delta = float(icir_delta)
        self.last_add_candidate_ic = float(candidate_ic)

    def _compute_ic_with_target(self, alpha_val: np.ndarray) -> float:
        """计算单个alpha与target的IC（输入已归一化）"""
        return calc_mean_ic(alpha_val, self.target, already_normed=True)

    def _compute_mutual_ic(
        self, alpha_val1: np.ndarray, alpha_val2: np.ndarray
    ) -> float:
        """计算两个alpha之间的互相关IC（输入已归一化）"""
        return calc_mean_ic(alpha_val1, alpha_val2, already_normed=True)

    # ============================================================
    # Theorem 3.1: 高效MSE损失计算
    # ============================================================
    @staticmethod
    def _loss_from_cache(
        w: np.ndarray, ic_vector: np.ndarray, ic_matrix: np.ndarray
    ) -> float:
        return 1.0 - 2.0 * np.dot(w, ic_vector) + w @ ic_matrix @ w

    @staticmethod
    def _loss_gradient_from_cache(
        w: np.ndarray, ic_vector: np.ndarray, ic_matrix: np.ndarray
    ) -> np.ndarray:
        return -2.0 * ic_vector + 2.0 * ic_matrix @ w

    def compute_loss(self, w: np.ndarray) -> float:
        """
        论文公式7:
        L(w) = 1/n * (1 - 2*Σ w_i*σ_y(f_i) + Σ_i Σ_j w_i*w_j*σ(f_i,f_j))

        只用ic_vector和ic_matrix计算，不需要重新算z。
        """
        k = len(w)
        if k == 0:
            return 1.0

        return self._loss_from_cache(w, self.ic_vector[:k], self.ic_matrix[:k, :k])

    def compute_loss_gradient(self, w: np.ndarray) -> np.ndarray:
        """
        L(w) 对 w 的梯度:
        ∂L/∂w_i = 1/n * (-2*σ_y(f_i) + 2*Σ_j w_j*σ(f_i,f_j))
        """
        k = len(w)
        if k == 0:
            return np.array([])

        return self._loss_gradient_from_cache(
            w, self.ic_vector[:k], self.ic_matrix[:k, :k]
        )

    def _solve_weights(
        self,
        ic_matrix: np.ndarray,
        ic_vector: np.ndarray,
        n_steps: int = 200,
        lr: float = 0.01,
    ) -> np.ndarray:
        """给定互相关矩阵与 IC 向量，求解一组稳定权重。"""
        k = len(ic_vector)
        if k == 0:
            return np.array([], dtype=np.float64)

        reg_lambda = 0.1
        try:
            a = ic_matrix + reg_lambda * np.eye(k)
            w = np.linalg.solve(a, ic_vector)
            if np.all(np.isfinite(w)):
                return w
        except np.linalg.LinAlgError:
            pass

        w = ic_vector.copy()
        best_loss = self._loss_from_cache(w, ic_vector, ic_matrix)
        best_w = w.copy()

        for _ in range(n_steps):
            grad = self._loss_gradient_from_cache(w, ic_vector, ic_matrix)

            if not np.all(np.isfinite(grad)):
                break

            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1.0:
                grad = grad / grad_norm

            w -= lr * (grad + 2 * reg_lambda * w)

            loss = self._loss_from_cache(w, ic_vector, ic_matrix)
            if np.isfinite(loss) and loss < best_loss:
                best_loss = loss
                best_w = w.copy()

        return best_w

    def _compute_combo_ic(
        self, alpha_values: List[np.ndarray], weights: np.ndarray
    ) -> float:
        if len(alpha_values) == 0 or len(weights) == 0:
            return 0.0

        combo = np.zeros_like(alpha_values[0], dtype=np.float64)
        for val, weight in zip(alpha_values, weights):
            combo += weight * val

        return calc_mean_ic(combo, self.target, already_normed=True)

    # ============================================================
    # 权重优化（梯度下降）
    # ============================================================
    def optimize_weights(self, n_steps: int = 200, lr: float = 0.01) -> float:
        """
        优化权重。优先用解析解，失败时回退到梯度下降。

        解析解: w* = (ic_matrix + λI)^{-1} @ ic_vector
        其中 λ 为 L2 正则项，防止矩阵奇异。

        返回: 优化后的组合模型IC
        """
        if self.pool_size == 0:
            return 0.0

        k = self.pool_size
        self.weights = self._solve_weights(
            self.ic_matrix[:k, :k], self.ic_vector[:k], n_steps=n_steps, lr=lr
        )
        return self.get_combination_ic()

    # ============================================================
    # 组合模型IC计算
    # ============================================================
    def get_combination_ic(self) -> float:
        """计算当前组合模型的IC: σ_y(Σ w_i * f_i)"""
        if self.pool_size == 0:
            return 0.0

        return self._compute_combo_ic(self.alpha_values, self.weights)

    def get_combination_icir(self) -> float:
        """计算当前组合模型的ICIR: mean(daily_ic) / std(daily_ic)"""
        if self.pool_size == 0:
            return 0.0

        combo = np.zeros_like(self.alpha_values[0], dtype=np.float64)
        for val, weight in zip(self.alpha_values, self.weights):
            combo += weight * val

        daily_ic = calc_daily_ic(combo, self.target, already_normed=True)
        valid = daily_ic[~np.isnan(daily_ic)]
        if len(valid) < 2:
            return 0.0
        std = np.std(valid, ddof=1)
        if std < 1e-10:
            return 0.0
        return float(np.mean(valid) / std)

    def _compute_icir_from_state(
        self, alpha_values: List[np.ndarray], weights: np.ndarray
    ) -> float:
        """给定 alpha_values 和 weights，计算组合 ICIR（不修改内部状态）"""
        if len(alpha_values) == 0 or len(weights) == 0:
            return 0.0
        combo = np.zeros_like(alpha_values[0], dtype=np.float64)
        for val, weight in zip(alpha_values, weights):
            combo += weight * val
        daily_ic = calc_daily_ic(combo, self.target, already_normed=True)
        valid = daily_ic[~np.isnan(daily_ic)]
        if len(valid) < 2:
            return 0.0
        std = np.std(valid, ddof=1)
        if std < 1e-10:
            return 0.0
        return float(np.mean(valid) / std)

    # ============================================================
    # 核心：添加新alpha
    # ============================================================
    def add_alpha(
        self,
        expr: ExprNode,
        n_opt_steps: int = 200,
        lr: float = 0.01,
        alpha_values: Optional[np.ndarray] = None,
        already_normed: bool = True,
        baseline_ic: Optional[float] = None,
    ) -> float:
        """
        Algorithm 1 的完整流程：
        1. 计算新alpha的值并归一化
        2. 计算新alpha与target的IC、与池中所有alpha的互相关
        3. 试探性加入：扩展ic_vector和ic_matrix，优化权重
        4. 若组合IC提升则保留，否则回滚
        5. 若超过max_pool_size，移除最弱alpha

        参数:
            expr: 新alpha的表达式树
            alpha_values: 可选的预计算alpha值；用于避免重复求值
            already_normed: 当提供 alpha_values 时，是否已完成归一化
            baseline_ic: 可选的当前组合IC；用于避免重复计算

        返回: 更新后的组合模型IC
        """
        # 1. 计算并归一化
        if alpha_values is None:
            # AlphaCalculator.evaluate 已返回归一化后的结果
            norm_val = self.calculator.evaluate(expr)
        else:
            norm_val = alpha_values if already_normed else normalize_alpha(alpha_values)
        if norm_val is None:
            self._set_last_add_result("invalid_eval", False)
            return self.get_combination_ic()

        # 检查是否有效（非全nan）
        if np.all(np.isnan(norm_val)):
            self._set_last_add_result("invalid_all_nan", False)
            return self.get_combination_ic()

        # 2. 计算IC
        new_ic_with_target = self._compute_ic_with_target(norm_val)
        if not np.isfinite(new_ic_with_target):
            self._set_last_add_result("invalid_ic", False)
            return self.get_combination_ic()

        # |IC|太低直接拒绝（节省计算，无预测能力的alpha）
        if abs(new_ic_with_target) < 0.005:
            self._set_last_add_result(
                "rejected_low_ic", False, candidate_ic=new_ic_with_target
            )
            return self.get_combination_ic()

        # 与池中所有alpha的互相关
        new_mutual_ics = []
        for existing_val in self.alpha_values:
            mic = self._compute_mutual_ic(norm_val, existing_val)
            new_mutual_ics.append(mic)
        new_self_ic = self._compute_mutual_ic(norm_val, norm_val)

        # 与已有alpha相关性太高则拒绝（冗余）
        if new_mutual_ics and max(abs(m) for m in new_mutual_ics) > 0.95:
            self._set_last_add_result(
                "rejected_redundant", False, candidate_ic=new_ic_with_target
            )
            return self.get_combination_ic()

        # 保存完整旧状态用于回滚
        old_ic = self.get_combination_ic() if baseline_ic is None else baseline_ic
        old_weights = self.weights.copy()
        old_ic_vector = self.ic_vector.copy()
        old_ic_matrix = self.ic_matrix.copy()
        old_exprs = list(self.alpha_exprs)
        old_values = list(self.alpha_values)

        # 3. 扩展缓存矩阵
        k = self.pool_size

        self.ic_vector = np.append(self.ic_vector, new_ic_with_target)

        if k == 0:
            self.ic_matrix = np.array([[new_self_ic]])
        else:
            new_col = np.array(new_mutual_ics)
            new_matrix = np.zeros((k + 1, k + 1))
            new_matrix[:k, :k] = self.ic_matrix
            new_matrix[k, :k] = new_col
            new_matrix[:k, k] = new_col
            new_matrix[k, k] = new_self_ic
            self.ic_matrix = new_matrix

        # 添加到池
        self.alpha_exprs.append(expr)
        self.alpha_values.append(norm_val)
        self.weights = np.append(self.weights, np.random.randn() * 0.01)

        # 4. 梯度下降优化权重
        self.optimize_weights(n_steps=n_opt_steps, lr=lr)

        # 5. 如果超出容量，移除最弱alpha
        if self.pool_size > self.max_pool_size:
            self._remove_weakest()

        # 6. 检查是否真正提升了组合IC，否则回滚
        new_ic = self.get_combination_ic()
        if (not np.isfinite(new_ic)) or (new_ic < old_ic - 1e-6 and k > 0):
            # 完整回滚所有状态
            self.alpha_exprs = old_exprs
            self.alpha_values = old_values
            self.weights = old_weights
            self.ic_vector = old_ic_vector
            self.ic_matrix = old_ic_matrix
            self._set_last_add_result(
                "rejected_no_improve", False, candidate_ic=new_ic_with_target
            )
            return old_ic

        # 计算 ICIR delta（用于多目标奖励）
        new_icir = self.get_combination_icir()
        old_icir = self._compute_icir_from_state(old_values, old_weights)
        icir_delta = new_icir - old_icir

        self._set_last_add_result(
            "accepted",
            True,
            ic_delta=new_ic - old_ic,
            icir_delta=icir_delta,
            candidate_ic=new_ic_with_target,
        )
        return new_ic

    def _remove_weakest(self):
        """移除对组合IC贡献最小的alpha（逐个试移除，保留使IC最高的方案）"""
        k = self.pool_size
        if k <= 1:
            return

        best_ic = -np.inf
        best_remove = None
        best_weights = None

        for p in range(k):
            keep_mask = np.ones(k, dtype=bool)
            keep_mask[p] = False
            trial_vals = [v for j, v in enumerate(self.alpha_values) if j != p]
            trial_ic_vec = self.ic_vector[keep_mask]
            trial_ic_mat = self.ic_matrix[np.ix_(keep_mask, keep_mask)]
            trial_w = self._solve_weights(trial_ic_mat, trial_ic_vec)
            ic = self._compute_combo_ic(trial_vals, trial_w)
            if np.isfinite(ic) and ic > best_ic:
                best_ic = ic
                best_remove = p
                best_weights = trial_w

        if best_remove is None:
            best_remove = int(np.argmin(np.abs(self.weights)))
            keep_mask = np.ones(k, dtype=bool)
            keep_mask[best_remove] = False
            best_weights = self._solve_weights(
                self.ic_matrix[np.ix_(keep_mask, keep_mask)],
                self.ic_vector[keep_mask],
            )

        p = best_remove

        # 从所有列表/矩阵中移除索引p
        self.alpha_exprs.pop(p)
        self.alpha_values.pop(p)
        self.weights = best_weights
        self.ic_vector = np.delete(self.ic_vector, p)
        self.ic_matrix = np.delete(np.delete(self.ic_matrix, p, axis=0), p, axis=1)

    # ============================================================
    # 从token序列添加alpha（供RL使用）
    # ============================================================
    def add_alpha_from_tokens(
        self, tokens: List[Token], n_opt_steps: int = 200, lr: float = 0.01
    ) -> Optional[float]:
        """
        从token序列解析并添加alpha。
        返回: 更新后的组合模型IC，解析失败返回 None
        """
        stripped = strip_special_tokens(tokens)
        tree = parse_rpn_to_tree(stripped)
        if tree is None:
            return None
        return self.add_alpha(tree, n_opt_steps=n_opt_steps, lr=lr)

    # ============================================================
    # 展示当前alpha池状态
    # ============================================================
    def summary(self):
        """打印当前alpha池的状态"""
        print(f"\n{'='*70}")
        print(f"Alpha池状态: {self.pool_size}/{self.max_pool_size} 个alpha")
        print(f"组合模型IC: {self.get_combination_ic():.4f}")
        print(f"{'='*70}")

        for i, expr in enumerate(self.alpha_exprs):
            formula = tree_to_formula(expr)
            ic_val = self.ic_vector[i]
            w_val = self.weights[i]
            print(f"  [{i+1:2d}] IC={ic_val:+.4f}  W={w_val:+.4f}  {formula}")

        if self.pool_size > 1:
            print(f"\n互相关矩阵 (部分):")
            k = min(self.pool_size, 5)
            header = "      " + "".join(f"  α{j+1:2d}  " for j in range(k))
            print(header)
            for i in range(k):
                row = f"  α{i+1:2d} "
                for j in range(k):
                    row += f" {self.ic_matrix[i,j]:+.3f}"
                print(row)


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    import pandas as pd
    from tokens import (
        Token,
        TokenType,
        OP_MEAN,
        OP_STD,
        OP_DIV,
        OP_SUB,
        OP_DELTA,
        OP_CORR,
        OP_REF,
        OP_LOG,
    )

    # 1. 创建数据
    from data import Data
    from common import ADJUST_PREV

    reader = Data()
    df = reader.daily(start_date="20250101", end_date="20260501", adjust=ADJUST_PREV)
    df = df[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "vwap"]]
    print("读取数据成功")
    print(df.head())

    sd = StockData(df)
    target = sd.get_target(horizon=20)

    # 2. 创建组合模型
    model = AlphaCombinationModel(sd, target, max_pool_size=10)

    # 3. 依次添加几个alpha
    alphas_tokens = [
        # Alpha 1: Mean($close, 20)
        [
            Token(TokenType.FEATURE, "close"),
            Token(TokenType.TIME_DELTA, 20),
            Token(TokenType.OPERATOR, OP_MEAN),
        ],
        # Alpha 2: Std($close, 20)
        [
            Token(TokenType.FEATURE, "close"),
            Token(TokenType.TIME_DELTA, 20),
            Token(TokenType.OPERATOR, OP_STD),
        ],
        # Alpha 3: Delta($close, 20)
        [
            Token(TokenType.FEATURE, "close"),
            Token(TokenType.TIME_DELTA, 20),
            Token(TokenType.OPERATOR, OP_DELTA),
        ],
        # Alpha 4: Corr($close, $vol, 20)
        [
            Token(TokenType.FEATURE, "close"),
            Token(TokenType.FEATURE, "vol"),
            Token(TokenType.TIME_DELTA, 20),
            Token(TokenType.OPERATOR, OP_CORR),
        ],
        # Alpha 5: Div($close, Mean($close, 20))
        [
            Token(TokenType.FEATURE, "close"),
            Token(TokenType.FEATURE, "close"),
            Token(TokenType.TIME_DELTA, 20),
            Token(TokenType.OPERATOR, OP_MEAN),
            Token(TokenType.OPERATOR, OP_DIV),
        ],
    ]

    print("\n逐步添加alpha到组合模型:")
    for i, tokens in enumerate(alphas_tokens):
        ic = model.add_alpha_from_tokens(tokens)
        if ic is None:
            print(f"  第{i+1}个alpha解析失败")
        else:
            print(f"  添加第{i+1}个alpha后，组合IC = {ic:.4f}")

    # 4. 展示完整状态
    model.summary()
