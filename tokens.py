"""
Token和操作符定义
"""

from enum import Enum
from typing import List


# ============================================================
# 1. 操作符类型枚举
# ============================================================
class OpCategory(Enum):
    """操作符分类：CS=cross-section, TS=time-series, U=unary, B=binary"""

    CS_U = "CS-U"  # 截面一元
    CS_B = "CS-B"  # 截面二元
    TS_U = "TS-U"  # 时序一元 (需要时间窗口参数)
    TS_B = "TS-B"  # 时序二元 (需要时间窗口参数)


# ============================================================
# 2. 操作符定义
# ============================================================
class Operator:
    """单个操作符的定义"""

    def __init__(self, name: str, category: OpCategory, description: str = ""):
        self.name = name
        self.category = category
        self.description = description

    @property
    def arity(self) -> int:
        """操作数个数（不含时间窗口参数）"""
        if self.category in (OpCategory.CS_U, OpCategory.TS_U):
            return 1
        else:  # CS_B, TS_B
            return 2

    @property
    def need_time_delta(self) -> bool:
        """是否需要时间窗口参数"""
        return self.category in (OpCategory.TS_U, OpCategory.TS_B)

    def __repr__(self):
        return f"Op({self.name}, {self.category.value})"


# --- 截面一元操作符 ---
OP_ABS = Operator("Abs", OpCategory.CS_U, "|x|")
OP_LOG = Operator("Log", OpCategory.CS_U, "log(x)")

# --- 截面二元操作符 ---
OP_ADD = Operator("Add", OpCategory.CS_B, "x + y")
OP_SUB = Operator("Sub", OpCategory.CS_B, "x - y")
OP_MUL = Operator("Mul", OpCategory.CS_B, "x * y")
OP_DIV = Operator("Div", OpCategory.CS_B, "x / y")
OP_GREATER = Operator("Greater", OpCategory.CS_B, "max(x, y)")
OP_LESS = Operator("Less", OpCategory.CS_B, "min(x, y)")

# --- 时序一元操作符 ---
OP_REF = Operator("Ref", OpCategory.TS_U, "x在t天前的值")
OP_MEAN = Operator("Mean", OpCategory.TS_U, "近t天均值")
OP_MED = Operator("Med", OpCategory.TS_U, "近t天中位数")
OP_SUM = Operator("Sum", OpCategory.TS_U, "近t天求和")
OP_STD = Operator("Std", OpCategory.TS_U, "近t天标准差")
OP_VAR = Operator("Var", OpCategory.TS_U, "近t天方差")
OP_MAX = Operator("Max", OpCategory.TS_U, "近t天最大值")
OP_MIN = Operator("Min", OpCategory.TS_U, "近t天最小值")
OP_MAD = Operator("Mad", OpCategory.TS_U, "近t天均值绝对偏差")
OP_DELTA = Operator("Delta", OpCategory.TS_U, "x - Ref(x,t)")
OP_WMA = Operator("WMA", OpCategory.TS_U, "加权移动平均")
OP_EMA = Operator("EMA", OpCategory.TS_U, "指数移动平均")

# --- 时序二元操作符 ---
OP_COV = Operator("Cov", OpCategory.TS_B, "近t天协方差")
OP_CORR = Operator("Corr", OpCategory.TS_B, "近t天相关系数")


# 所有操作符列表
ALL_OPERATORS: List[Operator] = [
    OP_ABS,
    OP_LOG,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_GREATER,
    OP_LESS,
    OP_REF,
    OP_MEAN,
    OP_MED,
    OP_SUM,
    OP_STD,
    OP_VAR,
    OP_MAX,
    OP_MIN,
    OP_MAD,
    OP_DELTA,
    OP_WMA,
    OP_EMA,
    OP_COV,
    OP_CORR,
]


# ============================================================
# 3. 特征 (原始股票数据的列)
# ============================================================
FEATURES = ["open", "close", "high", "low", "vol", "vwap"]


# ============================================================
# 4. 常数
# ============================================================
CONSTANTS = [-30, -10, -5, -2, -1, -0.5, -0.01, 0.01, 0.5, 1, 2, 5, 10, 30]


# ============================================================
# 5. 时间窗口
# ============================================================
TIME_DELTAS = [10, 20, 30, 40, 50]


# ============================================================
# 6. 特殊标记
# ============================================================
SEP_TOKEN = "SEP"  # 序列结束
BEG_TOKEN = "BEG"  # 序列开始


# ============================================================
# 7. Token类型枚举
# ============================================================
class TokenType(Enum):
    OPERATOR = "operator"
    FEATURE = "feature"
    CONSTANT = "constant"
    TIME_DELTA = "time_delta"
    SEP = "sep"
    BEG = "beg"


# ============================================================
# 8. Token类：统一封装各类型token
# ============================================================
class Token:
    def __init__(self, token_type: TokenType, value):
        self.token_type = token_type
        self.value = value  # Operator对象 / str / float / int

    def __repr__(self):
        if self.token_type == TokenType.OPERATOR:
            return f"T({self.value.name})"
        elif self.token_type == TokenType.FEATURE:
            return f"T(${self.value})"
        elif self.token_type == TokenType.CONSTANT:
            return f"T({self.value})"
        elif self.token_type == TokenType.TIME_DELTA:
            return f"T({self.value}d)"
        else:
            return f"T({self.value})"

    def __eq__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return self.token_type == other.token_type and self.value == other.value

    def __hash__(self):
        if self.token_type == TokenType.OPERATOR:
            return hash((self.token_type, self.value.name))
        return hash((self.token_type, self.value))


# ============================================================
# 9. 构建完整的Token词表
# ============================================================
def build_token_vocabulary() -> List[Token]:
    """构建所有可用token的列表，每个token有唯一索引"""
    vocab = []

    # 特殊标记
    vocab.append(Token(TokenType.BEG, BEG_TOKEN))
    vocab.append(Token(TokenType.SEP, SEP_TOKEN))

    # 特征
    for feat in FEATURES:
        vocab.append(Token(TokenType.FEATURE, feat))

    # 常数
    for const in CONSTANTS:
        vocab.append(Token(TokenType.CONSTANT, const))

    # 时间窗口
    for td in TIME_DELTAS:
        vocab.append(Token(TokenType.TIME_DELTA, td))

    # 操作符
    for op in ALL_OPERATORS:
        vocab.append(Token(TokenType.OPERATOR, op))

    return vocab


# 构建词表和索引映射
VOCAB = build_token_vocabulary()
TOKEN_TO_IDX = {tok: idx for idx, tok in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)


# ============================================================
# 10. 测试/展示
# ============================================================
if __name__ == "__main__":
    print(f"词表大小: {VOCAB_SIZE}")
    print(f"\n所有Token:")
    for idx, tok in enumerate(VOCAB):
        print(f"  [{idx:2d}] {tok}")

    print(f"\n操作符数量: {len(ALL_OPERATORS)}")
    print(f"特征数量: {len(FEATURES)}")
    print(f"常数数量: {len(CONSTANTS)}")
    print(f"时间窗口数量: {len(TIME_DELTAS)}")
