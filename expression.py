"""
表达式树与RPN解析

RPN示例: [BEG, 5, $vol, Add, 2d, Sum, SEP]
表达式:  Sum(Add(5, $volume), 2d)
"""

from typing import List, Optional
from tokens import (
    Token,
    TokenType,
    Operator,
    OpCategory,
    VOCAB,
)


# ============================================================
# 1. 表达式树节点
# ============================================================
class ExprNode:
    """表达式树的节点"""

    pass


class FeatureNode(ExprNode):
    """叶节点：原始特征，如 $close"""

    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    def __repr__(self):
        return f"${self.feature_name}"


class ConstantNode(ExprNode):
    """叶节点：常数，如 0.5"""

    def __init__(self, value: float):
        self.value = value

    def __repr__(self):
        return str(self.value)


class UnaryOpNode(ExprNode):
    """一元操作符节点 (CS-U)，如 Abs(x)"""

    def __init__(self, operator: Operator, operand: ExprNode):
        self.operator = operator
        self.operand = operand

    def __repr__(self):
        return f"{self.operator.name}({self.operand})"


class BinaryOpNode(ExprNode):
    """二元操作符节点 (CS-B)，如 Add(x, y)"""

    def __init__(self, operator: Operator, left: ExprNode, right: ExprNode):
        self.operator = operator
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{self.operator.name}({self.left}, {self.right})"


class TSUnaryOpNode(ExprNode):
    """时序一元操作符节点 (TS-U)，如 Mean(x, 20)"""

    def __init__(self, operator: Operator, operand: ExprNode, time_delta: int):
        self.operator = operator
        self.operand = operand
        self.time_delta = time_delta

    def __repr__(self):
        return f"{self.operator.name}({self.operand}, {self.time_delta})"


class TSBinaryOpNode(ExprNode):
    """时序二元操作符节点 (TS-B)，如 Corr(x, y, 20)"""

    def __init__(
        self, operator: Operator, left: ExprNode, right: ExprNode, time_delta: int
    ):
        self.operator = operator
        self.left = left
        self.right = right
        self.time_delta = time_delta

    def __repr__(self):
        return f"{self.operator.name}({self.left}, {self.right}, {self.time_delta})"


class _TimeDeltaPlaceholder:
    """解析过程中的临时占位符，用于在栈中暂存时间窗口值"""

    def __init__(self, value: int):
        self.value = value


# ============================================================
# 2. RPN序列 -> 表达式树
# ============================================================
def parse_rpn_to_tree(tokens: List[Token]) -> Optional[ExprNode]:
    """
    将RPN token序列解析为表达式树。

    RPN的构建规则：
    - 后序遍历表达式树得到RPN
    - 操作数在前，操作符在后
    - 时序操作符的时间窗口紧跟在操作符后面

    注意：输入tokens应该去掉BEG和SEP

    示例：
      tokens = [$vol, 5, Add, 20d, Sum]
      => Sum(Add($vol, 5), 20)
    """
    stack: List[ExprNode] = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok.token_type == TokenType.FEATURE:
            stack.append(FeatureNode(tok.value))

        elif tok.token_type == TokenType.CONSTANT:
            stack.append(ConstantNode(tok.value))

        elif tok.token_type == TokenType.OPERATOR:
            op: Operator = tok.value

            if op.category == OpCategory.CS_U:
                # 一元截面操作符：弹出1个操作数
                if len(stack) < 1:
                    return None
                operand = stack.pop()
                stack.append(UnaryOpNode(op, operand))

            elif op.category == OpCategory.CS_B:
                # 二元截面操作符：弹出2个操作数
                if len(stack) < 2:
                    return None
                right = stack.pop()
                left = stack.pop()
                stack.append(BinaryOpNode(op, left, right))

            elif op.category == OpCategory.TS_U:
                # 时序一元：弹出时间窗口和1个操作数（标准RPN：operand, time_delta, operator）
                if len(stack) < 2:
                    return None
                td_node = stack.pop()
                if not isinstance(td_node, _TimeDeltaPlaceholder):
                    return None
                operand = stack.pop()
                stack.append(TSUnaryOpNode(op, operand, td_node.value))

            elif op.category == OpCategory.TS_B:
                # 时序二元：弹出时间窗口和2个操作数（标准RPN：left, right, time_delta, operator）
                if len(stack) < 3:
                    return None
                td_node = stack.pop()
                if not isinstance(td_node, _TimeDeltaPlaceholder):
                    return None
                right = stack.pop()
                left = stack.pop()
                stack.append(TSBinaryOpNode(op, left, right, td_node.value))

        elif tok.token_type == TokenType.TIME_DELTA:
            stack.append(_TimeDeltaPlaceholder(tok.value))

        elif tok.token_type in (TokenType.BEG, TokenType.SEP):
            pass  # 跳过特殊标记

        i += 1

    # 最终栈中应该只剩一个完整表达式（且不能是时间窗口占位符）
    if len(stack) == 1 and isinstance(stack[0], ExprNode):
        return stack[0]
    return None


# ============================================================
# 3. 辅助：从token序列中去掉BEG/SEP
# ============================================================
def strip_special_tokens(tokens: List[Token]) -> List[Token]:
    """去掉BEG和SEP标记"""
    return [t for t in tokens if t.token_type not in (TokenType.BEG, TokenType.SEP)]


# ============================================================
# 4. 表达式树 -> 可读公式字符串
# ============================================================
def tree_to_formula(node: ExprNode) -> str:
    """将表达式树转换为可读的公式字符串"""
    if isinstance(node, FeatureNode):
        return f"${node.feature_name}"

    elif isinstance(node, ConstantNode):
        return str(node.value)

    elif isinstance(node, UnaryOpNode):
        return f"{node.operator.name}({tree_to_formula(node.operand)})"

    elif isinstance(node, BinaryOpNode):
        left_str = tree_to_formula(node.left)
        right_str = tree_to_formula(node.right)
        # 算术运算用中缀更易读
        if node.operator.name in ("Add", "Sub", "Mul", "Div"):
            op_sym = {"Add": "+", "Sub": "-", "Mul": "*", "Div": "/"}
            return f"({left_str} {op_sym[node.operator.name]} {right_str})"
        return f"{node.operator.name}({left_str}, {right_str})"

    elif isinstance(node, TSUnaryOpNode):
        return (
            f"{node.operator.name}({tree_to_formula(node.operand)}, {node.time_delta})"
        )

    elif isinstance(node, TSBinaryOpNode):
        left_str = tree_to_formula(node.left)
        right_str = tree_to_formula(node.right)
        return f"{node.operator.name}({left_str}, {right_str}, {node.time_delta})"

    return "???"


# ============================================================
# 4b. 公式字符串 -> 表达式树（tree_to_formula 的逆操作）
# ============================================================
def parse_formula(s: str) -> Optional[ExprNode]:
    """
    将 tree_to_formula 输出的公式字符串解析回表达式树。

    支持格式：
      $close                       -> FeatureNode
      0.5  /  -1                   -> ConstantNode
      (x + y)  (x - y)  (x * y)  (x / y)  -> BinaryOpNode
      Abs(x)  Log(x)              -> UnaryOpNode
      Mean(x, 20)                 -> TSUnaryOpNode
      Corr(x, y, 20)             -> TSBinaryOpNode
      Greater(x, y)  Less(x, y)  -> BinaryOpNode
    """
    import re

    # 构建操作符名称到 Operator 对象的映射
    _op_map: dict = {}
    for tok in VOCAB:
        if tok.token_type == TokenType.OPERATOR:
            _op_map[tok.value.name] = tok.value

    s = s.strip()
    if not s:
        return None

    try:
        node, pos = _parse_expr(s, 0, _op_map)
        if pos == len(s):
            return node
        return None
    except Exception:
        return None


def _skip_ws(s: str, pos: int) -> int:
    while pos < len(s) and s[pos] == " ":
        pos += 1
    return pos


def _parse_expr(s: str, pos: int, ops: dict) -> tuple:
    """递归下降解析，返回 (ExprNode, next_pos)。"""
    pos = _skip_ws(s, pos)
    if pos >= len(s):
        raise ValueError("unexpected end")

    # 括号包裹的中缀二元：(x op y)
    if s[pos] == "(":
        return _parse_infix(s, pos, ops)

    # $feature
    if s[pos] == "$":
        return _parse_feature(s, pos)

    # 尝试匹配函数名 Name(...)
    m = _match_ident(s, pos)
    if m and pos + len(m) < len(s) and s[pos + len(m)] == "(":
        return _parse_func_call(s, pos, m, ops)

    # 数字常量
    return _parse_number(s, pos)


def _match_ident(s: str, pos: int) -> Optional[str]:
    """从 pos 开始匹配一个标识符（字母开头）。"""
    if pos >= len(s) or not s[pos].isalpha():
        return None
    end = pos
    while end < len(s) and (s[end].isalnum() or s[end] == "_"):
        end += 1
    return s[pos:end]


def _parse_feature(s: str, pos: int) -> tuple:
    """解析 $name。"""
    assert s[pos] == "$"
    pos += 1
    end = pos
    while end < len(s) and (s[end].isalnum() or s[end] == "_"):
        end += 1
    name = s[pos:end]
    if not name:
        raise ValueError("empty feature name")
    return FeatureNode(name), end


def _parse_number(s: str, pos: int) -> tuple:
    """解析数字常量（含负号）。"""
    end = pos
    if end < len(s) and s[end] == "-":
        end += 1
    while end < len(s) and (s[end].isdigit() or s[end] == "."):
        end += 1
    # 科学计数法
    if end < len(s) and s[end] in ("e", "E"):
        end += 1
        if end < len(s) and s[end] in ("+", "-"):
            end += 1
        while end < len(s) and s[end].isdigit():
            end += 1
    text = s[pos:end]
    if not text or text == "-":
        raise ValueError(f"invalid number at {pos}")
    val = float(text)
    if val.is_integer():
        val = int(val)
    return ConstantNode(val), end


def _parse_infix(s: str, pos: int, ops: dict) -> tuple:
    """解析 (left OP right)，OP 是 + - * /。"""
    assert s[pos] == "("
    pos += 1  # skip '('
    left, pos = _parse_expr(s, pos, ops)
    pos = _skip_ws(s, pos)
    # 读操作符
    op_char = s[pos]
    op_name_map = {"+": "Add", "-": "Sub", "*": "Mul", "/": "Div"}
    if op_char not in op_name_map:
        raise ValueError(f"unexpected infix op '{op_char}'")
    pos += 1
    pos = _skip_ws(s, pos)
    right, pos = _parse_expr(s, pos, ops)
    pos = _skip_ws(s, pos)
    if pos >= len(s) or s[pos] != ")":
        raise ValueError("missing ')'")
    pos += 1
    return BinaryOpNode(ops[op_name_map[op_char]], left, right), pos


def _parse_args(s: str, pos: int, ops: dict) -> tuple:
    """解析括号内逗号分隔的参数列表，返回 (arg_list, next_pos)。"""
    assert s[pos] == "("
    pos += 1
    args = []
    while True:
        pos = _skip_ws(s, pos)
        if pos < len(s) and s[pos] == ")":
            pos += 1
            break
        arg, pos = _parse_expr(s, pos, ops)
        args.append(arg)
        pos = _skip_ws(s, pos)
        if pos >= len(s):
            raise ValueError("unexpected end while parsing args")
        if s[pos] == ",":
            pos += 1
            continue
        if s[pos] == ")":
            pos += 1
            break
        raise ValueError(f"expected ',' or ')', got '{s[pos]}'")
    return args, pos


def _parse_func_call(s: str, pos: int, name: str, ops: dict) -> tuple:
    """解析 Name(arg1, arg2, ...)。"""
    pos += len(name)  # skip name
    args, pos = _parse_args(s, pos, ops)

    if name not in ops:
        raise ValueError(f"unknown operator '{name}'")

    op = ops[name]
    cat = op.category

    if cat == OpCategory.CS_U:
        if len(args) != 1:
            raise ValueError(f"{name} expects 1 arg, got {len(args)}")
        return UnaryOpNode(op, args[0]), pos

    elif cat == OpCategory.CS_B:
        if len(args) != 2:
            raise ValueError(f"{name} expects 2 args, got {len(args)}")
        return BinaryOpNode(op, args[0], args[1]), pos

    elif cat == OpCategory.TS_U:
        if len(args) != 2:
            raise ValueError(f"{name} expects 2 args, got {len(args)}")
        td = args[1]
        if not isinstance(td, ConstantNode):
            raise ValueError(f"{name} time_delta must be constant")
        return TSUnaryOpNode(op, args[0], int(td.value)), pos

    elif cat == OpCategory.TS_B:
        if len(args) != 3:
            raise ValueError(f"{name} expects 3 args, got {len(args)}")
        td = args[2]
        if not isinstance(td, ConstantNode):
            raise ValueError(f"{name} time_delta must be constant")
        return TSBinaryOpNode(op, args[0], args[1], int(td.value)), pos

    raise ValueError(f"unhandled category for '{name}'")


# ============================================================
# 5. 检查RPN序列是否合法（用栈模拟）
# ============================================================
def is_valid_rpn(tokens: List[Token]) -> bool:
    """
    检查token序列（不含BEG/SEP）是否是合法的RPN。
    合法条件：
    1. 按RPN规则解析后栈中恰好剩1个表达式
    2. 不能等价于单个常数（论文 Appendix C）
    """
    stripped = strip_special_tokens(tokens)
    if len(stripped) == 0:
        return False

    tree = parse_rpn_to_tree(stripped)
    if tree is None:
        return False

    # 不能是纯常数
    if isinstance(tree, ConstantNode):
        return False

    return True


# ============================================================
# 6. 测试
# ============================================================
if __name__ == "__main__":
    from tokens import OP_ADD, OP_SUM, OP_CORR, OP_LOG, OP_GREATER, OP_VAR

    print("=" * 60)
    print("测试1:")
    print("  公式: Sum(Add(5, $vol), 20)")
    print("  RPN:  [5, $vol, Add, 20d, Sum]")
    print("=" * 60)

    # 构造RPN token序列（不含BEG/SEP）
    test_tokens_1 = [
        Token(TokenType.CONSTANT, 5),
        Token(TokenType.FEATURE, "vol"),
        Token(TokenType.OPERATOR, OP_ADD),
        Token(TokenType.TIME_DELTA, 20),
        Token(TokenType.OPERATOR, OP_SUM),
    ]

    tree1 = parse_rpn_to_tree(test_tokens_1)
    print(f"  解析结果: {tree1}")
    print(f"  公式字符串: {tree_to_formula(tree1)}")
    print(f"  合法性: {is_valid_rpn(test_tokens_1)}")

    print()
    print("=" * 60)
    print("测试2: 更复杂的表达式")
    print("  公式: Corr(Greater($close, $open), Log($vol), 30)")
    print("=" * 60)

    test_tokens_2 = [
        Token(TokenType.FEATURE, "close"),
        Token(TokenType.FEATURE, "open"),
        Token(TokenType.OPERATOR, OP_GREATER),
        Token(TokenType.FEATURE, "vol"),
        Token(TokenType.OPERATOR, OP_LOG),
        Token(TokenType.TIME_DELTA, 30),
        Token(TokenType.OPERATOR, OP_CORR),
    ]

    tree2 = parse_rpn_to_tree(test_tokens_2)
    print(f"  解析结果: {tree2}")
    print(f"  公式字符串: {tree_to_formula(tree2)}")
    print(f"  合法性: {is_valid_rpn(test_tokens_2)}")

    print()
    print("=" * 60)
    print("测试3: 非法序列（操作数不够）")
    print("=" * 60)

    test_tokens_3 = [
        Token(TokenType.FEATURE, "close"),
        Token(TokenType.OPERATOR, OP_ADD),  # 二元操作符但只有1个操作数
    ]

    tree3 = parse_rpn_to_tree(test_tokens_3)
    print(f"  解析结果: {tree3}")
    print(f"  合法性: {is_valid_rpn(test_tokens_3)}")

    print()
    print("=" * 60)
    print("测试4: 纯常数（不合法）")
    print("=" * 60)

    test_tokens_4 = [
        Token(TokenType.CONSTANT, 5),
    ]
    print(f"  合法性: {is_valid_rpn(test_tokens_4)}")

    print()
    print("=" * 60)
    print("测试5: Ref($low, 50)")
    print("=" * 60)

    from tokens import OP_REF

    test_tokens_5 = [
        Token(TokenType.FEATURE, "low"),
        Token(TokenType.TIME_DELTA, 50),
        Token(TokenType.OPERATOR, OP_REF),
    ]
    tree5 = parse_rpn_to_tree(test_tokens_5)
    print(f"  解析结果: {tree5}")
    print(f"  公式字符串: {tree_to_formula(tree5)}")
    print(f"  合法性: {is_valid_rpn(test_tokens_5)}")
