"""
RPN 合法动作掩码

核心规则：
  - 栈只包含两种类型：'E'(表达式) 和 'D'(时间窗口占位)
  - D 入栈后，下一步必须是 TS 操作符（否则 D 会被埋没无法消费）
  - SEP 仅当栈恰好剩 1 个含特征的 E 时合法
  - 每步检查剩余 token 数是否足够关闭表达式
"""

import numpy as np
from tokens import (
    VOCAB,
    VOCAB_SIZE,
    TOKEN_TO_IDX,
    TokenType,
    OpCategory,
    Token,
    BEG_TOKEN,
    SEP_TOKEN,
)


class RPNBuilder:
    """跟踪 RPN 生成状态，提供合法动作掩码。"""

    def __init__(self, max_len: int = 20):
        self.max_len = max_len
        self.reset()

    def reset(self):
        self.seq: list[int] = []
        self.stack: list[str] = []  # 'E' 或 'D'
        self.stack_has_feat: list[bool] = []
        self.stack_last_op: list[str] = []  # 栈顶E的最后一个操作符名，叶节点为""
        self.stack_log_safe: list[bool] = []  # 当前表达式是否可安全作为 Log 输入
        self.has_feature = False
        self.done = False
        self._d_on_top = False
        self._last_token_type = None  # 上一步token类型

    def step(self, token_idx: int):
        """执行一步动作，更新栈状态。"""
        tok = VOCAB[token_idx]
        self.seq.append(token_idx)
        tt = tok.token_type
        self._last_token_type = tt

        if tt == TokenType.BEG:
            pass
        elif tt == TokenType.SEP:
            self.done = True
        elif tt == TokenType.FEATURE:
            self.stack.append("E")
            self.stack_has_feat.append(True)
            self.stack_last_op.append("")
            self.stack_log_safe.append(True)
            self.has_feature = True
            self._d_on_top = False
        elif tt == TokenType.CONSTANT:
            self.stack.append("E")
            self.stack_has_feat.append(False)
            self.stack_last_op.append("")
            self.stack_log_safe.append(tok.value > 0)
            self._d_on_top = False
        elif tt == TokenType.TIME_DELTA:
            self.stack.append("D")
            self.stack_has_feat.append(False)
            self.stack_last_op.append("")
            self.stack_log_safe.append(False)
            self._d_on_top = True
        elif tt == TokenType.OPERATOR:
            op = tok.value
            cat = op.category
            if cat == OpCategory.CS_U:
                hf = self.stack_has_feat[-1]
                self.stack.pop()
                self.stack_has_feat.pop()
                self.stack_last_op.pop()
                self.stack_log_safe.pop()
                self.stack.append("E")
                self.stack_has_feat.append(hf)
                self.stack_last_op.append(op.name)
                if op.name == "Abs":
                    self.stack_log_safe.append(True)
                else:
                    self.stack_log_safe.append(False)
            elif cat == OpCategory.CS_B:
                hf = self.stack_has_feat[-1] or self.stack_has_feat[-2]
                left_log_safe = self.stack_log_safe[-2]
                right_log_safe = self.stack_log_safe[-1]
                self.stack.pop()
                self.stack_has_feat.pop()
                self.stack_last_op.pop()
                self.stack_log_safe.pop()
                self.stack.pop()
                self.stack_has_feat.pop()
                self.stack_last_op.pop()
                self.stack_log_safe.pop()
                self.stack.append("E")
                self.stack_has_feat.append(hf)
                self.stack_last_op.append(op.name)
                if op.name in ("Add", "Mul", "Div"):
                    self.stack_log_safe.append(left_log_safe and right_log_safe)
                elif op.name == "Sub":
                    self.stack_log_safe.append(False)
                elif op.name == "Greater":
                    self.stack_log_safe.append(left_log_safe or right_log_safe)
                elif op.name == "Less":
                    self.stack_log_safe.append(left_log_safe and right_log_safe)
                else:
                    self.stack_log_safe.append(False)
            elif cat == OpCategory.TS_U:
                self.stack.pop()
                self.stack_has_feat.pop()
                self.stack_last_op.pop()  # D
                self.stack_log_safe.pop()
                hf = self.stack_has_feat[-1]
                operand_log_safe = self.stack_log_safe[-1]
                self.stack.pop()
                self.stack_has_feat.pop()
                self.stack_last_op.pop()  # E
                self.stack_log_safe.pop()
                self.stack.append("E")
                self.stack_has_feat.append(hf)
                self.stack_last_op.append(op.name)
                if op.name in ("Ref", "Mean", "Med", "Sum", "Max", "Min", "WMA", "EMA"):
                    self.stack_log_safe.append(operand_log_safe)
                else:
                    self.stack_log_safe.append(False)
            elif cat == OpCategory.TS_B:
                self.stack.pop()
                self.stack_has_feat.pop()
                self.stack_last_op.pop()  # D
                self.stack_log_safe.pop()
                hf = self.stack_has_feat[-1] or self.stack_has_feat[-2]
                self.stack.pop()
                self.stack_has_feat.pop()
                self.stack_last_op.pop()  # E
                self.stack_log_safe.pop()
                self.stack.pop()
                self.stack_has_feat.pop()
                self.stack_last_op.pop()  # E
                self.stack_log_safe.pop()
                self.stack.append("E")
                self.stack_has_feat.append(hf)
                self.stack_last_op.append(op.name)
                self.stack_log_safe.append(False)
            self._d_on_top = False

        if len(self.seq) >= self.max_len:
            self.done = True

    def get_valid_mask(
        self, allowed_op_categories: set[str] | None = None
    ) -> np.ndarray:
        """返回 shape=(VOCAB_SIZE,) 的布尔掩码。

        参数:
            allowed_op_categories: 允许的操作符类别集合，如 {"TS-U", "TS-B"}。
                                   None 表示允许所有类别（向后兼容）。
        """
        mask = np.zeros(VOCAB_SIZE, dtype=bool)
        if self.done:
            return mask

        sz = len(self.stack)
        remaining = self.max_len - len(self.seq)

        # D 在栈顶时，只允许 TS 操作符
        if self._d_on_top:
            for idx, tok in enumerate(VOCAB):
                if tok.token_type != TokenType.OPERATOR:
                    continue
                op = tok.value
                # 课程学习过滤：只允许指定类别的操作符
                if allowed_op_categories is not None:
                    if op.category.value not in allowed_op_categories:
                        continue
                if op.category == OpCategory.TS_U:
                    if sz >= 2 and self.stack[-2] == "E":
                        # 操作数必须含特征，不允许对纯常数做时序操作
                        if self.stack_has_feat[-2]:
                            if remaining >= sz - 1 + 1:
                                mask[idx] = True
                elif op.category == OpCategory.TS_B:
                    if sz >= 3 and self.stack[-2] == "E" and self.stack[-3] == "E":
                        # 至少一个操作数含特征
                        if self.stack_has_feat[-2] or self.stack_has_feat[-3]:
                            if remaining >= sz - 2 + 1:
                                mask[idx] = True
            return mask

        # 正常状态（栈全为 E）
        n_e = sz  # 正常状态下全是 E

        # 上一步是否推入了常数（用于限制连续常数）
        last_was_const = self._last_token_type == TokenType.CONSTANT

        for idx, tok in enumerate(VOCAB):
            tt = tok.token_type

            if tt == TokenType.BEG:
                continue

            if tt == TokenType.SEP:
                if n_e == 1 and self.has_feature and self.stack_has_feat[0]:
                    mask[idx] = True
                continue

            if tt == TokenType.FEATURE:
                if remaining >= sz + 2:
                    mask[idx] = True
                continue

            if tt == TokenType.CONSTANT:
                # 规则: 不允许连续推入常数（浪费token长度，应先用操作符消费）
                if last_was_const:
                    continue
                if remaining >= sz + 2:
                    mask[idx] = True
                continue

            if tt == TokenType.TIME_DELTA:
                # 需要 n_e >= 1 才有 E 配对，且栈中至少有一个含特征的 E
                has_any_feat = any(self.stack_has_feat[i] for i in range(n_e))
                if n_e >= 1 and has_any_feat and remaining >= sz + 2:
                    mask[idx] = True
                continue

            if tt == TokenType.OPERATOR:
                op = tok.value
                cat = op.category
                # 课程学习过滤：只允许指定类别的操作符
                if allowed_op_categories is not None:
                    if cat.value not in allowed_op_categories:
                        continue
                if cat == OpCategory.CS_U:
                    if n_e >= 1 and self.stack_has_feat[-1]:
                        # 规则: 禁止 Abs(Abs(x)) — 冗余
                        # 规则: 禁止 Log(Log(x)) — 几乎全NaN
                        top_op = self.stack_last_op[-1]
                        if op.name == top_op:
                            continue
                        if op.name == "Log" and not self.stack_log_safe[-1]:
                            continue
                        if remaining >= sz + 1:
                            mask[idx] = True
                elif cat == OpCategory.CS_B:
                    if n_e >= 2:
                        if self.stack_has_feat[-1] or self.stack_has_feat[-2]:
                            if remaining >= sz:
                                mask[idx] = True

        return mask


if __name__ == "__main__":
    from tokens import OP_ADD, OP_MEAN, OP_CORR

    builder = RPNBuilder(max_len=20)
    beg_idx = TOKEN_TO_IDX[Token(TokenType.BEG, BEG_TOKEN)]
    builder.step(beg_idx)

    m = builder.get_valid_mask()
    valid_tokens = [VOCAB[i] for i in range(VOCAB_SIZE) if m[i]]
    print(f"BEG 后合法动作 ({sum(m)} 个):")
    for t in valid_tokens:
        print(f"  {t}")
