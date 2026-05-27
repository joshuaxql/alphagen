# AlphaGen — 基于强化学习的Alpha因子生成系统

> 基于深度强化学习（PPO + GAE）自动挖掘股票Alpha因子的量化研究框架。

---

## 1. 项目概述

AlphaGen 使用 **PPO（Proximal Policy Optimization）** 算法训练一个序列生成模型（LSTM 或 Transformer），通过生成逆波兰表达式（RPN）来构建股票Alpha因子。系统支持：

- **自动因子生成**：策略网络生成RPN token序列，解析为表达式树后计算alpha值
- **多目标奖励**：同时优化IC增量、ICIR稳定性、Rank IC鲁棒性、正负平衡、冗余控制和复杂度约束
- **组合模型**：通过梯度下降动态优化因子权重，构建多因子组合
- **课程学习**：分阶段逐步开放操作符和时间窗口复杂度
- **断点续训**：完整保存/恢复训练状态，支持随时中断和恢复
- **回测验证**：基于验证集进行组合回测，对比基准指数

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练管线  (train.py)                      
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  PPO Agent  │───▶│ RPN Builder │───▶│ evaluate_episode()  │  │
│  │  (LSTM/TF)  │    │  (masking)  │    │  (奖励计算 + 池子)    │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         ▲                                         │             │
│         │                                         ▼             │
│  ┌──────┴──────┐                        ┌─────────────────────┐ │
│  │   Update    │◀───────────────────────│ AlphaCombination    │ │
│  │  (PPO+GAE)  │                        │ Model               │ │
│  └─────────────┘                        └─────────────────────┘ │
│                                                                 │
│  外围：CurriculumManager（课程学习）/ Checkpoint（断点续训）         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      评估回测 (backtest.py)                      
│  验证集因子IC ──▶ 组合权重优化 ──▶ 日频回测 ──▶ 对比基准指数           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 核心模块

| 模块             | 文件                       | 职责                                                 |
| ---------------- | -------------------------- | ---------------------------------------------------- |
| **数据加载**     | `data.py`                  | 从数据源获取日线数据（open/high/low/close/vol/vwap） |
| **特征计算**     | `calculator.py`            | 将表达式树解析为numpy计算，支持numba加速             |
| **表达式树**     | `expression.py`            | RPN token序列 ↔ 表达式树 ↔ 字符串公式 的互相转换     |
| **Token词表**    | `tokens.py`                | 定义49个token（特征/常数/操作符/时间窗口/特殊标记）  |
| **掩码生成**     | `masking.py`               | 基于RPN语法规则动态生成合法token掩码                 |
| **策略网络**     | `generator.py`             | LSTM-based AlphaGenNet + PPOAgent（含GAE）           |
| **策略网络(TF)** | `generator_transformer.py` | Transformer-based 替代网络                           |
| **组合模型**     | `combination.py`           | Alpha组合、权重优化、IC/IR计算、互相关矩阵           |
| **训练管线**     | `train.py`                 | 完整训练循环、课程学习、断点续训、CLI入口            |
| **回测引擎**     | `backtest.py`              | 日频组合回测、夏普/年化/最大回撤计算                 |
| **报告可视化**   | `reporting.py`             | 训练曲线、收益曲线、benchmark对比图                  |

---

## 3. 改进记录

### P0 紧急修复（防止Agent作弊）

| 修改               | 说明                                                                    |
| ------------------ | ----------------------------------------------------------------------- |
| **接受阈值收紧**   | `1e-4 → 1e-6`，阻止ic_delta≈0的无意义因子进入池子                       |
| **IC权重提升**     | `10.0 → 50.0`，让IC增量成为绝对主导目标                                 |
| **正负平衡增强**   | `balance_bonus 0.05 → 1.0`，强力鼓励正IC方向                            |
| **方向基础奖励**   | 新增 `direction_bonus = sign(candidate_ic) * 0.5`，所有因子都有方向激励 |
| **复杂度惩罚重构** | 原始特征罚5.0（极重），复杂公式仅罚 `len/200`（极轻）                   |
| **原始特征去重**   | 同类型原始特征（$close/$vol等）池中最多只能有1个                        |
| **ic_delta硬门槛** | `ic_delta <= 1e-6` 的接受会被强制回滚并返回负奖励                       |

### P1 重要改进（质量与稳定性）

| 修改              | 说明                                                            |
| ----------------- | --------------------------------------------------------------- |
| **状态保存/恢复** | `combination.save_state()` / `restore_state()` 保存完整池子快照 |
| **连续下降回退**  | 验证IC连续3轮下降时，回退到**下降开始前的池子状态**（非清空）   |
| **断点续训**      | 完整保存/恢复：网络权重、优化器状态、池子、训练历史、验证IC追踪 |

---

## 4. 快速开始

### 4.1 安装依赖

```bash
pip install torch numpy pandas matplotlib tqdm
# 可选：numba 加速
pip install numba
```

### 4.2 基础训练

```bash
python train.py \
  --iterations 50 \
  --episodes 64 \
  --pool_size 15 \
  --horizon 3 \
  --lr 3e-4 \
  --save_dir outputs
```

### 4.3 使用Transformer模型

```bash
python train.py \
  --model transformer \
  --tf_embed_dim 64 \
  --tf_nhead 4 \
  --tf_num_layers 3 \
  --iterations 50
```

### 4.4 断点续训

```bash
# 方式1：指定目录（自动查找 checkpoint_latest.pt）
python train.py --iterations 100 --save_dir outputs --resume outputs

# 方式2：指定具体checkpoint
python train.py --iterations 100 --resume outputs/checkpoint_iter40.pt

# 方式3：基于最佳模型继续
python train.py --iterations 100 --resume outputs/checkpoint_best.pt
```

### 4.5 自定义数据区间

```bash
python train.py \
  --train_start 20190101 \
  --train_end 20231231 \
  --val_start 20240101 \
  --val_end 20251231
```

---

## 5. CLI参数详解

### 5.1 数据参数

| 参数            | 默认值     | 说明                      |
| --------------- | ---------- | ------------------------- |
| `--train_start` | `20190101` | 训练集起始日期 (YYYYMMDD) |
| `--train_end`   | `20241231` | 训练集结束日期            |
| `--val_start`   | `20250101` | 验证集起始日期            |
| `--val_end`     | `20251231` | 验证集结束日期            |

### 5.2 训练参数

| 参数                  | 默认值 | 说明                |
| --------------------- | ------ | ------------------- |
| `--iterations`        | `20`   | 训练轮数            |
| `--episodes`          | `64`   | 每轮episode数       |
| `--pool_size`         | `15`   | Alpha池最大容量     |
| `--horizon`           | `3`    | 目标收益前瞻天数    |
| `--lr`                | `3e-4` | 学习率              |
| `--gamma`             | `0.99` | GAE折扣因子         |
| `--gae_lambda`        | `0.95` | GAE lambda          |
| `--enable_curriculum` | `True` | 启用课程学习        |
| `--no_curriculum`     | -      | 禁用课程学习        |
| `--device`            | `auto` | 训练设备 (cpu/cuda) |
| `--resume`            | `None` | 断点续训路径        |

### 5.3 模型参数

| 参数                   | 默认值 | 说明                            |
| ---------------------- | ------ | ------------------------------- |
| `--model`              | `rnn`  | 策略网络类型: rnn / transformer |
| `--tf_embed_dim`       | `64`   | Transformer embedding维度       |
| `--tf_nhead`           | `4`    | Transformer 注意力头数          |
| `--tf_num_layers`      | `3`    | Transformer Encoder层数         |
| `--tf_dim_feedforward` | `256`  | 前馈网络维度                    |
| `--tf_dropout`         | `0.1`  | Dropout概率                     |

### 5.4 回测参数

| 参数               | 默认值      | 说明         |
| ------------------ | ----------- | ------------ |
| `--n_hold`         | `20`        | 持仓股票数   |
| `--n_swap`         | `3`         | 每期换仓数   |
| `--commission`     | `0.001`     | 交易佣金率   |
| `--benchmark_code` | `000300.SH` | 基准指数代码 |

### 5.5 输出参数

| 参数         | 默认值    | 说明     |
| ------------ | --------- | -------- |
| `--save_dir` | `outputs` | 输出目录 |

---

## 6. 输出文件说明

训练完成后，`save_dir` 目录下会生成以下文件：

```
outputs/
├── checkpoint_latest.pt      # 最新训练状态（断点续训用）
├── checkpoint_best.pt        # 验证IC最佳时的状态
├── checkpoint_iter20.pt      # 每20轮自动保存
├── checkpoint_final.pt       # 最终状态
├── net_latest.pt             # 最新网络权重
├── net_best.pt               # 最佳网络权重
├── net_final.pt              # 最终网络权重
├── pool_latest.json          # 最新因子池
├── pool_best.json            # 最佳因子池
├── pool_final.json           # 最终因子池
├── training_summary.json     # 训练摘要（train/val IC、ICIR）
├── training_history.json     # 每轮详细指标
├── training_history.csv      # 同上，CSV格式
├── training_metrics.png      # 训练曲线图
├── validation_report.json    # 验证集回测详情
├── backtest_vs_benchmark.png # 回测对比图
└── equity_curve_vs_benchmark.png  # 收益曲线图
```

---

## 7. 奖励函数设计

当前奖励函数（多目标）：

```python
reward = (
    50.0 * ic_delta                 # IC增量（压倒性主导）
    + 3.0 * icir_delta              # ICIR增量（稳定性）
    + 2.0 * candidate_rank_ic       # Rank IC（鲁棒性）
    + direction_bonus               # 基础方向奖励：sign(candidate_ic) * 0.5
    + balance_bonus                 # 正负平衡：1.0（若方向与多数相反）
    - redundancy_penalty            # 冗余惩罚：max_mutual > 0.7时触发
    - complexity_penalty            # 复杂度：原始特征=5.0，复杂公式=len/200
)
```

---

## 8. 课程学习策略

默认启用4阶段课程：

| 阶段 | 轮次  | 可用操作符              | 最大序列长度    |
| ---- | ----- | ----------------------- | --------------- |
| 1    | 1-5   | TS-U（时序一元）        | `max_len * 0.4` |
| 2    | 6-10  | TS-U + TS-B（时序二元） | `max_len * 0.6` |
| 3    | 11-15 | + CS-U（截面一元）      | `max_len * 0.8` |
| 4    | 16+   | 全部操作符              | `max_len`       |

---

## 9. 断点续训机制

### 9.1 自动保存

每轮训练结束后自动保存 `checkpoint_latest.pt`，包含：

```python
{
    "iteration": 当前轮次,
    "net_state": 神经网络权重,
    "optimizer_state": Adam优化器状态,
    "pool_state": Alpha池完整快照（因子/权重/IC矩阵）,
    "best_score": 最佳验证分数,
    "history": 完整训练历史,
    "val_ic_history": 验证IC追踪,
}
```

### 9.2 恢复内容

```bash
python train.py --resume outputs
```

恢复时会加载：
- 网络权重 → 继续训练
- 优化器状态 → 保持学习率调度
- Alpha池 → 保留已有因子
- 训练进度 → 从断点轮次继续
- 最佳分数 → 保留历史最优

---

## 10. 验证IC下降回退

当验证IC连续3轮下降时，系统会：

1. 检测：`val_ic[t-3] > val_ic[t-2] > val_ic[t-1] > val_ic[t]`
2. 回退：恢复到 `t-3` 时刻的池子状态（下降开始前）
3. 丢弃：下降期间（t-2, t-1, t）加入的所有因子
4. 重启：从回退后的状态继续训练

这比直接清空池子更温和，保留了下降前的有效探索成果。

---

## 11. 常见问题

**Q: 为什么原始特征（$close/$vol等）被严厉惩罚？**  
A: 实验发现agent会利用宽松的奖励函数作弊——生成最短的原始特征来最小化复杂度惩罚。原始特征去重+5.0重罚迫使agent学习构建有意义的复合因子。

**Q: 训练中断后如何恢复？**  
A: 直接添加 `--resume outputs` 即可，会自动加载 `checkpoint_latest.pt` 并从中断轮次继续。

**Q: 如何延长训练轮数？**  
A: 使用 `--resume` 加载已有checkpoint，同时增加 `--iterations` 参数，例如：`--iterations 100 --resume outputs`。

**Q: 课程学习有什么作用？**  
A: 防止早期探索阶段生成过长的无效表达式。先让agent学会简单操作，再逐步开放复杂操作符，提高样本效率。

---

## 12. Citation

本项目参考了以下工作：
- AlphaGen: 基于强化学习的Alpha因子挖掘
- PPO + GAE 训练策略（Schulman et al., 2017）

---
