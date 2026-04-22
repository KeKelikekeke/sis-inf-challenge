# Wang 2025 restricted-SVP 实现规范（面向当前 SIS∞ 项目）

## 0. 文档用途

这份文档不是论文笔记，而是**工程实现规范**。

目标是：

- 把 Wang 2025 的 restricted-SVP 主线，落地到当前 SIS infinity-norm 项目；
- 统一 Codex / 人工实现方向；
- 明确“哪些是主线，哪些只能作为诊断基线”。

---

## 1. 项目主线目标

### 1.1 当前主线

当前主线必须切换为：

> 复现 Wang 2025 的 restricted-SVP solver，用于求解齐次 SIS∞。

更具体地说：

1. 先把齐次 SIS∞ 建模为 restricted-SVP；
2. 用 infinity norm 的 `P_B(len)` 连接欧氏长度与限制通过概率；
3. 复现 TwoStepSolver；
4. 复现 FlexibleD4F；
5. 复现 Sieve-Then-Slice；
6. 复现 Algorithm 8 的统一调度。

### 1.2 第二阶段主线

待齐次主线跑通后，再进入：

> 非齐次 SIS∞ = Kannan embedding + restricted-SVP

也就是：

1. 构造 embedded lattice；
2. restriction 增加最后一维等于 `±M`；
3. 在嵌入格上复用 restricted-SVP solver。

---

## 2. 当前项目中不再作为主线的内容

以下内容可以保留，但只能降级为 **diagnostic baseline**：

- 直接取 BKZ reduced basis 前几行
- 朴素 pairwise combination
- small-coeff linear combination
- 仅依赖 `top_k / pair_budget / pair_max_base` 的 search

这些方法可以继续保留用来：

- 观察 BKZ 产出的候选分布；
- 做工程 smoke test；
- 快速 sanity check；
- 与 Wang 主线做对照。

但它们**不能再作为项目主算法继续扩展**。

---

## 3. 第一阶段：只做齐次 SIS∞

### 3.1 数学问题

当前阶段只处理：

$$
Av + u \equiv 0 \pmod q,
\qquad
\|u\|_\infty,\|v\|_\infty \le \gamma .
$$

其中最终 restriction 可统一理解为：

```text
R(x) = 1  <=>  ||x||_∞ <= gamma
```

这里 `x` 可以是直接的格向量表示，也可以是解码后的 `[u; v]` 组合对象，具体由现有仓库的数据模型决定。

### 3.2 当前阶段禁止项

在齐次主线完成前，不要：

- 直接加入非齐次 embedding 逻辑
- 把 MISIS / Dilithium 的 rescaling 一起实现
- 为题 5/8 这种额外欧氏范数条件开新支线
- 把 Gaussian sampling 当当前第一优先级

---

## 4. 必做模块

### 模块 A：restricted-SVP 建模层

#### 目标

实现如下抽象：

- `restriction predicate R(v)`
- `related probability function P(len)`
- `target list size` 计算

#### 对齐次 SIS∞ 的具体要求

使用 Wang 2025 Section 6.2 的概率近似：

```text
P_B(len) ≈ (1 - 2 Φ(-sqrt(d) * B / len))^d
```

其中：

- `d` 为格维数或与具体 restriction 对应的有效维数；
- `B = gamma`。

#### 必须暴露的接口

建议至少有：

- `restriction_infinity_norm(candidate, bound)`
- `prob_infinity_norm_pass(dim, bound, len)`
- `required_list_size(p_success, p_single)`

#### 验收标准

- 概率函数值域正确，且对 `len` 单调下降；
- `size = log(1-p_success)/log(1-p_single)` 正确实现；
- 有单元测试；
- 有最小 CLI/demo。

---

### 模块 B：TwoStepSolver 底座

#### 目标

实现 Wang 3.2 / Algorithm 4 对应的骨架：

1. BKZ 到目标 basis quality `δ`；
2. 在 `κ` 维 projected sublattice 上做 sieve；
3. 输出短向量列表，而不是单个答案。

#### 工程要求

- 尽量复用现有 BKZ / lattice / decode / validation 代码；
- 但不得退化成“直接取 reduced rows 前几条”；
- 需要明确当前是 paper-accurate 版本还是 engineering approximation。

#### 验收标准

- 能输出非空短向量列表；
- 输出向量可进入 restriction 检查；
- 日志包含：
  - reduction 参数
  - `κ`
  - 输出列表大小
  - 长度统计

---

### 模块 C：FlexibleD4F

#### 目标

当目标列表规模小于普通 sieve 列表规模时，实现 Wang 5.1 的分支。

#### 必要结构

- 输入：`δ, κ, f'`
- 在 `L[f':κ]` 上生成列表
- Babai lift 回原格
- 按 Wang 的长度阈值筛选

核心阈值：

```text
γ = sqrt(4/3) * δ^(-f' d / (d-1))
len = γ * gh(L[0:κ])
```

#### 验收标准

- `f'=0` 可退化到无 d4f 情况；
- 随 `f'` 增加，输出列表规模合理变化；
- 输出向量长度统计合理；
- 有单元测试与 smoke test。

---

### 模块 D：Sieve-Then-Slice

#### 目标

当目标列表规模大于普通 sieve 列表规模时，实现 Wang 5.2 的分支。

#### 必要结构

1. `Lsieve <- Sieve(L[0:κ])`
2. 计算 `φ`
3. `T' <- Sieve(L[κ:κ+φ])`
4. `T <- Lift(T')`
5. `Lout <- ModifiedRandomizedSlicer(T, Lsieve, S)`

#### 禁止项

- 不允许用 pairwise search 冒充 Sieve-Then-Slice；
- 不允许把 small-coeff 组合作为 slicer 的替代品而不注明。

#### 如果当前工程无法严格复现

允许先做一个“工程近似版”，但必须：

- 明确标记为 `engineering approximation`
- 说明与论文完整版的差距
- 保持接口设计与最终论文版本兼容

#### 验收标准

- 输出列表规模可超过普通 sieve 的基准规模；
- 列表不只是简单重复旧向量；
- 输出向量仍然可以做 restriction 筛查；
- 日志包含：
  - `Lsieve` 大小
  - `T'` 大小
  - `T` 大小
  - slicer 输出大小
  - 长度统计

---

### 模块 E：Algorithm 8 统一调度层

#### 目标

复现 Wang 5.3 的统一调度逻辑。

#### 统一输入

至少包含：

- lattice / instance
- restriction `R`
- related probability `P(len)`
- target success rate `p_success`
- BKZ / basis-quality 参数
- `κ`

#### 决策逻辑

1. 计算 `len`
2. 计算 `P(len)`
3. 计算所需列表大小
   
   `S = log(1-p_success)/log(1-P(len))`
4. 若 `S > p^(4κ/3)`，走 Sieve-Then-Slice
5. 否则走 FlexibleD4F
6. 在线性扫描中找满足 `R(v)=1` 的向量

#### 验收标准

日志必须包含：

- `len`
- `P(len)`
- target size `S`
- 选择了哪条支路
- 实际生成的列表大小
- pass restriction 的数量

必须提供：

- 一个最小 end-to-end 示例
- 一个集成测试

---

## 5. 推荐模块布局

建议新增或重构为下面的模块形态：

```text
src/sisinf/
  restricted_svp.py          # restriction + size logic + top-level interfaces
  probability.py             # P_B(len), later embedding probability
  two_step.py                # TwoStepSolver
  flexible_d4f.py            # FlexibleD4F
  sieve_then_slice.py        # Sieve-Then-Slice + slicer wrapper
  solver_restricted_hom.py   # homogeneous restricted-SVP solver
  solver_restricted_inhom.py # stage 2 only, not now
```

已有模块尽量复用：

- `io.py`
- `types.py`
- `validate.py`
- `lattice.py`
- BKZ / reduction 相关代码

但当前旧的 pairwise / search 代码建议：

- 保留；
- 标注为 `diagnostic_baseline`；
- 不再作为主 solver 入口。

---

## 6. 所有实现都必须遵守的规则

### 6.1 不允许静默替换论文主线

如果某一步不能严格按论文实现，必须显式标注为：

- `heuristic approximation`
- `engineering simplification`

不能直接把别的启发式方法改名后当作 Wang 方案实现。

### 6.2 复杂任务先出计划，再改代码

每次进入一个大模块前，先给出：

- 代码结构变化
- 关键函数
- 测试计划
- 与论文的对应关系

### 6.3 每个阶段都必须有日志

至少输出：

- 输入参数
- 生成列表大小
- 长度统计
- restriction 通过统计
- 最终是否找到解

### 6.4 每个阶段都必须有测试

至少包括：

- 单元测试
- 一个 smoke command
- 一个 verbose 运行示例

---

## 7. 当前仓库中已有模块的定位建议

### 保留并复用

- 数据读取
- 候选验证
- lattice basis 构造与 decode
- BKZ 调用与环境
- 基础诊断脚本

### 降级为诊断基线

- 旧 `search.py`
- pairwise search
- small-coeff search
- 任何“直接看 reduced row prefix”的逻辑

这些模块只允许：

- 做对照实验；
- 做 sanity check；
- 提供开发期快速反馈。

不得再作为主 solver 的扩展方向。

---

## 8. 当前阶段的阅读与实现优先级

### 先读

1. Section 1 + 1.1
2. Section 4
3. Section 5.3
4. Section 5.1
5. Section 5.2
6. Section 6.2

### 后读

7. Section 6.1
8. Section 6.3

---

## 9. 当前阶段的停止标准

只有当下面几件事都成立，才算“齐次 restricted-SVP 主线基本落地”：

1. 已实现统一 restriction + `P(len)` 接口
2. 已实现真正的 TwoStepSolver
3. 已实现 FlexibleD4F
4. 已实现 Sieve-Then-Slice 或清晰标注的近似版
5. 已实现 Algorithm 8 调度
6. solver 能基于目标成功率自动选择支路
7. end-to-end 日志可解释
8. 测试可运行且通过

在此之前，不要宣称已经“复现 Wang 2025 主算法”。

---

## 10. 一句话总目标

> 当前仓库的主线目标，是把 SIS∞ 求解问题从“BKZ + 人工后处理”正式提升为“基于 Wang 2025 的 restricted-SVP solver”：以 two-step 为底座，以 `P(len)` 控制所需列表规模，以 FlexibleD4F / Sieve-Then-Slice 为两条列表生成支路，以 Algorithm 8 为统一调度器。
