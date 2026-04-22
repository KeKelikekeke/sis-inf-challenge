# Wang 2025 阅读笔记：Heuristic Algorithm for Solving Restricted SVP and its Applications

## 0. 这篇论文到底在解决什么问题

作者想解决的问题不是普通的 approximate-SVP，而是这样一类问题：

- 我们既想找**足够短**的格向量；
- 又要求这个向量满足某个**额外限制**；
- 这个限制通常不能仅由欧氏范数是否足够小来表达。

论文给出的统一抽象是 **restricted SVP**。它的核心思想是：

> 不再把“找短向量”和“满足额外限制”混成一个黑盒问题，而是把“限制是否通过”显式建模成一个谓词 `R(v)`，再用一个与长度相关的通过概率函数 `P(len)` 来控制需要生成多大的短向量列表。

作者强调，这类问题的典型例子包括：

1. **SVP under infinity norm**：限制是 `||v||_∞ <= B`；
2. **Kannan embedding for approximate CVP**：限制是嵌入后向量最后一维必须等于 `±M`。

---

## 1. 论文的三项核心贡献

### 1.1 新问题定义：restricted SVP

restricted SVP 被定义为：

- 给定格 `L`；
- 给定限制谓词 `R : R^d -> {0,1}`；
- 目标是找到一个格向量 `v in L` 使 `R(v)=1`；
- 并且要求通过限制的概率 `P(len)` 随着长度界 `len` 变小而增大。

这比普通 approximate-SVP 更适合描述：

- infinity norm 约束；
- Kannan embedding 的最后一维约束；
- 两者叠加后的 restricted-CVP / MISIS 型问题。

### 1.2 两类“输出指定规模短向量列表”的算法

论文不是直接重复跑 approximate-SVP，而是设计两类列表生成算法：

1. **Flexible Dimension for Free (FlexibleD4F)**
   - 用于输出一个**比普通 sieve 列表更小**的短向量列表；
2. **Sieve-Then-Slice**
   - 用于输出一个**比普通 sieve 列表更大**的短向量列表。

### 1.3 一个统一求解器：Algorithm 8 / RestrictedSVPSolver

最后作者把 two-step、FlexibleD4F、Sieve-Then-Slice 统一起来，构造出一个 heuristic restricted-SVP solver。

它的关键不是“只求一个最短向量”，而是：

- 先用 BKZ + sieve 建立短向量生成器；
- 再根据 `P(len)` 和目标成功率，计算需要的候选列表大小；
- 决定走哪一条列表生成分支；
- 最后在列表中查找满足 `R(v)=1` 的向量。

---

## 2. Section 4：为什么 restricted SVP 是这篇论文的真正起点

### 2.1 固定长度版本（Definition 4.1）

最简单版本是：

- 给定长度界 `len`；
- 给定限制谓词 `R(v)`；
- 在 `L ∩ B^d(len)` 中找一个满足 `R(v)=1` 的向量。

这对应一个自然但低效的朴素算法：

1. 先用 two-step approximate-SVP 生成一个短向量列表；
2. 再在列表里线性扫描检查 `R(v)`。

作者指出：这个朴素方法的问题在于，**列表大小不一定对**：

- 可能太大，浪费；
- 也可能太小，成功率不够。

### 2.2 一般版本（Definition 4.2）

论文最后使用的是更一般的版本：

- 不再把 `len` 当作固定输入；
- 而是显式定义一个 related probability function
  
  `P(len) = Pr(R(v)=1 | v in L and ||v|| <= len)`

并要求：

- `len` 越小，`P(len)` 越大；
- 也就是说，向量越短，越容易通过限制。

这使 restricted-SVP 成为 approximate-SVP 的一个更灵活的变体。

### 2.3 这节最重要的思想

这一节最关键的不是定义本身，而是下面这条思维转换：

> 我们不再问“最短向量是谁”，而是问“为了以目标成功率找到一个满足限制的向量，需要生成多大的短向量列表”。

这会直接导向下面这个核心公式：

```text
size = log(1 - p_success) / log(1 - P(len))
```

这里：

- `P(len)` 是单个候选在长度界 `len` 下通过 restriction 的概率；
- `size` 是为了达到成功率 `p_success` 所需的列表大小。

---

## 3. Section 3.2：Two-step solver 是整个主线的底座

### 3.1 TwoStepSolver 的形式

论文把 approximate-SVP 的 two-step 模式写成：

1. 先对格做 BKZ，到目标 RHF `δ`；
2. 再在 `κ` 维 projected sublattice 上做一次 sieve；
3. 输出一个短向量列表 `S`。

这和“只看 reduced basis 前几行”是根本不同的：

- two-step 的输出是**短向量列表**；
- 而不是“BKZ 基向量前几行”这种非常保守的候选源。

### 3.2 两个核心参数

two-step solver 的核心参数是：

- `δ`：BKZ 后的 basis quality；
- `κ`：最后调用 sieve 的维度。

目标是：

- 让输出向量足够短；
- 同时总成本最小。

### 3.3 为什么 two-step 在这里重要

因为 restricted-SVP 的主线不是“求解一个答案”，而是“生成一个可控大小的短向量列表”。

所以 two-step 不是配角，而是整篇论文所有后续算法的底座。

---

## 4. Section 5.1：Flexible Dimension for Free（小列表分支）

### 4.1 它要解决的问题

如果所需列表大小 `S` 比普通 sieve 的列表规模 `p^(4κ/3)` **小很多**，那直接 full sieve 会浪费。

这时论文提出：

- 不是非黑即白地“全 d4f”或“无 d4f”；
- 而是允许取一个中间的 free dimensions 数 `f'`；
- 在 `L[f':κ]` 上做 sieve；
- 再 Babai lift 回原格。

### 4.2 算法结构（Algorithm 6）

输入：

- BKZ-reduced basis
- RHF `δ`
- sieving dimension `κ`
- free dimensions `f'`

输出：

- 一批 lifted short vectors

核心步骤：

1. 在 `L[f':κ]` 上做 sieve，得到 `Lsieve`；
2. 令
   
   `γ = sqrt(4/3) * δ^{-f'd/(d-1)}`；
3. 对每个 `v in Lsieve` 做 Babai lift；
4. 如果 lift 后长度不超过 `γ * gh(L[0:κ])`，则保留。

### 4.3 这条支路的意义

作者证明它输出的列表规模大约是：

```text
|L| ≈ γ^κ
```

所以它可以用来**连续调节**：

- 时间成本；
- 输出列表规模；
- 向量长度。

### 4.4 这一节你该抓住什么

最重要的不是公式细节，而是下面这条思想：

> d4f 不是“能用多少免费维度就尽量用多少”，而是一个调节列表大小与成本的旋钮。

---

## 5. Section 5.2：Sieve-Then-Slice（大列表分支）

### 5.1 它要解决的问题

如果所需列表大小 `S` **大于** 普通 sieve 列表规模 `p^(4κ/3)`，那单次 sieve 不够。

这时不能靠简单重复跑 approximate-SVP；作者希望：

- 尽量复用已经得到的 BKZ basis 和 sieve 结果；
- 构造一个**更大的短向量列表**；
- 且列表中的向量仍然尽量短。

### 5.2 论文为什么不直接用“选不同基向量子集重复 sieve”

作者专门批评了一种看似自然的方法：

- 从 BKZ basis 里换不同的 `κ` 个向量组成子格；
- 每次在这些子格上重新 sieve。

问题是：

- 后面的 basis 向量一般更长；
- 用它们生成的新列表，向量长度会显著变差；
- 与最前面的 `L[0:κ]` 相比，会损失候选质量。

### 5.3 Sieve-Then-Slice 的结构（Algorithm 7）

输入：

- BKZ-reduced lattice
- `κ`
- 目标输出规模 `S > p^(4κ/3)`

核心步骤：

1. 在 `L[0:κ]` 上做 sieve，得到 `Lsieve`；
2. 计算一个上层维数参数 `φ`；
3. 在 `L[κ:κ+φ]` 上做 sieve，得到 `T'`；
4. 将 `T'` lift 回原格得到 `T`；
5. 调用 **Modified Randomized Slicer(T, Lsieve, S)**；
6. 输出更大的短向量列表。

### 5.4 Modified Randomized Slicer 在这里扮演什么角色

这不是 pairwise search，也不是简单线性组合枚举。

它的作用是：

- 以 `Lsieve` 作为“短向量库”；
- 以 `T + L` 为目标集合；
- 通过类似 randomized slicing / approximate Voronoi 的方式，逐步生成大量短向量。

### 5.5 这节你该抓住什么

最重要的思想是：

> 当所需列表大于普通 sieve 输出时，不应靠朴素 pairwise 或重复 approximate-SVP 去硬凑，而应改用能系统地产生“大量仍较短候选”的 slicer 分支。

---

## 6. Section 5.3：Algorithm 8 是整篇论文最重要的总调度

这一节是全篇最关键的实现蓝图。

### 6.1 Algorithm 8 的主逻辑

输入：

- 格 `L`
- 目标 RHF `δ`
- sieving dimension `κ`
- restriction `R`
- related probability function `P`
- 目标成功率 `p`

算法流程：

1. 对 `L` 做 BKZ 到 `δ`；
2. 先计算一个长度阈值 `len`；
3. 用 `P(len)` 算出所需列表大小：
   
   `size = log(1-p) / log(1-P(len))`
4. 若 `size > (sqrt(4/3))^κ`，则走 **Sieve-Then-Slice**；
5. 否则走 **FlexibleD4F**，并调节 `f'` 直到输出列表规模刚好够；
6. 最后在线性扫描中寻找满足 `R(v)=1` 的向量。

### 6.2 这一节最该记住的结论

restricted-SVP 的主线不是：

- 先 BKZ
- 然后再人工想各种候选生成技巧

而是：

- 先把 `P(len)` 和目标成功率转成一个“需要多大的列表”；
- 再由这个列表规模，决定该走哪种短向量列表生成算法。

### 6.3 “Why not combine the two?” 这段很重要

作者还专门解释了：

- 一般不应该把 FlexibleD4F 和 Sieve-Then-Slice 混用；
- 更好的策略是根据所需列表规模择一使用。

原因是：

- 减少 free dimensions 带来的列表增长通常更划算；
- 与增加 slicer 侧列表规模相比，前者在时间/输出规模比上更优；
- 除非 `P(len)` 随长度增长指数级下降，否则没必要混用两种技术。

这段是实现主调度时的关键原则。

---

## 7. Section 6.2：为什么这节对 SIS∞ 最重要

### 7.1 这里给出了 infinity norm 的 `P_B(len)`

如果 restriction 是：

```text
R_B(v) = 1  <=>  ||v||_∞ <= B
```

论文给出的 related probability approximation 是：

```text
P_B(len) ≈ (1 - 2 Φ(-sqrt(d) * B / len))^d
```

其中：

- `Φ` 是标准高斯分布的累积分布函数；
- 这个公式来自“球内均匀分布近似为球对称高斯”的启发式近似。

### 7.2 这节对实现的直接意义

对齐次 SIS∞，你不再应该拍脑袋定：

- `top_k`
- `pair_budget`
- `search base size`

而是应该：

1. 给定 `B = gamma`；
2. 给定长度阈值 `len`；
3. 用 `P_B(len)` 算单个候选通过 restriction 的概率；
4. 再由目标成功率推出需要多大的向量列表。

### 7.3 这一节你该抓住什么

这是把“欧氏球中的短向量列表”和“无穷范数下通过概率”真正连接起来的桥梁。

对当前的 SIS∞ 主线，这一节比 6.1 更优先。

---

## 8. Section 6.1：非齐次问题以后如何接 Kannan embedding

### 8.1 这节做了什么

对 approximate CVP，作者把 Kannan embedding 写成一个 restricted-SVP：

- 构造嵌入格 `L'`；
- restriction 变成：嵌入后向量最后一维必须等于 `±M`；
- 然后再用 restricted-SVP solver 去解。

### 8.2 关键点

作者给了：

- 最后一维等于 `±M` 的通过概率近似；
- embedding 参数 `M` 的一个启发式选择；
- 进一步把 embedding 限制与 infinity norm 限制结合起来。

### 8.3 对你当前项目的意义

这说明非齐次 SIS∞ 的主线应当是：

1. 先做 Kannan embedding；
2. 再在 embedded lattice 上应用 restricted-SVP solver；
3. 而不是单独发明另一套后处理搜索器。

这一步属于第二阶段，不是当前齐次主线的第一优先级。

---

## 9. 这篇论文对当前 SIS∞ 项目的真正启发

### 9.1 应当放弃的主线

不应再把下面这种思路当主线：

- BKZ 后取 reduced rows 前若干条；
- 再做人工 pairwise / small-coeff 组合；
- 再看有没有掉进 infinity norm 约束。

这最多只能当 diagnostic baseline。

### 9.2 应当转向的主线

正确主线应当是：

1. 先把齐次 SIS∞ 写成 restricted-SVP；
2. 用 `P_B(len)` 建模通过限制的概率；
3. 复现 two-step solver；
4. 实现 FlexibleD4F；
5. 实现 Sieve-Then-Slice；
6. 用 Algorithm 8 统一调度；
7. 最后再做非齐次 embedding 分支。

---

## 10. 推荐阅读顺序

建议按下面顺序读：

1. **Section 1 + 1.1**：先把问题意识和贡献抓住
2. **Section 4**：先理解 restricted-SVP 的定义与 `P(len)`
3. **Section 5.3**：先看 Algorithm 8，建立总调度图
4. **Section 5.1**：理解 FlexibleD4F 是什么
5. **Section 5.2**：理解 Sieve-Then-Slice 与 slicer
6. **Section 6.2**：认真看 infinity norm 下的 `P_B(len)`
7. **Section 6.1**：第二阶段再看 embedding
8. **Section 6.3**：最后再看 Dilithium/MISIS 应用

Section 2 不需要一开始精读，把它当工具箱查阅即可：

- GH
- BKZ / RHF / GSA
- projected sublattice
- Babai lift
- sieving

---

## 11. 一句话总结这篇论文该怎么吸收

> 这篇论文最重要的，不是某个单独技巧，而是它把“带额外限制的格问题”统一写成 restricted-SVP，并提出：先用 two-step approximate-SVP 产生短向量列表，再根据 `P(len)` 决定走 FlexibleD4F 还是 Sieve-Then-Slice，最后在列表中筛出满足 restriction 的向量。

这就是后续复现算法时应该严格对齐的主线。
