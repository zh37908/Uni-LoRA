# IGU_UniLoRA

> 说明：本文档描述的是 **IGU-inspired UniLoRA 改写方案**，用于在 UniLoRA 框架中引入 IGU 的打分思想。它**不是** IGU-LoRA 论文算法（奇异值级 top-b 剪枝）的逐字复现。

## 1. 核心想法

IGU-LoRA 的原始做法是：
- 用 **parameter-space Integrated Gradients (IG)** 估计每层的重要性；
- 用 **uncertainty-aware score** 平滑噪声并做稳定的层间 rank allocation。

这里将它改写为 **Uni-LoRA** 视角：

- 不再给第 \(m\) 层直接分配 LoRA rank \(r_m\)；
- 而是给第 \(m\) 层分配两类容量：
  1. **shared latent slots**：来自全局共享子空间；
  2. **residual latent slots**：给该层的额外专属补偿容量。

于是，rank allocation 被改写成：

> **slot allocation in a global latent space**

而不是传统的：

> **rank allocation in independent LoRA blocks**

---

## 2. 参数化

设所有 LoRA 参数拼接后的向量为

\[
\theta_D \in \mathbb{R}^D.
\]

IGU_UniLoRA 将其写为

\[
\theta_D
=
P_{\text{sh}} z_{\text{sh}}
+
\sum_{m=1}^{M} M_m P_m z_m,
\]

其中：

- \(P_{\text{sh}} \in \mathbb{R}^{D \times d_{\text{sh}}}\)：全局共享投影；
- \(z_{\text{sh}} \in \mathbb{R}^{d_{\text{sh}}}\)：共享 latent 向量；
- \(P_m \in \mathbb{R}^{D_m \times d_m^{\text{res}}}\)：第 \(m\) 层 residual 投影；
- \(z_m \in \mathbb{R}^{d_m^{\text{res}}}\)：第 \(m\) 层 residual latent；
- \(M_m\)：把第 \(m\) 层 residual 参数放回全局参数位置的掩码/嵌入算子；
- \(d_{\text{sh}}\)：共享 slots 数；
- \(d_m^{\text{res}}\)：第 \(m\) 层 residual slots 数。

总预算为

\[
B = d_{\text{sh}} + \sum_{m=1}^{M} d_m^{\text{res}}.
\]

其中：
- **shared part** 负责捕获跨层共享更新；
- **residual part** 负责补偿少数高重要层的额外创新方向。

---

## 3. IGU 打分改写为 slot 分配

### 3.1 层重要性估计

对第 \(m\) 层，记其 LoRA 参数为 \(\theta_m\)。

参考 IGU-LoRA，用参数空间 IG 定义层内参数重要性：

\[
\mathrm{IG}_{m,i}
=
(\theta_{m,i}-\theta_{m,i}^{0})
\int_{0}^{1}
\frac{\partial \mathcal{L}(\theta^0 + \alpha(\theta-\theta^0))}{\partial \theta_{m,i}}
\, d\alpha.
\]

实际训练中，用小批量随机采样路径点近似该积分，并在层内聚合：

\[
g_m^{\text{raw}} = \operatorname{Agg}\big(|\mathrm{IG}_{m,i}|\big).
\]

其中 \(\operatorname{Agg}(\cdot)\) 可以取 mean / sum / top-k mean。

---

### 3.2 uncertainty-aware 稳定化

为降低 noisy mini-batch 的影响，对每层维护：

- EMA 均值 \(\mu_m\)
- EMA 偏差 \(\sigma_m\)

更新形式可写为：

\[
\mu_m \leftarrow \beta \mu_m + (1-\beta) g_m^{\text{raw}},
\]

\[
\sigma_m \leftarrow \beta \sigma_m + (1-\beta)|g_m^{\text{raw}}-\mu_m|.
\]

定义最终层分数：

\[
s_m = \frac{\mu_m}{\sigma_m + \varepsilon}.
\]

解释：
- \(\mu_m\) 大：该层长期贡献更大；
- \(\sigma_m\) 大：该层打分更不稳定；
- \(s_m\) 大：该层“高贡献且稳定”，更值得给 residual slots。

---

## 4. shared / residual slots 分配

### 4.1 shared slots

shared slots 不按层拆开，而是全局共享：

\[
d_{\text{sh}} = \rho B,
\qquad 0 < \rho < 1.
\]

其中 \(\rho\) 是共享比例。

它对应 Uni-LoRA 的主干子空间，要求：
- **global**：跨层统一共享；
- **load-balanced**：每个 shared slot 承担近似均匀负载；
- **isometric**：列归一化后近似保持子空间内几何。

---

### 4.2 residual slots

剩余预算

\[
B_{\text{res}} = B - d_{\text{sh}}
\]

按层分配给 residual bank：

\[
d_m^{\text{res}}
=
\left\lfloor
B_{\text{res}}
\cdot
\frac{(s_m+\tau)^\gamma}{\sum_{j=1}^{M}(s_j+\tau)^\gamma}
\right\rfloor.
\]

其中：
- \(\tau > 0\)：平滑项，防止低分层完全归零；
- \(\gamma \ge 1\)：温度/尖锐度，越大越偏向高分层。

为了避免容量塌缩，可再设最小 residual 配额：

\[
d_m^{\text{res}} \leftarrow \max(d_{\min}, d_m^{\text{res}}).
\]

若总和超过预算，再从最低分层回收多余 slots。

---

## 5. 投影矩阵构造

### 5.1 shared projection

构造全局 one-hot random projection：

\[
P_{\text{sh}} \in \mathbb{R}^{D \times d_{\text{sh}}}.
\]

做法：
1. 对每个 full parameter index \(i \in \{1,\dots,D\}\)，均匀采样一个 slot \(h(i)\in\{1,\dots,d_{\text{sh}}\}\)；
2. 令第 \(i\) 行只有一个非零元素，落在第 \(h(i)\) 列；
3. 对每一列做归一化。

这样得到全局共享、近似 load-balanced、近似等距的 shared core。

---

### 5.2 residual projection

对第 \(m\) 层，再单独生成一个局部 residual projection：

\[
P_m \in \mathbb{R}^{D_m \times d_m^{\text{res}}}.
\]

它只服务于该层，不要求全局共享，但规模应远小于 shared core。

因此整体结构是：
- **大头容量** 放在全局 shared core；
- **小头补偿** 放在层特异 residual bank。

---

## 6. 训练流程

### 阶段 A：warmup scoring

先用少量训练步估计每层 IGU score：
- 计算参数空间 IG 的随机近似；
- 聚合成 \(g_m^{\text{raw}}\)；
- 用 EMA 得到 \(s_m\)。

此时只做打分，不频繁改变结构。

### 阶段 B：slot allocation

根据 \(s_m\) 分配：
- 固定 shared slots 数 \(d_{\text{sh}}\)；
- 计算各层 residual slots \(d_m^{\text{res}}\)。

然后初始化：
- \(P_{\text{sh}}, P_m\)
- \(z_{\text{sh}}, z_m\)

### 阶段 C：main training

固定投影结构，只训练 latent variables：

\[
\{z_{\text{sh}}, z_1,\dots,z_M\}.
\]

必要时每隔 \(T\) 步重新估计一次 \(s_m\)，做一次轻量 reallocation。

---

## 7. 简洁伪代码

```text
Algorithm: IGU_UniLoRA
Input:
    pretrained model W0
    total budget B
    shared ratio rho
    smoothing beta
    temperature gamma
    stability eps
    warmup steps Tw

1. Insert LoRA targets and flatten their trainable parameter space to size D.
2. Reserve shared budget:
       d_sh = floor(rho * B)
       B_res = B - d_sh
3. Warmup scoring for t = 1 ... Tw:
       sample minibatch
       sample one path point alpha ~ Uniform(0,1)
       compute stochastic parameter-space IG estimate
       aggregate per-layer raw score g_m^raw
       update EMA mean:
           mu_m <- beta * mu_m + (1-beta) * g_m^raw
       update EMA deviation:
           sigma_m <- beta * sigma_m + (1-beta) * |g_m^raw - mu_m|
       compute score:
           s_m = mu_m / (sigma_m + eps)
4. Allocate residual slots:
       d_m^res = floor(B_res * (s_m + tau)^gamma / sum_j (s_j + tau)^gamma)
       enforce minimum quota and budget correction
5. Build projections:
       build global shared projection P_sh in R^{D x d_sh}
       for each layer m:
           build local residual projection P_m in R^{D_m x d_m^res}
6. Initialize trainable latents:
       z_sh, {z_m}
7. Train:
       theta_D = P_sh z_sh + sum_m M_m P_m z_m
       optimize only z_sh and {z_m}
8. Output adapted model.
```

---

## 8. 为什么这个改写合理

IGU-LoRA 的核心思想在这里以“同思想、不同对象”的形式保留：
- 仍然用 **IG** 捕获“路径上的长期贡献”；
- 仍然用 **uncertainty-aware score** 压低 noisy / unstable 层的权重。

改变的是分配对象与机制：
- 原始 IGU-LoRA：给每层分 rank；
- IGU_UniLoRA：给每层分 **residual capacity**，而把大部分公共结构交给 **shared global latent space**。

因此它更适合 Uni-LoRA 的理论视角，但应与原论文严格区分：

> 大多数层共享同一个低维主干，
> 少数真正重要且稳定的层，再额外获得 residual slots 作为补偿。

补充说明（避免归因混淆）：
- 原论文主线是奇异值级评分 `S_i` 与 top-b 裁剪；
- 本文档方案是模块/层级评分后做 shared/residual 容量分配；
- 若需要“论文复现”，应另行实现其 SVD + top-b 流程。

---

## 9. 最简实现建议

若想先做一个最容易跑的版本，可以直接采用：

- 固定 \(d_{\text{sh}} = 0.7B\)
- 用 5%~10% 训练步做 warmup scoring
- 只在 warmup 结束后分配一次 \(d_m^{\text{res}}\)
- 后续不再动态改结构

即：

\[
\theta_D = P_{\text{sh}} z_{\text{sh}} + \sum_m M_m P_m z_m,
\qquad
d_m^{\text{res}} \propto (s_m+\tau)^\gamma.
\]

这是最干净、最稳、最容易和 Uni-LoRA 做对比实验的版本。

---

## 10. 一句话总结

**IGU_UniLoRA = 用 IGU 的稳定重要性打分，决定每层应获得多少 residual latent slots；同时保留 Uni-LoRA 的全局 shared latent core 来承载跨层共享更新。**
