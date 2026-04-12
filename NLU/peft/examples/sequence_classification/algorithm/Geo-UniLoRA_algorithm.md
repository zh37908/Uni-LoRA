# Geo-UniLoRA：基于“共享流形 + 创新流形”的几何自适应 Uni-LoRA

> **定位**：这是一个**新提出的算法草案**，不是现有论文中的标准算法名。  
> 它把 **GeLoRA 的几何 rank allocation 思想** 与 **Uni-LoRA 的全局随机投影共享子空间** 统一起来：  
> 不再把 rank allocation 理解为“给每一层单独分几个 LoRA rank”，而是改写为  
> **“给共享流形和各模块创新流形分配多少 latent capacity”**。

---

## 1. 核心目标

传统 LoRA / AdaLoRA / GeLoRA 的 rank allocation，通常是在每个层或每个模块内部决定低秩矩阵的 rank。  
而 Uni-LoRA 的统一视角是：把所有 LoRA 参数拼接成一个大向量，然后通过一个低维 latent 向量经过投影矩阵来生成：

\[
\theta_D = P \theta_d, \qquad d \ll D.
\]

Geo-UniLoRA 的核心思想是：

1. **共享流形（shared manifold）**：  
   对一组表示几何相似的模块，分配一部分**共享 latent 维度**，用来建模跨层/跨模块共享的更新方向。

2. **创新流形（innovation manifold）**：  
   对每个模块，再分配一部分**模块私有 latent 维度**，用来建模它偏离共享子空间的新增方向。

3. **几何分配原则**：  
   共享维度由“模块之间表示有多相似”决定；  
   创新维度由“该模块相对共享结构还需要多少额外几何自由度”决定。

因此，Geo-UniLoRA 不是简单的 non-uniform random projection，而是：

- 保留 Uni-LoRA 的 **共享、正交、随机投影** 主体；
- 在其上叠加一个 **几何驱动的 shared / residual budget allocation**。

---

## 2. 记号与参数化

设共有 \(M\) 个待插入 LoRA 的目标模块（如 attention 的 \(q,k,v,o\) 或 MLP 的 up/down/gate 等），编号为 \(m=1,\dots,M\)。

对第 \(m\) 个模块，设其标准 LoRA 参数模板为：

\[
\Delta W_m = B_m A_m,
\]

其中

- \(A_m \in \mathbb{R}^{r_0 \times d_m^{in}}\)
- \(B_m \in \mathbb{R}^{d_m^{out} \times r_0}\)

这里 \(r_0\) 是一个**模板 rank**，它不表示真实可训练自由度，而只是定义该模块 LoRA 参数向量的展开维度：

\[
D_m = r_0(d_m^{in} + d_m^{out}).
\]

把模块 \(m\) 的 LoRA 参数展平成向量：

\[
\theta_m \in \mathbb{R}^{D_m},
\]

所有模块拼接成总参数向量：

\[
\theta_D = [\theta_1; \theta_2; \dots; \theta_M] \in \mathbb{R}^{D}, \qquad D=\sum_{m=1}^M D_m.
\]

---

## 3. Geo-UniLoRA 的层次化参数化

### 3.1 分组

先把模块按表示几何相似性聚成 \(G\) 个组：

\[
\mathcal{G}_1,\mathcal{G}_2,\dots,\mathcal{G}_G,
\qquad
\bigcup_{g=1}^G \mathcal{G}_g = \{1,\dots,M\}.
\]

对于组 \(g\)，其总展开维度为

\[
D_g = \sum_{m \in \mathcal{G}_g} D_m.
\]

### 3.2 shared + innovation 参数化

Geo-UniLoRA 的默认形式为：

\[
\theta_D
=
\sum_{g=1}^{G} M_g P_g^{\text{sh}} z_g^{\text{sh}}
+
\sum_{m=1}^{M} M_m P_m^{\text{in}} z_m^{\text{in}}.
\]

其中：

- \(z_g^{\text{sh}} \in \mathbb{R}^{c_g}\)：第 \(g\) 组的**共享 latent 向量**
- \(P_g^{\text{sh}} \in \mathbb{R}^{D_g \times c_g}\)：第 \(g\) 组的**共享投影矩阵**
- \(z_m^{\text{in}} \in \mathbb{R}^{r_m}\)：模块 \(m\) 的**创新 latent 向量**
- \(P_m^{\text{in}} \in \mathbb{R}^{D_m \times r_m}\)：模块 \(m\) 的**创新投影矩阵**
- \(M_g\)：把组内向量散射回全局参数向量 \(\theta_D\) 的插入算子
- \(M_m\)：把模块 \(m\) 的向量插入到全局参数向量的插入算子
- \(c_g\)：组共享维度
- \(r_m\)：模块创新维度

总 latent 预算为

\[
d = \sum_{g=1}^{G} c_g + \sum_{m=1}^{M} r_m.
\]

---

## 4. 为什么这种结构适合 Uni-LoRA

这个结构的思想是：

- **shared manifold** 捕获  
  “多个模块共同需要的更新方向”
- **innovation manifold** 捕获  
  “某个模块偏离共享模式的新增方向”

如果直接对整个模型做一个 flat 的 non-uniform random projection，会破坏 Uni-LoRA 所强调的 uniform / isometric 结构。  
而 Geo-UniLoRA 采用的是：

- 组内共享空间：保留共享性
- 模块创新空间：只在必要的地方补充自由度

所以这是一个 **hierarchical shared projection**，而不是“任意偏置某些层的投影”。

---

## 5. 几何统计量的估计

Geo-UniLoRA 需要两个核心统计量：

1. 每个模块的几何复杂度（intrinsic dimensionality）
2. 模块之间的几何相似度（representation similarity）

---

## 6. 模块几何复杂度 \(\hat d_m\)

对每个模块 \(m\)，在一个 calibration set 上收集：

- 输入隐表示 \(H_m^{in} \in \mathbb{R}^{N \times d_m^{in}}\)
- 输出隐表示 \(H_m^{out} \in \mathbb{R}^{N \times d_m^{out}}\)

其中 \(N\) 是采样 token / sequence 的总数。

### 6.1 协方差谱

定义输入/输出协方差：

\[
\Sigma_m^{in} = \frac{1}{N}(H_m^{in})^\top H_m^{in}, \qquad
\Sigma_m^{out} = \frac{1}{N}(H_m^{out})^\top H_m^{out}.
\]

令它们的特征值分别为 \(\lambda_i^{in}\)、\(\lambda_i^{out}\)。

### 6.2 intrinsic dimension 估计

可以选择以下任一稳定估计器：

#### 方案 A：effective rank
\[
\mathrm{erank}(\Sigma) = \exp\!\left(-\sum_i p_i \log p_i\right),
\qquad
p_i = \frac{\lambda_i}{\sum_j \lambda_j}.
\]

#### 方案 B：participation ratio
\[
\mathrm{prank}(\Sigma)
=
\frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}.
\]

#### 方案 C：TwoNN / MLE intrinsic dimension
适合更严格的流形估计，但计算更慢。

### 6.3 模块需求维度

定义模块 \(m\) 的几何需求为：

\[
\hat d_m
=
\left\lceil
\gamma \cdot
\max\big(
\widehat{ID}(H_m^{in}),
\widehat{ID}(H_m^{out})
\big)
\right\rceil,
\]

其中 \(\gamma \in (0,1]\) 是缩放系数，用来把表示几何复杂度映射到可分配的 latent 维度。

> 实践上，\(\widehat{ID}\) 推荐先用 participation ratio 或 effective rank。  
> 如果你特别强调 GeLoRA 式“流形维度”解释，可以换成 TwoNN/MLE 版本。

---

## 7. 模块间几何相似性 \(S_{mn}\)

Geo-UniLoRA 需要知道哪些模块应该共享子空间。  
在 calibration set 上，对每对模块 \(m,n\) 计算相似性：

### 7.1 线性 CKA
\[
S_{mn} = \mathrm{CKA}(H_m, H_n),
\]

其中 \(H_m\) 可取输入隐表示、输出隐表示，或二者拼接。

### 7.2 余弦相似 / 协方差相似
也可以定义为：

\[
S_{mn}
=
\frac{\langle \mathrm{vec}(\Sigma_m), \mathrm{vec}(\Sigma_n) \rangle}
{\|\Sigma_m\|_F \|\Sigma_n\|_F}.
\]

得到相似矩阵 \(S \in \mathbb{R}^{M \times M}\) 后，用谱聚类 / 层次聚类 / KMeans 对模块分组，得到 \(\mathcal{G}_1,\dots,\mathcal{G}_G\)。

---

## 8. 共享维度 \(c_g\) 的估计

Geo-UniLoRA 的关键，不是简单把组内所有模块都分到同样 rank，而是估计：

> 这组模块之间到底有多少“可共享的公共维度”？

对组 \(g\)，定义其**共享维度估计**：

\[
c_g
=
\left\lfloor
\alpha
\cdot
\frac{
\sum_{m,n \in \mathcal{G}_g}
S_{mn}\,\min(\hat d_m,\hat d_n)
}{
\sum_{m,n \in \mathcal{G}_g} S_{mn} + \varepsilon
}
\right\rfloor.
\]

其中：

- \(S_{mn}\) 越大，说明模块 \(m,n\) 越相似
- \(\min(\hat d_m,\hat d_n)\) 表示它们可共享的重叠维度不会超过较小者
- \(\alpha \in (0,1]\) 是共享压缩系数
- \(\varepsilon\) 是数值稳定项

然后做裁剪：

\[
c_g \leftarrow \min\big(c_g,\ \min_{m\in \mathcal{G}_g} \hat d_m\big).
\]

这保证共享维度不超过组内最小需求。

---

## 9. 创新维度 \(r_m\) 的估计

对模块 \(m\)，设其所属组为 \(g(m)\)。  
定义其**创新需求**：

\[
u_m = \max(\hat d_m - c_{g(m)},\ 0).
\]

这表示：

- 共享空间已经覆盖了 \(c_{g(m)}\) 个公共方向
- 还剩下 \(u_m\) 个方向只能由该模块自己负责

给定残差预算 \(B_{res}\)，用 soft allocation 分给各模块：

\[
r_m
=
r_{\min}
+
\left\lfloor
B_{res}
\cdot
\frac{(u_m+\varepsilon)^\tau}
{\sum_{j=1}^{M}(u_j+\varepsilon)^\tau}
\right\rfloor.
\]

其中：

- \(r_{\min}\) 是每个模块保底创新维度
- \(\tau > 0\) 是温度系数  
  - \(\tau>1\)：更偏向高需求模块  
  - \(\tau<1\)：更平滑
- \(B_{res}\) 是 innovation manifold 的总预算

最后通过余数修正，使得

\[
\sum_{m=1}^{M} r_m = B_{res} + M r_{\min}.
\]

---

## 10. 总预算的设定

总 latent 预算写为：

\[
d = B_{sh} + B_{res},
\qquad
B_{sh} = \sum_{g=1}^{G} c_g.
\]

如果用户给定总预算 \(d_{tot}\)，则可以反过来用以下规则：

1. 先根据几何估计得到原始 \(\tilde c_g\)
2. 令
\[
\lambda = \frac{d_{tot}}{\sum_g \tilde c_g + \sum_m \tilde r_m}
\]
3. 再统一缩放：
\[
c_g = \max(1,\lfloor \lambda \tilde c_g \rfloor), \qquad
r_m = \max(r_{\min},\lfloor \lambda \tilde r_m \rfloor).
\]

最后做一次 budget correction，使总维度恰好等于 \(d_{tot}\)。

---

## 11. 投影矩阵的构造

### 11.1 组共享投影

对每个组 \(g\)，随机采样高斯矩阵：

\[
G_g \sim \mathcal{N}(0,1)^{D_g \times c_g},
\]

再做 QR 分解：

\[
P_g^{sh} = \mathrm{qr}(G_g),
\qquad
(P_g^{sh})^\top P_g^{sh} = I_{c_g}.
\]

### 11.2 模块创新投影

对每个模块 \(m\)，随机采样：

\[
H_m \sim \mathcal{N}(0,1)^{D_m \times r_m}.
\]

若希望 innovation 空间尽量不与 shared 空间重叠，则先做一次正交化：

\[
\widetilde H_m
=
H_m
-
P_{g(m)\to m}^{sh}
\big(P_{g(m)\to m}^{sh}\big)^\top
H_m,
\]

其中 \(P_{g(m)\to m}^{sh}\) 是组共享投影在模块 \(m\) 对应参数块上的限制矩阵。  
再做 QR 分解：

\[
P_m^{in} = \mathrm{qr}(\widetilde H_m),
\qquad
(P_m^{in})^\top P_m^{in} = I_{r_m}.
\]

这样可以近似保证：

\[
\big(P_{g(m)\to m}^{sh}\big)^\top P_m^{in} \approx 0.
\]

---

## 12. 从 latent 向量恢复 LoRA 参数

### 12.1 组共享部分
先得到组 \(g\) 的共享参数向量：

\[
\theta_g^{sh} = P_g^{sh} z_g^{sh} \in \mathbb{R}^{D_g}.
\]

再把它切分回组内各模块：

\[
\theta_{m,sh} = \mathrm{slice}_m(\theta_g^{sh}).
\]

### 12.2 模块创新部分
\[
\theta_{m,in} = P_m^{in} z_m^{in}.
\]

### 12.3 合成模块参数
\[
\theta_m = \theta_{m,sh} + \theta_{m,in}.
\]

再把 \(\theta_m\) reshape 回

- \(A_m \in \mathbb{R}^{r_0 \times d_m^{in}}\)
- \(B_m \in \mathbb{R}^{d_m^{out} \times r_0}\)

从而得到

\[
\Delta W_m = B_m A_m.
\]

---

## 13. 训练目标

Geo-UniLoRA 的训练变量只有：

\[
\{z_g^{sh}\}_{g=1}^G
\quad\text{和}\quad
\{z_m^{in}\}_{m=1}^M.
\]

所有投影矩阵 \(P_g^{sh}, P_m^{in}\) 固定不训练。

训练目标为常规下游任务损失：

\[
\mathcal{L}_{task}(\Theta)
\]

可选地加入 innovation 抑制正则：

\[
\mathcal{L}
=
\mathcal{L}_{task}
+
\lambda_{in}\sum_{m=1}^{M}\|z_m^{in}\|_2^2.
\]

这个正则鼓励模型优先利用共享流形，仅在必要时使用创新流形。

---

## 14. 完整算法流程

### 阶段 A：几何统计与预算分配
1. 选定目标模块集合 \(\{1,\dots,M\}\)
2. 在 calibration set 上收集每个模块的隐表示
3. 估计每个模块的 intrinsic dimension \(\hat d_m\)
4. 计算模块相似矩阵 \(S_{mn}\)
5. 对模块聚类，得到组 \(\mathcal{G}_1,\dots,\mathcal{G}_G\)
6. 用相似加权重叠公式计算每个组的共享维度 \(c_g\)
7. 用 \(u_m=\max(\hat d_m-c_{g(m)},0)\) 计算创新需求
8. 按总预算分配 innovation 维度 \(r_m\)

### 阶段 B：投影初始化
9. 为每个组采样共享投影 \(P_g^{sh}\)
10. 为每个模块采样创新投影 \(P_m^{in}\)
11. 初始化所有 latent 向量为零或小高斯噪声

### 阶段 C：训练
12. 每次前向时，从 latent 向量恢复全部 LoRA 参数
13. 注入到各模块中，计算任务损失
14. 反向传播，只更新 \(z_g^{sh}, z_m^{in}\)
15. 训练结束后导出对应的 LoRA 参数

---

## 15. 伪代码 1：预算分配

```python
# Algorithm 1: Geo-UniLoRA Budget Allocation

Inputs:
    target_modules = {1, ..., M}
    calibration_data
    total_budget d_tot
    grouping_method
    ID_estimator
    alpha          # shared compression factor
    tau            # innovation allocation temperature
    r_min          # minimum innovation per module
    eps

Outputs:
    groups G_1, ..., G_G
    shared dimensions {c_g}
    innovation dimensions {r_m}

# Step 1: collect activations
for each module m in target_modules:
    H_in[m], H_out[m] = collect_hidden_states(module=m, data=calibration_data)

# Step 2: estimate module geometric complexity
for each module m:
    d_in  = ID_estimator(H_in[m])
    d_out = ID_estimator(H_out[m])
    d_hat[m] = ceil(gamma * max(d_in, d_out))

# Step 3: compute similarity matrix
for each pair (m, n):
    S[m, n] = similarity(H_out[m], H_out[n])   # e.g., linear CKA

# Step 4: cluster modules into groups
groups = cluster_modules(S, method=grouping_method)

# Step 5: estimate shared dimensions per group
for each group g in groups:
    numerator   = 0.0
    denominator = 0.0
    for m in g:
        for n in g:
            numerator   += S[m, n] * min(d_hat[m], d_hat[n])
            denominator += S[m, n]
    c[g] = floor(alpha * numerator / (denominator + eps))
    c[g] = min(c[g], min(d_hat[m] for m in g))
    c[g] = max(1, c[g])

# Step 6: innovation demands
for each module m:
    g = group_id_of(m)
    u[m] = max(d_hat[m] - c[g], 0)

# Step 7: allocate innovation budget
B_sh_raw  = sum(c[g] for g in groups)
B_res_raw = sum((u[m] + eps)**tau for m in target_modules)

# If total budget is fixed, rescale shared and innovation parts together
raw_total = B_sh_raw + sum(r_min for _ in target_modules) + B_res_raw
lambda_ = d_tot / raw_total

for each group g:
    c[g] = max(1, floor(lambda_ * c[g]))

remaining = d_tot - sum(c[g] for g in groups) - M * r_min
weights = [(u[m] + eps)**tau for m in target_modules]
Z = sum(weights)

for each module m:
    r[m] = r_min + floor(max(0, remaining) * weights[m] / (Z + eps))

# Step 8: budget correction
adjust {r[m]} so that sum(c[g]) + sum(r[m]) = d_tot

return groups, c, r
```

---

## 16. 伪代码 2：投影初始化

```python
# Algorithm 2: Projection Initialization

Inputs:
    groups, {c_g}, {r_m}
    module_shapes = {D_m}
Outputs:
    {P_g_sh}, {P_m_in}

for each group g:
    D_g = sum(D_m for m in g)
    G = randn(D_g, c_g)
    P_g_sh = orthonormalize_columns(G)   # QR / Householder

for each module m:
    g = group_id_of(m)
    H = randn(D_m, r_m)

    # optional: orthogonalize innovation against shared block
    P_block = restrict_group_projection_to_module(P_g_sh, module=m)
    H = H - P_block @ (P_block.T @ H)

    P_m_in = orthonormalize_columns(H)

return {P_g_sh}, {P_m_in}
```

---

## 17. 伪代码 3：训练循环

```python
# Algorithm 3: Geo-UniLoRA Training

Inputs:
    frozen_backbone
    {P_g_sh}, {P_m_in}
    groups
    train_loader
    learning_rate
    lambda_in
Outputs:
    trained latent vectors {z_g_sh}, {z_m_in}

initialize z_g_sh = 0 for each group g
initialize z_m_in = 0 for each module m

for each training step:
    # reconstruct LoRA parameters
    for each group g:
        theta_g_sh = P_g_sh @ z_g_sh

    for each module m:
        g = group_id_of(m)
        theta_m_sh = slice_from_group(theta_g_sh, module=m)
        theta_m_in = P_m_in @ z_m_in
        theta_m = theta_m_sh + theta_m_in

        A_m, B_m = reshape_to_lora(theta_m)
        inject_lora(module=m, A=A_m, B=B_m)

    # forward / backward
    loss_task = task_loss(frozen_backbone, batch)

    reg = 0.0
    for each module m:
        reg += norm(z_m_in, 2)**2

    loss = loss_task + lambda_in * reg
    loss.backward()

    update_only_latent_vectors({z_g_sh}, {z_m_in})
    zero_grad()

return {z_g_sh}, {z_m_in}
```

---

## 18. 一个更简洁的“全局版”Geo-UniLoRA

如果你不想分组，可以取 \(G=1\)，即整个模型只用一个共享流形：

\[
\theta_D = P^{sh} z^{sh} + \sum_{m=1}^{M} M_m P_m^{in} z_m^{in}.
\]

这时：

- \(P^{sh} \in \mathbb{R}^{D \times c}\)
- \(z^{sh} \in \mathbb{R}^{c}\)

所有模块共享同一个 global manifold。  
该版本最接近 Uni-LoRA 原始“全局共享子空间”的叙事；  
grouped 版本则更灵活，更接近“shared manifold + innovation manifold”的精细实现。

---

## 19. 推荐超参数

### 19.1 几何估计
- ID estimator：`participation ratio` 或 `effective rank`
- similarity：`linear CKA`
- 聚类数 \(G\)：可设为层数的 \(1/4 \sim 1/8\)，或通过谱间隙自动选择

### 19.2 预算
- \(\alpha = 0.5 \sim 0.9\)
- \(r_{\min}=1\) 或 \(2\)
- \(\tau = 1.0 \sim 2.0\)
- 总预算 \(d_{tot}\)：与 VeRA / VB-LoRA / Uni-LoRA 同量级做公平比较

### 19.3 正则
- \(\lambda_{in}\) 可取 \(10^{-5} \sim 10^{-3}\)

---

## 20. 理论直觉

Geo-UniLoRA 的理论叙事可以总结成一句话：

> **如果多个模块的最优更新方向来源于同一个共享表示流形，那么它们应该共享 latent 维度；  
> 如果某个模块拥有超出共享流形的新增几何复杂度，那么只给它补充少量创新维度即可。**

因此，Geo-UniLoRA 做的是：

- 用 shared manifold 降低参数方差
- 用 innovation manifold 控制投影偏差
- 用 GeLoRA 式几何统计量来决定两者的预算分配

---

## 21. 与标准 GeLoRA / Uni-LoRA 的区别

### 相对 GeLoRA
GeLoRA 分的是“每层 LoRA rank”；  
Geo-UniLoRA 分的是“共享维度 + 创新维度”。

### 相对 Uni-LoRA
Uni-LoRA 通常用一个统一共享子空间；  
Geo-UniLoRA 则在其上增加一个几何驱动的 residual manifold。

### 关键新点
Geo-UniLoRA 的真正创新不在于“又一个 adaptive rank method”，而在于：

\[
\textbf{把 rank allocation 重新解释为 projection-space capacity allocation。}
\]

---

## 22. 可选增强版

### 22.1 动态重分配
训练若干 step 后，重新估计：

- 模块 latent 梯度范数
- innovation latent 的利用率
- 模块表示相似性是否变化

再对 \(r_m\) 做一次小规模 reallocation。

### 22.2 组内共享 + 全局共享
在上面的 grouped shared manifold 之上，再加一个全局共享核：

\[
\theta_D
=
P^{global} z^{global}
+
\sum_g M_g P_g^{sh} z_g^{sh}
+
\sum_m M_m P_m^{in} z_m^{in}.
\]

这会形成三级结构：

1. 全局共享
2. 组内共享
3. 模块创新

### 22.3 创新空间的稀疏门控
对 \(z_m^{in}\) 再施加 L1 或 group sparsity，使模型只在确实需要时使用创新空间。

---

## 23. 最小实现版本

如果你想先做一个最容易跑通的版本，建议按下面配置：

1. 目标模块：所有 attention 的 \(q,v\)  
2. ID estimator：participation ratio  
3. similarity：linear CKA  
4. grouping：按层相邻模块做 4~8 个组  
5. shared manifold：group-wise  
6. innovation manifold：每模块保底 1 维  
7. \(P\)：固定随机正交，不训练  
8. 只训练 \(z_g^{sh}, z_m^{in}\)

这个版本最容易验证：

- 是否比纯 Uni-LoRA 更强
- 是否比 GeLoRA 风格 layerwise rank 更省参数
- 是否真出现“共享维度负责共性，创新维度负责个性”的分工

---

## 24. 一句话总结

Geo-UniLoRA 可以写成：

\[
\boxed{
\theta_D
=
\underbrace{\sum_g M_g P_g^{sh} z_g^{sh}}_{\text{共享流形}}
+
\underbrace{\sum_m M_m P_m^{in} z_m^{in}}_{\text{创新流形}}
}
\]

其中：

- 共享流形的维度 \(c_g\) 由**组内几何相似性**决定
- 创新流形的维度 \(r_m\) 由**模块剩余几何需求**决定
- 所有投影矩阵均为固定随机正交矩阵
- 最终只训练低维 latent 向量

这就是“GeLoRA + Uni-LoRA = shared manifold + innovation manifold”的一个完整可实现版本。
