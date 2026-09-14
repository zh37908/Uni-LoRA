# ICLR 2027 论文修改清单：按优先级整理

> 论文：**Why Compression Makes LoRA Better: A Bias–Variance Theory of Compressed Adaptation and When It Fails**
>
> 目标：根据当前稿件的理论、实验设计和叙事结构，整理投稿前需要修改的关键问题。
>
> - **P0：必须修改**。如果不改，可能被审稿人视为理论错误、逻辑矛盾或 claim 过强。
> - **P1：强烈建议修改**。决定论文能否从 borderline / weak reject 提升到 weak accept / accept。
> - **P2：增强项**。主要提高可信度、表达严谨性和实验完整性。

---

# P0：必须修改

## P0-1. 修正 Proposition 2 中 random-support corollary 的数学矛盾

### 位置
- Section 4.1：Sparse Residual Correction under Matched Budgets
- Appendix A.3：Proof of the Matched-Budget Condition (Proposition 2)
- Section 5.6：Ablation Study 中对 random support 的理论解释

### 当前问题
Appendix A.3 先证明：
\[
\mathbb{E}\|(I-\Pi_S)\theta^\star\|_2^2=(D-B)v,
\]
因此对于 isotropic prior，只要子空间 \(S\) 的维度为 \(B\)，且与 \(\theta^\star\) 和 observation noise 独立，其 expected approximation error 只依赖维度 \(B\)。

这意味着在 matched-budget 条件下，如果 random sparse support 与目标独立，而且
\[
\dim \operatorname{col}([P,E_I])=B=d+K,
\]
那么 random hybrid 与 \(B\)-dimensional pure compression 的 expected risk 应当相同。

但当前 Appendix A.3 随后又写：
\[
\mathbb{E}[E_K]
=
\left(1-\frac dD\right)Kv
<
Kv,
\]
并据此声称 random support “slightly harmful”。这与前面的 isotropic-subspace 结论冲突。

### 为什么这是高风险问题
这是内部数学不一致，而不是单纯措辞问题。审稿人很可能直接质疑 Proposition 2 与其 corollary 是否可以同时成立。

### 建议修改
保留 Proposition 2：
\[
\mathbb{E}[R(\hat\theta_{\rm hyb})]
-
\mathbb{E}[R(\hat\theta_{U,B})]
=
\frac12
\left(
Kv-\mathbb{E}[E_K]
\right).
\]

但将 random-support 结论改为：
\[
\boxed{\mathbb{E}[E_K]=Kv}
\]
从而得到：
\[
\boxed{
\mathbb{E}[R(\hat\theta_{\rm random})]
=
\mathbb{E}[R(\hat\theta_{U,B})]
}
\]

即：
- informed support：若 \(\mathbb{E}[E_K]>Kv\)，hybrid 胜；
- random / uninformed support：若 \(\mathbb{E}[E_K]=Kv\)，期望上打平；
- systematically poor support：若 \(\mathbb{E}[E_K]<Kv\)，hybrid 更差。

### 建议替换文本
> **Corollary: uninformed supports provide no expected gain under isotropy.**  
> Under the isotropic target model, any \(B\)-dimensional subspace selected independently of \(\theta^\star\) has the same expected approximation error. Therefore, if the random support \(I\) is selected independently of the target and noise and \(\operatorname{rank}([P,E_I])=B\), then
> \[
> \mathbb{E}[E_K]=Kv,
> \]
> and the random hybrid has the same expected risk as a \(B\)-dimensional compressed-only estimator. Hence, the benefit of the hybrid parameterization must come from **data-dependent or target-informed support selection**, rather than from sparsity itself.

---

## P0-2. 删除 “random support worse exactly as Proposition 2 predicts”

### 位置
Section 5.6：Ablation Study

### 当前问题
按照修正后的 Proposition 2，在严格 isotropic matched-budget model 中，random support 应该与 pure compression 期望上打平，而不是系统性更差。

### 建议修改
将原来的 “exactly as Proposition 2 predicts” 改成：

> Under the idealized isotropic matched-budget model, random support is expected to provide no systematic gain over pure compression. In the real network, random support performs slightly worse, which may reflect anisotropic curvature, optimization effects, finite-sample variability, or conditioning of the augmented parameterization. In contrast, warmup-SNIP support consistently improves over random routing, supporting the central prediction that **informed residual allocation**, rather than sparsity alone, is responsible for the gain.

---

## P0-3. 处理 LoRA factor-space 的 non-identifiability

### 位置
- Section 3.1：定义 \(\theta_D=\operatorname{vec}(\{A_\ell,B_\ell\})\)
- Section 3.2：定义 \(v=D^{-1}\mathbb{E}\|\theta^\star\|_2^2\)
- Section 5.4：Transformer teacher–student experiment

### 当前问题
LoRA 更新满足：
\[
\Delta W=BA.
\]
但存在 gauge freedom：
\[
A\rightarrow C^{-1}A,\qquad B\rightarrow BC.
\]
尤其：
\[
A\rightarrow cA,\qquad B\rightarrow B/c,
\]
不会改变功能更新 \(\Delta W\)，却会改变 factor-space norm。

因此
\[
v=\frac1D\mathbb{E}\|\theta^\star\|_2^2
\]
并不是 parameterization-invariant task quantity。

### 建议修改
明确区分两层理论：

1. **一般理论**：真实网络中真正重要的是 curvature-weighted projection bias
\[
B_{\rm proj}
=
\frac12
(\theta^\star_U-\theta^\star)^\top
H
(\theta^\star_U-\theta^\star).
\]

2. **isotropic sequence model**：
\[
v<\sigma_{\rm eff}^2
\]
只是固定参数化下的可解释 specialization。

### 建议在 Proposition 1 后增加
> **Remark on parameterization dependence.**  
> The scalar \(v=D^{-1}\mathbb{E}\|\theta^\star\|_2^2\) is defined in the chosen local LoRA-factor parameterization and is not invariant to equivalent factorizations of the same effective update. We therefore use \(v<\sigma_{\rm eff}^2\) only as an interpretable characterization of the isotropic sequence model. In real neural networks, the corresponding quantity is the curvature-weighted projection error
> \[
> B_{\rm proj}
> =
> \frac12
> (\theta^\star_U-\theta^\star)^\top
> H(\theta^\star_U-\theta^\star),
> \]
> which is the primary object tested in the downstream experiments.

---

## P0-4. 处理 \(H\succeq 0\) 与 \(H^{-1}\) 的不一致

### 位置
- Section 3.1：\(H=\nabla^2R(\theta^\star)\succeq 0\)
- Section 3.3：\(\Sigma_L\approx \frac1nH^{-1}GH^{-1}\)

### 当前问题
前文只假设 \(H\) positive semidefinite，后文却直接使用 \(H^{-1}\)。LoRA factorization 因为 non-identifiability 很容易存在 Hessian null directions。

### 建议修改
将 Section 3.3 的 derivation 明确限制到 locally identifiable tangent space，或改用 Moore–Penrose pseudoinverse。

推荐文本：

> For the local asymptotic calculation, we restrict attention to the identifiable tangent space of the parameterization, on which the population Hessian is positive definite. Equivalently, the following expressions may be interpreted using the Moore–Penrose pseudoinverse when \(H\) has null directions induced by parameterization symmetries.

并将：
\[
\Sigma_L
\approx
\frac1nH^{-1}GH^{-1}
\]
改为：
\[
\Sigma_L
\approx
\frac1nH^\dagger G H^\dagger
\]
或注明 inverse 只在 identifiable tangent space 上定义。

---

## P0-5. 明确 Proposition 2 与 data-dependent SNIP support 之间的 gap

### 位置
- Proposition 2
- Section 4.3：Warmup support selection
- Section 5.2：synthetic support uses \(|y-PP^\top y|\)

### 当前问题
Proposition 2 假设 support selection 与 estimation noise 独立，但实际 ProLoSA 的 support 是从训练数据中选择的。

sequence experiment 中：
\[
I=\operatorname{TopK}|y-PP^\top y|,
\]
而
\[
y=\theta^\star+\sigma\epsilon,
\]
因此 \(I\) 显式依赖 observation noise。

### 建议修改
明确 Proposition 2 是 idealized matched-budget result，并补充：

\[
\operatorname{Cov}(\hat\theta)
=
\mathbb{E}_I
[
\operatorname{Cov}(\hat\theta\mid I)
]
+
\operatorname{Cov}_I
(
\mathbb{E}[\hat\theta\mid I]
).
\]

第二项可称为 selection variance：
\[
V_{\rm selection}.
\]

### 建议正文加入
> Proposition 2 isolates the idealized benefit of allocating budget to target-informed residual directions while excluding support-selection noise. In practice, warmup SNIP uses the same training data to select the support, which may introduce an additional selection-variance term. The Transformer and downstream experiments therefore test whether the captured bias reduction remains larger than this additional variance in realistic training.

### 可选实验
增加 support stability：
\[
\operatorname{Jaccard}(I_s,I_{s'})
\]
或不同 seed 下 captured-energy 的稳定性。

---

## P0-6. Synthetic Figure 2 增加真正的 matched-budget baseline

### 位置
Section 5.2：Blind Synthetic Validation

### 当前问题
当前设置：
\[
d=16,\qquad K=32.
\]
因此：
- Uni-LoRA：16 parameters
- ProLoSA：48 parameters

这可以证明 sparse augmentation 降低 bias，但不能直接验证 Proposition 2 的 matched-budget claim。

### 建议新增
总预算：
\[
B=d+K=48.
\]

增加：
\[
\boxed{\text{Compressed-only}(B=48)}
\]

推荐比较：
1. LoRA/full-space；
2. compressed-only \(B=48\)；
3. hybrid \(d=16,K=32\), random support；
4. hybrid \(d=16,K=32\), informed support；
5. oracle hybrid。

### 目标
Figure 2 同时回答：

**Q1：Compression 是否优于 full-space？**
\[
\text{variance reduction}
>
\text{projection bias}?
\]

**Q2：固定总预算后，是否值得分出 \(K\) 个维度做 sparse correction？**
\[
E_K>Kv?
\]

---

# P1：决定论文说服力的关键修改

## P1-1. 把 E1 做成真正的 sample-size crossover 实验

### 位置
Section 5.7，E1

### 理论预测
\[
\Delta(n)
=
L^{\rm LoRA}_{\rm dev}
-
L^{\rm Uni}_{\rm dev}
\approx
\frac{A_{\rm var}}{n}
-
B_{\rm proj}.
\]

因此应重点画：
\[
x=\frac1n,\qquad y=\Delta L_{\rm dev}.
\]

### 必须报告
1. 各 \(p\) 下 dev loss；
2. paired seed difference；
3. 拟合
   \[
   \Delta=\hat a/n-\hat b;
   \]
4. \(R^2\)；
5. slope confidence interval；
6. crossover
   \[
   \hat n^\star=\hat a/\hat b;
   \]
7. bootstrap CI。

### 推荐图
\[
\Delta L_{\rm dev}
\quad \text{vs.}\quad
1/n.
\]

关键不是证明“小数据时 Uni-LoRA 更好”，而是检验理论预测的 functional form 是否成立。

---

## P1-2. 把 E4 做成最强的 real-model theory validation

### 位置
Section 5.7，E4

### 理论核心
\[
\text{compression gap}
\propto
B_{\rm proj}.
\]

当前固定 task/data/\(d\)，只改变 projection seed 的设计很好。

### 建议增强
projection seed 从 8 增加到至少 16–32（算力允许时）。

### 必须画散点图
横轴：
\[
\hat B_{\rm off}^H(P)
\]
纵轴：
\[
L_{\rm dev}^{\rm Uni}(P)-L_{\rm dev}^{\rm LoRA}.
\]

报告：
- Pearson \(r\)；
- Spearman \(\rho\)；
- regression slope；
- confidence interval。

### 对照
同时画 unweighted energy：
\[
\|q\|_2^2.
\]

如果：
\[
r(B_{\rm off}^H,\text{gap})
>
r(\|q\|^2,\text{gap}),
\]
会成为非常强的证据：真实 LoRA 中关键的是 curvature-weighted geometry，而非简单 Euclidean energy。

---

## P1-3. 把 E5 直接连接到 Proposition 2

### 位置
Section 5.7，E5

### 理论预测
ProLoSA gain 应随 sparse support 捕获的 off-subspace energy 增加。

定义：
\[
q=(I-PP^\top)\Delta\theta.
\]

主图建议：

横轴：
\[
\rho_{I,H}^2
\]

纵轴：
\[
\text{Gain}
=
L_{\rm Uni}
-
L_{\rm ProLoSA}.
\]

理论预测：
\[
\boxed{
\rho_{I,H}^2\uparrow
\Rightarrow
\text{ProLoSA gain}\uparrow
}
\]

### 同时报告
- oracle top-\(K\) captured fraction；
- SNIP captured fraction；
- random expected fraction；
- enrichment：
  \[
  \mathrm{Enrich}_K
  =
  \frac{\rho_I^2}
  {K/(D-d)};
  \]
- ProLoSA gain。

### 目的
回答：
> ProLoSA 有效，是因为“增加了 sparse 参数”，还是因为 sparse 参数确实落在理论要求的 high-energy off-subspace directions 上？

---

## P1-4. Transformer teacher–student 必须产出清晰的 bias–variance 图

### 位置
Section 5.4

### 角色
sequence model 验证 exact threshold；Transformer experiment 验证 qualitative mechanism 是否在以下因素存在时仍成立：
- nonlinear factorization；
- anisotropic curvature；
- finite-step optimization；
- data-dependent support selection。

### 推荐至少展示两个 regime

#### Concentrated target
预期：
- LoRA variance high；
- compressed bias moderate；
- ProLoSA recovers much of bias。

#### Diffuse target
预期：
- compressed bias high；
- ProLoSA recovery limited。

### 推荐图
横轴：\(n\) 或 \(\sigma_y\)

纵轴分别画：
- total excess risk；
- functional bias；
- functional variance。

包含：
- LoRA；
- matched-budget compressed；
- ProLoSA；
- random hybrid；
- oracle hybrid。

### 最理想结果
出现：
\[
\text{LoRA}\leftrightarrow\text{compressed}
\]
crossover，同时 ProLoSA 在 concentrated regime 明显降 bias，而在 diffuse regime 改善有限。

---

## P1-5. 弱化 “RoBERTa-large 更支持理论” 的解释

### 位置
Section 5.5：GLUE main results

### 当前问题
从 RoBERTa-base 换到 RoBERTa-large，不只改变 \(D\)，还同时改变：
- representation；
- curvature \(H\)；
- gradient noise \(G\)；
- target update；
- optimization landscape。

因此不能将差异归因于 \(D\) alone。

### 建议替换
> The stronger relative performance of compressed methods on RoBERTa-large is qualitatively compatible with the variance-reduction interpretation, but changing backbone size also changes curvature, representation quality, optimization, and the target update. We therefore do not treat this comparison as a controlled test of Eq. (6).

### 如果要严格验证 \(D\)
在同一个 backbone 内改变：
- LoRA rank；
- adapted layer 数量；
- target modules。

---

## P1-6. 删除 GSM8K/MATH 的 “different update energy” 理论解释

### 位置
Section 5.5：Mathematical Reasoning

### 当前问题
Appendix Table 12 显示模型只在 MetaMathQA 上 fine-tune 一次，然后分别 evaluate GSM8K 和 MATH。

因此两者没有不同的 task-specific training update。

不能从：
\[
\text{MATH 上 compression gap 更大}
\]
直接推出：
> MATH requires a higher-energy / less-compressible update.

### 建议修改
当前只作为 practical benchmark：
> ProLoSA improves over Uni-LoRA on MATH while remaining competitive on GSM8K.

不要赋予强理论解释。

### 如果要理论化
分别训练：
\[
\text{GSM8K train}\rightarrow\text{GSM8K test}
\]
和
\[
\text{MATH train}\rightarrow\text{MATH test}.
\]

然后比较：
\[
B_{\rm off}^H
\]
与 compression gap。

---

## P1-7. 强化理论 novelty 的 framing

### 位置
- Abstract
- Introduction
- Related Work
- Contributions

### 审稿风险
restricted estimation 下的 bias–variance trade-off 是经典思想，不应把 novelty 押在 “发现 bias–variance” 上。

### 推荐 framing
强调贡献是：
1. 将 compressed LoRA 统一解释为 restricted estimation；
2. 导出可 falsify 的 sample-size / noise / alignment / recoverability predictions；
3. 同时给出成功与失败 regime；
4. 区分：
   - compression 是否值得；
   - projection bias 是否 sparse-recoverable；
5. 理论导出 ProLoSA；
6. 在 Transformer 与真实 fine-tuning 中验证。

### 推荐表述
> Our contribution is not a new generic bias–variance principle, but a theory-to-experiment account of compressed LoRA that turns subspace restriction into falsifiable predictions about sample size, noise, alignment, and residual recoverability.

---

# P2：增强可信度与完整性

## P2-1. Related Work 增加 restricted-estimation / statistical regularization

### 位置
Section 2：Related Work

### 建议补充方向
- restricted estimation；
- subspace-constrained estimation；
- dimension reduction as regularization；
- principal-component regression / random-subspace methods；
- statistical bias–variance trade-off under projection；
- intrinsic-dimensional optimization。

### 推荐新增段落标题
**Subspace restriction and statistical regularization.**

推荐表述：
> Classical statistical estimation shows that restricting an estimator to a lower-dimensional hypothesis space can reduce variance at the cost of approximation bias. Our work does not claim this generic principle as new; instead, we instantiate it for compressed LoRA, derive testable consequences specific to PEFT, and connect the resulting geometry to sparse residual correction.

---

## P2-2. 将 “blind synthetic” 改为更准确的 “projection-blind”

### 位置
- Abstract
- Introduction
- Section 5.2

### 当前问题
虽然 \(\theta^\star\) 独立于 \(P\)，但 heavy-tailed target 的 sparsity profile 与 \(K=32\) 很匹配。

### 建议改为
> **projection-blind synthetic validation**

并明确：
> The target is sampled independently of the projection matrix, while its energy profile is deliberately controlled to instantiate recoverable and diffuse off-subspace regimes.

### 推荐额外实验
扫描：
\[
\pi
\]
或：
\[
K
\]
得到 sparse recoverability phase diagram。

---

## P2-3. 核心 baseline 自己统一重跑

### 位置
Table 1：GLUE

### 建议至少统一重跑
\[
\boxed{
\text{LoRA, Uni-LoRA, ProLoSA}
}
\]

统一：
- data subsets；
- seeds；
- optimizer；
- target modules；
- stopping criterion；
- projection seeds。

### 推荐统计
报告 paired differences：
\[
\Delta_s
=
L_s^{\rm LoRA}
-
L_s^{\rm Uni}.
\]

---

## P2-4. Math 与 instruction tuning 增加 uncertainty

### 位置
- Table 2
- Table 13
- Appendix Tables 12/14

### 当前问题
当前基本是单 seed，约 1 point 的差异很难判断是否稳定。

### 推荐
如果算力允许：
\[
3\text{ seeds}
\]

否则至少：
- test-example bootstrap；
- 95% CI。

MT-Bench 还应明确：
- judge model exact version；
- judge sampling；
- 是否重复 judge；
- decoding randomness。

---

## P2-5. 标题考虑改成 “When and Why”

### 当前标题
> Why Compression Makes LoRA Better: A Bias–Variance Theory of Compressed Adaptation and When It Fails

### 推荐标题 1
> **When and Why Compression Makes LoRA Better: A Bias–Variance Theory of Compressed Adaptation**

### 推荐标题 2
> **When Does Compression Make LoRA Better? A Bias–Variance Theory of Compressed Adaptation**

第二个标题更符合 crossover / failure regime 的真正核心。

---

# 推荐的整体论文逻辑

## Level 1：一般理论
\[
\boxed{
R_{\rm compressed}
=
B_{\rm proj}
+
V_{\rm compressed}
}
\]
与
\[
R_{\rm LoRA}=V_{\rm full}.
\]

Compression helps when：
\[
V_{\rm full}-V_{\rm compressed}
>
B_{\rm proj}.
\]

## Level 2：isotropic sequence specialization
\[
B_{\rm proj}
=
\frac12(D-d)v,
\]
\[
V_{\rm full}-V_{\rm compressed}
=
\frac12(D-d)\sigma_{\rm eff}^2.
\]
所以：
\[
\boxed{v<\sigma_{\rm eff}^2.}
\]

这一层只提供可解释 threshold。

## Level 3：matched-budget sparse correction
给定：
\[
B=d+K,
\]
hybrid 相比 pure compression 的差异取决于：
\[
\boxed{E_K-Kv.}
\]

关键不是 sparsity itself，而是 selected residual directions 是否捕获 above-average off-subspace energy。

## Level 4：Transformer experiment
验证在：
- nonlinear factorization；
- anisotropic curvature；
- finite-step optimization；
- data-dependent support selection；
下 qualitative bias–variance pattern 是否仍然存在。

## Level 5：真实 fine-tuning
依次验证：
\[
\boxed{P1:\ 1/n}
\]
\[
\boxed{P2:\ \text{noise}}
\]
\[
\boxed{P3:\ B_{\rm off}^H}
\]
\[
\boxed{P4:\ \rho_{I,H}^2}
\]

---

# 最终优先级 Checklist

## P0：投稿前必须完成
- [ ] 修正 Proposition 2 random-support corollary。
- [ ] 删除 “random worse exactly as Proposition 2 predicts”。
- [ ] 明确 LoRA factor-space energy \(v\) 的 parameterization dependence。
- [ ] 处理 \(H\succeq0\) 与 \(H^{-1}\) 的问题。
- [ ] 明确 data-dependent support 带来的 selection variance。
- [ ] Figure 2 增加 matched-budget compressed baseline \(B=d+K\)。

## P1：决定论文能否进入 6–7 分区间
- [ ] E1：画 \(\Delta L_{\rm dev}\) vs. \(1/n\)，拟合理论 functional form。
- [ ] E4：验证 \(B_{\rm off}^H(P)\) 与 compression excess loss 的关系。
- [ ] E5：验证 captured off-subspace energy 与 ProLoSA gain 的关系。
- [ ] Transformer teacher–student 给出清晰 function-space bias–variance decomposition。
- [ ] 弱化 RoBERTa-base/large 的非控制性理论解释。
- [ ] 删除当前 GSM8K/MATH “different update energy” 的不成立解释。
- [ ] 将 novelty 定位为 theory-to-experiment framework，而不是 generic bias–variance principle。

## P2：增强完整性
- [ ] Related Work 补 restricted estimation / statistical regularization。
- [ ] 将 “blind synthetic” 改成 “projection-blind synthetic”。
- [ ] 增加 energy concentration / \(K\) sweep。
- [ ] 核心 LoRA / Uni-LoRA / ProLoSA baseline 自己统一重跑。
- [ ] math / instruction tuning 增加 uncertainty estimates。
- [ ] 考虑标题改为 “When and Why Compression Makes LoRA Better”。

---

# 建议修改顺序

建议严格按照：
\[
\boxed{
\text{理论正确性}
\rightarrow
\text{synthetic matched-budget}
\rightarrow
\text{Transformer mechanism}
\rightarrow
\text{downstream falsification}
\rightarrow
\text{benchmark completeness}
}
\]
进行。

当前最需要提升的不是 benchmark 数量，而是：
\[
\boxed{
\text{让理论、synthetic、Transformer 和真实 fine-tuning 使用同一套可检验变量闭环。}
}
\]

如果这一闭环建立起来，论文的核心卖点就不再只是“ProLoSA 比 Uni-LoRA 好一点”，而会变成：

> **Compressed LoRA 的收益来自可量化的 variance reduction，其失败来自 curvature-weighted projection bias，而 sparse correction 只在该 bias 具有可恢复的能量集中结构时有效。**
