# Bias–Variance 理论与 GLUE 实验修改建议

## 1. 理论部分建议

### 1.1 增加一个从理论到真实微调的桥梁公式

当前一般 bias–variance 分解可进一步写成：

\[
\mathcal R_{\mathrm{LoRA}}(n)-\mathcal R_{\mathrm{Comp}}(n)
\approx
\frac{A}{n}-B,
\]

其中：

- \(A\)：压缩相对 LoRA 减少的 estimation variance 系数；
- \(B\)：压缩引入的 projection bias；
- \(n\)：有效训练样本数。

因此存在 crossover：

\[
n^\star=\frac{A}{B}.
\]

结论：

- \(n<n^\star\)：variance 主导，压缩方法更好；
- \(n>n^\star\)：bias 主导，LoRA 更好。

这可以作为 Proposition 1 的真实模型推广，而 Proposition 1 的

\[
v<\sigma_{\mathrm{eff}}^2
\]

是其 isotropic 特例。

---

### 1.2 将 scalar noise 推广到 directional noise

真实神经网络中噪声不是各向同性的。可引入：

\[
G=\mathrm{Cov}[\nabla_\theta \ell_i(\theta^\star)]
\]

作为 gradient-noise covariance。

局部渐近下：

\[
V_{\mathrm{LoRA}}
\approx
\frac{1}{2n}\mathrm{tr}(H^{-1}G),
\]

压缩空间 \(P\) 中：

\[
V_{\mathrm{Comp}}
\approx
\frac{1}{2n}
\mathrm{tr}
\left[
(P^\top HP)^{-1}P^\top GP
\right].
\]

因此更一般地：

\[
\mathcal R_{\mathrm{Comp}}-\mathcal R_{\mathrm{LoRA}}
\approx
B_{\mathrm{proj}}
-
\frac{A_{\mathrm{var}}}{n}.
\]

解释应改为：

> 压缩是否有效，取决于它移除的高噪声方向所带来的 variance reduction，是否大于被丢弃方向中的任务相关能量所产生的 projection bias。

---

### 1.3 修正 ProLoSA 的 variance 预测

在 matched-budget、isotropic 理论下：

\[
V_{\mathrm{LoRA}}
>
V_{\mathrm{UniLoRA}}
\approx
V_{\mathrm{ProLoSA}}.
\]

而 bias 预期为：

\[
B_{\mathrm{LoRA}}
<
B_{\mathrm{ProLoSA}}
<
B_{\mathrm{UniLoRA}}.
\]

因此不要写：

> ProLoSA 的 variance 介于 LoRA 和压缩方法之间。

更准确的说法是：

> ProLoSA 在不显著增加一阶 estimation variance 的前提下，恢复压缩空间之外的高价值方向，从而降低 projection bias。

---

### 1.4 Heavy-tail 不是 Proposition 1 的前提

需要明确区分两件事：

- Proposition 1：解释 **compression 是否优于 LoRA**；
- Heavy-tail / sparsity：解释 **projection bias 是否能被少量 sparse directions 恢复**。

因此：

> “微调更新重尾”不是 Proposition 1 的应用前提，而是 ProLoSA sparse recovery 的前提。

更准确地，应分析 off-subspace residual：

\[
q=(I-\Pi_P^H)\theta^\star,
\]

而不是只分析总更新 \(\Delta\theta\)。

---

### 1.5 增加真实模型 matched-budget 条件

设总预算为 \(B\)。

- pure compression：使用 \(B\) 个 projected dimensions；
- ProLoSA：使用 \(d=B-K\) 个 projected dimensions + \(K\) 个 sparse directions。

记：

- \(B_B\)：预算为 \(B\) 时 pure compression 的 bias；
- \(B_d\)：减少到 \(d\) 个 projected dimensions 后的 bias；
- \(C_I\)：sparse support 恢复的 bias。

则 ProLoSA 更好的条件为：

\[
C_I>B_d-B_B.
\]

若考虑额外 variance：

\[
C_I>
(B_d-B_B)
+
(V_{\mathrm{ProLoSA}}-V_{\mathrm{Comp}}).
\]

其含义是：

> sparse branch 恢复的 projection bias 必须大于为了腾出 sparse budget 而牺牲 projected dimensions 所产生的额外 bias。

---

### 1.6 考虑 data-dependent support selection

真实 ProLoSA 的 SNIP support 依赖训练数据，因此会引入 selection variance：

\[
\mathrm{Cov}(\hat\theta)
=
\mathbb E_I[\mathrm{Cov}(\hat\theta\mid I)]
+
\mathrm{Cov}_I(\mathbb E[\hat\theta\mid I]).
\]

第二项可理解为：

\[
V_{\mathrm{selection}}.
\]

因此可以增加一个预测：

> 如果不同数据重采样下 SNIP support 很稳定，则 selection variance 很小；如果 support 很不稳定，则 ProLoSA 可能重新引入额外 variance。

---

## 2. GLUE 实验修改建议

### E1：Sample-size crossover

理论预测：

\[
\Delta(n)
=
\mathcal R_{\mathrm{LoRA}}
-
\mathcal R_{\mathrm{Comp}}
\approx
\frac{A}{n}-B.
\]

实验：

- 改变训练集比例 \(p\)，令 \(n=pN\)；
- 主要画：

\[
\Delta L_{\mathrm{dev}}
\quad \text{vs.}\quad
1/n.
\]

拟合：

\[
\Delta L_{\mathrm{dev}}
=
a/n-b.
\]

进一步估计：

\[
\hat n^\star=a/b.
\]

重点验证：

1. compression advantage 是否随 \(1/n\) 近似线性增大；
2. 是否存在预测到的 crossover。

**理论主指标建议用 dev loss，而不是 accuracy/Matthews。**  
GLUE 官方 metric 可作为 practical metric 辅助报告。

---

### E2：Noise scaling

原来的分类标签翻转实验不够严格，因为 label noise 可能同时改变 population optimum。

最推荐：

#### STS-B

加入 zero-mean additive label noise：

\[
\tilde y=y+\epsilon,\qquad
\epsilon\sim\mathcal N(0,\tau^2).
\]

这样 target 基本不变，而 estimation noise 增大。

理论预测：

\[
\Delta(\tau)
\approx
c_0+c_1\tau^2,
\qquad
c_1>0.
\]

分类任务上的 label flipping 可以保留，但应定位成机制的 robustness test，而不是 Proposition 1 的严格验证。

---

### E3：真实 bias–variance decomposition

不要只改变 optimizer seed。

理论中的 estimation variance 对应：

> 重新采样一个大小为 \(n\) 的训练集后，估计器发生多少变化。

建议使用：

- 多个 dataset bootstrap/subsample；
- 每个 dataset 再运行多个 optimization seeds。

定义：

\[
\bar\theta_b
=
\frac1S\sum_s\hat\theta_{b,s}.
\]

统计 variance：

\[
V_{\mathrm{stat}}
\approx
\frac1B
\sum_b
\|\bar\theta_b-\bar\theta\|_H^2.
\]

优化 variance：

\[
V_{\mathrm{opt}}
\approx
\frac1{BS}
\sum_{b,s}
\|\hat\theta_{b,s}-\bar\theta_b\|_H^2.
\]

bias proxy：

\[
B
\approx
\|\bar\theta-\theta_{\mathrm{ref}}\|_H^2.
\]

其中 \(\theta_{\mathrm{ref}}\) 可由更大数据量 / 更高预算 LoRA 得到。

不要优先使用普通 Euclidean parameter distance；推荐：

- diagonal Fisher / empirical Fisher weighted distance；
- 或固定 dev/probe set 上的 logits distance / predictive KL。

---

### E4：Off-subspace energy 与 compression performance

理论上的 projection bias 应定义为：

\[
B_{\mathrm{off}}
=
\frac12
\|
(I-\Pi_P^H)\theta^\star
\|_H^2.
\]

预测：

\[
B_{\mathrm{off}}\uparrow
\quad\Rightarrow\quad
\text{compression advantage}\downarrow.
\]

#### 最推荐的 controlled experiment

固定：

- task；
- dataset size；
- compressed dimension \(d\)。

只改变随机 projection \(P\)。

然后测：

\[
B_{\mathrm{off}}(P)
\]

与：

\[
L_{\mathrm{LoRA}}-L_{\mathrm{Comp}}(P)
\]

之间的相关性。

这比直接跨不同 GLUE task 做相关性更干净，因为跨 task 的 \(n\)、noise、difficulty 都不同。

跨任务相关性可以作为辅助实验。

---

### E5：Sparse recoverability

重点不要放在：

> 总 LoRA update 是否重尾。

而应该放在：

> off-subspace residual 的能量是否集中。

定义：

\[
q=(I-\Pi_P^H)\theta^\star.
\]

推荐主指标：

\[
C_K(q)
=
\frac{
\sum_{i\in\mathrm{TopK}(q)}q_i^2
}{
\|q\|_2^2
}.
\]

如果：

\[
C_K(q)\gg K/D,
\]

说明少量 sparse directions 能恢复大量 projection bias。

建议报告：

- top-\(K\) energy ratio；
- energy concentration curve；
- SNIP captured-energy ratio；
- random support captured-energy ratio；
- oracle top-\(K\) captured-energy ratio；
- kurtosis 作为辅助指标。

比 support overlap 更重要的是：

\[
\rho_I^2
=
\frac{
\text{SNIP support 捕获的 off-subspace energy}
}{
\text{总 off-subspace energy}
}.
\]

可以进一步定义 energy enrichment：

\[
\mathrm{Enrich}_K
=
\frac{\rho_I^2}{K/(D-d)}.
\]

若：

\[
\mathrm{Enrich}_K\gg1,
\]

说明 SNIP 选到的是明显高于平均能量的方向。

---

## 3. 最推荐新增的理论小节

建议在理论部分增加：

### From Bias–Variance Theory to Testable Downstream Predictions

核心写出：

\[
R_L(n)-R_C(n)
\approx
\frac{A_{\mathrm{var}}}{n}
-
B_{\mathrm{proj}}.
\]

然后明确三个 downstream predictions：

1. **Sample-size prediction**  
   \(\Delta R\) approximately scales with \(1/n\).

2. **Noise prediction**  
   增加 estimation noise 会扩大 variance-reduction benefit，并使 crossover 向更大的 \(n\) 移动。

3. **Subspace-alignment prediction**  
   在 variance 条件相近时：

   \[
   B_{\mathrm{proj}}\uparrow
   \Rightarrow
   \text{compression performance}\downarrow.
   \]

再增加 ProLoSA 的预测：

4. **Sparse-recoverability prediction**  
   ProLoSA 的收益由 off-subspace energy 的集中程度决定，而不是由总 update magnitude 单独决定。

---

## 4. 实验优先级

如果计算资源有限，优先做：

### 第一优先：E1

\[
\Delta L_{\mathrm{dev}}
\text{ vs. }1/n
\]

并拟合 crossover。

### 第二优先：E4

固定 task，改变 projection seed，验证：

\[
B_{\mathrm{off}}(P)
\rightarrow
\text{compression excess loss}.
\]

### 第三优先：E5

验证：

\[
\text{SNIP captured off-subspace energy}
\rightarrow
\text{ProLoSA gain}.
\]

### 第四优先：E3

用 dataset bootstrap 真正分解 statistical variance 和 optimization variance。

---

## 5. 最终理论—实验闭环

建议将整篇论文的真实模型验证组织为：

\[
\text{small }n / \text{large noise}
\rightarrow
\text{large estimation variance}
\rightarrow
\text{compression helps},
\]

\[
\text{subspace mismatch}
\rightarrow
\text{projection bias}
\rightarrow
\text{compression hurts},
\]

\[
\text{concentrated off-subspace energy}
\rightarrow
\text{sparse recoverability}
\rightarrow
\text{ProLoSA helps}.
\]

这样 GLUE 实验就不只是 benchmark comparison，而是直接验证论文的三个核心机制：

**variance reduction、projection bias、sparse bias recovery。**
