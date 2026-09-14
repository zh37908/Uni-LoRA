# 下一代合成实验设计：从高斯序列估计到单层 Transformer

本文档回答两个问题：(1) 现有合成实验与真实微调之间还差什么；(2) 如何在**保留"ground truth 已知、能量剖面可控"这一核心优势**的前提下，把合成实验升级为"单层 Transformer 挂载不同 LoRA 变体"的形式，并逐级逼近真实设置。

---

## 1. 现有合成实验的定位与差距

当前实验（`synthetic_spike_slab_noise_biasvar.py` 等）是一个高斯序列估计问题：直接观测 \(y=\theta^\star+\sigma_y/\sqrt{n_{\mathrm{eff}}}\,\epsilon\)，各方法解约束最小二乘。它精确对应论文 Section 3 的局部二次模型（\(H=I\)），因此是理论的最干净检验，**应当保留**。但它与真实 LoRA 微调之间有五个差距：

| # | 差距 | 现状 | 真实微调 |
|---|------|------|---------|
| 1 | 优化动力学 | 闭式最小二乘解 | SGD/AdamW 有限步，隐式正则化 |
| 2 | 参数化 | 直接在 \(\theta\in\mathbb{R}^D\) 上 | \(\Delta W=BA\) 双线性分解，非凸 |
| 3 | 曲率 | \(H=I\) 各向同性 | \(H\) 高度各向异性、跨层不均 |
| 4 | 支撑选择 | 残差 top-K（`noisy_residual`） | 真实的两阶段 warmup SNIP 打分 |
| 5 | 架构 | 无 | attention/MLP 结构、层间交互 |

审稿人可预期的质疑正是"理论只在线性回归里验证过"。下述设计逐级消除差距 1–5。

---

## 2. 相关工作的合成实验做法（调研）

- **Duranthon, Boncoraglio & Zdeborová (2026), "High-Dimensional Theory of LoRA Fine-Tuning in a Solvable Attention Model" (arXiv:2606.05899)**：与本文最接近的先例。单头 tied-attention 层，先在数据充足的 attention-indexed（AIM）teacher 上预训练，再用小数据学 rank-1 LoRA；teacher–student 全程合成，高维极限下用少量 order parameter 给出学习曲线。证明"单层 attention + 种植式 teacher"是理论论文公认的合成实验形态。
- **"When pre-training hurts LoRA fine-tuning" (arXiv:2602.02855)**：单指标模型 teacher–student，\(y=\phi(\omega^\star\cdot x)\)，学生在预训练方向上加低秩扰动，分析梯度流动力学。说明"预训练方向 + 小扰动"的种植式构造是标准做法。
- **"How Much Rank Does LoRA Need?" (arXiv:2608.26052)**：固定目标 attention 更新，研究各 rank 下的逼近误差；目标谱由任务 query/key 加权。启发我们在 attention 上控制**目标更新的谱/能量剖面**。
- **Garg et al. (2022) in-context 线性回归**及其后续：用小 Transformer 在合成函数族上训练，是"最小可用 Transformer 合成测试台"的通用范式。

共同点：**teacher–student + 种植式目标 + 可控数据量/噪声**。我们的差异化在于：先例都在研究 LoRA 本身的学习曲线或秩，而我们要比较 **LoRA / 压缩 LoRA / ProLoSA 三方**在受控能量剖面下的 bias–variance 行为——这正是现有文献没有的。

---

## 3. 设计原则

1. **ground truth 可知**：合成实验相对真实实验的根本优势是 \(\Delta\theta^\star\) 已知，可以直接测 projection bias、off-subspace 能量、支撑命中率。任何升级都不能丢掉这一点。
2. **能量剖面是唯一被操纵的自变量**：稀疏重尾（可恢复）vs 稠密高斯（不可恢复），其他条件严格一致，与论文 Fig. 2/3 的对照逻辑相同。
3. **理论预测不变**：升级后要检验的仍是四条可证伪预测（见 §5），若在非线性设置下依然成立，理论的外推有效性大大增强。
4. **逐级加复杂度**：每次只引入一个差距因素，否则无法归因。

---

## 4. 实验阶梯

### Level 1：线性回归 + 真实 BA 分解 + 梯度下降（引入差距 1、2）

- 模型：\(f(x)=(W_0+BA)x\)，teacher 为 \(W_0+\Delta W^\star\)。
- \(\Delta W^\star\) 的向量化 \(\theta^\star\) 仍从 spike-slab 或稠密高斯采样（复用现有生成代码）。
- 三方法：LoRA 训练 \(B,A\)；Uni-LoRA 训练 \(\theta_d\)（\(\theta_D=P\theta_d\) 再 reshape 成 \(B,A\)，与正文 Eq. 的参数化一致）；ProLoSA 加 routed sparse 分支并执行**真实的两阶段 warmup**（先 \(T_{\mathrm{pre}}\) 步纯压缩，再 128 步累积 SNIP 分数 \(|\theta_{D,i}\,g_i|\)，然后激活稀疏分支）。
- 优化器 AdamW，有限步数；数据 \((x_i,y_i)\)，\(x_i\sim\mathcal N(0,I)\)，标签加噪声 \(\sigma_y\)。
- 意义：检验闭式解结论在"GD + 双线性分解 + 真 SNIP"下是否保持。成本极低（CPU 分钟级），是 Level 2 的调试基础。

### Level 2（主推）：单层 Transformer teacher–student

**模型**
- 一个标准 Transformer block：单头（或 4 头）self-attention + 两层 MLP + LayerNorm + 残差连接。
- 建议规模：\(d_{\mathrm{model}}=64\)，序列长度 \(T=32\)，MLP 宽度 256。可训练矩阵 \(W_q,W_k,W_v,W_o,W_1,W_2\)，全部参数量约 \(5\times10^4\)，与现有 \(D=4096\) 同量级偏大，单卡秒级一个 trial。

**teacher 构造（两阶段，模拟"预训练 + 下游任务"）**
1. *预训练*：随机初始化后在一个通用合成任务上训练至收敛，得到 \(W_0\)。推荐任务：序列回归 \(y=\sum_t \alpha_t\,g(x_t)\)（保证 attention 与 MLP 都被用到），或 Garg 式 in-context 线性回归。
2. *种植下游更新*：在 \(W_q,W_v\) 上加 \(\Delta W^\star\)，其向量化坐标从两种能量剖面之一采样：
   - **稀疏重尾、低总能量**：\((1-\pi)\mathcal N(0,\tau_0^2)+\pi\,t_\nu(0,s)\)（对应论文 Eq. (12)，模拟微小分布变化的微调场景）；
   - **稠密高斯、高总能量**：\(\mathcal N(0,\tau^2)\)、\(\tau\) 较大（对应失效 regime，模拟需要大幅参数变化的任务）。
   两种剖面**不要求总能量相等**：能量大小与分布形状同时不同，正是"微调 vs 大分布偏移"的现实对应，也与论文 Fig. 2/3 的设置一致（spike-slab 实现 std≈0.05，稠密高斯 τ=0.125）。
   *可选的 2×2 解耦消融*：若审稿人质疑"稠密剖面下 ProLoSA 失效只是因为能量太大而非弥散"，补一个 {低/高能量}×{集中/弥散} 的 2×2 网格把两个因素分开。其中"高能量 + 集中"一格检验一条更锐利的预测：压缩本身失效（\(v>\sigma_{\mathrm{eff}}^2\)，Prop. 1）但稀疏修正仍可恢复——序列模型中 `results_spike_slab_synthetic_theory` 的 `strong_slab` 组已支持该预测（Uni-LoRA 风险 40.1，ProLoSA 0.35–0.80，全噪声段获胜）。
3. *下游数据*：\(y_i=f_{\mathrm{teacher}}(x_i;W_0+\Delta W^\star)+\varepsilon_i\)，\(\varepsilon_i\sim\mathcal N(0,\sigma_y^2)\)，训练集大小 \(n\) 可扫描。

**student（冻结 \(W_0\)，只训 PEFT 参数，matched budget \(B=d+K\)）**

| 方法 | 参数化 | 说明 |
|------|--------|------|
| Full FT | 全部 \(\Delta W\) | 上界参照 |
| LoRA(r) | \(\Delta W=BA\) | 标准基线 |
| Uni-LoRA(d) | \(\theta_D=P\theta_d\)，\(P\) 归一化一热 | 压缩基线 |
| ProLoSA(d,K) | \(P\theta_d + R\,a\)，真实 warmup SNIP 选支撑 | 本文方法 |
| ProLoSA-random(d,K) | 支撑随机 | Proposition 2 的对照 |
| Oracle(d,K) | 支撑取 \(\Delta\theta^\star\) 真实 top-K | 可恢复性上界 |

**操纵变量**：能量剖面（2 种）× 训练集大小 \(n\in\{64,128,256,512,1024,4096\}\) × 标签噪声 \(\sigma_y\in\{0.1,0.3,1.0\}\)；每格 ≥20 seeds。

**测量（合成设置独有的优势：\(\Delta\theta^\star\) 已知）**
- 测试损失（population 版本用大量 fresh 样本近似）；
- **参数空间分解**：跨种子均值 \(\overline{\Delta\theta}\)，\(\mathrm{Bias}=\|\overline{\Delta\theta}-\Delta\theta^\star\|^2\)、\(\mathrm{Variance}=\frac1M\sum_m\|\Delta\theta^{(m)}-\overline{\Delta\theta}\|^2\)（论文 Fig. 2B/C 的真模型版）；
- off-subspace 能量 \(\|(I-PP^\top)\Delta\theta^\star\|^2\) 与 top-K 能量占比；
- SNIP 支撑与 \(\Delta\theta^\star\) 真实高能坐标的重合率（recall）。

### Level 2 v1 结果诊断与 v2 必要修改（2026-08-30）

`results_transformer_lab_level1_full` 与 `results_transformer_lab_level2_full` 的结论：
**Level 1 完整复现理论**（sparse 剖面 Uni-LoRA bias 地板 ≈0.046、ProLoSA≈Oracle 降至 0.0004–0.03、SNIP≫随机支撑；dense 剖面 Uni 2.50/Pro 2.10 全面落后 LoRA；交叉方向正确），说明 GD + BA 分解不破坏理论。
**Level 2 v1 未能构成对 P2–P4 的有效检验**（Oracle 支撑也赢不了 Uni-LoRA；小 n 端出现"LoRA 反而更好"的反向交叉），诊断为三个构造缺陷，需在 v2 中修复：

1. **标量输出（根本原因）**。`head` 把 pooled 隐层压成 1 维：
   - Gauss–Newton 曲率 \(H\) 极度秩亏，参数空间 93.8% 的 off-subspace 能量在函数空间几乎不可见——理论的 bias 是 \(H\) 加权的（论文 Eq. (4)），\(H\) 加权 off-subspace 能量 ≈0 时"压缩全程有利、无 bias 可修"正是理论预测，因此 v1 结果**不证伪理论**，只说明裸参数能量不是正确的测量；
   - 每样本只有 1 个方程，各方法的插值阈值由 n/参数量决定：Uni-LoRA（64 参数）风险峰在 n≈64，LoRA（1024 参数）双下降峰在 n≈256–512（dense σ=2 时 LoRA 2.13→7.70→0.72 的驼峰），小 n 端胜负被插值/隐式正则现象主导，掩盖了理论的方差机制。
   - **修复**：输出改为 \(d_{\mathrm{model}}\) 维（去掉标量 head，直接回归 pooled 隐层或逐 token 隐层）。n=64 即有 64×64=4096 个方程 ≫ 全部预算，回到经典欠参数 regime，\(H\) 秩也大幅提升。此一项同时解决两个问题，**优先级最高**。
2. **因子空间种植扭曲能量剖面**。spike-slab 小因子经双线性映射后 \(\|\Delta W^\star\|=0.176\)（下游任务近乎平凡），dense 为 13.0——两剖面的实际差别变成"函数几乎不动 vs 大动"而非"集中 vs 弥散"。
   - **修复**：改为在 **\(\Delta W\) 空间**直接种植剖面（student 的 LoRA/压缩参数化不变），并对每个剖面**校准功能位移**：把 \(\Delta W^\star\) 缩放到目标输出扰动水平 \(\mathbb E[(f_{W_0+\Delta W^\star}(x)-f_{W_0}(x))^2]\)（例如 sparse 设为输出方差的 ~10%，dense 设为 ~100%）。注意这不是"总能量相等"，而是保证 sparse 剖面**低但非平凡**；
   - 进阶（可选）：在预训练模型的 GN/Fisher 曲率 top-\(m\) 特征方向张成的基上表达剖面，使 \(H\) 加权能量 \(v_H\) 按构造可控。
3. **预算未匹配**。v1 中 ProLoSA 为 d+K=128 参数对 Uni-LoRA 的 64，P4 无从谈起。**修复**：加 Uni-LoRA(d=128) 对照，matched budget 比较 Uni(128) vs ProLoSA(64+64) vs ProLoSA-random(64+64)。

**v2 新增的前置校准步骤（必做，防止再浪费整个网格）**：跑网格前，对每个（剖面 × student 类）先做一次"无穷数据"拟合（n 很大、无噪声、长训练），测量各参数化的**功能逼近地板** \(b_{\mathcal S}\)。要求 sparse 剖面下 \(b_{\mathrm{Uni}}\) 明显 >0 且 Oracle 支撑能显著降低它、dense 剖面下 \(b_{\mathrm{Uni}}\) 更大且 Oracle 无法降低——地板结构正确后再扫 n×σ 网格。v1 若先做此步，会立即发现 \(b_{\mathrm{Uni}}\approx 0\)（无 bias 可修）而免跑全网格。

**Oracle 定义修正**：raw \(|\theta^\star_i|\) top-K 在各向异性 \(H\) 下不是正确的上界；v2 的 oracle 支撑应取 **\(H\) 加权能量** top-K（用逐坐标梯度二阶矩的对角近似 \(\hat H_{ii}\)，按 \(\hat H_{ii}\theta_i^{\star 2}\) 排序）。SNIP 分数 \(|\theta_i g_i|\) 本身就是功能敏感度，无需改。

**对论文与真实实验的传导**：同样的教训适用于 `real_model_theory_validation.md` 的 E4/E5——裸 ℓ2 off-subspace 能量在真实网络上可能与性能差无相关，必须报告曲率加权版本（已在该文档中更新）。

### Level 3：由真实模型验证实验代替（不再单独做合成小 GPT）

证据链的第三级直接采用 `ICLR_2027/real_model_theory_validation.md` 中的 E1–E5（GLUE 数据量 crossover、标签噪声、真实模型 bias–variance 分解、off-subspace 能量、能量谱重尾性），理由：
- E1–E5 与 Level 2 检验的是同一组预测（P1↔E1/E2，P2/P3↔E4/E5，分解↔E3），覆盖面相同且证据强度更高（真实 backbone、真实任务）；
- E4/E5 复用已有 checkpoint，几乎零训练成本，性价比远高于再预训练一个合成小 GPT。

原 Level 3 唯一不可替代的部分是"多层设置下 \(\Delta\theta^\star\) 已知 + 跨层能量摆放可控"（用于检验跨层预算分配）。当前论文没有跨层分配的 claim，故不做；若未来需要：
1. 低成本方案：在 E4 中增加**按层分解**的 off-subspace 能量统计（对每个 target module 单独算 \(r_{\mathrm{off}}\)），检验"高能层收益大"的定性预测，无需新训练；
2. 完整方案：Level 2 的单层 block 堆成 2 层、只在其中一层种植高能量 \(\Delta W^\star\)，比较把 \(K\) 花在正确层 vs 均匀分摊。

---

## 5. 要检验的四条预测（与论文命题一一对应）

| 预测 | 来源 | 通过判据 |
|------|------|---------|
| P1 交叉点存在且随 \(n\)、\(\sigma_y\) 按理论方向移动 | Prop. 1 / Eq. (5) | 小 \(n\) 或大噪声时 Uni-LoRA 优于 LoRA，反之反转；交叉 \(n^\ast\) 随 \(\sigma_y^2/v\) 单调 |
| P2 稠密高能量目标使压缩均匀失效 | Prop. 1 失效条件 | 稠密剖面下 Uni-LoRA 全 \(n\) 落后 LoRA，且 bias 项主导（Fig. 3 的真模型版） |
| P3 稀疏修正的增益由能量集中度决定 | §3.2 可恢复性 | ProLoSA 增益与 top-K 能量占比正相关；稠密剖面下 ProLoSA≈Uni-LoRA（若需排除能量大小混淆，用 §4 的可选 2×2 消融：固定能量档位比较集中 vs 弥散） |
| P4 matched budget 下随机支撑劣于纯压缩 | Prop. 2 | ProLoSA-random 不超过 Uni-LoRA(d+K)，SNIP 支撑显著超过 |

---

## 6. 实现要点与预算

- 代码组织：新增 `synthetic/transformer_lab/`，包含 `model.py`（单层 block）、`teacher.py`（预训练 + 种植）、`students.py`（五种参数化，共享训练循环）、`run_level2.py`（网格扫描）、`plot_level2.py`（沿用现有三面板风格）。
- 训练：AdamW，lr 5e-3（PEFT 参数），500–2000 步早停；单个 (格点×seed) 在单 GPU 上约 10–30 秒，完整 Level 2 网格 2×6×3×20≈720 runs×6 方法 ≈ 数 GPU 小时。
- 复现性：沿用 `stable_seed` 机制；teacher、\(\Delta W^\star\)、投影 \(P\) 的种子与训练种子分离。

**风险与对策**
1. *非凸性使阈值从定量变定性*：不再要求交叉点数值精确匹配 \(\sqrt{n_{\mathrm{eff}}v}\)，改为检验方向性/单调性（P1–P4 均为序关系判据）；同时用局部二次拟合（对收敛点附近做 Gauss-Newton 近似 \(H\)）报告各向异性程度，作为理论适用性的诊断。
2. *LoRA 的 BA 隐式正则化混淆变量*：加一组"直接参数化 \(\Delta W\)（无 BA）"的对照，把"低秩分解效应"与"压缩投影效应"分离。
3. *预训练质量影响结论*（arXiv:2606.05899 的核心发现）：固定同一个 \(W_0\) 用于所有方法与所有格点；另做一个附加消融改变预训练步数，检验结论对 \(W_0\) 质量的稳健性。

---

## 7. 与论文的衔接

- 正文 §5.2/5.3（序列模型）保持不变，作为理论的精确实例化；Level 2 结果作为新小节"From the sequence model to a transformer block"或并入附录，回应"合成实验过于线性"的审稿意见。
- Level 2 的参数空间 bias–variance 分解图可与 `real_model_theory_validation.md` 中的 GLUE 数据量扫描（E1）形成"序列模型 → 单层 Transformer → 真实 GLUE"的三级证据链，这是目前文献中没有的完整验证路径。
- 建议叙述顺序：闭式模型给出定量预测 → 单层 Transformer 证明预测在非凸优化下仍以序关系成立 → 真实基准显示同样的模式。
