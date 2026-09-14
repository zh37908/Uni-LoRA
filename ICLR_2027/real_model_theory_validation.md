# 真实环境下 Bias–Variance 理论验证实验设计

本文档描述一组在 GLUE(以及可选的数学推理任务)上验证论文核心理论的实验。
目标是把 Proposition 1(energy–noise condition: 压缩有利 ⇔ $v<\sigma_{\mathrm{eff}}^2$)和
Proposition 2(matched-budget condition)的**可证伪预测**搬到真实模型上,
填充正文中已预留的注释小节 `sec:real_model_validation`(`iclr2027_conference.tex` 第 560 行附近)。

所有实验共享一个原则:**只改变理论中的一个变量**($\sigma_{\mathrm{eff}}^2$、$v$、能量分布),
其余训练配置(backbone、target modules、optimizer、budget、schedule)与论文附录 Table `tab:glue_hyperparams` 完全一致,
保证观测到的差异只能归因于理论机制。

> **来自 Level 2 合成实验的重要教训(2026-08-30)**:在真实网络中,曲率 $H$ 高度各向异性且秩亏,
> **裸参数空间的能量不预测功能 bias**——`results_transformer_lab_level2_full` 中参数空间 93.8% 的
> off-subspace 能量对应的功能 bias 地板几乎为 0。论文 Eq. (4) 的 bias 本来就是 $H$ 加权的,
> 因此凡是涉及"能量"的测量(E4、E5)都必须同时报告**曲率加权版本**,否则零结果无法解释
> (无法区分"理论错了"与"测错了空间")。加权方式见 E4。

---

## 理论预测一览(每个实验各检验一条)

| # | 理论预测 | 操纵变量 | 观测指标 |
|---|---------|---------|---------|
| E1 | 数据越少($\sigma_{\mathrm{eff}}^2\propto 1/n$ 越大),压缩相对全空间 LoRA 的优势越大,存在 crossover | 训练集比例 $p$ | 各方法 dev 性能、性能差 $\Delta(p)$、交叉点 $p^\*$ |
| E2 | 标签噪声增大 $\sigma_{\mathrm{eff}}^2$ 但不改变目标更新,压缩优势应随噪声单调扩大 | 标签翻转比例 $\eta$ | $\Delta(\eta)$ 的单调性 |
| E3 | 压缩方法估计方差显著更低、bias 更高;ProLoSA 介于两者之间 | 随机种子 | 跨种子方差(variance 项)、到参考解距离(bias 代理) |
| E4 | off-subspace 能量占比越大的任务,压缩相对 LoRA 越差 | 任务(自然变化) | 能量占比 vs. 性能差的相关性;SNIP 支撑重合率 |
| E5 | 微调更新近似稀疏重尾(Proposition 1 应用前提;也是 ProLoSA 可恢复性的前提) | 任务 | $\Delta\theta$ 能量谱 top-$k$ 占比、峰度 |

---

## E1 数据量 crossover(核心实验)

### 动机
Proposition 1 中 $\sigma_{\mathrm{eff}}^2\propto 1/n$。固定任务(固定 $v$)、只改变训练样本数 $n$,
是真实环境里最干净的 $\sigma_{\mathrm{eff}}^2$ 扫描:
理论预测存在交叉点 $n^\*$,当 $n<n^\*$ 时 Uni-LoRA/ProLoSA 优于 LoRA,当 $n>n^\*$ 时 LoRA 反超或追平。
这是合成实验 Figure 2 左图的真实模型版本。

### 设置
- **Backbone**: RoBERTa-large(与主表一致;RoBERTa-base 可作附加验证)。
- **任务**: 选训练集较大的任务才能扫出完整曲线,推荐 **SST-2(67k)与 QNLI(105k)** 为主,
  MNLI(393k)可选。RTE/MRPC/CoLA/STS-B 本身太小(全量即处于高噪声区),
  可只跑 $\{25\%,50\%,100\%\}$ 作为"曲线左端"的补充点。
- **数据比例**: $p\in\{1\%,2\%,5\%,10\%,25\%,50\%,100\%\}$,分层抽样(保持类别比例),
  同一 $p$ 下所有方法使用**同一个子集**(固定子集抽样种子)。
- **方法**: LoRA(full-space,$D$ 参数)、Uni-LoRA($d$)、ProLoSA($d+K$,配置同主表)。
- **训练预算**: 固定**更新步数**而非 epoch 数(小子集下 epoch 数不可比),
  例如统一为全量数据设置对应的总步数;lr 与 scheduler 沿用主表,不做 per-$p$ 调参
  (若担心 lr 敏感性,可对每个方法在 $p=10\%$ 上粗调一次后全程固定)。
- **种子**: 每个 $(p,\text{方法})$ 组合 5 个种子(与主表一致)。ProLoSA 的 warmup 仍为
  $\lfloor 0.1T\rfloor+128$ 步,注意小 $p$ 时 $T$ 变小、128 步占比变大,需在文中注明。

### 指标与产出
1. 主图:x 轴为 $p$(对数轴),y 轴为 dev 指标(SST-2/QNLI 用 accuracy),
   三条曲线(LoRA / Uni-LoRA / ProLoSA)± 跨种子标准差。
2. 差值曲线 $\Delta(p)=\text{Acc}_{\text{Uni-LoRA}}(p)-\text{Acc}_{\text{LoRA}}(p)$,
   报告 $\Delta(p)$ 由正转负(或收敛到 0)的交叉点 $p^\*$;
   对交叉点做 bootstrap 置信区间(对 5 个种子重采样)。
3. 显著性:每个 $p$ 处做配对(同子集同种子)t 检验或 Wilcoxon。

### 预期结果(理论成立时)
- 小 $p$:Uni-LoRA、ProLoSA 显著优于 LoRA(方差主导区);
- 大 $p$:LoRA 追平或反超 Uni-LoRA(bias 主导区),而 ProLoSA 因恢复了高能量方向,
  其相对 LoRA 的劣化应显著晚于/小于 Uni-LoRA——这是 Proposition 2 的附带检验;
- $\Delta(p)$ 单调递减。若 $\Delta(p)$ 与 $p$ 无关或反向,则理论被证伪。

### 反事实检查
把 dev 集固定为全量,确认性能变化来自训练集大小而非评估噪声;
并额外报告 train loss,确认大 $p$ 下 LoRA 反超不是欠拟合伪影。

---

## E2 标签噪声敏感性

### 动机
E1 中改变 $n$ 同时改变了很多东西(步数、过拟合程度)。标签噪声提供第二条独立通道:
以概率 $\eta$ 均匀翻转训练标签,提升有效估计噪声 $\sigma_{\mathrm{eff}}^2$,
但**不改变**目标更新 $\theta^\*$(dev 集保持干净)。理论预测压缩优势随 $\eta$ 单调扩大。

### 设置
- 任务:SST-2 与 MRPC(一大一小);backbone、超参同主表。
- 噪声比例 $\eta\in\{0\%,10\%,20\%,30\%\}$,对称翻转,所有方法共享同一份噪声标签
  (固定噪声种子);每组 5 个种子。
- 训练全量数据,其余配置不变。

### 指标与预期
- $\Delta(\eta)=\text{Acc}_{\text{Uni-LoRA}}(\eta)-\text{Acc}_{\text{LoRA}}(\eta)$
  应随 $\eta$ **单调增大**(即使 $\eta=0$ 时为负也应上升);
  ProLoSA 曲线应始终位于两者上方或与较优者持平。
- 报告 Spearman 相关(跨 $\eta$ 的 $\Delta$ 单调性)与每个 $\eta$ 的配对检验。

---

## E3 真实模型的经验 bias–variance 分解

### 动机
理论的核心是分解本身:压缩用 bias 换 variance。E3 直接在参数空间测量这两项,
是合成实验 Figure 2 中间/右图的真实模型版本。

### 设置
- 任务:MRPC 与 CoLA(小任务,方差效应明显;训练便宜)。
- 方法:LoRA、Uni-LoRA、ProLoSA,每个方法 $M=10$ 个种子
  (种子只改变初始化与数据顺序),训练后合并得到更新向量
  $\Delta\theta^{(m)}\in\mathbb{R}^{D}$(所有 target module 的 $B_\ell A_\ell$ 拉平拼接)。
- 参考解 $\theta^{\mathrm{ref}}$:高资源参考,推荐用**全量数据、大 rank 的 full LoRA
  (或全参微调)多种子平均**;参考解与被测方法使用相同 target modules 以保证同一空间。

### 指标
对每个方法计算(与正文 Eq. `empirical_bias_variance` 完全同式):

- $\bar{\Delta\theta}=\frac1M\sum_m \Delta\theta^{(m)}$
- $\mathrm{Variance}=\frac{1}{M}\sum_m\|\Delta\theta^{(m)}-\bar{\Delta\theta}\|_2^2$
- $\mathrm{Bias}^2$(代理)$=\|\bar{\Delta\theta}-\Delta\theta^{\mathrm{ref}}\|_2^2$

同时报告**功能层面**的对应量(避免"参数空间距离≠功能距离"的质疑):
- 功能方差:同一 dev 集上 $M$ 个种子预测 logits 的跨种子方差;
- 功能 bias:种子平均 logits 与参考模型 logits 的均方差。

### 预期结果
- Variance:LoRA ≫ ProLoSA ≳ Uni-LoRA;
- Bias:Uni-LoRA ≫ ProLoSA > LoRA;
- ProLoSA 的 (bias+variance) 总和最低——与其性能优势对应。
参数空间与功能层面两套指标应给出一致的排序;若不一致,**以功能层面为准**并在文中讨论
(Level 2 合成实验已证明参数空间距离可能与功能距离严重脱节,见文档开头教训)。

---

## E4 off-subspace 能量预测压缩何时失效

### 动机
Proposition 1 的逐实现版本:固定目标下压缩失效条件是
$\|(I-PP^\top)\theta^\*\|_2^2/(D-d)>\sigma_{\mathrm{eff}}^2$。
用 full LoRA 学到的 $\Delta\theta_{\mathrm{LoRA}}$ 作为 $\theta^\*$ 的代理,
可以对每个任务测出"off-subspace 能量占比",并预测 Uni-LoRA 相对 LoRA 的性能差。

### 设置
- 覆盖尽可能多的任务:6 个 GLUE 任务 + GSM8K + MATH(后两者用已有 checkpoint,不需重训)。
- 对每个任务:
  1. 取 full LoRA 的合并更新 $\Delta\theta$(多种子平均更稳);
  2. 用 Uni-LoRA 实际使用的投影 $P$(同一 $d$、同一分组种子)计算
     $r_{\mathrm{off}}=\|(I-PP^\top)\Delta\theta\|_2^2/\|\Delta\theta\|_2^2$,
     对多个分组种子取均值;
  3. **曲率加权版本(主指标)**:用下游数据估计逐坐标对角 Fisher/Gauss–Newton
     $\hat H_{ii}=\frac{1}{B}\sum_b\big(\partial\ell_b/\partial\theta_i\big)^2$
     (在微调后的模型上、LoRA 参数空间内,几百个 batch 即可),令 $q=(I-PP^\top)\Delta\theta$,计算
     $r_{\mathrm{off}}^{H}=\sum_i \hat H_{ii}q_i^2\,\big/\,\sum_i \hat H_{ii}\Delta\theta_i^2$;
  4. 记录该任务上 $\text{gap}=\text{Acc}_{\text{LoRA}}-\text{Acc}_{\text{Uni-LoRA}}$(主表已有)。

### 指标与预期
- 散点图:$x=r_{\mathrm{off}}$ 与 $x=r_{\mathrm{off}}^{H}$ 各画一张,
  $y=\text{gap}$;报告 Spearman/Pearson 相关系数。
  预期 **$r_{\mathrm{off}}^{H}$ 显著正相关**(这是理论 Eq. (4) 的直接对应量);
  $r_{\mathrm{off}}$(裸能量)相关性可能弱得多——两者的对比本身就是有价值的结果,
  可作为"压缩理论必须在曲率加权空间中理解"的真实模型证据。
  预期 MATH 的 $r_{\mathrm{off}}^{H}$ 应明显高于 GSM8K 与 GLUE
  (与正文"MATH 更接近 bias 主导端"的论述闭环)。
- **SNIP 支撑重合率**(直接验证 ProLoSA 机制):
  取 off-subspace 残差 $q=(I-PP^\top)\Delta\theta$ 能量最大的前 $K$ 个坐标,
  计算与 ProLoSA warmup 实际选出的支撑 $\mathcal I$ 的重合率
  $|\mathcal I\cap \mathrm{TopK}(q)|/K$,并与随机支撑的期望重合率 $K/D$ 对比。
  预期远高于 $K/D$;重合率越高的任务,ProLoSA 相对 Uni-LoRA 的增益应越大。

---

## E5 更新能量谱的重尾性(应用前提检验)

### 动机
正文多处假设"微调更新近似稀疏重尾"(可恢复 regime)。E5 用 E4 的 $\Delta\theta$
直接统计这一前提,不需要任何新训练。

### 指标
对每个任务的 $\Delta\theta_{\mathrm{LoRA}}$(以及 off-subspace 残差 $q$):
- top-$k$ 能量占比曲线:前 $0.1\%/1\%/5\%$ 坐标承载的能量比例;
- 峰度(kurtosis)与 Gini 系数;
- 与各向同性 Gaussian 基线(同维度、同总能量)对比画在同一图上;
- **曲率加权版本**:用 E4 的 $\hat H_{ii}$ 把每坐标能量替换为 $\hat H_{ii}\Delta\theta_i^2$
  后重算上述全部统计量。ProLoSA 可恢复性的正确前提是**加权能量**重尾
  (SNIP 分数 $|\theta_i g_i|$ 逼近的正是加权量),裸能量重尾只是近似说法。

### 预期
GLUE/GSM8K 更新显著重尾(少数坐标承载大部分 off-subspace 能量),
MATH 相对更弥散——同一组统计量同时解释了 ProLoSA 在前者收益大、在后者收益打折。

---

## 执行与写作说明

### 优先级与算力估计(单卡 L20 口径)
1. **E1(SST-2 + QNLI)**:约 2 任务 × 7 比例 × 3 方法 × 5 种子 ≈ 210 次训练,
   小比例训练很快,总量约等于 40–60 次全量训练,是最贵也最重要的实验。
2. **E4 + E5**:几乎免费(复用已有 checkpoint,只做线性代数统计),建议最先做。
3. **E3(MRPC + CoLA)**:2 × 3 × 10 = 60 次小任务训练,中等成本。
4. **E2**:2 × 4 × 3 × 5 = 120 次训练,可在 E1 结论明确后裁剪(只跑 SST-2)。

### 共同控制项(避免审稿人质疑)
- 同一 $(p,\eta)$ 配置下,三个方法使用**完全相同**的数据子集/噪声标签/评估集;
- 投影 $P$ 的分组种子固定并报告;改变 $P$ 种子重复一遍 E4 以证明结论对分组不敏感;
- 所有对比在 matched budget 语境下解释:LoRA($D$)、Uni-LoRA($d$)、
  ProLoSA($d+K$),并沿用主表的 $B=d+K$ 口径引用 Proposition 2;
- 报告均值 ± 标准差与配对检验,不只报单点。

### 结果如何进正文
- 取消注释 `sec:real_model_validation`(tex 第 560–575 行),
  五个 paragraph 与 E1–E5 一一对应,把 `[结果]` 占位符替换为实测数值;
- E1 主图建议与合成实验 Figure 2 左图并排(合成 vs. 真实同一形状的 crossover 是最有说服力的图);
- 若正文空间不足,E2/E3/E5 细节放附录,正文各留一句结论 + 引用;
- 若某条预测未被证实(例如 $\Delta(p)$ 不单调),不要隐藏:在 limitation 中如实讨论,
  说明真实微调与各向同性模型的偏离(如 $H\neq I$、优化非精确最小二乘)。
