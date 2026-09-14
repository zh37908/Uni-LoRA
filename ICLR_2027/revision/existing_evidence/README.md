# 第 4 部分执行报告：已有数据再分析与固定目标核查

2026-09-13。没有训练新模型；原始实验 CSV/JSON/PT 保持不变。本文稿采用此目录的再分析结果，原有 summary.md 中的拟合和 E5 表格属于历史口径。

## E1：配对不确定性与拟合

- 原始来源：`NLU/peft/examples/sequence_classification/results_theory_e1_crossover`。SST-2/CoLA 使用已存在的专用 LoRA 学习率重跑；QNLI/MRPC 使用现有基线。每个任务、比例、方法有 5 个种子，不混入被替代的旧 LoRA。
- `e1_paired_runs.csv` 保存每一对的路径、种子、实际 n、两种方法的 loss 及其差，loss 延续原协议：每次运行的最小 dev loss。跨方法按种子配对，绝不从边际标准差拼配对误差。
- 每个比例的训练子集使用固定 subset seed 12345，五个种子只改变训练随机性。点态 95% 区间为配对差均值 ± t(4, .975) × 配对差标准差 / √5；不覆盖重新采样训练集、dev 集或 checkpoint 选择的总体不确定性，也不是同时置信区间。
- 拟合使用同一组配对均值与实际训练样本数；旧脚本拟合的是两组边际中位数之差。新脚本分别拟合自由 `a/n-b` 和 `a≥0,b≥0` 模型。枚举 5^5=3125 个完整种子轨迹 bootstrap，保持跨方法及跨比例依赖；区间是小样本下的描述性敏感性结果，边界处 [0,0] 不是总体参数已知为零。
- SST-2：自由拟合 a=52.14、b=-0.01712、R²=0.485；约束拟合 a=68.69、b=0、R²=0.358。1% 的配对差 0.0671，区间 [-0.0206, 0.1549]；100% 为 0.0023，区间 [-0.0083, 0.0129]。
- QNLI：自由 a=-43.90，bootstrap 区间 [-63.14,-29.14]，R²=0.065；约束 R²=-0.363。这与理论的非负斜率条件冲突。
- CoLA：均值拟合 R²=0.014，撤回旧中位数拟合 R²=0.968 所支持的“强样本量证据”。MRPC 自由斜率也为负，但区间很宽。
- 四个任务的约束点拟合均为 b=0，没有有限正 crossover；不能把符号冲突与拟合失配统称为样本量不足。
- `e1.json` 包含系数、bootstrap 区间、逐比例预测与残差、RMSE 及其 bootstrap 区间。`e1_paired_ci.tex`、`e1_paired_fit.tex` 是论文附录表格。

## E3：随机性来源、重复修正与参考敏感性

- 原始来源：E3 保存的 predictive logits。每个方法 5 个 80% 分层子样本（101–105）× 2 个训练种子（0、1）；子样本内部无放回，不是 bootstrap 有放回抽样。
- 重算复现了原表六组数值。Vopt 使用 B(S−1) 分母；between 使用 B−1，再减 Vopt/S；保留减法前后和裁剪前的数值。MRPC ProLoSA 原始 Vstat=-0.001152，显示为 0 是裁剪，不是方差消失。
- 因两个 seed 在所有子样本中复用，额外给出 crossed data/seed 方差分量敏感性：用交互项/S 修正 between。主要排序不变；这不能消除仅有两个 seed 的估计局限。
- Vopt 包含初始化、训练顺序、优化及 compressed projection seed；正文图改为分别展示“偏差代理、数据重采样方差、训练种子方差”。CoLA 的总方差下降 87.8% 来自训练种子分量，不能据此验证 A/n。
- 参考为每任务 3 个 full-data rank-4 E4 模型 + 5 个 rank-64 E3 模型。沿用原质量规则：rank-64 dev score 至少为最佳参考分数的 85%；CoLA 排除 5 个失败参考，MRPC 无排除。具体路径与分数保存在 `e3.json`。
- 检查 rank-4 only、rank-64 only，以及 8 次 leave-one-reference-out，全部偏差排序保持：CoLA LoRA < Uni-LoRA < ProLoSA；MRPC LoRA < ProLoSA < Uni-LoRA。区间是参考敏感性范围，不是置信区间。
- 偏差仍是有限 reference ensemble 下的 plug-in 距离；未扣除参考噪声或大均值估计噪声。Bias² + corrected V 是描述性和，不宣称为精确经验恒等式或无偏总体风险。

## E5：候选集与几何

- 明确重新定义为 full-data rank-4 LoRA 目标代理：CoLA/MRPC 各 8 个，QNLI/RTE/STS-B 各 5 个；从 E1/E4 的全量训练运行取平均，不把 E3 子采样或 E1 小样本运行当成全量目标。使用固定的 full-data seed-0 ProLoSA 支持，不根据能量挑选 seed。每个目标和支持的完整路径记录于 `e5.json`。
- 旧表的 pooled-target/support 来源不够明确，不能仅换分母而宣称完整复现；新表以明确的 full-data 协议替代，故 CoLA/MRPC 的 oracle 和部分支持比例也更新。JSON 另外保留跨训练比例汇总目标的敏感性结果，它不是对旧表的精确复现。
- D=1,769,472；保存的 offsets 验证为所有原始坐标的双射。CoLA/MRPC：d=14,400，K=8,640；其余：d=22,320，实际 K=721（标称 720，按保存 mask 实数报告）。E5 MRPC 支持预算与 E3 MRPC 不同。
- 固定 e_i=H_ii q_i²，所有原始坐标等概率、无放回选 K 个，E[capture]=K/D。若候选集是 C，一般为 K/|C| × sum_C e / sum_all e；D−d 不能自动当作候选坐标数。
- 每任务 10,000 个真实随机支持；保存均值、标准差、中央 95% 分布区间和超越 SNIP 次数，并用有限总体公式计算精确标准差。Monte Carlo 均值均在 K/D 的两倍 Monte Carlo SE 内；RTE 有 8/10000 个随机支持达到或超过 SNIP，不能称随机支持绝无可能达到它。
- qE 是普通桶均值投影后的残差再做 Fisher 加权；qH 是 H 加权桶均值投影残差。两者差别很大。MRPC 的支持坐标比例分别为 18.04% 和 2.67%；在同一个 d 下共同重拟合桶参数和稀疏参数，降低 qH 加权偏差约 5.68%。后者仍没有扣除替代压缩容量，不能冒充等预算收益。
- H 是原脚本在一个 full-data LoRA checkpoint 上用至多 200 个大小 16 的 minibatch（RTE 实际为 156 个） 平均梯度平方估计的对角代理。不是逐样本 Fisher，也不是精确 Hessian。不同 LoRA 因子的均值还受 gauge 影响。这些诊断不直接识别未知最优解的功能偏差。

## 合成实验：固定 theta* 与不能混合的方差

- 当前序列模型的 matched-budget、原始 concentrated 和 diffuse 数据：目标和 P 都在 noise/trial 循环外生成。重建目标的 SHA-256 指纹、范数、残差能量见 `fixed_target_audit.json`。目标指纹相互独立；不把不同目标的数据拼成一个偏差—方差分解。
- 在固定目标下，经验 bias=1/2||mean(theta_hat)−theta*||²，variance=1/(2M)sum||theta_hat−mean||²；两者必须来自同一组估计。M 分母保证和等于经验 mean risk；plug-in bias 仍含有限重复噪声。
- 如果明确报告多个任务平均，只能先逐任务计算条件分解，再平均各自 B_t+V_t。跨目标的 estimator variance 含目标间均值变动，不能直接加到条件偏差上。
- 对当前 240 行序列汇总和 518 行 Transformer 汇总，检查各自 risk=bias+variance；最大误差 <8e-15。并逐组用 10,108 条 Transformer trial 重新复现 risk mean/std，验证全部主实验 504 个格子各 20 个不同种子、14 个校准格子各 2 个不同种子。profile/n/noise/method 全键唯一；没有将 calibration 与主实验、两类 teacher 混为一个分解。
- Transformer runner 的 `.py` 源码和逐次预测在当前目录中缺失。对留存 CPython 3.11 字节码的只读反汇编显示 `_plant` 在 profile 内且在 n/noise/seed 循环外、分解也在 profile 内完成；日志和主/校准目标范数、缩放、总能量一致。完整反汇编证据保存在 `transformer_bytecode_audit.txt`。这些证据支持固定目标，但汇总恒等式本身不能独立证明每个原始预测都对应同一目标；没有伪造原始预测重算。
- Transformer oracle 的留存实现按 `H_ii * theta_star_i²` 排序，在建立 learner projection 之前完成；它不是原文所写的 H 投影残差评分。附录公式、协议表已更正，并说明校准与主实验的曲率探针和 oracle 支持可能不同。
- 另发现 Transformer 的实际代码使用 mean squared error，**没有 1/2**，且平均所有输出坐标；正文、公式、图轴据此修改，保持原实验数值。该单位不与序列模型的 half-squared risk 混加。
- **旧 Hidden-P 问题已修复代码，旧实验未重跑**：旧目标 seed 包含 pm_mode/pm_angle，且投影构造消耗 teacher RNG，所以跨角度/模式目标会变。新协议 `fixed_hidden_teacher_v2` 分离教师、学习器投影、数据 RNG；同目标下修改投影不改变 theta*。新 CSV 会写 target_id 与 protocol。历史 Hidden-P 结果不作为同目标投影对照，也不能跨 scenario 汇总偏差/方差。当前论文主实验不引用这些旧扫描。
- 新的 `synthetic/fixed_target.py` 被两个公共汇总入口使用；会拒绝 per-trial target 矩阵和与固定目标不一致的 per-trial risk。`audit_fixed_targets.py` 包含混合目标反例与不同投影下目标逐元素相同的检查，不启动训练。

## 复现

在仓库根目录：

```bash
/home/hzhaobi/miniconda3/envs/unilora_nlu/bin/python ICLR_2027/revision/reanalyze_existing_evidence.py
python ICLR_2027/revision/render_existing_evidence.py
/home/hzhaobi/miniconda3/envs/unilora_nlu/bin/python ICLR_2027/revision/validate_existing_evidence.py
/home/hzhaobi/miniconda3/envs/unilora_modern/bin/python ICLR_2027/revision/inspect_transformer_bytecode.py
/tmp/iclr-layout-env/bin/python ICLR_2027/revision/audit_fixed_targets.py
/tmp/iclr-layout-env/bin/python ICLR_2027/figures/make_summary_figures.py
/tmp/iclr-layout-env/bin/python synthetic/make_transformer_paper_figures.py --results synthetic/results_transformer_lab_level2_v2_full/summary.csv --output-dir ICLR_2027
```

分析环境需要 NumPy/SciPy/PyTorch；绘图需要 NumPy/Matplotlib。`e5_main_rows.tex` 提供正文表格的生成行，其内容同步到正文表格；其余生成表格由附录直接 input。在 `ICLR_2027` 内编译：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
```

修改前备份：`ICLR_2027/archive/before_existing_evidence_20260913_170823/`。

最终编译检查：正文 8 页、全稿 44 页；无未定义引用、重复标签或 overfull。正文与附录图表已抽查。
