# ICLR 2026 LoRA(理论)论文实验配置调研与本文实验现代化建议

调研目的:本文目前的主实验组合(RoBERTa + GLUE、Gemma-7B + GSM8K/MATH、Alpaca + MT-Bench)
在 2026 年的 LoRA 论文中已显得老旧。本文档统计近期(以 ICLR 2026 为主)LoRA 及 LoRA 理论论文
的实验配置,并据此给出本文主实验的改进方案。

---

## 一、调研论文与实验配置汇总

### 1. 理论向论文(与本文定位最接近)

| 论文 | 理论内容 | 真实模型实验配置 |
|------|---------|----------------|
| **LoRA vs. Full Fine-Tuning: A Theoretical Perspective** (arXiv 2605.19018) | 线性回归下 LoRA vs FFT 的 excess risk 界;结论与本文同向:LoRA 的优势来自**方差缩减**,在任务偏移 $\Delta^\*$ 有效低秩、强噪声 regime 下 LoRA 胜出 | **Qwen2.5 (0.5B/1.5B/3B)**,BoolQ + CommonsenseQA;做了**标签噪声扫描**与**样本量扫描**,每配置 3 种子,lm-evaluation-harness zero-shot 评测;并逐层估计 $\Delta^\*$ 的奇异值累积能量谱验证低秩前提 |
| **High-Dimensional Theory of LoRA Fine-Tuning in a Solvable Attention Model** (arXiv 2606.05899) | 可解注意力模型中的 LoRA 高维学习曲线,刻画预训练/微调样本量与预训练质量的联合影响 | 以可解模型的数值模拟为主(合成为主、真实实验轻量) |
| **Tight Sample Complexity for Low-Rank Adaptation** (arXiv 2607.27680) | LoRA 的 $\tilde O(rd/n)$ 快率上界 + 匹配下界;over-ranking 增大估计误差的 rank-selection 二分定理 | 以统计模拟验证为主 |
| **Spectral Dynamics of Low-Rank Adaptation** (ICML 2026 workshop) | 梯度流下 $\Delta W$ 奇异值演化 $\sigma_i\approx\lambda_i\tanh^2(\lambda_i t/\sqrt2)$;过参数化 LoRA 良性;免训练 rank 选择规则 | **BERT-base (GLUE)、Llama-3-8B (MT-Bench)、ViT-B/16 (CIFAR-100)**——理论文也覆盖了语言+chat+视觉三线 |

### 2. 方法向论文(ICLR 2026 主流实验口径)

| 论文 | Reasoning 实验 | Chat / 指令实验 | 视觉实验 |
|------|---------------|----------------|---------|
| **LoFT** (ICLR 2026) | **LLaMA-7B / LLaMA2-7B / LLaMA3-8B / LLaMA3.1-70B**;Commonsense-170K 联合训练 → 8 个 commonsense 基准(BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-c/e, OBQA),含极低秩 r=1,2,4 | — | **ViT-Base**:医学影像 + DomainNet(不平衡/域偏移分类) |
| **OD-LoRA** (ICLR 2026 投稿) | **Gemma-2B、LLaMA3-8B**(另用 LLaMA2-7B 对比):Commonsense-170K → 8 基准;**MetaMathQA → MATH + GSM8K**;Code-Feedback → **HumanEval** | — | **ViT-Base** 图像分类 |
| **SDS-LoRA** (arXiv 2606.16454) | **Gemma-2B、LLaMA3-8B**:Commonsense-170K → 8 基准;MetaMathQA-100K → MATH + GSM8K;Code-Feedback → HumanEval;rank-32 可追平 full FT | — | ViT 图像分类(附录) |
| **TinyLoRA: Learning to Reason in 13 Parameters** (arXiv 2602.04118) | **Qwen2.5-Instruct (3B/7B/8B)、Llama-3** + **GRPO 强化学习**;GSM8K 及 SimpleRL 难题套件:MATH500, Minerva, GAOKAO, OlympiadBench, CollegeMath, AIME24, AMC23 | (RL 后指令模型) | — |
| **mtLoRA** (ICLR 2026) | Flan-v2 → **BBH** | **Dolly-15k → MMLU**(LLama2-7B) | **ViT**:DOTA(15 任务)、iNat2018(25 任务) |
| **LoRA-S** (ICLR 2026) | — | GPT-2 medium → E2E NLG(BLEU 等) | Mix-of-show 图像生成(CLIP/FID) |

### 3. 口径统计(出现频次)

- **Reasoning backbone**:LLaMA3-8B(4 篇)> Qwen2.5 系列(2 篇)> Gemma-2B(2 篇)> LLaMA2-7B。
  本文用的 **Gemma-7B 已无人使用**;RoBERTa 只在个别理论文中作为 GLUE 载体保留(BERT-base)。
- **2026 年中(ICLR 2027 同期)更新**:最新 arXiv 论文的标准双 backbone 已演进为
  **Llama-3.1-8B + Qwen3-8B**——Spectral Surgery(arXiv 2603.03995,MetaMath/Magicoder 训练,
  CommonsenseQA/HumanEval/IFEval 评测)与 Understanding LoRA as Knowledge Memory
  (arXiv 2603.01097,Llama-3.2-1B/3.1-8B、Qwen3-1.7B/8B 尺度扫描)均采用该组合;
  数据与评测口径(MetaMath、CommonsenseQA、HumanEval、IFEval)未变。
- **Reasoning 数据**:Commonsense-170K → 8 基准是最高频组合(3 篇);MetaMathQA → GSM8K+MATH 仍是标准数学口径(2 篇,与本文一致);
  更前沿的工作已把数学评测升级为 **MATH500 / OlympiadBench / AIME / AMC** 难题套件。
- **Chat / 指令**:MT-Bench 仍在用(Spectral Dynamics 用 Llama-3-8B + MT-Bench),但单独用 Alpaca→MT-Bench 已偏弱;
  近期工作补充 **MMLU / BBH**(能力保持)与 **IFEval**(可验证指令遵循,有论文专门指出 MT-Bench 类评测与可验证指令遵循脱节)。
- **视觉**:理论与方法文的标配是 **ViT-Base/16 图像分类**——CIFAR-100、DomainNet、医学影像、DOTA/iNat2018;
  compressed LoRA 一族(VeRA/FourierFT 传统)常用 **VTAB-1k 或 CIFAR-100/Food101 等小样本分类**。本文完全没有视觉实验。
- **种子与统计**:理论验证类实验普遍 3–5 种子并报告区间;本文数学与指令实验目前**单种子(42)**,是明显短板。

---

## 二、本文实验的问题诊断

1. **Backbone 老旧**:RoBERTa(2019)与 Gemma-7B(2024)在 ICLR 2026 论文中几乎绝迹,审稿人会直接质疑结论的时效性。
2. **任务覆盖窄**:缺少 commonsense reasoning 套件(当前最高频口径)与视觉实验(理论普适性的低成本证据)。
3. **chat 评测单薄**:只有 MT-Bench 一个 judge 分数,无能力保持(MMLU)与可验证指令遵循(IFEval)维度。
4. **统计强度不足**:数学/指令实验单种子,与"理论验证"的定位不匹配。
5. **竞品理论文已抢占真实验证口径**:LoRA vs FFT 一文已经在 Qwen2.5 上做了标签噪声与样本量扫描——本文若不做同类实验(见 `real_model_theory_validation.md` 的 E1/E2),等于把最直接的理论验证阵地让给竞品;做了则可以引用该文作为口径先例。

---

## 三、改进方案(按优先级)

### P0:升级 reasoning 主实验(必做)

- **Backbone**:Gemma-7B → **Llama-3.1-8B**(社区默认)+ **Qwen3-8B**(第二 backbone;
  2026 年中论文的标准组合,见"口径统计"。若为对齐竞品理论文的噪声/样本量扫描,可在 E1/E2 中
  用 Qwen2.5 小尺度模型,与主表 backbone 解耦)。
- **数学**:保留 MetaMathQA 训练集(口径与 OD-LoRA/SDS-LoRA 一致,方便横向引用),评测从 GSM8K+MATH 扩展为
  **GSM8K + MATH500**,可选加 OlympiadBench/AMC23 作为"高难度=高能量更新"的理论压力测试——
  这正好强化正文"MATH 更接近 bias 主导端"的论述:难度谱越宽,off-subspace 能量与性能差的相关性(E4)越有说服力。
- **新增 commonsense 套件**:Commonsense-170K 联合训练 → 8 基准(BoolQ/PIQA/SIQA/HellaSwag/WinoGrande/ARC-c/e/OBQA)。
  一次训练、八个测试,性价比高,且是与 LoFT/OD-LoRA/SDS-LoRA 直接可比的表格。
- **种子**:至少 3 个,报告均值±标准差。

### P1:补充视觉实验(强烈建议)

- **ViT-B/16 + 小样本图像分类**(VTAB-1k 子集,或 CIFAR-100 / DomainNet 少样本切分)。
- 对本文尤其有利:视觉小样本天然处于大 $\sigma_{\mathrm{eff}}^2$ 区,理论预测压缩方法(Uni-LoRA/ProLoSA)优势显著;
  再配一个全量数据点即可展示 crossover 的跨模态普适性——把 E1(数据量扫描)顺带在视觉上以 2–3 个点复现,
  比在 NLP 上重跑一整条曲线便宜得多。
- compressed LoRA 基线(VeRA、FourierFT)本身有 ViT 结果可引用,方便凑齐对照表。

### P2:升级 chat / 指令评测

- 训练集可保留 Cleaned Alpaca(成本不变),评测从单一 MT-Bench 扩展为:
  **MT-Bench + IFEval(可验证指令遵循)+ MMLU(能力保持/遗忘)**。
  三个维度分别对应理论中的"适配质量 / 更新是否落在必要方向 / 压缩是否因低方差而更少扰动原能力"——
  最后一点可作为新卖点:理论预测压缩方法方差小,遗忘(MMLU 掉点)应更少。
- 若预算允许,升级训练集为 UltraChat/Tulu-3 子集并加 **AlpacaEval 2.0(length-controlled)**;预算不允许则不必强求。

### P3:GLUE 的处理

- **不建议删除**:GLUE 是与 Uni-LoRA/VeRA/VB-LoRA/LoRA-XS 等 compressed 基线唯一直接可比的口径,且是 E1/E2/E3 理论验证实验的载体(小任务多种子成本低)。
- 建议**压缩其正文篇幅**(保留 RoBERTa-large 主表,base 表移附录),把腾出的空间给 commonsense/视觉新表;
  正文定位改为"与 compressed LoRA 文献的可比性 + 理论验证平台",而非主要战场。

### P4:理论验证实验与竞品对齐(与 `real_model_theory_validation.md` 联动)

- LoRA vs FFT 理论文已在 Qwen2.5 上做了**标签噪声扫描 + 样本量扫描 + $\Delta^\*$ 逐层能量谱**,
  与本文档 E1/E2/E5 几乎一一对应。建议:
  1. E1/E2 至少各做一个 Qwen2.5-7B(或 1.5B,更便宜)配置,可直接引用该文作为实验口径先例,
     并强调本文的差异化:对比的是 **compressed LoRA vs 标准 LoRA**(他们对比 LoRA vs FFT),
     且本文有 energy–noise 阈值的定量预测,而不只是定性趋势;
  2. E5(能量谱重尾性)可仿照其"逐层奇异值累积能量"的展示方式,但统计对象换成
     off-subspace 坐标能量,与 Proposition 1 的 $v$ 直接挂钩。

### 成本估算(单卡 L20 口径,粗略)

| 项目 | 训练次数 | 备注 |
|------|---------|------|
| P0 数学(2 backbone × 3 种子) | 6 次 MetaMathQA 微调 | 每次约 1–2 天(QLoRA 可再降) |
| P0 commonsense(2 backbone × 3 方法 × 3 种子) | 18 次 | Commonsense-170K 比 MetaMathQA 便宜 |
| P1 视觉(ViT-B,3 方法 × 3 数据集 × 3 种子) | 27 次 | 单次分钟~小时级,总成本很低 |
| P2 chat 评测扩展 | 0 次新训练 | 复用已有 checkpoint 补 IFEval/MMLU 评测 |

若算力紧张,最小可行组合为:**P0 只上 Llama-3.1-8B(3 种子)+ P1 视觉 + P2 评测扩展**;
commonsense 套件与 Qwen3-8B 第二 backbone 可作为 rebuttal 期间的补充弹药。

---

## 四、写作层面的连带修改

1. 实验设置小节(Sec 5.1)与附录 B 超参表按新 backbone 更新;
2. 正文"MATH 更接近 bias 主导端"的论述在新的难度谱(GSM8K→MATH500→Olympiad)下重写,理论联系更连续;
3. Related Work 增引上表中的理论文(尤其 LoRA vs FFT 与 sample complexity 两篇),
   明确本文定位差异:**他们解释 LoRA 相对 FFT 的收益,本文解释 compressed LoRA 相对 LoRA 的收益与失效条件**——
   这两层是互补的,引用它们反而强化本文的独特性;
4. 若加入视觉实验,摘要与结论中的 "GLUE, mathematical reasoning, and instruction tuning"
   更新为包含 vision 的表述。
