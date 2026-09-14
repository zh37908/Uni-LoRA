# Uni-LoRA / ProLoSA：双服务器两周微调补实验方案

制定日期：2026-09-14；执行窗口：2026-09-14 至 2026-09-27（香港时间）。

依据：当前 47 页论文 `iclr2027_conference.pdf`、仓库训练/评测脚本及结果、当前服务器 Slurm 查询、用户提供的另一服务器两张资源截图。本文是实验设计文档，尚未提交训练作业。另一服务器的信息来自截图，本次没有远程登录复核。

**推荐交付：Llama-3.1-8B 数学确认 + Qwen3-8B-Base 指令微调，两组各 3 方法 × 3 种子，共 18 次正式训练、12 次短调参。** 现服务器继续完成 GLUE/机制证据；新服务器优先承担指令实验。若新服务器稳定获得更多卡，再补 Qwen3-8B-Base 数学的完整 9 次训练。最后 3 天用于分析和写作。

本方案更新了 [原两周方案](two_week_experiment_plan_20260914.md) 的微调部分：新增算力后，现代指令微调从低优先级升为推荐必做项；原方案的公平 GLUE 与已完成合成实验分析继续保留。

## 1. 应优先补什么

| 当前论文的证据 | 实际缺口 | 本轮补充 |
|---|---|---|
| PDF 第 7–8、44 页：Llama-3.1-8B 数学已有结果 | seed 42；压缩配置按测试成绩事后汇总；MATH500 的 0.4 个百分点仅对应 2 题 | 独立内部验证选参，锁定后 3 个新种子，同时报告两个测试集 |
| 第 44 页 Table 22：Llama2 + Alpaca + MT-Bench | 指令模型/训练配方较旧，主要依赖 judge 分数 | Qwen3-8B-Base + SmolTalk 子集，IFEval + IFBench，全套三方法三种子 |
| 已有 GLUE 与实模型偏差/方差诊断 | 公平调优正在运行，需完成和整理 | 不迁移运行中协议；继续原服务器收尾 |
| 新的同功能位移合成对照 | 480/480 已完成，尚需转化为论文图表 | 不重跑网格，将算力用于真实微调 |
| 第 8 页 Table 3 已有视觉 | 并非“没有视觉实验” | 本轮不再扩充视觉套件 |

2026-09-14 本次读取时，公平 GLUE 已完成 **81/192 调参、0/40 最终训练**；合成对照 **480/480**。这些是进度快照，不能当作未来每天的状态。

本文的中心是压缩估计的偏差—方差权衡。新微调表用于检验实际收益、稳定性和失败边界；IFEval 分数上升本身不证明方差降低，困难数学题也不直接等于更大的投影偏差。近期 [Beyond LoRA（2026-06）](https://arxiv.org/html/2606.13767v1) 同样研究结构受限适配及泛化；对本文而言，严格的对照和成本口径比单纯增加任务数量更有价值。这是本方案的判断，不是该文替本文作出的结论。

## 2. 两台服务器如何分工

### 2.1 现服务器：保住已经投入的实验

账户 `shsong`、用户 `hzhaobi`。以下个人限制已在前一轮用 Slurm 核实；空闲卡随时间变化。

| 分区 | 每节点卡数 | 同时运行作业上限 | 未结束作业上限（含运行/排队/依赖） | 个人 GPU 上限 | 分工 |
|---|---:|---:|---:|---:|---|
| gpu-rtx5880 | 6 | 8 | 10 | 12 | 数学确认，目标 3 个单卡 worker |
| gpu-l20 | 4 | 2 | 4 | 8 | 已有公平 GLUE 两个 worker |
| gpu-rtx4090d | 6 | 2 | 4 | 2 | 数据/实现检查、能装下的评测 |
| gpu-a30 | 4 | 8 | 10 | 16 | 小模型/GLUE 备用 |

单作业时限均为 72h。现有数学网格还有调度器和运行作业，新增 3 个 worker 是资源目标，不代表已分配；调度时必须合计既有作业及调度器将继续提交的任务。已有试验不重复提交，正在运行的 GLUE 不修改环境和协议。

### 2.2 新服务器：资源证据及限制

截图中的账户为 `songust`，用户为 `hzhaobi`。

| 项目 | 截图显示 | 本方案采用的解释 |
|---|---|---|
| normal | 23 节点、每节点 8 卡，快照未分配 57 卡 | 是分区资源快照，不是个人可立即取得的 57 卡 |
| preempt | 22 节点、每节点 8 卡，快照未分配 9 卡 | 仅作可中断的额外资源；不把两个分区物理资源相加作保底 |
| normal_qos / preempt_qos | 各自每用户同时运行 8 作业、未结束 10 作业、72h | 作业上限不是 GPU 上限，也不是节点上限 |
| 账户共享上限 | 各上述 QoS 下账户合计 96 GPU、2688 CPU | 团队共享额度，不是个人独占额度 |
| normal_debug_qos | 1 作业、2h，个人至多 2 GPU/56 CPU | 只用于短检查，不承载正式训练 |
| 已有任务 | 575888 使用 dgx-05 的 4 卡；另有依赖任务 575890 | 是截图时状态；不能把这 4 张已占用卡计为新增资源 |
| 已核实硬件 | dgx-05 作业内可见的 4 卡均为 NVIDIA H800，81559 MiB/卡（80GB 级） | 仅这些卡型号经过核实 |
| 其余 normal/preempt 节点 | Slurm 记录每节点 8 卡，型号和显存未核实 | 分配到后在作业内运行 nvidia-smi，再决定能否承载正式实验 |

截图中的空闲整节点 dgx-23、dgx-46 可以作为排队时的候选资源线索，**不绑定它们，也不称其硬件已确认为 H800**。隐藏分区 dgx-07/19/32/45 的卡数分别为 1/4/7/0，不纳入本方案。

**推荐资源目标：新服务器 normal 分区稳定 4 张经核实的 80GB 级 GPU，4 个单卡独立任务；旧服务器 3 张 RTX5880 跑数学。** 新服务器剩余名额用于评测和重跑。若截图中的既有 1 个运行、1 个依赖任务仍在，新增 4 个单卡作业合计为 5 个运行、6 个未结束，低于截图所示 8/10 上限；仍受账户共享配额和排队约束。

不默认需要 8 卡分布式训练一个 8B 模型。优先并行方法和种子。单卡 BF16 + gradient checkpointing 能否容纳实际 ProLoSA 实现，须用完整模型检查，不能仅按有效适配参数量推算显存。

### 2.3 第一天的资源验收

在新服务器读取 `sinfo -N`、`scontrol show node`、用户 `squeue`、`sacctmgr show assoc` 和 `show qos`，刷新截图信息；在合法分配的作业内执行：

```bash
nvidia-smi --query-gpu=name,memory.total,uuid --format=csv
```

对实际使用的模型/方法运行至少 200 optimizer steps，覆盖 ProLoSA 支持激活阶段，记录激活前后速度、峰值显存、主机内存及 checkpoint 时间。用训练数据中的固定样本检查，不查看正式测试集来调配置。若单卡 OOM，先统一降低该实验组三方法的 microbatch、提高 accumulation，保持 global batch；仍不可行时整组改为相同的双卡训练布局并重新估时，不只给某一种方法换精度或量化。

本方案尚无 H800 实测速率。后文时间为容量规划假设，第 2 天结束前必须替换为实测预测。

## 3. 实验总矩阵与取舍

| 编号 | 优先级 | Backbone | 训练数据 | 方法 | 正式种子 | 正式训练次数 | 短调参数 | 主要评测 |
|---|---|---|---|---|---|---:|---:|---|
| M1 | 必做 | NousResearch/Meta-Llama-3.1-8B，当前使用的 base 镜像 | MetaMathQA 现有 100K 池，约 98K train / 2K val | LoRA r4 / Uni-LoRA / ProLoSA | 301、302、303 | 9 | 6 | GSM8K、MATH500 |
| I1 | 推荐必做 | **Qwen/Qwen3-8B-Base** | SmolTalk 分层抽样 100K，约 98K train / 2K val | 同上 | 301、302、303 | 9 | 6 | IFEval、IFBench、独立 SFT 测试 NLL |
| M2 | 有余量再做 | **Qwen/Qwen3-8B-Base** | 与 M1 相同的题目清单和划分 | 同上 | 301、302、303 | 9 | 6 | GSM8K、MATH500 |
| E1 | 随训练附带 | M1 / I1 | 同一固定训练片段 | 同上 | 固定一个 seed 做 profile | 0 次额外完整训练 | — | 显存、有效/实际存储、时间、tokens/s |

推荐版：**18 次完整训练 + 12 次 1/4 预算 pilot**，另含基座评测、正式 checkpoint 评测和短 smoke/profile；不能只报 18 次而漏算调参与评测成本。

扩展版：加 M2，共 **27 次完整训练 + 18 次短 pilot**。M2 使数学比较覆盖两个基座，并在 Qwen 同一基座上覆盖数学/指令两类适配。仅 M1+I1 时，任务和 backbone 同时变化，不能把跨表差异归因于单独的 backbone 或任务因素。

两周内不新增 32B/70B、全参数微调、GRPO、全量百万级 SFT、代码执行评测套件或大规模 rank/比例扫描。已有 VB-LoRA/VeRA/FourierFT 和 GLUE RoSA 先整理；如没有完整匹配协议，不塞进新三种子主表冒充公平对照。

## 4. 三方法共同协议

### 4.1 参数预算与实现

| 方法 | 设置 | Llama-3.1-8B 适配坐标 | Qwen3-8B-Base 适配坐标 |
|---|---|---:|---:|
| LoRA | r=4，所有 q/k/v/o/gate/up/down projection | 10,485,760 | 预计 10,911,744，须运行时核验 |
| Uni-LoRA | 同 r、同目标模块，压缩向量长度 B | 1,048,576 | 1,048,576 |
| ProLoSA | 同 r、同目标模块，d+K=B，约 12:1 | d=967,916，K=80,660 | d=967,916，K=80,660 |

Qwen 数量按官方 [config.json](https://huggingface.co/Qwen/Qwen3-8B-Base/blob/main/config.json) 的 36 层及投影形状计算，最终以实际注入模块/参数审计为准。三方法都冻结 embedding、LM head、norm 和基座权重，记录任何例外。

本次复核发现旧两周方案建议的整数拆分为 967,917/80,659，和现有脚本 12:1 分支差一个坐标。**本方案统一采用已实现的 967,916/80,660**，两者之和严格为 1,048,576；旧运行保留其原始配置，不追溯改写。

等预算主要比较 Uni-LoRA 与 ProLoSA；LoRA 是更大适配容量的参照，不称三方法等参数。LoRA 使用当前实现的 alpha=r、dropout=0；压缩方法保留其自身映射/缩放定义并保存实参，不能因同 rank 就假设算子完全一致。

ProLoSA 固定 warmup=128 optimizer steps、scoring=1 step、稀疏学习率 multiplier=0.2，继承当前 reset/activation 设置并写入锁定配置。不扫描比例、warmup 或 scoring 长度；这也是方法适用范围的限制。12:1 受历史探索启发，需如实披露。

### 4.2 数据、选参和统计

- 每组先固定训练/内部验证划分，再开始 pilot。分组种子用 20260914，pilot seed=101，正式 seed=301/302/303。正式种子控制初始化、数据顺序、压缩映射/支持选择所需随机性，具体随机流写入 manifest。
- 每方法 2 个 LR 候选，各跑约 1/4 正式训练预算，只按内部验证的 assistant/answer-token 平均 NLL 选择，越低越好；差值小于 1e-4 时选较小 LR。用未四舍五入值判定。
- pilot 使用独立的短训练调度；三方法同预算。它是有限搜索，不保证选出完整训练下最优 LR。不得根据正式测试结果补开单方法搜索。
- 选定后从原始基座重新训练 3 个种子，2 epochs，统一用最终 checkpoint。旧 checkpoint 若已见过新划出的 val，不用来进行“无泄漏”选参。
- 同一实验组、同一个 seed 的三方法共用完全相同的数据清单、顺序和 batch 划分。训练/监督 token 总数一致；新增 SNIP 评分访问量和时间单列。
- 正式测试每个 checkpoint 一套冻结解码协议。报告每个 seed 的分数、mean±sample std、ProLoSA−Uni-LoRA 配对差值。3 个种子统计能力有限，不据极小均值差宣称显著优势。
- 逐题 bootstrap 保留同题方法配对；多约束指令以 prompt 为重采样单元。该区间主要表达测试样本不确定性，不能替代训练种子方差，也不能把 3 个种子的重复题目当独立样本扩大样本量。
- 历史测试集已经影响方法开发，新 seeds 不会消除这种适应性偏差；新流程提供前瞻冻结的稳定性证据。新指令测试同样不宣称可排除未知预训练污染。

## 5. M1：数学确认实验

**目的：解决当前主表单种子和测试集选配置的问题，检验 MATH500 的小改进是否稳定。**

- 基座：保持当前论文/脚本使用的 `NousResearch/Meta-Llama-3.1-8B`，固定 revision 和权重哈希。若换官方来源，先确认权重一致，否则单独标注。
- 数据：`meta-math/MetaMathQA` 的现有 100K 池。先按原始题目 ID 或可复核的改写组分组，约 98K fit / 2K val；各组不跨集合。没有可靠原题 ID 时，记录去重/分组规则及剩余限制，不声称实现完美语义去重。
- 训练：BF16、无量化、gradient checkpointing、最大长度 512、global batch=64、2 epochs、cosine、scheduler warmup ratio=0.02、weight decay=0。最大长度保持历史口径；记录答案被截断比例，不能在新表里静默换长上下文后直接宣称方法进步。
- microbatch 从 1 开始，accumulation=64；同组如统一提高 microbatch，正式开跑前冻结，并保留 token 顺序/等效 batch。
- LR 候选：LoRA `{1e-4,2e-4}`；Uni `{1e-3,2e-3}`；ProLoSA theta LR `{1.6e-3,3.2e-3}`，base LR 固定 `2e-4`，其余按共同协议。
- 测试：GSM8K 全部 1319 题、MATH500 全部 500 题；沿用已验证 prompt、答案抽取，greedy 解码。最大新 token 分别 1024、2048，记录达到上限的比例；两者分开报告，不用平均分选择配置。
- 同协议评测未微调基座作为参照。旧 rank64 LoRA 单种子结果单独标作历史高容量结果，不拼入新三种子统计。

产物：一张 3 方法 × 2 benchmark 表、逐种子附表、逐题输出及配对差值、预算/训练成本表。即使 ProLoSA 不胜出，完整结果仍可用于本文的任务依赖性边界。

## 6. I1：现代指令微调，直接补强 MT-Bench

### 6.1 基座与训练集

用 [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base)，官方卡标注为预训练基座。仓库旧脚本 `submit_train_p0_math_llama31_qwen3_1gpu.sh` 默认值是 **Qwen/Qwen3-8B（后训练模型）**，必须显式覆盖，输出名称包含 `qwen3_8b_base`；历史 Qwen3-8B adapter 不直接复用于这组。

选择 [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) 的 `all/train`，按 source 比例分层抽取 100K 对话，分组后约 98K fit / 2K val。该数据已有 `messages` 与 `source`，包含通用对话、改写、摘要、约束等来源；本方案只用固定子集，不声称复现全量 SmolTalk 训练配方。[官方数据说明](https://huggingface.co/datasets/HuggingFaceTB/smoltalk/blob/main/README.md)

处理顺序：固定 revision → 对话/首轮问题去重并分组 → 按预定义规则排除与 IFEval/IFBench 测试 prompt 的近重复 → 固定 100K 样本及 fit/val → tokenize。去重只使用 prompt 重合信息，不使用测试得分或参考回答选样本。数据不足精确 100K 时记录实际数量，不拆组凑数。

保留各 source 的样本数。`smol-constraints` 包含指令约束训练内容，故 IFEval 不能被称为“约束类型完全未见”的测试；IFBench 是补充性的外部约束泛化评测，也不保证与该混合训练集在语义上完全无重合。

另从官方 `all/test` 按固定 seed 取 2K 对话，先对本轮训练/验证去重，作为**独立 SFT 测试 NLL**，不用于 LR 选择。这个 NLL 与 IFEval 的外部能力指标分开解释。

### 6.2 训练细节与工程工作量

- BF16、无量化、gradient checkpointing、最大长度 2048、global batch=64、2 epochs、cosine、2% scheduler warmup、weight decay=0。单卡 microbatch=1 起步；pilot 后可整组固定为 2 或 4 并相应减少 accumulation。
- 保留角色边界、多轮上下文；只对 assistant 内容及必要结束符监督，system/user/padding 全部 mask。过长对话优先在完整轮次边界截断；仍装不下的长首轮按统一规则截断，记录部分回答及无监督 token 样本的比例，后者排除并记录。
- 默认不引入跨对话 packing，以减少首轮迁移工作；如第 1 天决定使用，必须保证 loss mask 和隔离规则正确，并对三方法一致冻结。
- LR 候选：LoRA `{1e-4,2e-4}`；Uni `{5e-4,1e-3}`；ProLoSA theta LR `{8e-4,1.6e-3}`、base LR 固定 `1e-4`。这些是两周预算下的候选设计，不是已验证最优值。
- 复用现有 math 路径的三个 tuner，新增/核验对话数据 collator、独立验证集选择及新服务器 launcher。旧 `instruction_tuning/qlora*.py` 主要面向 Alpaca 等格式，不能认为换一个 dataset 名称就完成适配。
- Qwen 官方 tokenizer 当前含聊天模板；显式冻结模板文件和 `enable_thinking=False` 的序列化结果。检查训练完整 assistant 轮次与推理 assistant 前缀一致；模板中的空 think 标记属于已固定格式，不等于对 Base 模型的推理能力作保证。[官方 tokenizer 配置](https://huggingface.co/Qwen/Qwen3-8B-Base/blob/main/tokenizer_config.json)
- 固定 `im_end` / EOS 停止规则。审计真实 token IDs 和 10 个序列化样例，不直接使用旧 Llama2 的默认模板。已有 IFEval 脚本未显式指定这些设置，需要适配后才能作正式评测。

预留 **1–2 天工程与 smoke**：三方法梯度正常、有效参数计数、支持激活、保存/重载、合并前后 logits 一致性、续训恢复 optimizer/scheduler/RNG/稀疏支持、数据 mask 与模板正确。小模型通过后，仍需新服务器完整 8B 检查。

### 6.3 评测与 MT-Bench 的位置

| 项目 | 地位 | 冻结口径 | 回答什么 |
|---|---|---|---|
| IFEval | 主指标 | prompt-level strict accuracy；附录列 instruction-level strict 和 loose 指标 | 可程序核验的指令执行 |
| IFBench | 预定次指标 | 全部官方单轮测试，prompt-level loose accuracy 为主，strict 为补充 | 新的、更具挑战的外部约束泛化 |
| 独立 SFT test NLL | 辅助 | 2K 固定 test 对话，assistant-token NLL | 同分布生成拟合的补充诊断 |
| 未微调 Qwen Base | 参照行 | 与微调模型同模板、同解码、同评测 | 微调前后的变化 |
| 历史 MT-Bench | 附录保留 | 保留原 Llama2/Alpaca/judge 配置 | 与原 Uni-LoRA 实验衔接 |

IFEval 本身也是 2023 年的基准，选它主要为补充可验证指标，**不能把“换成 IFEval”写成单纯追新**。[IFEval 官方实现](https://github.com/google-research/google-research/tree/master/instruction_following_eval) IFBench 发布于 2025 年，论文提出 58 类新约束；本轮固定单轮版本，不再加多轮隔离训练/评测。[IFBench 论文](https://arxiv.org/abs/2507.02833)、[官方评测说明](https://github.com/allenai/IFBench)

所有正式种子使用 greedy/temperature=0，max_new_tokens=2048，至少容纳完整输入加输出预算的 context（起始设置 8192）；不截短测试 prompt。超过上下文的输入按统一预先声明的规则扩容并记录，不静默丢题。只截取模型新生成的回答并按冻结模板处理边界 token，不按约束内容事后删标点/改格式。记录空回答、EOS/长度截断和 verifier 异常；基础设施异常可重跑同一配置，不根据分数挑输出。

锁定官方评测数据和代码 commit。若 IFBench 在第 2 天前无法完成可信的 verifier 检查，整组降为可选、保留 IFEval 主指标并在计划版本中记录；不能看过低分再取消。两套程序判分都不依赖付费 judge。

**对“MT-Bench 太旧了吗”的结论：仍有历史对照价值，但不适合作为新稿唯一的指令证据。** 本轮优先完成三种子 IFEval/IFBench，不安排依赖外部 judge 的 Arena-Hard/AlpacaEval 大批评测；新表与历史 MT-Bench 分数不作直接数值比较。IFEval/IFBench 也不覆盖通用对话质量的全部维度。

## 7. M2：余量优先用于同一 Qwen 基座的数学复验

只在第 5 天决策时启动，条件同时满足：I1 数据/实现已冻结并完成至少一个完整的方法三元组；M1 进度按时；新服务器剩余承诺容量覆盖 M2 预算且另留至少 25% 余量；预测第 9 天前完成训练。

- 直接复用 M1 的题目清单、512 长度、训练 token 规则及 GSM8K/MATH500 解码预算，改为 Qwen3-8B-Base。不同 tokenizer 的 token 数可能不同，记录后不称跨 backbone 严格等 token。
- 三方法 × 两 LR，6 个短 pilot；采用 I1 的 Qwen LR 候选范围作为预先指定的起点，按数学内部验证 NLL 独立选择。三方法各 3 新种子，9 次完整训练。
- 数学 prompt 使用同一已冻结文本规范；各方法同 tokenizer，不从指令 SFT checkpoint 继续训，而从同一个 Qwen Base 重新开始。
- 未满足条件就取消整个扩展组。若已启动，完整性优先，失败格子按原协议补齐，不只保留最好一个 seed。

不同时再开 Qwen 7 档数据量扫描或另一个 code/commonsense 方向。若主要审稿风险已变为支持选择而非跨基座泛化，可在正式启动 M2 前修订计划，换成有完整对照的机制消融；两组不并列为两周必做。

## 8. 容量估算：按 GPU 小时和依赖关系共同判断

### 8.1 推荐版预算

| 工作 | 规划 GPU 小时 | 依据/限制 |
|---|---:|---|
| M1：9 正式 + 6 短 pilot + 评测/重跑余量 | **180–250 RTX5880 GPU·h** | 历史混合配置单卡 Llama Uni/Pro 常约 13–19h；不是新协议的保证时长 |
| I1：9 正式 + 6 短 pilot + 全套评测/检查/余量 | **120–350 80GB GPU·h** | 假设单次正式 8–24h；包含约 12–36h pilot、15–35h 评测及约 20% 检查/失败余量；须 H800 实测校正 |
| M2：额外 9 正式 + 6 短 pilot + 评测/余量 | **130–270 80GB GPU·h** | 保守规划假设；不由 H800 型号推导固定加速倍数 |
| 既有 GLUE/合成分析 | 单独列账 | 继续原计划，不隐含在上述新增微调预算中 |

两类 GPU 小时不按等效算力相加。I1 从 512 增至 2048 长度且更换数据格式，不能简单套数学历史每 run 时间。指标必须使用实际非 padding token 吞吐，ProLoSA 激活前后分别计时。

对方法 m，预测 `单 run 时间 ≈ 预处理/加载 + 激活前步数×前段秒/步 + 激活后步数×后段秒/步 + 验证/存盘`；生成评测另计。最后用方法最慢阶段、pilot→选参→正式训练的依赖关系排程，不能只用总 GPU·h 除以卡数。

### 8.2 三档可完成范围

统一保守按每张卡每天实际获得 **18h 有效窗口**，D3–D9 共 7 天安排正式训练，D10–D11 补测/失败格；D12–D14 不预占训练容量。

| 稳定新增资源 | 新服务器 D3–D9 容量 | 两周范围 | 触发条件 |
|---|---:|---|---|
| 2 张 80GB 级卡 | 252 GPU·h | M1 保留在旧服务器；I1 采用保底规模 | 第 2 天若 I1 上限预测超过可用容量，三方法统一改为预先分组抽样的 50K 池、约 48K fit / 2K val，仍 2 epochs、3 seeds；整组重新冻结，pilot 也使用此规模 |
| **4 张 80GB 级卡（推荐）** | **504 GPU·h** | **M1 + I1 100K，18 次完整训练** | I1 含余量上限 350h，保留排队/串行依赖缓冲；旧服务器实际取得约 3 张数学卡 |
| 6–8 张 80GB 级卡 | 756–1008 GPU·h | M1 + I1 + M2，27 次完整训练 | 第 5 天通过 M2 门槛；实际作业安排仍须满足每用户 8/10 限制及既有任务占用 |

8 张 GPU 不要求 8 个独立 Slurm 作业；如既有任务占名额，可在一个正常申请的多 GPU 作业内运行隔离的单卡 worker，但 CPU/内存/GPU 总量仍完整申请和记账，并先检查集群支持此布局。两周方案不依赖这种组合提交才能成立。

若旧服务器没有数学余量，则将 M1 未启动的完整三方法/种子块转到新服务器。此时 4 张卡的推荐版总需求可能超过 D3–D9 的 504h：先取消 M2，再按第 2 天冻结的 50K 指令保底分支执行，必要时降为 M1 完整交付、I1 后续工作；不能同时假设“旧服务器不可用”和“仍无条件完成两组全量”。同一个配对三元组尽量用相同硬件；硬件迁移需记录并单列成本。

任何 50K 降级都由资源、时长或数据工程原因触发，在正式测试前决定；不得因 100K 测试效果不佳改用 50K 挑结果。预训练权重获取或环境问题若第 2 天仍未解决，停止新增方向，保证 M1 和现有证据的完整性。

## 9. 两周排程与作业管理

| 时间 | 旧服务器 | 新服务器 | 当天验收 |
|---|---|---|---|
| D1–D2，9/14–15 | GLUE 持续；冻结 M1 数据与短 pilot；整理合成图 | 核实硬件/队列，迁移最小代码和权重，完成 I1 数据/模板/保存恢复检查；短 pilot | 模型/数据哈希、三方法 smoke、实测时长；选 100K 或保底分支 |
| D3–D4，9/16–17 | M1 正式训练，先完成 seed301 三方法再推进后续 seed | I1 锁定 LR，开始正式三种子；预先缓存 verifier 依赖 | 第一组三方法训练/完整性可核验；流水生成评测，不据分数改配置 |
| D5，9/18 | 核实 M1 剩余工时和 GLUE 状态 | 决定是否启动 M2；未达门槛则专注 I1 | 可复核的余量计算和固定任务清单 |
| D6–D9，9/19–22 | 完成 M1，补已有基线输出/成本记录 | 完成 I1；可选 M2；评测随 checkpoint 到达开始 | 第 9 天停止新增实验方向 |
| D10–D11，9/23–24 | 只补失败/缺失格、汇总 GLUE | 只补失败/缺失格、核对所有评测及输出完整性 | 冻结数值、统计表、失败说明 |
| D12–D14，9/25–27 | 分析、图表、写作、编译 | 同步轻量结果及必要 checkpoint，保留缓冲 | 正文表/附录协议/成本表与机器可读结果一致 |

正式任务优先使用 normal；preempt 用于可恢复的额外任务或生成评测，不计为保底。先验证抢占恢复后，再把正式 run 放到 preempt；若恢复只能加载 adapter 而缺 optimizer/支持状态，不能称其为连续同一次训练。

采用小批分波提交，计算未结束数组元素和依赖任务的数量；不要一次提交超过 10 个未结束任务。单作业申请不超过 72h，按周期保存恢复状态，至少保存最终 adapter、一个可恢复训练 checkpoint、manifest 和训练轨迹。两个服务器使用唯一 run_id 和任务归属清单，避免重复跑相同配置。

新服务器先同步版本化代码及配置、必要模型和数据缓存；不要复制整个含数百 checkpoint 的工作目录。每轮回传 manifest、日志、指标和逐题 JSONL；大型合并权重在本地评测后按既定存储策略处理，原始 adapter/恢复状态按实验归档规则保留。

## 10. 必须保存的文件与论文呈现

建议单独建立执行目录（以下是将来执行时的产物，不表示本次已创建）：

```text
ICLR_2027/revision/two_server_ft/
  protocol.md
  environment.lock.txt
  resources/                     # 两服务器快照和完整模型 profile
  manifests/                     # model/data/code revision、样本划分和哈希
  locked_configs/                # 每组每方法最终实参及选参依据
  pilot_results.csv
  runs/<experiment>/<method>/<seed>/
    run_manifest.json
    trainer_state.json
    resource_usage.json
    eval_predictions/*.jsonl
    metrics.json
  tables/                        # per_seed、aggregate、paired_delta、cost
```

每个 run_manifest 至少记录：服务器/Slurm job/GPU 型号、模型 ID/revision、代码 SHA 与本地补丁、实际 PEFT import 路径、数据及划分哈希、模板/评测代码版本、方法/rank/d/K/目标模块、各组 LR/缩放/支持选择/初始化规则、所有随机种子、batch/accumulation/精度/token 数、checkpoint 选择规则、恢复历史、评测解码和完整性。

效率表同时报告有效适配坐标、实际可训练张量和优化器状态存储、冻结映射/索引、adapter 文件体积、peak GPU memory、墙钟/GPU 小时和调参总成本。稠密掩码实现不能用 K 个有效坐标代替实际稠密存储；H800 与 RTX5880 日志不能直接做方法速度排名。

论文最终放置：

1. 正文保留公平 GLUE 和机制主图；加入 M1 三种子数学表。
2. 用 I1 的三种子 IFEval/IFBench 表补充指令证据；历史 MT-Bench 留附录并标明旧模型/训练/judge。
3. 若 M2 完整完成，数学表按 backbone 分块；缺失则不以单 seed 替代完整表。
4. 附录给出所有 seed、有限搜索空间、数据处理、模板、预算、运行失败及完整成本。验证损失不同于人口投影偏差，不将其直接代入理论阈值。

**验收以预定对照完整、选择流程可复核、两周内按时写入论文为准，不以 ProLoSA 必须胜过 Uni-LoRA 为准。**
