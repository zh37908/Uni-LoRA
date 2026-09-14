# 新实验总览(按 experiment_modernization_survey.md 的 P0–P4 改进方案)

本文档记录 2026-08 新增的五组实验的入口、预算口径与提交命令。
所有新实验共用 conda 环境 **`unilora_modern`**(Python 3.11 + 新版 torch/transformers/vllm,
editable 安装 `math_instruction_tuning/peft`;老环境 `math_instruction_tuning` 的
transformers 4.40 / vllm 0.4 不支持 Llama-3.1 与 Qwen3)。

统一方法与预算口径(与论文主表一致):

| 方法 | 可训练参数 | 8B/1.5B LLM 配置 | ViT-B 配置 |
|------|-----------|------------------|-----------|
| LoRA | 全空间 D(r=4, all-linear / q,v) | lr 2e-4 | lr 1e-3 |
| Uni-LoRA | d = 524288 | lr 2e-3 | d=24600, lr 4e-3 |
| ProLoSA (unilora_rosa_snip) | θ_d + K 恒等于 524288(比例待每任务扫描确定) | base 2e-4, θ_d 8e-4, sparse mult 0.2 | θ_d + K 恒等于 24600 |

**两阶段流程(2026-08-28 起)**:
阶段1 用单种子(42)跑三方法,并对 ProLoSA 扫描 θ_d:sparse ∈ {2:1, 4:1, 8:1, 12:1}
(旧 Gemma-7B 数学扫描:12:1 最优 GSM8K 74.99/MATH 29.92,8:1 次之,2:1 最差,新任务需重扫);
每任务确定最优比例后,阶段2 只对 lora/unilora/prolosa(best) 补种子 43/44。
`meta-llama/Llama-3.1-8B` 是 gated 仓库,默认使用同权重镜像
`NousResearch/Meta-Llama-3.1-8B`,可用 `LLAMA31_BASE_MODEL` 覆盖。

ProLoSA 比例表(θ_d + sparse,LLM 预算 524288 / ViT 预算 24600):

| 比例 | LLM θ_d + sparse | ViT θ_d + sparse |
|------|------------------|------------------|
| 2:1  | 349525 + 174763 | 16400 + 8200 |
| 4:1  | 419430 + 104858 | 19680 + 4920 |
| 8:1  | 466034 + 58254  | 21867 + 2733 |
| 12:1 | 483958 + 40330  | 22708 + 1892 |

分区与资源:所有脚本默认 `gpu-rtx5880`(8 节点 × 6 卡 48GB)。**全部作业单卡**
(8B 模型 + 梯度检查点 + bf16 单卡即够,与旧 Gemma-7B 单卡扫描同配置;vLLM 评测
tensor_parallel=1)。若 5880 排队变长,可提交时覆盖 `--partition=gpu-l20`。

---

## 实验 A(P0 数学升级)

Llama-3.1-8B + Qwen3-8B,MetaMathQA-100K(2 epochs)→ GSM8K + MATH500(可选完整 MATH)。

- 目录:`math_instruction_tuning/`
- 数据:`data/prepare_math500.py` 已生成 `data/math_eval/MATH500_test.jsonl`(500 条)
- 训练:`sbatch submit_train_p0_math_llama31_qwen3_1gpu.sh`(阶段1 array 0–11,单卡)
- 评测:`sbatch submit_eval_p0_math_llama31_qwen3_1gpu.sh`(同映射,merge→vLLM tp=1,评完默认删 merged)
- 汇总:`python summarize_p0_math.py`
- TASK_ID 映射:`backbone(0=llama31,1=qwen3)*6 + config(0=lora,1=unilora,2..5=prolosa 2/4/8/12:1)`
- 阶段2(定最优比例 best_cfg 后):`SEED=43 sbatch --array=0,1,<best>,6,7,<6+best> ...`(44 同理)
- 输出目录带种子后缀:`output/p0_math/<model>_<method>_s<seed>`

## 实验 B(P0 常识推理套件)

同 2 backbone,Commonsense-170K(LLM-Adapters,3 epochs)→ 8 基准
(BoolQ/PIQA/SIQA/HellaSwag/WinoGrande/ARC-e/ARC-c/OBQA),与 LoFT/OD-LoRA/SDS-LoRA 直接可比。

- 目录:`math_instruction_tuning/`
- 数据:`data/commonsense/`(训练集 + 8 个 test.json 已下载;curl 自 LLM-Adapters 仓库)
- 训练:`sbatch submit_train_p0_commonsense_llama31_qwen3_1gpu.sh`(阶段1 array 0–11,单卡,映射同 A)
- 评测:`sbatch submit_eval_p0_commonsense_llama31_qwen3_1gpu.sh`
  (评测脚本 `instruction_tuning_eval/commonsense_eval.py`,vLLM 生成 + LLM-Adapters 同款答案抽取)
- 汇总:`python summarize_p0_commonsense.py`

## 实验 C(P1 视觉)

ViT-B/16(in21k)+ CIFAR-100 / Food101 / DTD 少样本 1k(VTAB-1k 口径)
+ CIFAR-100 数据量扫描 {1k, 5k, full}(E1 的跨模态复现点)。

- 目录:`ViT/`
- 训练:`sbatch submit_train_p1_vision_rtx5880.sh`(阶段1 array 0–29,每任务 1×RTX5880,分钟~小时级)
  - TASK_ID = setting*6 + config;setting: 0=cifar100@1k, 1=food101@1k, 2=dtd@1k,
    3=cifar100@5k, 4=cifar100@full;config 同 A(0=lora,1=unilora,2..5=prolosa 比例)
- 入口:`train_vit_p1.py`(复用 math 目录的 RoSA-SNIP Trainer 机制;分类头始终可训练,head_lr 3e-3)
- 汇总:`python summarize_p1_vision.py`

## 实验 D(P2 chat 评测扩展)

补 IFEval(可验证指令遵循)+ MMLU 5-shot(能力保持/遗忘)。
**2026-08-28 更新:全盘搜索确认旧 Alpaca checkpoint 已不存在**(全仓库无 adapter_model.*),
因此需先补训:用与 P0 完全一致的现代化管线在 Cleaned Alpaca 上重训 Llama-2-7b。

- 补训数据:`math_instruction_tuning/data/prepare_alpaca_cleaned.py`
  → `data/alpaca/alpaca_cleaned.json`(51,760 条,含 input 的样本已并入 instruction)
- 补训入口:`math_instruction_tuning/submit_train_p2_alpaca_llama2_1gpu.sh`
  (阶段1 array 0-5 = config,单卡;阶段2 SEED=43/44 补 0,1,<best>)
- checkpoint 就位后:每个 adapter 的 `ft/` 目录作为 ADAPTER_PATH 提交下方评测脚本;
  MT-Bench 需另用 FastChat(`fastchat_eval` 环境)重评,并配 GPT judge API。

评测依赖 `lm_eval`(已装入 unilora_modern)。

- 目录:`instruction_tuning/`
- 用法(每个 checkpoint 提交一次):

```bash
ADAPTER_PATH=/path/to/adapter RUN_TAG=llama2_unilora_s42 \
  BASE_MODEL=meta-llama/Llama-2-7b-hf \
  sbatch submit_eval_p2_ifeval_mmlu_1gpu.sh
# base model 对照行: RUN_TAG=llama2_7b_base sbatch submit_eval_p2_ifeval_mmlu_1gpu.sh
```

- 结果:`results/p2_chat_eval/<RUN_TAG>/{mmlu,ifeval}/`(lm-eval json)+ 同名 .log
- 理论卖点:压缩方法方差小 ⇒ MMLU 掉点(遗忘)应更少。

## 实验 E(P4 理论对齐,对标 LoRA-vs-FFT 论文的 Qwen2.5 扫描)

Qwen2.5-1.5B + CommonsenseQA,SFT 后按选项似然打分评测(验证集干净)。

- 目录:`theory_validation_llm/`
- E1 样本量扫描:`sbatch submit_e1_sample_sweep_qwen25_rtx5880.sh`
  (完整网格 0–41 = p{1,2,5,10,25,50,100}%×6config;阶段1默认 array 只含
  lora/unilora 全部 p + prolosa 比例扫描 p∈{1%,10%,100%};固定 900 步;子集种子 1234 共享)
- E2 标签噪声扫描:`sbatch submit_e2_label_noise_qwen25_rtx5880.sh`
  (完整网格 0–23 = η{0,10,20,30}%×6config;阶段1默认 array 只含
  lora/unilora 全部 η + prolosa 比例扫描 η∈{0,30%};噪声种子 5678 共享)
- 汇总(直接输出 Δ(p)/Δ(η)):

```bash
python summarize_theory_results.py --result_root results/e1_sample_sweep --sweep frac
python summarize_theory_results.py --result_root results/e2_label_noise --sweep eta
```

- 预期:Δ = Acc_compressed − Acc_LoRA 随 p 单调下降(存在 crossover p*),随 η 单调上升;
  ProLoSA 曲线应位于 Uni-LoRA 与 LoRA 的较优者之上。

---

## P3(GLUE)

按调研建议不新增实验:保留 RoBERTa-large 主表,base 表移附录,正文定位改为
"与 compressed LoRA 文献的可比性 + E1–E5 理论验证平台"。

## QOS 限额与分波提交(重要)

集群限额(`sacctmgr show qos`):`5880_qos` 每用户最多**排队 10 个 / 运行 8 个 / 12 张卡**;
`l20_qos` 仅 4 排 / 2 跑 / 8 卡。因此大 array 无法一次提交,统一用仓库根目录的
`submit_in_waves.sh`(轮询队列,有空位才提交下一个 array 元素,`nohup` 挂在登录节点):

```bash
cd math_instruction_tuning
nohup ../submit_in_waves.sh submit_train_p0_math_llama31_qwen3_1gpu.sh 0-11 > wave_p0_math.log 2>&1 &
# 环境变量: PARTITION(默认 gpu-rtx5880), MAX_QUEUED(默认 9), POLL_SECS(默认 180)
# array_spec 支持逗号列表, 如 E1 阶段1: 0,1,2,3,4,5,6,7,12,13,18,19,20,21,22,23,24,25,30,31,36,37,38-41
```

## 提交状态(2026-08-28,两阶段重启后)

- 12:45 已取消全部多种子作业与旧 2 卡作业(含互相覆盖输出目录的 llama31_lora 三种子,
  目录已删除重跑;卡在权重加载的 qwen3 冒烟 1794158_9 也已取消),旧分波提交器全部杀掉;
- **阶段1(单种子 42 + ProLoSA 比例扫描)分波提交器已重启**:
  A 数学(0–11)、C 视觉(0–29)、E1(26 个 id)、E2(16 个 id);
- 实验 B 阶段1(0–11)与 p2 alpaca 补训(0–5)待 A 消化后启动;
- 每个任务的最优比例出来后(先跑 `summarize_*` 看阶段1结果),再:
  1) 补 prolosa(best) 在其余扫描点(仅 E1/E2);2) `SEED=43/44` 补多种子;
- 实验 D 等 Alpaca checkpoint 就位后按 checkpoint 逐个提交(lm_eval 已装入 unilora_modern);
- 训练全部结束后按各目录的 `submit_eval_*` 评测(同样受限额,可用分波提交器)。

## 资源记录(时间 / 显存)

三个训练入口在 `trainer.train()` 结束后自动记录并落盘:

- 数学/常识(`intruction_tuning_unilora_multi_gpu.py`):写 `<output_dir>/resource_usage.json`,
  并在 slurm 日志打印 `RESOURCE: {...}` 行;
- 视觉(`ViT/train_vit_p1.py`)与理论(`theory_validation_llm/train_eval_csqa_qwen.py`):
  字段直接并入各自的 `RESULT` json(`train_runtime_sec`、`gpu_peak_alloc_GiB`、`gpu_peak_reserved_GiB`)。

记录内容:训练墙钟时间、samples/steps per second、每张 GPU 的
`torch.cuda.max_memory_allocated/reserved` 峰值(两阶段重启后所有作业均带记录)。

作业级时长统一用仓库根目录 `collect_job_stats.sh`(基于 `sacct`,过滤 p0_/p1_/p2_/e1_/e2_ 作业,
输出 JobID、分区、AllocTRES、提交/开始时间、Elapsed、State):

```bash
bash collect_job_stats.sh 2026-08-27 > job_stats.txt
```

vLLM 评测作业的显存无参考意义(`gpu_memory_utilization` 会预占固定比例),只记时长。

## 超参数设定依据

**原则:lr / warmup / rank 等直接继承仓库既有(已调好的)配置;ProLoSA 的 θ_d:sparse
比例每任务用阶段1单种子扫描确定(2:1 / 4:1 / 8:1 / 12:1),总预算严格对齐三方法。**

| 超参 | LLM 实验 (A/B/E) | ViT 实验 (C) | 来源 |
|---|---|---|---|
| LoRA rank | 4 (q,k,v,o,gate,up,down) | 4 (query,value) | 仓库原实验约定 |
| 总可训练预算 | 524,288 | 24,600(不含分类头) | 原 Uni-LoRA `vector_length` |
| LoRA lr | 2e-4 | 1e-3 | 仓库 profile 脚本默认 |
| Uni-LoRA lr | 2e-3 | 4e-3 | 原 unilora 提交脚本 |
| ProLoSA θ_d : sparse | {2,4,8,12}:1 扫描后取最优 | 同左(预算 24,600) | 阶段1扫描(见下) |
| ProLoSA θ_d lr | 8e-4 | 4e-3 | 原扫描基线 / 与 unilora 对齐 |
| sparse lr 乘子 | 0.2 | 0.2 | 原扫描基线 |
| RoSA warmup / mask steps | 128 / 1 | few-shot 32,全量 128 / 1 | 原脚本;few-shot 步数少按比例缩 |

比例扫描的历史证据(Gemma-7B + MetaMathQA,`unilora_rosa_snip_hparam*` 日志):
GSM8K/MATH 上 **12:1 最优(74.99/29.92)**,8:1 次之(74.83/29.78),4:1 中间,
2:1 各变体最差(72.8~73.7/28.8~29.6);16:1 回落(73.84/29.14)。说明稀疏分量占比
过大浪费预算,但完全砍掉也不行。新 backbone(Llama-3.1/Qwen3)与新任务
(常识/视觉/理论)最优点未必相同,故每任务在阶段1重扫 {2,4,8,12}:1,
θ_d lr 变体(1e-3)与 sparse 乘子变体(0.1)在旧扫描中无收益,不再重扫。
三方法预算相同,差异只来自参数化方式,这是论文对比的公平性前提。

## 写作联动(见 survey 第四节)

- Sec 5.1 与附录 B 超参表换新 backbone;
- "MATH 更接近 bias 主导端"改写为 GSM8K→MATH500 难度谱下的论述;
- Related Work 增引 LoRA-vs-FFT(2605.19018)与 sample complexity(2607.27680);
- 摘要/结论加入 vision 表述。
