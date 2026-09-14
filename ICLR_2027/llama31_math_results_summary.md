# Llama-3.1-8B 数学推理结果核对（2026-09-11）

核对 33 个完整评测配置，均为 seed=42；每个配置包含 GSM8K 1,319 题与 MATH500 500 题。未发现完整 MATH 评测，不能与 Gemma 的 MATH 列混用。LoRA r=4 仅作为补充参考，主表使用 r=64。

主表在共同的 1,048,576 参数预算内，分别展示 Uni-LoRA / ProLoSA 两项准确率算术平均最高的已评测配置；这是基于测试结果的事后描述性选择，不是独立验证集选参。每行两项成绩均来自同一个配置。没有多种子均值、标准差或显著性结论。

| 方法 | r | 参数量 | d:K | 学习率（ProLoSA 为 theta_d） | GSM8K (%) | MATH500 (%) | 结果目录（相对 math_instruction_tuning/results） | 训练日志 |
|---|---:|---:|---|---:|---:|---:|---|---|
| LoRA | 4 | 10,485,760 | -- | 0.0002 | 75.59 | 26.80 | `p0_math/llama31_lora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1794414_0.out` |
| LoRA | 64 | 167,772,160 | -- | 0.0002 | 79.61 | 33.20 | `p0_math_lora_r64/llama31_lora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1833921_0.out` |
| Uni-LoRA | 4 | 262,144 | -- | 0.002 | 70.28 | 23.40 | `p0_math_d/uni_d262144/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1807624_1.out` |
| Uni-LoRA | 4 | 524,288 | -- | 0.0005 | 68.39 | 25.40 | `p0_math_lr/uni_lr5e-4/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1814558_1.out` |
| Uni-LoRA | 4 | 524,288 | -- | 0.001 | 72.33 | 23.00 | `p0_math_lr/uni_lr1e-3/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1805717_1.out` |
| Uni-LoRA | 4 | 524,288 | -- | 0.002 | 73.84 | 24.80 | `p0_math/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1794415_1.out` |
| Uni-LoRA | 4 | 524,288 | -- | 0.004 | 74.15 | 26.60 | `p0_math_lr/uni_lr4e-3/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1805719_1.out` |
| Uni-LoRA | 4 | 524,288 | -- | 0.008 | 74.75 | 24.20 | `p0_math_lr/uni_lr8e-3/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1807644_1.out` |
| Uni-LoRA | 4 | 524,288 | -- | 0.01 | 72.40 | 21.00 | `p0_math_lr/uni_lr1e-2/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1814560_1.out` |
| Uni-LoRA | 4 | 1,048,576 | -- | 0.0005 | 72.25 | 25.00 | `p0_math_d/uni_d1048576_lr5e-4/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1819408_1.out` |
| Uni-LoRA | 4 | 1,048,576 | -- | 0.001 | 74.22 | 26.20 | `p0_math_d/uni_d1048576_lr1e-3/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1819409_1.out` |
| Uni-LoRA | 4 | 1,048,576 | -- | 0.002 | 75.44 | 26.20 | `p0_math_d/uni_d1048576/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1807629_1.out` |
| Uni-LoRA | 4 | 1,048,576 | -- | 0.004 | 74.07 | 25.60 | `p0_math_lr/uni_lr4e-3_d1048576/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1814543_1.out` |
| Uni-LoRA | 4 | 2,097,152 | -- | 0.0005 | 74.45 | 26.00 | `p0_math_d/uni_d2097152_lr5e-4/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1819411_1.out` |
| Uni-LoRA | 4 | 2,097,152 | -- | 0.001 | 75.97 | 25.60 | `p0_math_d/uni_d2097152_lr1e-3/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1819410_1.out` |
| Uni-LoRA | 4 | 2,097,152 | -- | 0.002 | 76.95 | 26.00 | `p0_math_d/uni_d2097152/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1814559_1.out` |
| Uni-LoRA | 4 | 2,097,152 | -- | 0.004 | 74.30 | 25.80 | `p0_math_d/uni_d2097152_lr4e-3/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1819412_1.out` |
| Uni-LoRA | 4 | 4,194,304 | -- | 0.002 | 76.12 | 26.20 | `p0_math_d/uni_d4194304/llama31_unilora_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1819399_1.out` |
| ProLoSA | 4 | 524,288 | 2:1 | 0.0008 | 69.75 | 24.80 | `p0_math/llama31_prolosa_r2to1_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1794416_2.out` |
| ProLoSA | 4 | 524,288 | 4:1 | 0.0008 | 69.07 | 21.40 | `p0_math/llama31_prolosa_r4to1_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1794417_3.out` |
| ProLoSA | 4 | 524,288 | 8:1 | 0.0008 | 69.83 | 21.00 | `p0_math/llama31_prolosa_r8to1_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1794418_4.out` |
| ProLoSA | 4 | 524,288 | 12:1 | 0.0004 | 68.23 | 21.00 | `p0_math_lr/prolosa12_td4e-4/llama31_prolosa_r12to1_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1805725_5.out` |
| ProLoSA | 4 | 524,288 | 12:1 | 0.0008 | 71.27 | 21.00 | `p0_math/llama31_prolosa_r12to1_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1795646_5.out` |
| ProLoSA | 4 | 524,288 | 12:1 | 0.0016 | 73.31 | 20.60 | `p0_math_lr/prolosa12_td1.6e-3/llama31_prolosa_r12to1_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1805718_5.out` |
| ProLoSA | 4 | 524,288 | 12:1 | 0.0032 | 73.92 | 23.80 | `p0_math_lr/prolosa12_td3.2e-3/llama31_prolosa_r12to1_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1814542_5.out` |
| ProLoSA | 4 | 1,048,576 | 2:1 | 0.0016 | 73.77 | 24.80 | `p0_math_prolosa1M/td1.6e-3/llama31_prolosa_r2to1_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1824910_2.out` |
| ProLoSA | 4 | 1,048,576 | 2:1 | 0.0032 | 74.98 | 24.60 | `p0_math_prolosa1M/td3.2e-3/llama31_prolosa_r2to1_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1824913_2.out` |
| ProLoSA | 4 | 1,048,576 | 4:1 | 0.0016 | 75.66 | 24.80 | `p0_math_prolosa1M/td1.6e-3/llama31_prolosa_r4to1_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1824916_3.out` |
| ProLoSA | 4 | 1,048,576 | 4:1 | 0.0032 | 74.60 | 25.00 | `p0_math_prolosa1M/td3.2e-3/llama31_prolosa_r4to1_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1824984_3.out` |
| ProLoSA | 4 | 1,048,576 | 8:1 | 0.0016 | 74.07 | 26.20 | `p0_math_prolosa1M/td1.6e-3/llama31_prolosa_r8to1_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1824985_4.out` |
| ProLoSA | 4 | 1,048,576 | 8:1 | 0.0032 | 74.37 | 25.60 | `p0_math_prolosa1M/td3.2e-3/llama31_prolosa_r8to1_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1824986_4.out` |
| ProLoSA | 4 | 1,048,576 | 12:1 | 0.0016 | 75.06 | 26.60 | `p0_math_prolosa1M/td1.6e-3/llama31_prolosa_r12to1_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1824987_5.out` |
| ProLoSA | 4 | 1,048,576 | 12:1 | 0.0032 | 74.53 | 21.60 | `p0_math_prolosa1M/td3.2e-3/llama31_prolosa_r12to1_s42` | `math_instruction_tuning/logs/p0_math_train_5880_1824988_5.out` |

每个结果目录内的 `gsm8k.log` / `math500.log` 最后一条准确率及题目数已经核对；训练日志已核对完成标记、秩、学习率和参数量。ProLoSA 参数量按 d+K 计算，避免使用稀疏分支激活前日志打印的 d。

主表来源：
- LoRA: `p0_math_lora_r64/llama31_lora_s42`；79.61 / 33.20。
- Uni-LoRA: `p0_math_d/uni_d1048576/llama31_unilora_s42`；75.44 / 26.20。
- ProLoSA: `p0_math_prolosa1M/td1.6e-3/llama31_prolosa_r12to1_s42`；75.06 / 26.60。
