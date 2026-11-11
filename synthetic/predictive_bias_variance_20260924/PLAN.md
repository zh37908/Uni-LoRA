# 固定 teacher 的预测偏差—方差实验（2026-09-24）

用户原始四档方案完整保存在 `USER_PROPOSAL.md`。本次先完成实验 1、2，并附小型优化控制；实验 3、4 留作后续。原始方案中的会议截止日期是原文记录，本次不用于排程依据。

## 已锁定的执行协议

- 复用 `synthetic/results_revision5_fixed_teacher/teachers.pt` 的 concentrated teacher、冻结底座及 1024 个测试输入，输出维度 16。独立 calibration 数据上的输出位移 MSE 为底座输出方差的 10%，不是输出幅度的 10%。
- 经源码核对，D=8192 是直接 Q/V 权重更新坐标。因此比较 **full-space adaptation** 与 **compressed-only adaptation**；本结果不能标作 standard LoRA 对压缩 LoRA。
- 每个条件 n=256，M=20 个独立训练集；重复编号 r 对应固定数据 RNG `920000+r`。跨方法、维度、噪声水平共用该编号的训练输入和标准高斯噪声。
- 所有主实验固定初始化为零更新、优化采样 RNG=81000、投影、teacher 和测试集。仅训练样本与标签噪声随重复编号变化。无 dropout、无 support selection。
- 延续现有 R2 协议：AdamW，lr=0.005，weight_decay=0，batch=128（有放回采样），梯度范数裁剪 1，1000 步。预先固定，未按最终测试结果挑选超参数或检查点。
- FP32，关闭 TF32，启用确定性算法。记录代码/输入/预测哈希及 GPU、PyTorch、Slurm 作业信息。
- 实验 1：sigma={0,0.25,0.5,1} × d={128,8192} × 20 = 160 次。
- 实验 2：sigma={0.25,1} × d={32,128,512,8192} × 20 = 160 次，其中 80 次复用实验 1，总计 **240 次主训练**。
- P_d 为归一化 one-hot sharing。恢复原始 RNG=913 的共同置换，令 assignment=perm mod d，验证 d=128 与历史投影完全一致；32/128/512 逐级细分分组，空间严格嵌套。
- 小型长训练控制：sigma={0,1}、d={128,8192}、r=0..4，主训练继续至 3000 步；用同 5 个重复比较 1000/3000 步，不将小样本控制混入主结果。
- 优化器随机性控制：sigma=1、d={128,8192}、r=0..2，另外使用优化采样种子 81001、81002，各 1000 步，共 12 次新增训练。

## 统计与报告

保存每次训练的 1024×16 clean-test 预测和 Q/V 更新。设 M 个预测的均值为 mean，V=mean_test(sum_r((pred_r-mean)^2)/(M-1))；bias²=mean_test((mean-truth)^2)-V/M；R=mean_r(mean_test((pred_r-truth)^2))。检查 R=bias²+V，修正后的负值保留。

所有区间按训练集重复编号配对 bootstrap，使用 10000 次重采样、固定 RNG；测试 token/坐标不充当独立训练重复。报告 95% 点态 percentile 区间，不作多重比较调整。同时报告 delta_bias=bias²_compressed-bias²_full、delta_var=V_full-V_compressed、net_gain=R_full-R_compressed。

经验预测偏差包含表示、有限样本与优化因素，不能直接命名为理论 B_proj；分解恒等式本身不是机制证据。完整保留有利和不利结果，不要求 U 型曲线。

## 文件

- `run.py` / `model_frozen.py`：可复现运行器及模型源码快照。
- `manifest.json`：不可静默变更的协议、输入哈希、完整任务网格。
- `main/`、`optimizer/`：逐运行记录、预测及更新。
- `summary/`：结果表、配对区间、图和中文总结（完成后生成）。

现有历史实验随重复编号变化了优化随机性，与本次协议不同，不直接混用其预测。

## 统计实现更正（训练完成后的解析核验）

首版按上述原定方法直接取 bootstrap percentile，发现分量区间系统性偏移。原因可解析推导，与结果正负无关：有限 M 无偏修正后的 B_hat、V_hat 不等于经验分布泛函；在 bootstrap 中 E[B*]=B_hat+V_hat/M，E[V*]=(M−1)V_hat/M。因此最终改为对 T*−T_emp 的配对重采样误差取分位点，以 T_hat−该误差分位点形成 basic 区间。风险与分量的点估计、训练设置、原始预测均不改变；配对重采样及 10000 次次数保持不变。

首版输出保存为 `initial_uncentered_bootstrap_diagnostic/`（仅作诊断，**不作为最终区间引用**）。`verify_statistics.py` 使用 M=2 的全部可能重采样枚举验证居中公式，最终结果位于 `summary/`。这是统计实现修正，不是依据显著性筛选区间方法。最终区间仍属 M=20 下的近似非参数推断。
