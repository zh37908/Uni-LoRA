# 真实 LoRA 因子机制实验（2026-09-24）

## 事先锁定的主问题

使用真实 BA 参数化检验完整 LoRA 与压缩 LoRA 的预测偏差—方差关系。旧实验的直接 Q/V 权重更新结果不改名、不混用。本次采用仓库 NLU/peft 的 Uni-LoRA 全局均衡 sharing、逆平方根计数缩放和 BA 重构；保存原源码快照，直接提取调用 generate_index 与 _get_lora_matrices 方法，并核验原层 forward 的数值及梯度等价性。

## 模型、参数化和初始化

- 冻结原 Transformer 底座，Q/V 各 rank=4；完整 A/B 共1024个可训练参数，Uni-LoRA全局向量d=128。LoRA alpha=rank=4，即BA缩放1，与仓库Uni-LoRA一致。无dropout、无support。
- 固定 projection seed=42，bank seed=970010，bank按仓库U(−0.02,0.02)初始化；sharing顺序为q.A/q.B/v.A/v.B。
- 主参照 lora_matched：完整可训练A/B复制Uni-LoRA的初始展开值，因此初始因子与函数完全一致。两边A/B均非零，不能称为默认LoRA初始化。
- 初始化敏感性参照 lora_zero：常规A Kaiming uniform、B=0，独立固定seed=970011；它不满足与Uni-LoRA因子相同，但单独报告可观察默认初始化的影响。
- 三方法都实际训练LoRA因子或其压缩坐标，计算deltaW=B@A，冻结底座。

## Teacher 与数据

固定rank4 teacher，A/B从独立seed=970020生成，与learner投影独立。用独立1024个calibration inputs（seed970030）校准BA更新的幅度，使输出位移MSE为底座输出方差的10%。缩放A/B的平方根使teacher始终可由rank4完整LoRA表达。

噪声sigma=teacher calibration位移RMS×{0,0.5,1,2,4}。这是按信号尺度事先规定的网格，不保证出现交叉；不因最终测试结果改变teacher、噪声或投影。

开发阶段使用独立验证输入（seed970040）、3个开发训练集（seed971000+r，n256），在噪声倍数{0,1,4}上对每方法分别选择全局学习率。候选{0.001,0.005,0.02}，每格1000步，以9格clean验证MSE均值最小为准；相同LR均值相同时选较小LR。共81次开发拟合，不计入正式M20分解。

开发选择锁定后才生成新测试输入（seed970050，1024例）及clean输出。正式重复使用20组全新训练数据（seed972000+r，n256），各方法/噪声配对X及标准噪声。Teacher、projection、初始因子、minibatch RNG=81000全部固定。

## 训练与工作量

AdamW，weight_decay0，batch128有放回，clip范数1，FP32关闭TF32并使用确定性算法；主结果固定1000步。正式5噪声×3方法×20重复=300次。噪声倍数1、4下前5重复的全部3方法延长到3000步，共30次延长控制，与同5重复的1000步配对。

所有grid、种子、快照及配置在正式结果产生前记录哈希。不得选择测试最优检查点。保存每次逐样本预测、最终因子、训练/测试/标签指纹与优化日志。

## 统计和解释

以lora_matched为主参照、lora_zero为初始化敏感性参照，各自报告压缩增加bias²、减少variance、风险净收益。有限M修正bias²，M−1样本方差；10000次训练重复编号配对的经验泛函居中basic区间，与原统计协议一致。M20主结果、M5长训练分别计算。

报告全部正负结果。不同因子参数化存在非凸优化和gauge自由度，因此只做预测空间分解，不把平均因子差称B_proj；即使出现完整模式，也不证明所有预训练LLM任务或所有初始化都符合它。
