# 全 Transformer block 线性层的 LoRA / Uni-LoRA / ProLoSA 预测偏差—方差实验

在新teacher的dev/test结果产生前锁定本协议；基于已知Q/V实验设计，不覆盖旧数据。

## 模型范围与预算

保留既有已预训练且冻结的单个Transformer block：width64、4 heads、MLP256、序列32、16维读出。六类目标为q/k/v/o/ff1/ff2，分别对应Q/K/V/attention output/MLP up/MLP down。全部六个矩阵都加rank4 BA，scaling1（alpha=r），dropout0；bias、LayerNorm与readout冻结。这是block内全部线性层，不是多层大语言模型，也不是在归一化、embedding或输出头加LoRA。

仓库分类入口 `NLU/peft/examples/sequence_classification/run_unilora_variants_glue.py` 的 target_modules 包含query/key/value/output.dense/intermediate.dense（output.dense同时匹配attention及FFN输出）；classifier另行modules_to_save。指令微调入口 `instruction_tuning/qlora_unilora.py` 扫描linear并排除lm_head。本实验对齐block内目标模块范围，任务头仍冻结以控制估计目标，未复现全部下游训练细节。

四个64×64矩阵各512个因子参数；ff1(256×64)、ff2(64×256)各1280，总D=4608，是旧Q/V因子D1024的4.5倍。压缩/残差预算按同倍数扩展，保持压缩比例及稀疏密度：

| 方法 | bank | sparse K | 有效自由度 |
|---|---:|---:|---:|
| full_matched | — | — | 4608 |
| full_zero | — | — | 4608 |
| uni288 | 288 | 0 | 288 |
| uni576 | 576 | 0 | 576 |
| uni864 | 864 | 0 | 864 |
| pro288_288 | 288 | 288 | 576 |
| pro576_288 | 576 | 288 | 864 |

full_matched复制Uni576展开后的初始因子；full_zero为标准A Kaiming/B0敏感性对照。同bank的Pro和Uni完全相同初始预测。等576预算比较改变bank投影；同bank576比较额外增加288参数；Uni864为总预算控制。有效自由度不等于dense实现存储量。全局sharing跨六个模块和A/B，按源码balanced assignment、seed42、inverse-sqrt-count归一化。

## 固定teacher、数据与优化

- 从同一冻结底座出发，六个目标矩阵各生成独立rank4 BA teacher（seed980020）。单个全局scale在独立calibration输入(seed980030)上使预测位移MSE为底座输出方差的0.1。teacher固定，不随方法、噪声或训练重复改变。记录每层非零更新及输出影响。与旧Q/V不同teacher，禁止把两组MSE差值归因于适配范围本身或合并计算方差。
- 所有bank初始化seed980010、U[-.02,.02]；标准LoRA seed980011。
- dev输入seed980040；训练seed981000+r，3重复。候选LR {0.0002,0.001,0.005,0.02}，噪声倍数{0,1,4}，每方法4×3×3=36拟合，共252次。选择平均9次dev MSE最小的单一LR，所有噪声共用；平局小LR。降低LR下界用于检验更多模块训练的步长敏感性，不根据test调参。
- 保存dev选择后才生成最终test(seed980050，1024样本)。正式训练seed982000+r，r0..19。每次训练n256、序列32、输出16。先生成x再生成独立标准高斯noise，方法和噪声档配对复用。噪声倍数{0,.5,1,2,4}×calibration位移RMS。
- AdamW、weight_decay0、batch128有放回、裁剪范数1、优化器RNG81000、FP32/TF32off，主训练1000步。7方法×5噪声×20重复=700正式拟合。
- ProLoSA按已审计仓库方法重构A/B与捕获SNIP=sum|factor×gradient|；128步warmup+128步score，第256步更新后global top288固定mask，保留bank/optimizer继续联合训练。选择步骤计入总预算；score在clip前捕获；残差初始0；不重置优化器。mask随数据变化引入的变异属于预测方差。
- 噪声1、4各前5重复延长到3000步，共70个延长控制，不挑test最优checkpoint。

## 验证、统计与报告

- 验证零更新模型与旧底座一致、所有六个模块的LoRA梯度非零、source factor reconstruction及gradient等价、SNIP sum、mask外无更新、matched初始预测完全一致；记录teacher各模块更新norm与移除单模块的预测影响。
- 每次保存预测、因子、support、scores、训练输入/噪声/标签/test/teacher标识、源码与产物hash。所有训练跑完再总结；不得因负结果重新选teacher/投影/预算。
- 重用经过检验的样本方差(M−1)、bias²扣V/M、risk=bias²+variance。10000次配对bootstrap、经验泛函居中的basic 95%点态近似区间，未作多重修正。
- 比较Uni576对full_matched/full_zero，Pro288+288对Uni576（等预算），Pro576+288对Uni576（同投影），Pro576+288对Uni864（等预算），Pro288+288对Uni288（同投影）。报告全部噪声、正负结果、长训练及support在六模块的分布。
- 图标题显式写明“all block linear maps: Q/K/V/O + FFN up/down”，提供完整分解、相对改善图PNG/PDF及中文解释。单teacher/投影/单block、固定训练协议，不能声称实际大模型理论已普遍验证。
