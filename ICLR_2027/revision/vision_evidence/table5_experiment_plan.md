# 对齐 Uni-LoRA Table 5 的 ProLoSA 补充方案

## 已核对的原始设置

来源：[Uni-LoRA NeurIPS 2025 正文第 9 页，Section 4.4 / Table 5](https://proceedings.neurips.cc/paper_files/paper/2025/file/3e596a70b60dcf7c6e0ac1a2d73e7470-Paper-Conference.pdf)。

- 模型：ViT-Base、ViT-Large；rank 4，20 epochs，每项实验重复 5 次，报告均值和标准差。
- 数据集（保持原列顺序）：OxfordPets、StanfordCars、CIFAR10、DTD、EuroSAT、FGVC-Aircraft、RESISC45、CIFAR100；另有八任务平均值。
- Table 5 的 Uni-LoRA 参数量：Base 72K，Large 144K。原文附近正文将 Base 写成 74,000，与表和本地 `fine_tuning_ViT_base.py` 的 72,000 不一致；对齐表格时以 72,000 为工作预算，并在复现实验记录中保留这个差异。
- 原文头部学习率搜索集合为 {0.001, 0.002, 0.005, 0.01}，向量学习率集合为 {0.002, 0.005, 0.01, 0.02, 0.05}。实际采用的配置还应与可获得的原始记录核对。

## 现有结果为何不能填入

`p1_vision` 和 `p1_vision_d72k` 只完成了 ViT-Base、seed 42 的共同配置扫描，数据包括 CIFAR-100@1k/@5k/@full、Food-101@1k、DTD@1k。Food-101 不属于 Table 5。没有可直接填入原表协议的完整 ProLoSA 单元格。

具体差异不仅是种子数和参数量：

- 本地旧 CIFAR 脚本将原训练集的 10% 留作验证集，并用验证集选择 checkpoint 后评价 test；P1 的 full CIFAR-100 使用全部 50,000 张训练图像，只训练 10 epochs，报告最终模型。
- 旧 DTD 脚本从本地图像目录按固定种子分割；P1 使用 Hugging Face 数据的 train/test 和 1,000 张训练子集，不能假定划分一致。
- 旧脚本 batch 128、梯度累积 4、FP16；P1 batch 64、无累积、BF16。旧 CustomTrainer 的 weight decay 为 0.01；P1 为 0。
- 旧启动脚本只列了种子 1/2/3/4，不能据此认定论文五次运行都已在本地找到。
- 旧 StanfordCars 分支把整个 DatasetDict 赋给 `test_ds`；恢复训练入口时须明确取 test split。FGVC/DTD 的本地数据源也需要核实。

这些旧文件可作为复现线索，但不是已经验证完毕的完整实验协议。若原始 split/checkpoint 记录不能恢复，应保存新的共享 split manifests，同时重跑 Uni-LoRA，把结果标为本次复现。

## 建议实施顺序

1. 恢复八数据集 train/validation/test，固定并保存划分索引；双方共享模型权重、数据、rank、更新模块、图像变换、有效 batch、epochs 和评估流程。记录 adapter-only 的 d+K 及包含分类头的总参数量。
2. 先在 ViT-Base 的 CIFAR-100 和 DTD 做小规模验证：每个数据集运行 Uni-LoRA 和 ProLoSA 四种比例的一个训练种子，共 10 次。验证数据流程、稀疏激活和显存，并在验证集上筛选比例；这些试运行不替代正式五种子结果。
3. 以 d+K=72,000 / 144,000 匹配 Base / Large。候选 d:K 为 2:1、4:1、8:1、12:1，稀疏预算必须从向量预算中扣除。先固定 warmup 和 sparse learning-rate multiplier，分阶段调节向量/头部学习率，避免直接开展庞大的笛卡尔积搜索。
4. 每个模型/数据集在验证集上选定配置后，固定配置运行五个训练种子（建议明确列出 42--46），用验证集选择每次运行的 checkpoint，最后计算独立 test accuracy。不用 test 选比例、学习率或最佳 epoch。Uni-LoRA 使用相同的调参原则和共享种子。
5. 八列分别报告五次 test accuracy 的 mean ± sample std；Avg. 为八个任务均值的算术平均。复现 Uni-LoRA 与 ProLoSA 的差值按共享 seed 配对分析。若报告 Avg. 的标准差，应先求每个 seed 的跨任务平均再计算，不能平均八个标准差。

## 正式训练次数（均不含调参和试运行）

| 覆盖范围 | ProLoSA | 同协议重跑 Uni-LoRA | 合计 |
|---|---:|---:|---:|
| 仅 ViT-Base，8 数据集 × 5 seeds | 40 | 40 | 80 |
| Base + Large，2 × 8 数据集 × 5 seeds | 80 | 80 | 160 |

若只补 ProLoSA，两行是 80 次正式训练。鉴于当前协议差异，推荐最终补齐同协议 Uni-LoRA 对照，总计 160 次。LP/FF/FourierFT 可以保留为明确引用的历史结果；若实现协议无法与原实验一致，必须区分历史结果与新复现结果。

已经准备 `drafts/vit_table5_extension.tex`，保留原 Table 5 的数值并增加 Base/Large 两行 ProLoSA（待实验）。草稿不进入正式投稿 PDF；结果确认后再替换或扩展当前独立的 P1 视觉表。本次仅核对并准备草稿/方案，尚未提交训练任务。
