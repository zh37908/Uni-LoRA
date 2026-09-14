# ProLoSA: Uni-LoRA Table 5 extension

目标：先使用 validation 筛选配置，然后运行 ViT-Base / ViT-Large × 八数据集 × 五训练种子的 **80 次 ProLoSA 正式实验**。本批次不重跑 Uni-LoRA。

## 已启动

- Slurm job：`1858910`，`gpu-rtx5880`，单节点 4 GPU；4 个单卡 worker 并行。
- CPU 汇总/续跑作业：`1858991`，依赖上述 GPU 作业结束。若 72 小时分配结束时尚有任务且无失败/缺失数据，则自动续接一次新的分配；完成时导出 CSV、统计和 TeX 行。
- 搜索和正式实验由持久化队列自动衔接；正式阶段有全局验证完成门槛。
- 数据/模型在登录节点预下载；训练严格使用本地固定快照和离线模式。
- `~/.bashrc` 设置了本机代理 `127.0.0.1:7890`。计算节点脚本清除代理；登录节点测试 direct 连接可用后，下载进程也使用局部清除代理的环境。没有修改用户 `.bashrc`。

## 超参数搜索（候选范围内的最优，非全局最优保证）

每个模型/数据集单独筛选：

| 阶段 | 配置数/组合 | 训练轮数 | 训练种子 | 选择依据 |
|---|---:|---:|---|---|
| 初筛 | 24 | 5 | 101 | best validation accuracy |
| 完整验证 | 初筛前三 | 20 | 101 | best validation accuracy |
| 第二种子确认 | 完整验证前二 | 20 | 102 | 两个种子的 validation accuracy 均值 |
| 正式实验 | 已选定的一组 | 20 | 42,43,44,45,46 | validation 选 checkpoint，随后独立 test |

初筛网格：向量 LR `{0.002,0.01,0.05}` × head LR `{0.002,0.01}` × `d:K={2:1,4:1,8:1,12:1}`。Sparse LR multiplier 固定 0.2，support warmup 固定训练步数的 10%，随后一 optimizer step 收集 SNIP 分数。学习率/比例的候选范围及分阶段策略在 `search_protocol.json` 中提前记录。短训练初筛可能遗漏晚收敛配置；所选结果只称为该搜索流程下的最佳验证配置。

共 384 次短初筛、48 次完整验证、32 次第二种子确认、80 次正式实验；另有 4 次微型 smoke。Smoke 的 Base 两项先通过后，Base 初筛与 Large 权重下载重叠执行，避免 GPU 空等。

## 与原实验的对齐和差异

- 固定 rank 4，query/value adapters；Base `d+K=72,000`，Large `d+K=144,000`。
- 20 epochs 正式训练；有效 batch 512。Base micro-batch 64 × accumulation 8；Large 32 × 16，等价于旧脚本有效 batch 128 × 4。
- FP16，AdamW，weight decay 0.01，linear schedule，无 scheduler warmup（旧 CustomTrainer 的实际设置）。Support warmup 与 scheduler warmup 是不同参数。
- 图像变换采用随机 224 裁剪/翻转训练和中心裁剪验证/测试。
- 数据清单 `data/<dataset>/manifest.json` 固定源版本、标签、拆分及索引哈希；所有训练种子共享拆分 42。旧文章原始逐种子拆分未恢复，因此结果应注明为按旧脚本重建并固定拆分的复现。
- DTD 按标签/文件名排序重建 imagefolder 顺序，再应用旧脚本的 shuffle/split，得到 4060/450/1130。不会将 P1 的 DTD@1k test 结果混入。
- CIFAR10/100、OxfordPets、StanfordCars 从官方 train 留出 10% validation；独立原 test。修正旧 StanfordCars 分支将整个 DatasetDict 当 test 的问题。
- EuroSAT、RESISC45、FGVC 使用对应源的 train/validation/test；FGVC 使用 torchvision 官方 100 类 variant 数据。
- Table 5 原文 Base budget 正文写 74,000，而表和旧代码为 72,000；本批按表和旧代码的 72,000。

## 安全的结果选择与可复现文件

Tuning/smoke 不加载 test split。保存每个 epoch 的 validation 日志及最优参数快照，正式实验仅在恢复该 checkpoint 后评价 test。代码检查 sparse support 的实际 K、非零更新和 d+K；参数表记录 adapter 和含 head 总量，区分有效自由度与实际分配的 dense sparse-buffer。

`vendor/` 固定本批使用的 PEFT 实现与回调，避免其他实验修改公共代码影响正在运行的任务。`code_manifest.json` 记录代码哈希。每次尝试保存独立日志和 best checkpoint；结果通过 JSON 原子落盘，失败最多自动重试一次。

## 查看状态

```bash
python ViT/table5/pipeline.py status
squeue -j 1858910
tail -n 20 ViT/table5/logs/worker_1858910_0.log
```

- `state.json`：完整任务队列、尝试记录及状态。
- `selected_configs.json`：每个组合的验证集选择结果（生成后才存在）。
- `runs/<trial>/attemptN/result.json`：实际结果和配置哈希。
- `final_summary.json`：80 次全部完成后才生成，各任务五种子均值和样本标准差。
- `final_runs.csv`、`table5_statistics.json`、`table5_prolosa_rows.tex`、`RESULTS.md`：汇总作业在 80 次完成后生成，包含八任务平均值。
- `PAUSE`：如需暂停领取新任务，可在本目录创建该空文件；在跑任务会正常结束。

未完成/失败实验不写成论文测量结果。最终论文应使用正式五种子结果，而不是初筛或验证分数。
