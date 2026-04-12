# UniLoRA-DeepK（分层聚类）任务规划

## 1. 目标与范围

- 目标：在 `unilora_deepk` 方案中，先采用**分层聚类**（layer-wise clustering），不做跨层共享。
- 约束：
  - 仅针对 LoRA 增量参数 `A` 与 `B`。
  - `A` 使用**按列聚类**（column-wise k-means）。
  - `B` 使用**按行聚类**（row-wise k-means）。
- 训练策略：先加入 Deep k-Means 风格正则进行重训练，再在训练末期执行硬聚类与共享（可选）。

---

## 2. 参数形状与聚类单元定义

- 对于 LoRA 线性层，通常有：
  - `A`: shape `[r, in_features]`
  - `B`: shape `[out_features, r]`
- 聚类单元：
  - `A`：把每一列视为一个样本向量，样本数 `N_A = in_features`，向量维度 `d_A = r`。
  - `B`：把每一行视为一个样本向量，样本数 `N_B = out_features`，向量维度 `d_B = r`。
- 说明：这种定义与论文的“结构化向量样本聚类”一致，且天然契合 LoRA 的低秩结构。

---

## 3. 总体实施步骤

### 阶段 A：框架接入（不改训练行为）

1. 新增 `unilora_deepk` tuner 目录与基础文件：
   - `config.py`：定义 DeepK 超参数。
   - `layer.py`：定义层接口（可先复用现有 UniLoRA Layer 逻辑）。
   - `model.py`：实现正则计算与 finalize。
   - `__init__.py`：注册导出。
2. 注册到 PEFT 类型系统与导出入口：
   - `PeftType` 新增 `UNILORA_DEEPK`。
   - `peft/tuners/__init__.py`、`peft/__init__.py` 补充 import 与 `__all__`。
3. 保存/加载接入：
   - `save_and_load.py` 增加 `UNILORA_DEEPK` 分支，确保可保存并恢复 DeepK 参数与状态。

### 阶段 B：训练期正则（核心）

1. 在 `model.py` 中实现 `compute_deepk_loss(adapter_name, global_step)`：
   - 遍历所有 DeepK 层，分别提取 `A` 列样本与 `B` 行样本。
   - 逐层计算谱松弛 k-means 正则项并累加。
2. 实现 `F` 的 lazy update：
   - 每隔 `deepk_f_update_interval` step 更新一次 `F_A/F_B`。
   - 其他 step 复用缓存，降低开销。
3. 在训练脚本中把 `deepk_loss` 加到主损失：
   - `loss = task_loss + ramp * deepk_loss`。
   - 加 warmup，避免训练早期被聚类约束过强。

### 阶段 C：训练后硬共享（可开关）

1. 实现 `finalize_deepk_assignment()`：
   - 每层独立执行：
     - `A` 列聚类，得到 `A_codebook` + `A_col_indices`；
     - `B` 行聚类，得到 `B_codebook` + `B_row_indices`。
2. 回写硬共享权重用于最终评估（可选保留原值分支）。
3. 导出统计信息：
   - codebook 大小、索引大小、唯一向量数、变化比例等。

---

## 4. 训练脚本改动清单（`run_unilora_variants_glue.py`）

- `--variant` 增加：`unilora_deepk`。
- 新增参数（建议）：
  - `--deepk_num_clusters_a`
  - `--deepk_num_clusters_b`
  - `--deepk_tau`
  - `--deepk_warmup_ratio`
  - `--deepk_f_update_interval`
  - `--deepk_assign_stage`（`none|end`）
  - `--deepk_svd_rank_cap`（可选）
- `peft_config` 分支新增 `UniLoRADeepKConfig(...)`。
- 训练 loop 新增：
  - 调用 `compute_deepk_loss`；
  - 记录 TensorBoard 指标：`DeepK/Loss`, `DeepK/Reg_A`, `DeepK/Reg_B`, `DeepK/Tau`。
- 训练结束新增：
  - 若 `assign_stage=end`，调用 `finalize_deepk_assignment`；
  - 将 `deepk_stats` 与 `deepk_finalize_info` 写入结果 JSON。

---

## 5. 关键实现要点

### 5.1 每层独立（当前阶段）

- 不跨层拼接样本，不共享跨层 codebook。
- 每层都有自己的：
  - `A` 聚类辅助变量 `F_A`
  - `B` 聚类辅助变量 `F_B`
  - 聚类结果与导出缓存

### 5.2 A/B 分开处理

- `A` 列样本与 `B` 行样本维度都为 `r`，但统计分布通常不同，必须分开建模和聚类。
- `K_A` 与 `K_B` 允许不同，便于调参。

### 5.3 数值稳定与复杂度控制

- 正则强度 `tau` 先小后大（warmup）更稳。
- `F` 更新不要每步做，建议 interval 更新。
- 对超大层可加 `svd_rank_cap` 或随机子样本更新策略。

---

## 6. 注意事项与风险清单

1. **聚类方向不能写反**：
   - `A` 必须按列；
   - `B` 必须按行。
2. **维度约定要统一**：
   - 构造样本矩阵时，确保样本轴与特征轴一致，避免转置错误导致聚类失真。
3. **正则项尺度不平衡**：
   - `A` 与 `B` 样本数不同，建议做归一化（如按样本数取均值）再加权求和。
4. **硬赋值时精度突降**：
   - 建议先在 `assign_stage=none` 验证正则有效，再开启 `end`。
5. **保存加载兼容性**：
   - 新增参数名要带统一前缀，避免与既有 UniLoRA 变体冲突。
6. **统计口径一致性**：
   - 报告里区分“训练时软约束效果”与“训练后硬共享效果”，避免混淆。

---

## 7. 最小可运行里程碑（MVP）

- M1：能跑通 `variant=unilora_deepk`，但先不加正则，验证配置接线与训练流程。
- M2：加入 layer-wise DeepK 正则（`A` 列、`B` 行），并完成日志记录。
- M3：加入 `assign_stage=end` 的硬聚类 finalize 与结果导出。
- M4：补充结果汇总脚本，输出不同 `K_A/K_B/tau` 的对比表。

---

## 8. 首轮推荐超参（smoke）

- `deepk_num_clusters_a=16`
- `deepk_num_clusters_b=16`
- `deepk_tau=1e-5`（先稳）
- `deepk_warmup_ratio=0.1`
- `deepk_f_update_interval=100`
- `deepk_assign_stage=none`（先看训练稳定性）

稳定后再尝试：
- `deepk_tau` 提到 `5e-5` / `1e-4`
- 再打开 `deepk_assign_stage=end` 评估硬共享损失。
