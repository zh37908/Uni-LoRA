# Slurm 提交脚本编写经验总结

本文档基于本仓库的实际提交脚本总结，主要参考：

- `NLU/peft/examples/sequence_classification/submit_prolosa_ablation_best_cola_mrpc_extend5seeds.sh`（单作业内 4 GPU 并行跑多个小实验）
- `math_instruction_tuning/submit_instruction_tuning_unilora_rosa_snip_multi_gpu_2to1_two_models_l20_2gpu.sh`（Job Array 拆分多个模型，每个任务独占 2 GPU）

---

## 1. SBATCH 头部：资源申请

```bash
#!/bin/bash
#SBATCH --job-name=xxx            # 简短的作业名，方便 squeue 查看
#SBATCH --nodes=1                 # 节点数，单机训练固定为 1
#SBATCH --ntasks-per-node=1       # 任务数；用 srun 并行跑多个子任务时相应调大
#SBATCH --cpus-per-task=16        # 每个任务的 CPU 核数（dataloader 需要）
#SBATCH --gpus-per-node=2         # 申请的 GPU 数
#SBATCH --time=48:00:00           # 时间上限，宁可给足，超时会被直接杀掉
#SBATCH --partition=gpu-l20       # 分区：amd / intel / gpu-a30 / gpu-l20
#SBATCH --account=shsong          # 计费账户
#SBATCH --output=logs/xxx_%j.out  # 标准输出（%j = JobID）
#SBATCH --error=logs/xxx_%j.err   # 标准错误
```

要点：

- **日志目录必须先存在**。`--output` 指向 `logs/` 时，如果目录不存在作业会直接失败且没有任何日志。所以脚本开头必须 `mkdir -p logs`（Slurm 在脚本运行前就要写日志文件，但实践中先建目录再排队最稳妥的做法是：提交前就保证 `logs/` 存在）。
- **`ntasks-per-node` 与并行方式匹配**：
  - 单进程训练（哪怕用多卡 device_map）：`--ntasks-per-node=1`；
  - 作业内用 `srun` 并行多个单卡任务：`--ntasks-per-node=N`，N = 并行槽位数 = GPU 数。
- 用 Job Array 时日志名用 `%A_%a`（主 JobID + 数组下标），不用 `%j`。

## 2. 环境初始化的固定顺序

顺序很重要，推荐固定为：

```bash
mkdir -p logs

# 1) 先激活 conda，再开 set -u
source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate my_env

# 2) 再开严格模式
set -euo pipefail

# 3) 清理代理，避免计算节点上网络请求挂死
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

# 4) 限制 CPU 线程数，避免多任务互相争抢
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
```

原因：

- **conda 激活要放在 `set -euo pipefail` 之前**：conda 的激活脚本会引用未定义变量，在 `set -u` 下会报错退出。
- **计算节点必须清理代理变量**：登录节点的代理配置会被继承，HuggingFace 下载等请求在计算节点会因代理不通而无限挂起，表现为作业"卡住不动"。
- **限制线程数**：单节点跑多个并行任务时，默认每个进程都会开满全部核，互相争抢反而变慢。

## 3. 工作目录

两种可靠写法：

```bash
# 写法 A：支持覆盖，默认用提交时的目录，最后兜底写死绝对路径
WORK_DIR="${WORK_DIR:-${SLURM_SUBMIT_DIR:-/home/hzhaobi/Uni-LoRA/NLU/peft/examples/sequence_classification}}"
cd "${WORK_DIR}" || exit 1

# 写法 B：直接写死绝对路径（简单可靠）
cd /home/hzhaobi/Uni-LoRA/math_instruction_tuning
```

如果用仓库内的本地包（比如本地 PEFT），记得设置：

```bash
export PYTHONPATH="${PWD}/peft/src:${PYTHONPATH:-}"
```

## 4. 超参数全部用环境变量 + 默认值

所有可调参数统一写成 `${VAR:-默认值}`：

```bash
MODEL="${MODEL:-roberta-large}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEEDS=(${SEEDS:-3 4})
```

好处：

- 不改脚本就能覆盖：`SEEDS="0 1 2" BATCH_SIZE=16 sbatch submit_xxx.sh`；
- 脚本本身就是默认配置的记录，可复现。

**启动前 echo 一遍全部关键配置**，写进 `.out` 日志，事后排查/复现时非常有用：

```bash
echo ">>> model=${MODEL} tasks=${TASKS[*]} seeds=${SEEDS[*]} batch_size=${BATCH_SIZE}"
echo ">>> output=${OUTPUT}"
```

## 5. 参数合法性预检查

启动训练前先做便宜的一致性检查，避免排了几小时队之后才因配置错误失败：

```bash
if [[ $((THETA_D_LENGTH + SPARSE_BUDGET)) -ne "${TOTAL_TRAINABLE_BUDGET}" ]]; then
    echo "theta_d_length + sparse_budget must equal total_trainable_budget." >&2
    exit 1
fi
```

## 6. 网络：先预热缓存，再切离线模式

计算节点网络不可靠，正确姿势是**先联网把模型和数据集下载进缓存，然后强制离线**：

```bash
echo ">>> Pre-warming cache..."
python - <<PY
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
AutoTokenizer.from_pretrained("${MODEL}")
load_dataset("nyu-mll/glue", "cola")
PY

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

这样即使中途 HuggingFace 抽风，训练也不会挂起。

## 7. 两种并行模式的选择

### 模式 A：Job Array —— 少量大任务，每个任务独占若干 GPU

适合"跑 2 个模型 / N 个配置，每个都要占满整个作业资源"的场景：

```bash
#SBATCH --array=0-1
#SBATCH --output=logs/xxx_%A_%a.out

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "${TASK_ID}" in
    0) MODEL_TAG="gemma";   BASE_MODEL="google/gemma-7b" ;;
    1) MODEL_TAG="mistral"; BASE_MODEL="mistralai/Mistral-7B-v0.1" ;;
    *) echo "Unsupported task id ${TASK_ID}" >&2; exit 1 ;;
esac
```

优点：每个配置是独立作业，互不影响，可单独重跑（`sbatch --array=1 xxx.sh`）。

### 模式 B：单作业内并行队列 —— 大量小任务共享多卡

适合"几十个 (task, seed, lr) 组合，每个只需 1 张卡"的场景。先把所有命令生成到临时文件，再用 `xargs -P` 做固定槽位的并行队列：

```bash
CMD_LIST="$(mktemp)"
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    FULL_CMD="srun --ntasks=1 --nodes=1 --exclusive --gres=gpu:1 \
      --cpus-per-task=16 --cpu-bind=none --gpu-bind=single:1 \
      python ${SCRIPT} ... > ${LOG_FILE} 2>&1"
    echo "${FULL_CMD}" >> "${CMD_LIST}"
  done
done

# 4 个并行槽位 = 申请的 4 张 GPU
xargs -I {} -P 4 bash -c "{}" < "${CMD_LIST}"
rm -f "${CMD_LIST}"
```

关键点：

- `srun --exclusive --gres=gpu:1 --gpu-bind=single:1` 让每个子任务独占 1 张卡，Slurm 自动分配空闲的那张；
- `xargs -P N` 的 N 必须等于 GPU 数（也等于 `--ntasks-per-node`），跑完一个自动补位下一个；
- 每个子任务把输出重定向到自己的 `log_*.txt`，主 `.out` 日志只留调度信息。

## 8. 幂等与断点续跑

脚本要支持"失败后原样重新 sbatch"：

- **跳过已完成的结果**（模式 B）：

```bash
if [[ -s "${RESULT_JSON}" ]]; then
  echo "Skip existing result: ${RESULT_JSON}"
  continue
fi
```

- **从 checkpoint 恢复**（长训练）：用数组安全地拼可选参数，避免空值传参问题：

```bash
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
python train.py ... "${RESUME_ARGS[@]}"
```

配合 `--save_strategy steps --save_steps 100 --save_total_limit 5`，即使超时被杀也能续跑。

## 9. 输出目录组织

按层级组织，路径本身就携带实验信息：

```text
${OUT_ROOT}/${MODEL}/${TASK}/${SUBEXP_DIR}/${METHOD_NAME}/seed_${SEED}/
├── log_lr_5e-3.txt        # 该 run 的完整训练日志
└── <variant>_<task>_<model>_lr<lr>_seed<seed>.json   # 结果文件（也是幂等判断依据）
```

每一层用 `mkdir -p` 创建；结果文件名包含全部关键超参，后续汇总脚本可以直接解析。

## 10. 提交与监控常用命令

```bash
sbatch submit_xxx.sh                 # 提交
SEEDS="0 1" sbatch submit_xxx.sh     # 覆盖默认参数提交
squeue -u hzhaobi                    # 看自己所有作业
squeue -j <jobid>                    # 看指定作业（R=运行中，PD=排队）
scancel <jobid>                      # 取消
tail -f logs/xxx_<jobid>.out         # 跟踪主日志
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # 事后查资源使用
```

## 11. 常见坑清单

| 问题 | 现象 | 解法 |
| --- | --- | --- |
| `logs/` 不存在 | 作业秒失败且无日志 | 提交前 `mkdir -p logs` |
| conda 激活放在 `set -u` 之后 | 脚本一开始就退出 | 先激活再 `set -euo pipefail` |
| 计算节点继承了代理变量 | 下载模型/数据时无限挂起 | 脚本内 `unset` 全部代理变量 |
| 训练中途访问 HuggingFace | 随机挂起或报网络错误 | 预热缓存后设 `*_OFFLINE=1` |
| `xargs -P` 数量 > GPU 数 | 子任务排队等卡或 OOM | 并行度 = GPU 数 = ntasks |
| 多进程抢 CPU 线程 | 并行后反而变慢 | 设 `OMP/MKL_NUM_THREADS` |
| 超时被杀且无 checkpoint | 全部白跑 | 给足 `--time` + 定期存 ckpt |
| 配置错误但排队几小时后才发现 | 浪费队列时间 | 启动前做参数预检查并 `exit 1` |

## 12. 最小模板

```bash
#!/bin/bash
#SBATCH --job-name=my_exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-l20
#SBATCH --account=shsong
#SBATCH --output=logs/my_exp_%j.out
#SBATCH --error=logs/my_exp_%j.err

mkdir -p logs

source /home/hzhaobi/miniconda3/etc/profile.d/conda.sh
conda activate my_env

set -euo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

cd /home/hzhaobi/Uni-LoRA/<subdir>

MODEL="${MODEL:-roberta-large}"
SEED="${SEED:-42}"
OUTPUT="${OUTPUT:-output/my_exp_seed${SEED}}"
mkdir -p "${OUTPUT}"

echo ">>> model=${MODEL} seed=${SEED} output=${OUTPUT}"

python train.py \
    --model_name "${MODEL}" \
    --seed "${SEED}" \
    --out_dir "${OUTPUT}"

echo "Done."
```
