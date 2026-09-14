# 另一台服务器获取实验分支

日期：2026-09-14。远端：`https://github.com/zh37908/Uni-LoRA.git`。
本次新增分支：`iclr2027-two-server-experiments`，基于 `it-final-version` 的 `c314fd3`。
原有 `main`、`it-final-version` 不改写；采用正常新增提交，不 force-push、不清理历史对象。

## 1. 首次获取：推荐浅克隆

在新服务器选择一个容量足够的工作目录，执行下面三条命令。`Uni-LoRA` 目标目录应尚不存在：

```bash
git clone --depth 1 --single-branch --branch iclr2027-two-server-experiments https://github.com/zh37908/Uni-LoRA.git Uni-LoRA
cd Uni-LoRA
git log -1 --oneline
```

浅克隆只取得该分支当前快照，避免下载包含历史日志和权重的大量祖先对象。服务器已有 GitHub SSH 认证时，也可把 URL 换为 `git@github.com:zh37908/Uni-LoRA.git`；读取需要远端仓库访问权限，推送还需写权限。不要在命令或脚本中写入 token。

## 2. 新服务器已有这个仓库

先在其仓库目录运行 `git status --short`。工作区干净时：

```bash
git fetch origin iclr2027-two-server-experiments
git switch --track -c iclr2027-two-server-experiments origin/iclr2027-two-server-experiments
```

若此前使用了 `--single-branch` 克隆，远端 fetch 规则可能不包含此分支，先增加跟踪规则再获取：

```bash
git remote set-branches --add origin iclr2027-two-server-experiments
git fetch origin iclr2027-two-server-experiments
git switch --track -c iclr2027-two-server-experiments origin/iclr2027-two-server-experiments
```

若本地分支已经存在，改为 `git switch iclr2027-two-server-experiments`，再执行 `git pull --ff-only origin iclr2027-two-server-experiments`。若有未提交的本地改动，优先在另一个新目录浅克隆，不执行 `reset --hard` 或覆盖已有工作区。

后续两服务器同步同一分支时，确保当前在该分支，且没有正在依赖这些源码的训练进程，再运行：

```bash
git pull --ff-only origin iclr2027-two-server-experiments
```

每次正式运行记录 `git rev-parse HEAD`。在另一台服务器开发新实现时，建议另建个人实现分支，减少两台服务器同时推送同一分支产生的冲突：

```bash
git switch -c iclr2027-h800-implementation
```

## 3. 此分支上传了什么

| 内容 | 处理 |
|---|---|
| NLU / math / instruction / ViT / synthetic 的源码、Slurm 脚本、custom PEFT 后端 | 上传当前源文件与新增实现 |
| ICLR 2027 论文 TeX/Bib/样式、所需图、约 1.9 MiB 的当前 PDF | 上传，方便两边引用同一论文版本 |
| 两周计划、冻结 specs/manifests、精简 CSV/JSON/TeX 汇总 | 上传；运行进度文字和汇总是对应日期快照 |
| ViT/table5/vendor 和 revision/snapshots 的冻结源码 | 上传，保留实现来源和版本；排除其中缓存 |
| 当前 `unilora_modern` 环境包版本 | 上传只含版本信息的 JSON；不复制 conda 环境或认证信息 |
| safetensors / pt / pth / bin / checkpoint、Fisher 张量、逐运行大型产物 | 不上传；已跟踪的对应产物仅从新分支索引中移除，本地保留 |
| 日志、TensorBoard、pyc、egg-info、下载数据/压缩包、旧论文备份 | 不上传；新增 `.gitignore` 规则 |
| `math_instruction_tuning/output`、`output_merged` | 是本机指向 `/project/shsong/hzhaobi/` 的软链接，不上传 |
| `RoSA/` | 是独立第三方 Git 仓库，不当作普通目录或无来源 gitlink 嵌入本仓库 |

旧分支另外含 8 个没有 `.gitmodules` 配置的外部仓库指针（参考方法和旧 bitsandbytes checkout），无法靠普通 clone 还原。本分支移除这些无配套配置的指针，保留本地目录，并将原路径/commit 记录在 [external_checkouts_20260914.json](external_checkouts_20260914.json)；未核实的 upstream 不猜测填写。它们不是本轮 BF16 微调的依赖。

盘点发现：原 HEAD 文件总大小约 **1,542 MiB**，主要是已跟踪结果日志/张量；`.git` 本地约 **6.2 GiB**，包括历史对象与既有临时 pack。新未跟踪数据里有约 **2.6 GiB** 的 FGVC 压缩包。权重软链接目标的真实容量不包含在这几个数中。本次不删除历史、不运行清理命令；浅克隆用于避免在新服务器下载旧历史。

源码上传不等于原服务器所有实验状态的镜像：`results/`、`state.json`、worker 运行状态和模型文件不会随 Git 同步。历史 Transformer 的 `synthetic/transformer_lab/` 目前只有旧字节码/日志，没有可读源码，本次不将它包装为可复现源码；已有审计报告及新 `synthetic/revision_controls/run_controls.py` 保留。若需复核该历史运行，另行传输原始证据归档。

## 4. Git 之外还需要准备的资源

### 模型、数据与恢复状态

- 新训练重新获取指定 revision 的 `NousResearch/Meta-Llama-3.1-8B` 和 **`Qwen/Qwen3-8B-Base`**，以及 MetaMathQA/SmolTalk；另行生成并冻结计划要求的题目分组、内部验证集和哈希。
- 旧脚本默认 `Qwen/Qwen3-8B`，与 Base 不同，必须显式指定。新两周计划的 100K/98K/2K 划分与现代指令 collator 尚需按计划实现；旧脚本不是已完成的新协议入口。
- 原仓库 `math_instruction_tuning/data/math_eval/` 中已有的小型数学评测输入保留在 Git；MATH500 要核对实际指定文件，不能把默认 `MATH_test.jsonl` 自动当作 MATH500。
- 原 `instruction_tuning/data/mmlu/` 四个生成格式 JSON 合计约 54 MiB，已从此分支移除跟踪。若使用旧 QLoRA MMLU 回调，需要单独传输这四个文件；新的 lm-eval 路径从其数据源获取数据，不等同于这些旧文件。
- 如需接续旧训练，用单独文件传输拷贝该 run 的 adapter、optimizer/scheduler/RNG、稀疏支持及 manifest，核对哈希；只有 adapter 不足以实现精确续训。新协议要求从原始基座训练的实验，不用旧 checkpoint 替代。
- 在新服务器自选项目存储位置建立 output/output_merged 或使用 launcher 的输出参数，不照搬旧服务器 `/project/shsong/...` 的软链接目标。

### 环境和 custom PEFT

[environment_unilora_modern_20260914.json](environment_unilora_modern_20260914.json) 是本机实测的包版本清单，Python 3.11.16，关键包为 torch 2.9.0、transformers 4.57.6、accelerate 1.14.0、datasets 5.0.1、vllm 0.11.2。它是环境快照，**不是已经在新服务器验证过的通用 lockfile**；CUDA wheel/驱动兼容性需在新服务器确认。

`math_instruction_tuning/requirements.txt` 是旧环境配方，不应直接用它覆盖新服务器的现代环境。多个目录包含不同版本的 custom PEFT，不能只安装 PyPI `peft` 就认为支持 Uni-LoRA/ProLoSA。数学和新指令开发可在仓库根目录按以下方式检查：

```bash
export UNILORA_REPO="$(pwd)"
export PYTHONPATH="$UNILORA_REPO/math_instruction_tuning/peft/src${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import peft; from peft import LoraConfig, UniLoRAConfig, UniLoRARoSASnipConfig; print(peft.__file__)'
```

输出应指向本次 clone 的 `math_instruction_tuning/peft/src/peft/__init__.py`。NLU、公平 GLUE 和 VB-LoRA/VeRA/FourierFT 现有数学基线使用 `NLU/peft/src`；为这些进程分别设置对应 PYTHONPATH，不在同一个已导入 peft 的 Python 进程中切换后端。

原始第三方 RoSA 本次记录来源为 `https://github.com/IST-DASLab/RoSA.git`，revision `5f70c49e67a43beaa09eeeb8287d4bd3554af365`。当前 GLUE 用的是本仓库 `NLU/peft/src/peft/tuners/lora_rosa/` 实现，不依赖这个外部 checkout；只有需要研究原实现时才单独克隆该版本。

### Slurm 设置

原 `.sh` 常写死 `--account=shsong`、`gpu-rtx5880/gpu-l20`、`/home/hzhaobi/miniconda3` 和仓库绝对路径。新服务器截图为账户 `songust`、`normal/normal_qos`，这些值必须重新核实后写入新的 launcher；不要原样提交旧脚本。先在分配的作业内检查实际 GPU 型号/显存，再做完整模型 smoke，保持实验精度、数据和有效 batch 不变。

Git 分支连接完成后，优先阅读 [双服务器两周方案](../ICLR_2027/two_server_finetuning_plan_20260914.md)。本次工作只整理并同步 Git，不启动或取消 Slurm 训练。

## 5. 本次打包验证

- 当前 Git 快照约 95 MiB，最大文件约 4 MiB；没有 checkpoint、模型权重、输出目录软链接或无 `.gitmodules` 的 gitlink。
- 196 个新增/修改 Python 文件通过 AST 语法检查，60 个 Shell 脚本通过 `bash -n`。
- 已检查新文档本地链接，确认 1,565 个从索引排除的原跟踪产物仍保留在原服务器磁盘。
- 在仅含 Git 内容的临时快照内尝试 CPU 导入 custom PEFT/训练入口，但 120 秒超时，未完成运行时导入验证；不能将本次语法检查当作新服务器依赖或 GPU 训练验证。
- 原有文档、样式及冻结源码中存在行尾空白，本次未为格式检查批量改写这些实验快照。未运行完整训练测试。
