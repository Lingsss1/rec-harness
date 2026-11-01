# BERT4Rec：RecHarness、Self-EvolveRec 与 MLE-STAR 实验记录

归档 2026-10-29 至 2026-10-31 完成的三个固定 BERT4Rec 家族、单模型搜索。2026-11-01 导出。每种方法各一次独立运行；MLE-STAR 与 Self-EvolveRec 包含原预算内的中断恢复，不能把恢复前后当成独立重复实验。

## 最终结果与预算

全部使用 `gpt-6.1-sol` / `medium`（用户最初称 middle），每项搜索上限 **43,200 GPU 秒**，4 张 GPU 并行。候选和测试详情分别见 [iterations.csv](iterations.csv)、[per-dataset.csv](per-dataset.csv)；机器可读汇总见 [comparisons.json](comparisons.json)。测试仅用于最终选中的模型，选择依据为验证集 HR@10。

| 方法 | 验证 HR@10 | 测试 HR@10 | 测试 NDCG@10 | 实际搜索 GPU 秒 | 成功 / 超时 / 崩溃 |
| --- | ---: | ---: | ---: | ---: | --- |
| RecHarness | 0.6275 | 0.5451 | 0.3536 | 43,194.0438 | 86 / 4 / 0 |
| Self-EvolveRec | 0.5988 | 0.5341 | 0.3386 | 42,585.1559 | 26 / 7 / 2 |
| MLE-STAR | 0.5737 | 0.5266 | 0.3324 | 41,997.1383 | 96 / 0 / 0 |

选中的候选代码：

- [RecHarness：round-0016-slot-1-attempt-1](searches/recharness/trials/round-0016-slot-1-attempt-1/train.py)
- [Self-EvolveRec：candidate-gpu0-dff3da699e0d52c5f471c9ae](searches/self_evolverec/trials/candidate-gpu0-dff3da699e0d52c5f471c9ae/train.py)
- [MLE-STAR：ab-4-0-t3](searches/mlestar/trials/ab-4-0-t3/train.py)

### 逐数据集测试

每格为 HR@10 / NDCG@10。总体结果为四类等权宏平均，使用源评估器报告的四位小数；不按用户数加权。

| 数据集 | 测试用户数 | RecHarness | Self-EvolveRec | MLE-STAR |
| --- | ---: | --- | --- | --- |
| Movies_and_TV | 7,467 | 0.5775 / 0.3883 | 0.5654 / 0.3700 | 0.5677 / 0.3711 |
| Industrial_and_Scientific | 10,000 | 0.5328 / 0.3281 | 0.5372 / 0.3292 | 0.5221 / 0.3189 |
| Electronics | 10,000 | 0.4426 / 0.2742 | 0.4224 / 0.2624 | 0.4199 / 0.2555 |
| CDs_and_Vinyl | 10,000 | 0.6276 / 0.4240 | 0.6115 / 0.3929 | 0.5966 / 0.3840 |

[per-dataset.json](per-dataset.json) 同时包含验证、测试的 HR@5/10/20、NDCG@5/10/20 和实际用户数。

## Token 与调用次数

| 方法 | 输入 tokens | 其中缓存输入 | 输出 tokens | 其中 reasoning | 输入+输出 | 调用数 | 有成功回执的调用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RecHarness | 2,925,342 | 705,792 | 46,399 | 10,993 | 2,971,741 | 117 | 117 |
| Self-EvolveRec | 71,553,764 | 37,039,232 | 1,885,401 | 170,682 | 73,439,165 | 1,906 | 1,901 |
| MLE-STAR | 2,497,207 | 95,104 | 508,951 | 17,265 | 3,006,158 | 110 | 110 |

调用数指 `calls/*/request.json` 对应的逻辑 CLI 请求，含失败请求；不等于底层 HTTP 请求数。输入已包含缓存输入，输出已包含 reasoning，不能再次相加。统计包括报告了 usage 的重试和恢复调用，不仅统计最终成功回执。

Self-EvolveRec 有 **1,915** 个尝试事件日志，其中 **14** 个未报告 usage，故其 token 数是**可核验的下界**。另外两种方法的全部尝试日志均有 usage。[calls.csv](calls.csv) 可按 scope/角色分析花费；[llm-attempts.csv](llm-attempts.csv) 保留缺失 usage 的尝试；[token-events.json](token-events.json) 保留每条 usage 的字段、线程/轮次、原文件 SHA-256 与行号。以 `(thread_id, turn_index)` 去重。

## 公平性口径与冷启动修复

- 数据为 Amazon Reviews 2023 四类别；参考 [6lyc/RecHarness](https://github.com/6lyc/RecHarness) 的 `run2.sh`，上游固定提交 `74a927fefaf7d120c52b6e3a0809958bd33984f0`。预处理 seed 0、训练/评估 seed 42，验证每类 200 用户，99 个不重复负样本，测试每类最多 10,000 用户。
- 三种方法冷启动 `train.py` 字节相同，SHA-256 为 `d51d266361556daac7a1c7e3dfbbc1aa6e46bcb7117f6cdbc71954385530c4ca`。已修复左填充时按长度减一索引的问题，并改为不泄露目标的 masked Cloze 训练；预测在已知历史后附加 MASK 并读取该位置。见 [修复差异](protocol/family/baselines/bert4rec/changes.diff) 与 [起点代码](searches/mlestar/baseline.py)。原始起点不是此次对照的起点。
- 单模型指**每个类别一个联合训练的 BERT4Rec 模型**；禁止独立模型融合。共享 worker、grader、结构审计及四类聚合口径。审计属于结构约束检查，其证据随候选保存。
- 搜索 GPU 秒为训练、验证、诊断、审计及失败/超时 worker 的实际耗时之和。LLM 等待、预检和最终测试另计；账本中保留真实剩余额度，未通过额外训练补齐到精确 43,200 秒。最终测试回执在各方法 `final-test/`。
- 每个任务 4 张 A800-SXM4-80GB；训练环境 Torch 2.5.1+cu121，每个 worker 4 线程，每任务最多 64 CPU 核。以各运行 `config.json` 与每候选 `resources.json` 为准。进程通过环境隔离测试入口，但不是操作系统级文件系统沙箱。
- 本地方法实现有明确适配：MLE-STAR 的 merge 必须输出一个 BERT4Rec 程序；Self-EvolveRec 用相同四类别指标和预算驱动原生演化；RecHarness 使用六臂、组合 jump 与 evidence memory v2。对应源代码与原始注册文件一并归档。

## 迭代与恢复证据

每个候选保留实际训练源码、预测入口（若生成）、训练/worker 日志、验证回执、结构审计与资源信息。`changes-from-baseline/` 是相对共同冷启动的完整代码差异，不代表每步的父子差异。

### RecHarness

[recharness-rounds.csv](recharness-rounds.csv) 覆盖最终 **24 轮**，执行编号 0–23 对应原生日志 1–24；[recharness-edits.json](recharness-edits.json) 保留每次提案的解释与精确修改。`searches/recharness/actual-changes/` 根据 memory 中的实际输入/输出源码生成父子 diff。完整选择、晋升、jump、memory 证据位于 [native](searches/recharness/native)。未使用搜索中途生成的第 18 轮分析快照冒充最终轨迹。

主要验证里程碑：冷启动 0.5425 → 全目录 masked CE 0.5537 → 嵌入归一化 0.5912 → FFN 宽度调整 0.6038 → 位置表示 0.6088 → 组合 jump 及后续调参 0.6188 → MASK 嵌入归一化豁免 0.6225 → 频率特征幅度 0.04 降至 0.02，达到最终 0.6275。其后候选与超时均保留。

### MLE-STAR

原运行在消耗 20,516.2520 GPU 秒后因 `Invalid MLE-STAR block selection` 中断：四条轨迹的选择回复只包含两条。恢复保留有效选择，仅以精确轨迹身份和允许符号约束补齐缺失两条；已完成训练和缓存响应复用，原账本继续累计。

见 [恢复前失败](protocol/recovery/jobs/bert4rec-mlestar/before/failure.json)、[恢复前账本](protocol/recovery/jobs/bert4rec-mlestar/before/budget.json)、[选择修复记录](searches/mlestar/recovery-selection)、[实际恢复代码](source-at-export/amazon_family_recovery/controller.py)。失败记录是历史证据，最终状态以完整 `summary.json` 为准。

### Self-EvolveRec

原运行在消耗 25,126.9234 GPU 秒后因找不到 `codex` 可执行文件而中断。恢复使用原轨迹 checkpoint（进度 5、5、3、4），保留 pending parent、同次运行缓存和原账本；通过固定 CLI 路径与启动异常重试继续。最终 35 个候选含恢复前记录。

见 [恢复前失败](protocol/recovery/jobs/bert4rec-self_evolverec/before/failure.json)、[恢复前账本](protocol/recovery/jobs/bert4rec-self_evolverec/before/budget.json) 及四条轨迹恢复元数据。`native/**/programs/` 保留程序代码、想法、父节点和指标，明确删去包含用户反馈/诊断文本的字段；聚合 directional feedback 保留，逐用户文本不发布。

## 文件与校验

- `comparisons.*`、`per-dataset.*`：最终方法与逐数据集结果。
- `iterations.*`：221 个已结算候选、真实状态与预算；失败时未形成回执的源文件仍可在候选目录找到。
- `calls.*`、`llm-attempts.*`、`token-events.*`：2,133 次调用与逐尝试 usage。
- `searches/{recharness,mlestar,self_evolverec}/`：配置、最终总结、账本、候选源码/日志、最终测试及方法原生记录。
- `protocol/`：冷启动差异、原始 freeze/恢复注册、数据准备指纹与许可。注册文件来自多模型 campaign，其中其他模型条目仅作为原始注册上下文，并不代表本次发布了其他模型结果。
- `source-at-export/`：注册代码与公共适配代码快照；[registered-source-audit.json](registered-source-audit.json) 逐项核验原注册指纹。绝对路径仅为原服务器的来源证据，迁移时需要重新配置。
- `copy-sources.json`：每份原样复制文件的来源、大小和 SHA-256；`derived-sources.json`：脱敏/派生文件的源 SHA-256 和转换规则；`manifest.json`：本目录文件清单及 SHA-256，manifest 自身不递归纳入哈希。

本归档不包含数据集内容、模型权重、虚拟环境、认证文件、原始 CLI 事件流、完整请求提示词、逐用户模拟输入/回复。原始 CLI usage 提取为独立记录并保留来源指纹，提示词保留哈希/长度/schema。因缺少数据和权重，本目录用于审计和重建搜索实现，不能直接复现已训练模型的推理；需另行准备相同数据、环境并重训。

离线校验：

```bash
python3 iteration-records/amazon-bert4rec-20261029-31/verify_records.py
```

校验覆盖所有导出文件哈希、共同起点、选中候选源码/结构审计、221 个候选的预算结算、逐类别宏平均、调用/尝试/事件 token 三方汇总及注册源码指纹。`export_records.py` 依赖原服务器本地实验资产，重新导出不会训练或调用 LLM。
