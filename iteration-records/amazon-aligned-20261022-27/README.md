# Amazon 四类别同预算迭代记录（2026-10-22 至 27）

本记录包归档 MLEvolve、MLE-STAR、Self-EvolveRec 和 RecHarness 的新增实验。它是现有运行证据的发布快照，不会启动、恢复或修改实验。完整状态以 [实验索引](comparisons.csv)、[逐候选索引](iterations.csv) 和 [调用索引](calls.csv) 为准。

## 协议

Amazon Reviews 2023，四类别为 Movies_and_TV、Industrial_and_Scientific、Electronics、CDs_and_Vinyl。预处理种子 0，训练和评测种子 42，SASRec per-dataset 冷启动。每次独立搜索预算为累计 43,200 GPU 秒，4 个单卡并行槽、64 CPU 核；训练、验证、诊断、失败和超时 worker 均计入，LLM 时间与最终测试推理开销分别记录。

LLM 为 `gpt-6.1-sol / medium`（用户所说 middle 对应配置值 medium）。公共宿主验证为每类别 200 用户、99 个负例，宏平均 HR@10；最终测试每类别最多 10,000 用户，Movies_and_TV 实际 7,467 用户。先冻结验证选择，再测试。不同重复保持相同数据与训练种子，使用独立 LLM 生成；这些重复不是多个训练随机种子的独立验证。

## 搜索运行

原始搜索允许集成，不能将其测试成绩直接当作单模型成绩。`reruns-20261026-mlevolve-single-*` 在搜索时关闭原生 fusion/aggregation，并在接受候选前检查预测器。Self-EvolveRec 保留原生搜索，最终按验证集过滤选择合规单模型。

| 实验 | 状态 | 验证 HR@10 | 测试 HR@10 | 已用 GPU 秒 |
| --- | --- | ---: | ---: | ---: |
| [original-mlevolve](searches/original-mlevolve) | complete | 0.6250 | 0.5575 | 43129.8 |
| [original-mlestar](searches/original-mlestar) | complete | 0.5862 | 0.5079 | 43178.2 |
| [original-self_evolverec](searches/original-self_evolverec) | complete | 0.5863 | 0.5161 | 42465.2 |
| [original-recharness](searches/original-recharness) | complete | 0.5950 | 0.5342 | 43177.7 |
| [original-recharness-replica](searches/original-recharness-replica) | complete | 0.5988 | 0.5291 | 43178.9 |
| [repeats-20261025-self_evolverec-run-02](searches/repeats-20261025-self_evolverec-run-02) | interrupted | 0.5687 | — | 42379.9 |
| [repeats-20261025-self_evolverec-run-03](searches/repeats-20261025-self_evolverec-run-03) | interrupted | 0.5487 | — | 42856.8 |
| [repeats-20261025-mlestar-run-02](searches/repeats-20261025-mlestar-run-02) | complete | 0.6412 | 0.5662 | 42691.3 |
| [repeats-20261025-mlestar-run-03](searches/repeats-20261025-mlestar-run-03) | complete | 0.6450 | 0.5716 | 43186.1 |
| [reruns-20261026-self_evolverec-fixed-run-01](searches/reruns-20261026-self_evolverec-fixed-run-01) | interrupted | 0.5300 | — | 10908.7 |
| [reruns-20261026-self_evolverec-fixed-run-02](searches/reruns-20261026-self_evolverec-fixed-run-02) | interrupted | 0.5050 | — | 1749.7 |
| [reruns-20261026-mlevolve-single-run-01](searches/reruns-20261026-mlevolve-single-run-01) | interrupted | 0.5650 | — | 7547.1 |
| [reruns-20261026-mlevolve-single-run-02](searches/reruns-20261026-mlevolve-single-run-02) | complete | 0.6275 | 0.5475 | 43176.4 |
| [reruns-20261026-mlevolve-single-run-03](searches/reruns-20261026-mlevolve-single-run-03) | complete | 0.5800 | 0.5113 | 43058.8 |
| [self-evolverec-resilient-01](searches/self-evolverec-resilient-01) | prepared_or_preflight_only | — | — | 0.0 |
| [self-evolverec-resilient-02](searches/self-evolverec-resilient-02) | prepared_or_preflight_only | — | — | 0.0 |

`complete` 表示有正式最终汇总；`interrupted` 表示异常退出，已有最佳验证成绩不等于完成一次搜索；`prepared_or_preflight_only` 表示只有配置或短预检，不能计作正式运行。最新两次 resilient 重跑在本快照时尚未正式启动。

## 原预算内的单模型重评

这一组是对已有搜索的事后资格筛选：保留全部原搜索消耗，包括集成候选，按宿主验证 HR@10 选最佳合规单模型，再重评或复用完全相同预测器的已有测试。它不是从头禁止集成的新搜索。每类别一个独立模型允许；跨模型、跨种子、检查点分数/排名融合，以及与独立拟合的转移/流行度分数相加均排除；一个网络内联合训练的特征和层允许。

| 重评记录 | 原搜索状态 | 测试 HR@10 |
| --- | --- | ---: |
| [mlestar-01](single-model-reselection/mlestar-01/summary.json) | complete | 0.5079 |
| [mlestar-02](single-model-reselection/mlestar-02/summary.json) | complete | 0.5311 |
| [mlestar-03](single-model-reselection/mlestar-03/summary.json) | complete | 0.5323 |
| [mlevolve-01](single-model-reselection/mlevolve-01/summary.json) | complete | 0.4855 |
| [recharness-01](single-model-reselection/recharness-01/summary.json) | complete | 0.5342 |
| [recharness-02](single-model-reselection/recharness-02/summary.json) | complete | 0.5291 |
| [self_evolverec-01](single-model-reselection/self_evolverec-01/summary.json) | complete | 0.5161 |
| [self_evolverec-02](single-model-reselection/self_evolverec-02/summary.json) | interrupted | 0.5051 |
| [self_evolverec-03](single-model-reselection/self_evolverec-03/summary.json) | interrupted | 0.4944 |

Self-EvolveRec 旧重复中断后得到的单模型测试成绩仍保留 interrupted 标记，不应算作完整成功重复。仅凭这些少量重复不作统计显著性结论。

## 中断和修复

早期 Self-EvolveRec 的演化诊断文件遗漏 `diagnosis_interpreter_prompt`，另有诊断原始明细使提示词超过 1,048,576 字符。2026-10-26 修复了提示函数保护、诊断明细压缩和历史长度限制；其后两次重跑的退出原因为 LLM 1,200 秒调用超时。MLEvolve 去集成第 1 次也被 LLM 超时打断，另外两次正常完成。

2026-10-27 新增的 resilient 协议包含有界调用重试、成功请求/反馈缓存、原生种群和迭代检查点、待处理父节点/灵感持久化、进程组清理，以及两次运行共享的 4 路 LLM 并发上限。配置和已完成预检已归档；未启动的正式实验没有最终成绩。发布动作没有恢复中断搜索或追加计算预算。

## 源码和发布边界

[导出时适配器](source-at-export/amazon_aligned) 是上传时快照，不代表每次历史运行都用了相同源码。各历史冻结清单和 protocol-before/protocol-after 快照保留在 [协议目录](protocol)。原生方法固定提交、具体任务适配差异、预算账本和外部评测器哈希见各运行 config、frozen、budget、receipt 及汇总。

保留候选源代码、训练/评测日志、宿主指标、预算、选择、最终公开回答和结构化方案；原始 LLM 提示词仅保存哈希/长度索引。原始 CLI 事件、逐用户模拟输入/原始诊断明细、数据集、权重、二进制索引和凭据留在本地。JSON 中绝对路径只作溯源，不应在其他机器直接执行。模型权重未上传，因此记录包可审阅和核验，但不能独立重放已有预测。

每个逐字复制文件的来源和原始哈希见 [copy-sources.json](copy-sources.json)，整个发布包的文件校验见 [manifest.json](manifest.json)。离线核验：

```bash
python3 iteration-records/amazon-aligned-20261022-27/verify_records.py
```
