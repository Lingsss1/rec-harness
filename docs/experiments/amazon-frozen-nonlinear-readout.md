# Amazon：冻结隔离环境下的非线性残差读出

候选完成全部四类别开发训练，平均 HR@10 从 0.55350 降至 0.53425，差值 −0.01925，未通过预先冻结的确认门槛。四个 checkpoint 的独立 CPU 重评分与训练时归档排名完全一致。此结果不支持晋升候选，也不证明其假设机制不存在。

## 改动与对照

在 incumbent 的当前状态与历史上下文融合后，增加无偏置的两层残差读出 `h + W2(GELU(W1(h)))`，再保留有效位置掩码。输出层零初始化，初始函数保持与 incumbent 一致；48 维配置增加 4,608 个参数。训练从头初始化，无 checkpoint 迁移。源码 hash 为 `5c5be13b04297f0d51e0bf37c04f1372252adc2ca6f512c39d6a4b1fd1e29a2f`。

比较使用同一冻结执行版本 `execution-f40bbccd81ea5824320c803d`，运行环境 archive hash 为 `497a9aa72b35ff3dc7c4881bedeb46eb889683326f1ec7d7bbf62a6b4f752330`。每类别独立初始化；开发 seed 为 42，验证选 checkpoint。训练、候选集和评分协议均保持冻结。配对 baseline 是 `legacy_incumbent`，不是重新定义的弱基线。

## 开发集结果

| 类别 | incumbent HR@10 | 候选 HR@10 | 差值 |
|---|---:|---:|---:|
| Movies_and_TV | 0.601 | 0.555 | −0.046 |
| Industrial_and_Scientific | 0.531 | 0.536 | +0.005 |
| Electronics | 0.445 | 0.426 | −0.019 |
| CDs_and_Vinyl | 0.637 | 0.620 | −0.017 |
| 等权类别均值 | 0.55350 | 0.53425 | −0.01925 |

预先冻结的门槛要求平均差值严格大于 0.005，且优于各声明对照。候选未达门槛，不能启动候选确认训练。单个类别的 +0.005 不构成确认收益。本报告只涉及开发集，不涉及最终测试。

## 成本与执行状态

同执行版本的 12 个 incumbent 基线任务计费 2,880.9106531739235 GPU 槽位秒；本轮四个候选任务计费 1,030.4348702207208 GPU 槽位秒。槽位费用包含受控作业开销，不等同于 GPU 内核执行时间。基线和候选成本分列，均保留在账本中；旧研究成本不因这次隔离验收消失。

七个 Codex 角色已全部完成，evaluation 与 lead 判定淘汰，知识记录已写出，未启动确认实验。以下是事件账本中的实际外层调用次数和角色耗时；耗时包含隔离启动开销，不等同于模型推理时间，调用次数不包含会话内部搜索或底层模型请求。

| 角色 | 调用次数 | 耗时（秒） |
|---|---:|---:|
| evidence | 1 | 72.08 |
| measurement | 1 | 37.19 |
| hypothesis | 1 | 86.67 |
| experiment | 1 | 61.10 |
| evaluation | 1 | 59.08 |
| lead | 1 | 60.19 |
| knowledge | 1 | 39.89 |
| 合计 | 7，无重试 | 416.20 |

独立终态审计已通过：七角色均有已提交的产物与费用；四类别同执行版本、独立排名一致；恢复未新增调用、训练、事件或成本。incumbent 保持 `legacy_incumbent`，候选未晋升，确认和最终测试均未执行。该隔离 study 总计 3,911.3455233946443 GPU 槽位秒，包括基线与候选。此结果证明本次开发阶段自动研究完整执行与恢复，不证明候选收益或整个方案所有阶段已完成。

## 本地证据

运行根目录为 `/home/liuzhishan/lhr/rec-harness-runs/amazon-frozen-autoresearch-acceptance-v1`：

- `baseline-acceptance.json`：12 个基线及独立重评分审计。
- `research/rounds/round-1/development-readout.json`：冻结协议下的四类别开发比较。
- `research/rounds/round-1/independent-rescores/*/acceptance.json`：四次独立排名复核。
- `codex-call-summary.json`：实际七次角色调用与耗时。
- `research-audit.py`、`watch-research-audit.py`：终态验收脚本及监督流程；`research-acceptance.json` 与 `audit-watch-result.json` 均为 passed。
- `development-memory.json`、`ready-next-development-memory.json`、`memory-acceptance.json`：已审计的本轮开发记忆及合并后的 5 条历史记录，不包含确认或最终测试结果。

旧控制器将隔离运行库逐项展开到图证据，导致导入开销较大。当前开发工作树已改为保留完整 runtime lock 的紧凑引用并逐字节验证；此次运行的冻结代码未中途替换。
