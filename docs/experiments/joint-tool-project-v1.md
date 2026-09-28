# 工具分支受控真实验收 v1

运行目录 `joint-tool-project-v1`，源码冻结于 `600a0da9bf86fc62bb3e0414946e891f0b0fe80f`。这是预声明测量需求触发的工程验收，不是 Agent 自主发现缺口。真实路径已完成模型暂停→工具生成→离线检查→影子验证，三组在线效用比较正在运行，工具尚未晋升。

## 冻结设置

初始 toy 模型种子为 2901、2902，14 BCU、最多一轮；项目最多两个模型回合、一次工具更新。需求是 metric_spread，公开输出合同明确为 mean_metric、minimum_metric、metric_spread 及 limitations，不提供隐藏参考值。新的范围检查在工具生成前核对声明名称。

复用 `joint-project-acceptance-v1` 中尚未执行的工具 pipeline：开发种子 2501、2502，确认 2601、2602；影子为两个已审计的 Amazon 开发模型条件，各重复两次。原阶段阈值和费用未降低，旧配置 hash 未变。初始未运行准备稿在明确输出合同后另存为 prepared-draft，未发起任何角色或训练调用，不作为第二次研究尝试。

## 已完成证据

- 模型侧完成一次真实 evidence 调用和基线评估，以 measurement_required 停止；incumbent 仍为 root，源码未更改。
- 项目自动进入工具分支，真实 Codex 生成 diagnostic.py。实现 hash 为 `163e6fc32d4440d438e65eb3f7322f9a3459c229080ed836cd4a5a47aba5413e`。
- correctness、synthetic、robustness 各两次真实 CPU 隔离参考执行全部通过。
- 两个真实开发条件各两次 CPU 隔离影子执行通过，合计四次；没有重训 Amazon 或启动最终测试。
- 已完成阶段共 2 次 Codex 调用、10 次本地工具验证执行。后续在线比较调用另计，不能把这个阶段数字当作最终总数。

`completed-stages-acceptance.json` 核对候选/报告 hash、四阶段门槛与已结算子账本；模型终态在禁止 Agent 调用和评估的情况下恢复，报告、事件和费用均不变。审计确认没有 measurement_reviewed 事件，明确保留操作者预声明来源。

三组在线比较为 no_tool、existing_tool、candidate_tool，使用独立研究图与同一冻结条件。尚无完整效用结果，不能从离线或影子通过推导晋升、机制证明或后继模型改善。
