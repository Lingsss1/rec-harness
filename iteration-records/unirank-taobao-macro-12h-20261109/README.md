# UniRank 淘宝：12 小时平均 AUC 自主研究迭代记录

归档时间：2026-11-10T04:06:44+00:00。研究 `unirank-taobao-macro-12h-20261109T043023Z` 已结束；74 轮研究、267 次课题训练尝试、85 次 Agent 调用、11 个动态课题。DIN 基线和成本校准另列，仍计入原研究预算。

从新的 DIN 起跑，四任务联合训练，验证集点击、加购、收藏、购买 AUC 的等权平均贯穿 checkpoint、父模型、课题收益和最终选模。本轮与之前以点击选模的 48 小时实验不同。每 GPU 一个课题，每 6 个完成轮次决策继续、切换或合并。研究模型为 gpt-6.1-sol / high；个人知识未启用，行业参考为 UniRank 的 15 种对比方法。

## 最终淘宝测试

前三从每个真实课题完成轮次后保留的最优模型中选出，按验证平均 AUC 排序，并对相同代码、配置和 checkpoint 去重。整组先冻结，再回放验证集，通过后执行测试；没有依据测试成绩重选。DIN 为额外参照。

| 模型 | 验证平均 AUC | 测试平均 AUC | 测试平均 Logloss | 参数(M) | 训练(s) |
|---|---:|---:|---:|---:|---:|
| g3-r0018-pair_occupancy_task_private_linear | 0.83241374 | 0.83209366 | 0.05357085 | 33.177 | 610.1 |
| g1-r0012-pair_compact_din | 0.83236493 | 0.83197889 | 0.05367173 | 33.150 | 512.5 |
| g0-r0017-pair_rank_occupancy | 0.83188684 | 0.83168092 | 0.05360667 | 33.130 | 514.8 |
| baseline-din | 0.80992339 | 0.80734774 | 0.05566622 | 32.059 | 293.1 |

逐任务 AUC、Logloss、正例数和样本数见 [final-test-tasks.csv](final-test-tasks.csv)。前三均为单种子；第一、第二名测试平均 AUC 仅相差约 0.000115，不能视为显著优势或跨数据集泛化证明。

## 最优模型的记录路线

以下按试验的 parent 字段回溯；它反映记录中的承接关系。跨课题导入和实际代码继承请结合 proposal 与源码差异查看，连线不构成机制归因。

| 模型 | 验证平均 AUC | 角色 |
|---|---:|---|
| baseline-din | 0.80992339 | baseline |
| g0-r0000-recency_slots | 0.81838049 | control |
| g0-r0001-grouped_recency | 0.81939167 | candidate |
| g0-r0002-replica_seven | 0.81978673 | control |
| g0-r0003-relation_content_counts | 0.82757048 | candidate |
| g0-r0005-linear_moment | 0.82832627 | candidate |
| g0-r0007-duplicate_full | 0.82858467 | control |
| g0-r0008-relation_bank | 0.82911961 | candidate |
| g0-r0009-relation_content | 0.82996259 | candidate |
| g0-r0010-feedback_factorized | 0.82999920 | control |
| g0-r0011-memory_joint_collapsed | 0.83037892 | control |
| g0-r0013-pair_joint_both | 0.83140952 | candidate |
| g0-r0016-pair_rank_joint | 0.83185845 | candidate |
| g0-r0017-pair_rank_occupancy | 0.83188684 | control |
| g3-r0018-pair_occupancy_task_private_linear | 0.83241374 | candidate |

## 跨数据集评测状态

- [g3-r0006 已完成结果](generalization/g3-r0006-completed/report.txt)：QK-Video、KuaiRand、TAAC-25、MerRec 的四次固定结构重训练全部完成；保留 DIN 和此前模型对照、逐任务退步及成本。它是中途模型，不是最终第一名。
- [g3-r0018 当前快照](generalization/g3-r0018-snapshot/report.txt)：最终第一名的四路结构迁移正在运行，尚不能报告最终泛化成绩。20 步校准、正式训练进度和原始协议已归档；状态仅代表归档期间。
- 两组均为各数据集原生词表/字段/任务头，从头训练；这不是把淘宝权重直接用于其他数据集的零样本迁移。任务平均 AUC 在各数据集内计算，不合并为跨数据集总分。

## 轨迹入口

- [iterations.csv](iterations.csv)、[iterations.json](iterations.json)：全部课题训练、角色、父模型、四任务验证指标和成本；未晋升或被否定的变体保留。
- [topic-incumbents.csv](topic-incumbents.csv)、[topic-decisions.json](topic-decisions.json)：逐轮保留模型、课题切换和合并；原始状态在 research/topics/gpu-N/state.json。
- research/topics/gpu-N/rounds/rXXXX/propose/proposal.json：前瞻问题、假设、相近方法、对照、知识采用与拒绝依据。
- 每轮 codex.jsonl、runtime/task/workspace 导出、knowledge-access/usage：当时上下文、工具调用与知识访问证据。
- research/topics/gpu-N/trials/TRIAL/：实际模型、配置、result 与 changes-from-parent.patch；最终前三代码在 research/final-assessment-top3/suite/。
- [calls.csv](calls.csv)：Agent 调用和 Token 用量；input 包含 cached，reasoning 是 output 的子集，不能重复相加。
- research/source/：冻结运行器、训练器、通用 Agent 指令与任务设计；research/config/、protocol.json、dataset-source-manifest.json：复现配置与数据来源。
- [REPRODUCE.md](REPRODUCE.md)：重建与评测入口；copy-sources.json 和 manifest.json：来源与逐文件完整性。

## 证据边界

使用固定 UniRank 官方切分与 official_offline 历史反馈语义，最近 100 次历史。原测试集曾用于此前模型，本轮测试属于补充离线评估；研究期间用户要求的中途补测另有记录，不作为研究 Agent 的选模反馈。未完成跨数据集评测不计作成功证据；未做多种子或参数/算力匹配消融的结论不能解释为稳定泛化或单机制因果结论。

原始绝对路径用于来源审计，迁移环境需重建路径。归档不含原始样本、词表大缓存、模型权重、逐样本预测、认证文件、环境目录或论文全文 HTML/PDF 缓存。运行中实验仅做只读快照；此次上传不会启动、停止或修改训练。

原始 CLI 流保留了 112 行非 JSON 诊断（含时间戳日志和回溯），包括工具补丁失败和模型列表刷新超时；见 [cli-diagnostics.json](cli-diagnostics.json)。这些是调用内部诊断，85 次 Agent 调用最终均成功；没有删除重试和错误文本。
