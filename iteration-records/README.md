# 迭代记录

按 campaign 单独归档，供人工审阅假设、实现、对照实验、归因和完整调用证据。

| 记录 | 范围 | 当前结论 |
| --- | --- | --- |
| [旧版 RecHarness＋修正版 NextItNet](amazon-nextitnet-recharness-original-20261031/README.md) | 2026-10-31，29 轮、107 次训练尝试、两次 jump | 验证 HR@10 0.5425 → 0.5737；损失 jump 保留、架构 jump 回滚；最终测试 HR@10 0.4990、NDCG@10 0.3126 |
| [Amazon NextItNet 20 轮迭代](amazon-nextitnet-v9-20-rounds/README.md) | 2026 年 10 月 10 日至 11 日，v9，第 1 至 20 轮 | 18 轮完成实验，3 次晋升；开发 HR@10 0.59475 → 0.59975；收尾审计未通过，最终测试尚无成绩 |
| [Amazon 四类别同预算迭代](amazon-aligned-20261022-27/README.md) | 2026 年 10 月 22 至 27 日；MLEvolve、MLE-STAR、Self-EvolveRec、RecHarness；459 条试验记录 | 9 次搜索完成、5 次中断；另 2 次仅准备/预检；含原预算单模型重评与去集成重跑 |
| [BERT4Rec 三方法完整实验](amazon-bert4rec-20261029-31/README.md) | 2026 年 10 月 29 至 31 日；RecHarness、Self-EvolveRec、MLE-STAR；221 个候选、2,133 次调用 | 三次搜索均完成；每项 43,200 GPU 秒；含逐数据集结果、源码差异、恢复证据及 token 用量 |
| [UniRank 固定模型跨数据集测试](unirank-generalization-20261108-09/README.md) | 2026年11月8日至9日，QK-Video、KuaiRand、TAAC-25、MerRec；候选与DIN共8组，17项任务对照 | 四数据集平均AUC均改善；TAAC转化、MerRec加购等三项任务AUC退步；单种子，报告参数量和实测成本 |
