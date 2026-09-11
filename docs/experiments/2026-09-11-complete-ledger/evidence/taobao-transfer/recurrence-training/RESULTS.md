# 循环训练目标对照

只改变训练深度分布或加入中间监督；默认PSGDKron、clip1、学习率、宽度和seed固定。

| 结构 | 固定两轮平均AUC | 循环平均AUC | 增量 |
|---|---:|---:|---:|
| middle_mixed123 | 0.827503 | 0.826473 | -0.001030 |
| middle_supervised12 | 0.827503 | 0.825485 | -0.002018 |

完整四任务AUC/logloss、显存和训练耗时见[结果](reports/summary.json)。[审计](reports/final-audit.json)。单seed结论，未评估测试集。两组验证均固定两轮；辅助监督重算前向，训练计算量不同。
